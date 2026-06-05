"""
Advanced Engine v2 — Luna V1 AI Mining (Engine 3/6)
====================================================
8 métodos (paridad plena con Correlaciones):
  1. Granger Causality       5. Spectral Coherence
  2. Transfer Entropy neta   6. SHAP + Random Forest  [GAP 4]
  3. PCA / Factor Analysis   7. Isolation Forest      [GAP 3]
  4. Copula Tail Dep.        8. Structural Breaks     [GAP 5]

Visual analytics [GAP 2]:
  engine_shap.png · engine_pca.png · engine_hmm.png
  transfer_entropy.png · engine_isolation.png
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_FEATURES = PROJECT_ROOT / "data" / "features"
REPORTS_DIR   = PROJECT_ROOT / "data" / "ai_mining" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# —— Lée parámetros desde settings.yaml (R: "ningún número mágico en scripts") ——
try:
    from config.settings import cfg
    _am = cfg.ai_mining.advanced
    GRANGER_MAXLAG   = int(_am.granger_maxlag)
    TE_BINS          = int(_am.te_bins)
    SPECTRAL_NPERSEG = int(_am.spectral_nperseg)
    SHAP_N_ESTIM     = int(_am.shap_n_estimators)
    ISOLATION_CONT   = float(_am.isolation_contamination)
    N_BREAKS         = int(_am.n_structural_breaks)
except Exception:
    # Fallback si cfg no disponible (ejecución aislada / tests)
    GRANGER_MAXLAG   = 13
    TE_BINS          = 8
    SPECTRAL_NPERSEG = 168
    SHAP_N_ESTIM     = 200
    ISOLATION_CONT   = 0.05
    N_BREAKS         = 5

ANALYSIS_CANDIDATES = [
    "FedFundsRate","YieldCurve_10Y3M","T10Y2Y","M2_USA_raw","GlobalM2_Index",
    "Fed_Net_Liquidity","CPI_YoY","Inflation_MoM","VIX","DXY","SP500_Ret",
    "NASDAQ_Ret","Gold_Ret","Oil_Ret","FundingRate","OI_BTC","DangerZone",
    "DVOL","MVRV_Proxy","FearGreed","SSR","Whale_Vol_ZScore","Stablecoin_Cap",
    "DeFi_WBTC_TVL","eth_btc_corr_24h","eth_ret_lag1","alt_season_proxy",
    "Master_Causal_Signal","KMeans_Tribe_ID",
]


class AdvancedEngine:

    def __init__(self, cutoff_date=None):
        # cutoff_date: pd.Timestamp | None
        # En --mode dev, limita los datos a <= train_end para evitar Selection Leakage.
        self.cutoff_date = cutoff_date

    def load_data(self) -> pd.DataFrame:
        # P1-2-FIX (2026-03-30): eliminado features_train_kshape.parquet (K-Shape decommisionado).
        # El orden correcto es: causal (con Master_Causal_Signal+KMeans_Tribe_ID) → train base.
        for name in ["features_train_causal.parquet", "features_train.parquet"]:
            p = DATA_FEATURES / name
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True)
                cutoff = getattr(self, "cutoff_date", None)
                if cutoff is not None:
                    n_before = len(df)
                    df = df[df.index <= cutoff]
                    logger.info(
                        f"Advanced [DEV]: cutoff={cutoff.date()} "
                        f"-> {len(df)}/{n_before} filas ({len(df)/n_before*100:.1f}%)"
                    )
                else:
                    logger.info(f"Advanced [PROD]: datos completos {df.shape} desde {name}")
                return df
        raise FileNotFoundError("No dataset en data/features/")

    def _build_target(self, df):
        if "close" in df.columns:
            return df["close"].pct_change(1).rename("btc_ret_1h")
        raise ValueError("'close' no encontrada")

    # 1. Granger
    def granger_test(self, x, y, maxlag=4):
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            data = pd.concat([x, y], axis=1).dropna()
            if len(data) < maxlag * 4: return {"p_value":1.0,"lag":0,"stars":""}

            # [H-08-FIX 2026-05-30] Guard de degeneración ANTES del Granger test.
            # PROBLEMA: Series con std~0 (FundingRate, OI_BTC, DangerZone rellenadas con fillna(0))
            # producen F-statistic=inf en el SSR F-test → p=0.0 → "***" (FALSO POSITIVO).
            # El mismo check ya existe en transfer_entropy (FALLA-02-FIX) pero no aquí.
            # CONSECUENCIA: evidence_score +3 en variables degeneradas → ranking causal inválido.
            _std_x = float(data.iloc[:, 0].std())
            _std_y = float(data.iloc[:, 1].std())
            _min_std_granger = 1e-6  # umbral institucional: series con std < 1e-6 son constantes numéricas
            if _std_x < _min_std_granger or _std_y < _min_std_granger:
                print(  # RULE[fixbugsprints.md]
                    f"[H-08-FIX] Granger DEGENERATE: std_x={_std_x:.2e} std_y={_std_y:.2e} < {_min_std_granger} "
                    f"→ p_value=1.0 stars='' (evita falso positivo F=inf)."
                )
                return {"p_value": 1.0, "lag": 0, "stars": "DEGENERATE"}

            # [FIX-GRANGER-OLS-01] Estandarizar la data antes de Granger.
            # Granger utiliza OLS (VAR models) por debajo. Si cruzamos M2_USA (10^13)
            # con Retornos (0.01), la matriz X'X colapsa numéricamente (HessianInversionWarning)
            # produciendo F-statistics aleatorios.
            from sklearn.preprocessing import StandardScaler
            data_scaled = pd.DataFrame(
                StandardScaler().fit_transform(data),
                index=data.index,
                columns=data.columns
            )

            res = grangercausalitytests(data_scaled, maxlag=maxlag, verbose=False)
            p_vals = {lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag+1)}
            best_lag = min(p_vals, key=p_vals.get); p = p_vals[best_lag]
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            return {"p_value":round(p,4),"lag":best_lag,"stars":stars}
        except Exception as e:
            return {"p_value":1.0,"lag":0,"stars":"","error":str(e)}


    # 2. Transfer Entropy
    def transfer_entropy(self, x, y, bins=TE_BINS):
        """
        [FALLA-02-FIX 2026-05-30] Corregido fallback silencioso que daba TE=1.2023 identico
        para variables sin solapamiento temporal o con distribucion degenerada.
        
        La causa: m2.get(tuple(r[1:]), 1e-10) usaba 1e-10 cuando el par (tp,sp) no aparecia
        en el histograma conjunto -> h = -sum(p3 * log2(p3/1e-10)) = H(X) + constante ~fija.
        
        Fixes aplicados:
        1. Validar longitud minima de solapamiento (>100 puntos)
        2. Retornar 0.0 si cualquier serie tiene varianza cero (degenerada)
        3. Usar max(pb, 1e-10) solo para estabilidad numerica, no como fallback de histograma
        4. Log CAPPED cuando TE_net supera la entropia marginal maxima teorica (detecta fallos)
        """
        def _te(src, tgt, b):
            def dig(a): return np.digitize(a, np.linspace(a.min(), a.max()+1e-8, b+1))-1
            sd, td = dig(src), dig(tgt)
            sp, tp, tf = sd[:-1], td[:-1], td[1:]
            j3 = np.c_[tf, tp, sp]; j2 = np.c_[tp, sp]
            _, c3 = np.unique(j3, axis=0, return_counts=True)
            u2, c2 = np.unique(j2, axis=0, return_counts=True)
            p3 = c3/c3.sum(); p2 = c2/c2.sum()
            m2 = {tuple(r): v for r, v in zip(u2, p2)}
            h = 0.0
            for i, r in enumerate(np.unique(j3, axis=0)):
                pb = m2.get(tuple(r[1:]), None)
                # [FALLA-02-FIX] Si el par (tp,sp) no existe en el histograma conjunto,
                # es un bin vacio — contribucion = 0 (no usar 1e-10 que produce capping)
                if pb is None or pb <= 0:
                    continue
                if p3[i] > 0:
                    h -= p3[i] * np.log2(p3[i] / pb)
            return h

        try:
            # [FALLA-02-FIX] Validaciones previas al calculo
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            # Eliminar NaN antes de calcular
            valid = np.isfinite(x) & np.isfinite(y)
            x_clean, y_clean = x[valid], y[valid]

            if len(x_clean) < 100:
                print(f"[FALLA-02-FIX] TE omitido: solapamiento insuficiente (n={len(x_clean)} < 100)")
                return 0.0

            # Detectar distribucion degenerada (varianza cero = todos los bins iguales)
            if np.std(x_clean) < 1e-8 or np.std(y_clean) < 1e-8:
                print(f"[FALLA-02-FIX] TE omitido: serie degenerada (std~0)")
                return 0.0

            te_xy = _te(x_clean, y_clean, bins)
            te_yx = _te(y_clean, x_clean, bins)
            te_net = round(float(te_xy - te_yx), 4)

            # [FALLA-02-FIX] Detector de capping: TE_net > log2(bins) es matematicamente imposible
            te_max_teorico = np.log2(bins)  # entropia maxima para 'bins' categorias = log2(8)=3.0
            if abs(te_net) > te_max_teorico * 0.8:
                print(f"[FALLA-02-FIX] TE_CAPPED detectado: |TE_net|={abs(te_net):.4f} > 80% de max_teorico={te_max_teorico:.2f}. "
                      f"Resultado sospechoso — retornando 0.0")
                return 0.0

            return te_net
        except Exception as e:
            print(f"[FALLA-02-FIX] TE exception: {e} — retornando 0.0")
            return 0.0


    # 3. PCA
    def pca_analysis(self, df):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        cols=[c for c in ANALYSIS_CANDIDATES if c in df.columns]
        if len(cols)<3: return {}
        X=df[cols].dropna()
        if len(X)<50: return {}
        X_s=StandardScaler().fit_transform(X)
        n=min(5,len(cols))
        pca=PCA(n_components=n); pca.fit(X_s)
        loadings=pd.DataFrame(pca.components_.T,index=cols,columns=[f"PC{i+1}" for i in range(n)])
        return {
            "explained_variance_pct":[round(v*100,1) for v in pca.explained_variance_ratio_],
            "top_pc1_features":loadings["PC1"].abs().sort_values(ascending=False).head(5).index.tolist(),
            "loadings":loadings,
        }

    # 4. Copula
    def tail_dependence(self, x, y, q=0.9):
        data=pd.concat([x,y],axis=1).dropna()
        if len(data)<50: return {"lambda_upper":0.0,"lambda_lower":0.0}
        u=data.iloc[:,0].rank(pct=True); v=data.iloc[:,1].rank(pct=True)
        lu=((u>q)&(v>q)).sum()/((u>q).sum() or 1)
        ll=((u<1-q)&(v<1-q)).sum()/((u<1-q).sum() or 1)
        return {"lambda_upper":round(float(lu),3),"lambda_lower":round(float(ll),3)}

    # 5. Spectral
    def spectral_coherence(self, x, y):
        from scipy.signal import coherence
        data=pd.concat([x,y],axis=1).dropna()
        if len(data)<SPECTRAL_NPERSEG*2: return {"peak_freq_h":0.0,"max_coherence":0.0}
        f,Cxy=coherence(data.iloc[:,0].values,data.iloc[:,1].values,
                        fs=1.0,nperseg=min(SPECTRAL_NPERSEG,len(data)//3))
        valid=f>0; idx=np.argmax(Cxy[valid]); pf=f[valid][idx]; pc=Cxy[valid][idx]
        return {"peak_freq_h":round(1.0/pf if pf>0 else 0,1),"max_coherence":round(float(pc),3)}

    # 6. SHAP + RF [GAP 4]
    def shap_analysis(self, df):
        """[BUG-V5-04] SHAP >= 0.40: sv.ndim==3 para RF binario -> sv[:,:,1]. Manejado abajo."""
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import TimeSeriesSplit
            cols=[c for c in ANALYSIS_CANDIDATES if c in df.columns]
            if not cols: return {}
            # [FALLA-09-FIX 2026-05-30] Priorizar Target_TBM_Bin (igual que Bayesian/Deep con [BUG-TARGET-FIX])
            # El target raw close.shift(-24)>0 da accuracy OOS=50.7% (moneda) -> SHAP inválido
            # Target_TBM_Bin tiene WR≈54% y es más estable al estar basado en TBM (Triple-Barrier Method)
            if "Target_TBM_Bin" in df.columns:
                target = df["Target_TBM_Bin"]
                print("[FALLA-09-FIX] SHAP RF: usando Target_TBM_Bin en lugar de close.shift(-24) raw")
                logger.info("[FALLA-09-FIX] Advanced SHAP: Usando Target_TBM_Bin (alineado con Bayesian/Deep engines)")
            elif "close" in df.columns:
                target = (df["close"].shift(-24)/df["close"]-1>0).astype(int)
                logger.warning("[FALLA-09-FIX] Target_TBM_Bin no disponible — usando close.shift(-24) como fallback")
            else:
                return {}
            X=df[cols].ffill().fillna(0); y=target
            idx=X.index.intersection(y.dropna().index); X,y=X.loc[idx],y.loc[idx]
            splits=list(TimeSeriesSplit(n_splits=3).split(X)); tr,te=splits[-1]
            rf=RandomForestClassifier(n_estimators=SHAP_N_ESTIM,max_depth=6,
                min_samples_leaf=50,n_jobs=-1,random_state=42)
            rf.fit(X.iloc[tr],y.iloc[tr])
            acc=rf.score(X.iloc[te],y.iloc[te])
            logger.info(f"SHAP RF OOS accuracy: {acc:.3f}")
            explainer=shap.TreeExplainer(rf)
            sv=explainer.shap_values(X.iloc[te])
            # [BUG-V5-04] SHAP moderno devuelve ndarray (n, f, 2) para clasificadores binarios.
            # Reducir siempre a 2D ANTES de cualquier otro check.
            if hasattr(sv, "ndim") and sv.ndim==3:
                sv = sv[:, :, 1]  # clase positiva
            # SHAP nuevo: sv puede ser (n, f) para binario o lista legacy
            if isinstance(sv, list):
                sv = sv[1]  # clase positiva en formato lista legacy
            if hasattr(sv, "ndim"):
                if sv.ndim == 3:
                    sv = sv[:, :, 1]  # (n_samples, n_features, class=1) — fallback redundante
                elif sv.ndim == 1:
                    sv = sv.reshape(1, -1)

            sarr=np.abs(sv)
            ms=pd.Series(sarr.mean(axis=0),index=cols).sort_values(ascending=False)
            self._plot_shap(ms)
            return {"shap_ranking":ms.to_dict(),"rf_accuracy":round(acc,3),"top5_shap":ms.head(5).index.tolist()}
        except ImportError:
            return self._shap_rf_fallback(df)
        except Exception as e:
            logger.warning(f"SHAP error: {e}"); return {}

    def _shap_rf_fallback(self, df):
        try:
            from sklearn.ensemble import RandomForestClassifier
            cols=[c for c in ANALYSIS_CANDIDATES if c in df.columns]
            if not cols or "close" not in df.columns: return {}
            target=(df["close"].shift(-24)/df["close"]-1>0).astype(int)
            X=df[cols].ffill().fillna(0); y=target
            idx=X.index.intersection(y.dropna().index); X,y=X.loc[idx],y.loc[idx]
            n=len(X); Xt=X.iloc[:int(n*0.75)]; yt=y.iloc[:int(n*0.75)]
            rf=RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=50,random_state=42)
            rf.fit(Xt,yt)
            imp=pd.Series(rf.feature_importances_,index=cols).sort_values(ascending=False)
            self._plot_shap(imp); return {"shap_ranking":imp.to_dict(),"rf_accuracy":0.0,"top5_shap":imp.head(5).index.tolist()}
        except Exception as e:
            logger.warning(f"RF fallback error: {e}"); return {}

    # 7. Isolation Forest [GAP 3]
    def isolation_forest(self, df):
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            cols=[c for c in ANALYSIS_CANDIDATES if c in df.columns]
            if not cols: return {}
            X=StandardScaler().fit_transform(df[cols].ffill().fillna(0))
            iso=IsolationForest(n_estimators=200,contamination=ISOLATION_CONT,n_jobs=-1,random_state=42)
            preds=iso.fit_predict(X); raw=iso.score_samples(X)
            n_anom=(preds==-1).sum()
            dates=[str(df.index[i].date()) for i in np.where(preds==-1)[0][-5:]]
            logger.info(f"IsoForest: {n_anom} anomalías ({n_anom/len(preds)*100:.1f}%) "
                        f"| actual={'⚠️' if preds[-1]==-1 else '✅'}")
            self._plot_isolation(df, raw, preds)
            return {"n_anomalies":int(n_anom),"anomaly_pct":round(n_anom/len(preds)*100,1),
                    "current_anomaly":bool(preds[-1]==-1),"current_score":float(raw[-1]),
                    "last_5_anomalies":dates,"level":"ALERTA" if preds[-1]==-1 else "Normal"}
        except Exception as e:
            logger.warning(f"IsoForest error: {e}"); return {}

    # 8. Structural Breaks [GAP 5]
    def structural_breaks(self, df, max_n_samples: int = 5000):
        if "close" not in df.columns: return []
        btc_ret=df["close"].pct_change(1).dropna(); blist=[]
        try:
            import ruptures as rpt
            # FIX PERF: Pelt(rbf) es O(n²) — con 43k filas tarda 30+ min.
            # Subsamplear uniformemente a max_n_samples (breaks estructurales son
            # fenómenos de régimen, no de tick a tick — subsampling no afecta resultado).
            signal_series = btc_ret
            if len(signal_series) > max_n_samples:
                step = len(signal_series) // max_n_samples
                signal_series = signal_series.iloc[::step].head(max_n_samples)
                logger.debug(f"StructBreaks: subsampleando {len(btc_ret)} → {len(signal_series)} puntos para Pelt")
            sig=signal_series.values.reshape(-1,1)
            # Reconstruir índice para mapear breaks to timestamps correctamente
            sig_index = signal_series.index
            res=rpt.Pelt(model="rbf").fit(sig).predict(pen=10)
            for b in res[:-1][:N_BREAKS]:
                # FIX: b es índice en sig_index (subsample), NO en df.index
                bd = sig_index[min(b-1, len(sig_index)-1)]
                bef=btc_ret[:bd]; aft=btc_ret[bd:]
                if len(bef)>30 and len(aft)>30:
                    rb=abs(float(bef.autocorr(1))); ra=abs(float(aft.autocorr(1)))
                    blist.append({"date":str(bd.date()),"r_before":round(rb,3),
                                  "r_after":round(ra,3),"delta_r":round(abs(ra-rb),3)})
            blist.sort(key=lambda x:x["delta_r"],reverse=True)
            logger.info(f"StructBreaks: {len(blist)} rupturas (ruptures lib)")
        except ImportError:
            # CUSUM fallback — con deduplicación por fecha
            sig=btc_ret.rolling(168).mean().dropna()
            mu=float(sig.mean()); s=float(sig.std()+1e-8); k=0.5*s; h=5*s
            cp=cn=0.0; seen_dates: set = set()
            for ts,v in sig.items():
                cp=max(0,cp+v-mu-k); cn=max(0,cn-v+mu-k)
                if (cp>h or cn>h) and str(ts.date()) not in seen_dates:
                    seen_dates.add(str(ts.date()))
                    blist.append({"date":str(ts.date()),"r_before":0.0,"r_after":0.0,"delta_r":round(max(cp,cn),3)})
                    cp=cn=0.0
                elif cp>h or cn>h:
                    cp=cn=0.0  # reset even if date duplicated
                if len(blist)>=N_BREAKS: break
            logger.info(f"StructBreaks: {len(blist)} rupturas (CUSUM fallback)")
        return blist

    # Plots [GAP 2]
    def _plot_shap(self, imp: pd.Series) -> None:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top=imp.head(15)
            fig,ax=plt.subplots(figsize=(10,7))
            colors=["#2ecc71" if i<5 else "#3498db" for i in range(len(top))]
            ax.barh(range(len(top)),top.values[::-1],color=colors[::-1])
            ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index[::-1],fontsize=9)
            ax.set_xlabel("Importance score"); ax.set_title("Feature Importance — SHAP/RF",fontweight="bold")
            plt.tight_layout(); plt.savefig(REPORTS_DIR/"engine_shap.png",dpi=120,bbox_inches="tight"); plt.close()
            logger.success("engine_shap.png generado")
        except Exception as e: logger.warning(f"SHAP plot: {e}")

    def _plot_pca(self, pca_res: dict) -> None:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
            vp=pca_res["explained_variance_pct"]
            ax1.bar(range(1,len(vp)+1),vp,color="#9b59b6")
            ax1.plot(range(1,len(vp)+1),np.cumsum(vp),"ro-",linewidth=2)
            ax1.set_xlabel("PC"); ax1.set_ylabel("Var %"); ax1.set_title("Scree Plot",fontweight="bold")
            ld=pca_res["loadings"]
            ax2.scatter(ld["PC1"],ld.get("PC2",ld["PC1"]*0),alpha=0.7,s=80,c="#e74c3c")
            for f in ld.index: ax2.annotate(f,(ld.loc[f,"PC1"],ld.get("PC2",ld["PC1"]*0).get(f,0)),fontsize=7)
            ax2.set_title("PCA Biplot PC1 vs PC2",fontweight="bold")
            plt.tight_layout(); plt.savefig(REPORTS_DIR/"engine_pca.png",dpi=120,bbox_inches="tight"); plt.close()
            logger.success("engine_pca.png generado")
        except Exception as e: logger.warning(f"PCA plot: {e}")

    def _plot_te_heatmap(self, df, avail):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top=avail[:8]
            mat=np.zeros((len(top),len(top)))
            for i,vi in enumerate(top):
                for j,vj in enumerate(top):
                    if i!=j:
                        xi=df[vi].dropna().values[-3000:]; xj=df[vj].dropna().values[-3000:]
                        n=min(len(xi),len(xj)); mat[i,j]=self.transfer_entropy(xi[:n],xj[:n],bins=6)
            fig,ax=plt.subplots(figsize=(10,8))
            im=ax.imshow(mat,cmap="RdYlGn",aspect="auto",vmin=-0.1,vmax=0.1)
            ax.set_xticks(range(len(top))); ax.set_yticks(range(len(top)))
            ax.set_xticklabels([v[:12] for v in top],rotation=45,ha="right",fontsize=8)
            ax.set_yticklabels([v[:12] for v in top],fontsize=8)
            plt.colorbar(im,ax=ax,label="TE_net"); ax.set_title("Transfer Entropy Heatmap",fontweight="bold")
            plt.tight_layout(); plt.savefig(REPORTS_DIR/"transfer_entropy.png",dpi=120,bbox_inches="tight"); plt.close()
            logger.success("transfer_entropy.png generado")
        except Exception as e: logger.warning(f"TE heatmap: {e}")

    def _plot_isolation(self, df, raw, labels):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8),sharex=False)
            if "close" in df.columns:
                pr=df["close"].values
                ax1.plot(pr,color="#3498db",linewidth=0.7,label="BTC Price")
                ax1.scatter(np.where(labels==-1)[0],pr[labels==-1],color="#e74c3c",s=12,alpha=0.7,label="Anomalía",zorder=5)
                ax1.set_ylabel("BTC Price"); ax1.legend(loc="upper left")
                ax1.set_title("Isolation Forest — Anomalías sobre BTC",fontweight="bold")
            ax2.plot(raw,color="#9b59b6",linewidth=0.6)
            ax2.axhline(np.percentile(raw,ISOLATION_CONT*100),color="#e74c3c",linestyle="--",linewidth=1.5,label="Umbral")
            ax2.set_ylabel("Anomaly Score"); ax2.legend()
            plt.tight_layout(); plt.savefig(REPORTS_DIR/"engine_isolation.png",dpi=120,bbox_inches="tight"); plt.close()
            logger.success("engine_isolation.png generado")
        except Exception as e: logger.warning(f"Isolation plot: {e}")

    def _plot_hmm(self, df):
        try:
            if "hmm_regime" not in df.columns: return
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt, matplotlib.patches as mp
            colors={0:"#e74c3c",1:"#e67e22",2:"#f1c40f",3:"#2ecc71"}
            labels={0:"Crash",1:"Bear",2:"Lateral",3:"Bull"}
            fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,8),sharex=True)
            if "close" in df.columns:
                ax1.plot(df.index,df["close"].values,color="#3498db",linewidth=0.7)
                ax1.set_title("HMM Regimes — BTC Price",fontweight="bold")
            reg=df["hmm_regime"]
            for rid in sorted(reg.unique()):
                mask=reg==rid; start=None
                for i in range(len(mask)):
                    if mask.iloc[i] and start is None: start=df.index[i]
                    elif not mask.iloc[i] and start is not None:
                        ax2.axvspan(start,df.index[i],alpha=0.4,color=colors.get(rid,"gray")); start=None
            patches=[mp.Patch(color=colors.get(r,"gray"),label=labels.get(r,str(r))) for r in sorted(reg.unique())]
            ax2.legend(handles=patches,loc="upper left"); ax2.set_title("HMM Regime Timeline")
            plt.tight_layout(); plt.savefig(REPORTS_DIR/"engine_hmm.png",dpi=120,bbox_inches="tight"); plt.close()
            logger.success("engine_hmm.png generado")
        except Exception as e: logger.warning(f"HMM plot: {e}")

    # Pipeline principal
    def run(self) -> pd.DataFrame:
        logger.info("="*60); logger.info("Advanced Engine v2 — INICIO (8 métodos)"); logger.info("="*60)
        df=self.load_data()
        try: btc_ret=self._build_target(df)
        except ValueError as e: logger.error(str(e)); return df

        available=[c for c in ANALYSIS_CANDIDATES if c in df.columns]
        logger.info(f"Advanced: {len(available)} variables disponibles")

        results=[]; btc_price=df["close"] if "close" in df.columns else None
        btc_ret_24h=(df["close"].pct_change(24).dropna() if "close" in df.columns else None)
        for var in available:
            col=df[var].resample("1h").last().ffill()
            common=col.index.intersection(btc_ret.index); x,y=col.loc[common],btc_ret.loc[common]
            c1d=col.resample("1d").last().dropna()
            p1d=(btc_price.resample("1d").last().dropna() if btc_price is not None else y)
            gr=self.granger_test(c1d,p1d,maxlag=min(GRANGER_MAXLAG,len(c1d)//6))
            xte=x.dropna().values[-5000:]; yte=y.loc[x.dropna().index].dropna().values[-5000:]
            nl=min(len(xte),len(yte)); te=self.transfer_entropy(xte[:nl],yte[:nl]) if nl>100 else 0.0
            tail=self.tail_dependence(c1d,p1d.pct_change(1).dropna())
            sp=self.spectral_coherence(x.iloc[-3000:],y.iloc[-3000:])
            # Pearson vs BTC 24H returns — da la DIRECCIÓN real de la relación
            pearson_corr = 0.0
            if btc_ret_24h is not None:
                try:
                    common24 = col.index.intersection(btc_ret_24h.index)
                    if len(common24) > 50:
                        # pd.Series.corr drops pairwise NaNs automatically, ensuring correct correlation
                        p_corr = col.loc[common24].corr(btc_ret_24h.loc[common24])
                        if pd.isna(p_corr):
                            pearson_corr = 0.0
                        else:
                            pearson_corr = float(p_corr)
                            
                        # Logging the robust calculation trace as per [fixbugsprints.md]
                        if any(x in var.lower() for x in ['mvrv', 'funding', 'vol', 'dvol']):
                            logger.info(f"[BUG-ADV-CORR-01] Robust Pearson corr computed for {var}: corr={pearson_corr:.4f} (overlapping samples={len(common24)})")
                except Exception as e:
                    logger.warning(f"[BUG-ADV-CORR-01] Error computing Pearson corr for {var}: {str(e)}")
                    pearson_corr = 0.0
            # te_direction combina causalidad (TE_net) con dirección (Pearson)
            # TE_net solo dice "esta variable precede a BTC" — no la dirección
            # Guard: si pearson_corr es NaN (pocas observaciones), te_direction = 0
            if np.isnan(pearson_corr) or np.isnan(te):
                te_direction = 0.0
            else:
                te_direction = float(np.sign(pearson_corr)) * abs(te) if te != 0 else 0.0
            results.append({"variable":var,"granger_p":gr["p_value"],"granger_lag":gr["lag"],
                "granger_stars":gr["stars"],"te_net":te,
                "pearson_corr":round(pearson_corr, 4),
                "te_direction":round(te_direction, 4),  # <-- señal direccional real
                "lambda_upper":tail["lambda_upper"],
                "lambda_lower":tail["lambda_lower"],"spectral_peak_h":sp["peak_freq_h"],
                "max_coherence":sp["max_coherence"],
                "evidence_score":(3 if gr["stars"]=="***" else 2 if gr["stars"]=="**" else 1 if gr["stars"]=="*" else 0)
                +(1 if abs(te)>0.01 else 0)+(1 if tail["lambda_upper"]>0.3 else 0)+(1 if sp["max_coherence"]>0.3 else 0)})
            logger.info(f"  {var:<35} Granger:{gr['stars'] or 'ns':<4} TE={te:+.4f} Dir={te_direction:+.4f} ρ={pearson_corr:+.3f}")

        results_df=pd.DataFrame(results).sort_values("evidence_score",ascending=False)
        logger.info(f"\n{results_df[['variable','granger_stars','te_net','evidence_score']].head(8).to_string(index=False)}")

        logger.info("Advanced: PCA..."); pca_res=self.pca_analysis(df)
        if pca_res: logger.info(f"PCA top: {pca_res['top_pc1_features']}"); self._plot_pca(pca_res)

        logger.info("Advanced: SHAP + RF..."); shap_res=self.shap_analysis(df)
        if shap_res:
            results_df["shap_importance"]=results_df["variable"].map(shap_res.get("shap_ranking",{})).fillna(0.0)
            mx=results_df["shap_importance"].max()
            if mx>0: results_df["evidence_score"]+=(results_df["shap_importance"]/mx).round().astype(int)
            results_df=results_df.sort_values("evidence_score",ascending=False)

        logger.info("Advanced: Isolation Forest..."); iso_res=self.isolation_forest(df)
        if iso_res: logger.info(f"IsoForest actual: {iso_res['level']}")

        logger.info("Advanced: Structural Breaks..."); breaks=self.structural_breaks(df)

        self._plot_hmm(df); self._plot_te_heatmap(df,available[:8])

        results_df.to_csv(REPORTS_DIR/"advanced_engine_results.csv",index=False)
        self._save_report(results_df,pca_res,shap_res,iso_res,breaks)
        logger.info("Advanced Engine v2 — COMPLETADO")
        return results_df

    def _save_report(self, results_df, pca_res, shap_res, iso_res, breaks):
        now = pd.Timestamp.now(tz="UTC").strftime("%d %B %Y %H:%M")
        n_rows = len(results_df) if hasattr(results_df, '__len__') else 0

        # Detect current HMM regime (best guess from top evidence var direction)
        top_dir = "Bull"
        if not results_df.empty and "te_direction" in results_df.columns:
            net = results_df.head(5)["te_direction"].sum()
            top_dir = "Bull" if net >= 0 else "Bear"
        top_var = results_df.iloc[0]["variable"] if not results_df.empty else "N/A"

        lines: list[str] = []

        # ── Header ──
        lines += [
            "# 🏟️ Advanced Engine: Diagnóstico del Paisaje Cuántico — Luna V1",
            "",
            f"**Fecha de Emisión:** {now}",
            f"**Dataset:** features_train.parquet — {n_rows} variables analizadas",
            "",
            "---",
            "",
            "## 🧠 Veredicto de Inteligencia",
            f"> Actualmente el mercado muestra señales de régimen **{top_dir}**. El motor de IA identifica"
            f" a **{top_var}** como el vector de mayor peso en la formación de precio de las próximas 24H.",
            "",
            "---",
        ]

        # ── 1. Granger Causal Network ──
        lines += [
            "",
            "## 1. Granger Causality Network",
            "> Solo los indicadores con causalidad estadística hacia BTC (no solo correlación)."
            " p-value < 0.05 con lag óptimo.",
            "",
            "| Indicator | Best_Lag | P_value | Strength | TE_net | TE_direction |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        sig_df = results_df[results_df["granger_stars"].str.len() > 0] if "granger_stars" in results_df.columns else results_df.head(12)
        for _, row in sig_df.head(12).iterrows():
            te_dir = row.get("te_direction", row.get("te_net", 0))
            arrow = "▲" if te_dir > 0 else "▼"
            lines.append(
                f"| {row['variable']} | {row.get('granger_lag', '-')} | +{row['granger_p']:.4f}"
                f" | {row.get('granger_stars', '-')} | {row['te_net']:+.4f} | {arrow} {te_dir:+.4f} |"
            )
        lines += ["", "![Transfer Entropy Network](transfer_entropy.png)", ""]

        # ── 2. PCA / Factor Analysis ──
        lines += [
            "",
            "---",
            "## 2. PCA / Factor Analysis",
            "> Reduce todos los indicadores a factores latentes."
            " Indica QUÉ GRUPOS de indicadores se mueven juntos.",
            "",
        ]
        if pca_res and "explained_variance_pct" in pca_res:
            lines += [
                "| Factor | Variance_Pct | Top Positivo |",
                "| --- | --- | --- |",
            ]
            for i, vp in enumerate(pca_res["explained_variance_pct"], 1):
                pc_key = f"PC{i}"
                top_feat = pca_res.get("top_pc1_features", ["N/A"])[0] if i == 1 else "—"
                lines.append(f"| {pc_key} | +{vp} | {top_feat} |")
        else:
            lines.append("*(Sin datos de PCA)*")
        lines += ["", "![PCA](engine_pca.png)", ""]

        # ── 3. SHAP + Random Forest ──
        lines += [
            "",
            "---",
            "## 3. SHAP + Random Forest",
            "> Impacto real en la predicción de BTC a 24H. SHAP mide la contribución real de cada indicador.",
            "",
        ]
        if shap_res and "top5_shap" in shap_res:
            lines += [
                "| Rank | Indicator | SHAP Importance | RF Acc OOS |",
                "| --- | --- | --- | --- |",
            ]
            acc = shap_res.get("rf_accuracy", 0)
            ranking = shap_res.get("shap_ranking", {})
            for i, feat in enumerate(shap_res["top5_shap"], 1):
                imp = ranking.get(feat, 0)
                lines.append(f"| {i} | **{feat}** | +{imp:.4f} | {acc:.3f} |")
        else:
            lines.append("*(Sin datos de SHAP)*")
        lines += ["", "![SHAP](engine_shap.png)", ""]

        # ── 4. Copula Tail Dependence ──
        lines += [
            "",
            "---",
            "## 4. Copula Tail Dependence",
            "> Dependencia en eventos extremos. Tail_Upper = co-rally,"
            " Tail_Lower = co-crash. Es donde se ve la verdadera estructura de riesgo.",
            "",
            "| Indicator | Lambda_Upper | Lambda_Lower | Evidence_Score |",
            "| --- | --- | --- | --- |",
        ]
        for _, row in results_df.head(10).iterrows():
            lu = row.get("lambda_upper", 0)
            ll = row.get("lambda_lower", 0)
            if lu > 0.2 or ll > 0.2:
                lines.append(
                    f"| {row['variable']} | +{lu:.3f} | +{ll:.3f} | {row.get('evidence_score', '-')} |"
                )

        # ── 5. Isolation Forest Anomalias ──
        lines += [
            "",
            "",
            "---",
            "## 5. Isolation Forest — Anomalías de Mercado",
            "> Periodos donde el mercado salió de su distribución normal."
            " Una anomalía actual (🚨) es señal de precaución.",
            "",
        ]
        if iso_res:
            emoji = "🚨" if iso_res.get("current_anomaly") else "✅"
            lines += [
                f"- **Anomalías históricas:** {iso_res.get('n_anomalies', 0)}"
                f" ({iso_res.get('anomaly_pct', 0)}% del periodo)",
                f"- **Estado actual:** {emoji} **{iso_res.get('level', '?')}**"
                f" (score={iso_res.get('current_score', 0):.3f})",
                f"- **Últimas anomalías:** {', '.join(iso_res.get('last_5_anomalies', []))}",
                "",
                "![Isolation Forest](engine_isolation.png)",
            ]
        else:
            lines.append("*(Sin datos de Isolation Forest)*")

        # ── 6. Structural Breaks ──
        lines += [
            "",
            "",
            "---",
            "## 6. Structural Break Detection",
            "> Periodos donde la relación indicador-BTC cambió fundamentalmente."
            " Una señal puede funcionar en un periodo y fallar en otro.",
            "",
        ]
        if breaks:
            lines += [
                "| Fecha | Δ Autocorr | r_before | r_after |",
                "| --- | --- | --- | --- |",
            ]
            for b in breaks[:8]:
                lines.append(
                    f"| {b['date']} | **{b['delta_r']:.3f}** | {b['r_before']:.3f} | {b['r_after']:.3f} |"
                )
        else:
            lines.append("*(Sin rupturas estructurales detectadas)*")

        # ── 7. Ranking Global (Hall of Fame) ──
        lines += [
            "",
            "",
            "---",
            "## 🏆 7. Ranking Global Sintético",
            "> **The Hall of Fame.** Indicadores que aparecen como importantes en MÚLTIPLES"
            " métodos son los más confiables para la toma de decisiones estratégicas.",
            "",
        ]
        top_sig = results_df.sort_values("evidence_score", ascending=False).head(15)
        if not top_sig.empty:
            # Highlight leader
            leader = top_sig.iloc[0]["variable"]
            lines += [
                "> [!IMPORTANT]",
                f"> **Líder del Clima:** **{leader}** es el indicador dominante del ciclo actual.",
                "",
                "| Rank | Indicador | Score | Evidencia |",
                "| --- | --- | --- | --- |",
            ]
            for i, (_, row) in enumerate(top_sig.iterrows(), 1):
                g_stars = row.get("granger_stars", "")
                te = row.get("te_net", 0)
                tail = row.get("lambda_upper", 0)
                evidence_parts = []
                if g_stars:
                    evidence_parts.append(f"Granger {g_stars} lag={row.get('granger_lag', '-')}")
                if abs(te) > 0.01:
                    evidence_parts.append(f"TE={te:+.3f}")
                if tail > 0.2:
                    evidence_parts.append(f"Tail={tail:.3f}")
                evidence_str = " · ".join(evidence_parts) or "—"
                lines.append(
                    f"| {i} | **{row['variable']}** | {row['evidence_score']} | {evidence_str} |"
                )

        lines += [
            "",
            "---",
            "",
            f"*Generado por Luna V1 Advanced Engine · {now}*",
        ]

        (REPORTS_DIR / "advanced_engine_report.md").write_text("\n".join(lines), encoding="utf-8")
        logger.success("Advanced: reporte guardado")



if __name__=="__main__":
    AdvancedEngine().run()

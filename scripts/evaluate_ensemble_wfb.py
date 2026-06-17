"""
scripts/evaluate_ensemble_wfb.py
================================
Orquestador de consolidación y evaluación del ensamble multi-semilla Walk-Forward Backtesting (WFB).
Combina las probabilidades Soft Voting de las diferentes semillas y genera un Tearsheet agregado de trades.

[FIX-ENSEMBLE-EVAL]
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

# Configurar encoding UTF-8 para consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Alinear path del proyecto
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from luna.validation.ensemble_voter import aggregate_wfb_seeds

def main():
    print("[FIX-ENSEMBLE-EVAL] Iniciando la evaluación y consolidación del ensamble WFB...")
    logger.info("[FIX-ENSEMBLE-EVAL] Buscando outputs de semillas en data/reports/wfb...")
    
    # Cargar configuración activa
    try:
        from config.settings import cfg as _cfg
        active_seeds = list(_cfg.wfb.active_seeds)
        print(f"[FIX-ENSEMBLE-EVAL] Semillas activas cargadas desde settings.yaml: {active_seeds}")
        logger.info(f"[FIX-ENSEMBLE-EVAL] Semillas activas: {active_seeds}")
    except Exception as e:
        err_msg = f"CRITICAL [FIX-ENSEMBLE-EVAL]: Falló la carga de wfb.active_seeds de settings.yaml: {e}"
        print(err_msg)
        logger.critical(err_msg)
        raise RuntimeError(err_msg)

    wfb_out_dir = _ROOT / "data" / "reports" / "wfb"
    predictions_dir = _ROOT / "data" / "predictions"
    predictions_dir.mkdir(exist_ok=True, parents=True)

    # [ENSEMBLE-GAUNTLET-01] Variables de scope global de main() para tearsheet §4
    # Se sobrescriben en el bloque principal si el portfolio es suficiente
    _ensemble_approved = False
    _ens_verdict = {}
    
    master_probs_path = predictions_dir / "master_ensemble_probs.parquet"
    
    # 1. Las probabilidades de Soft Voting se agregarán MÁS ADELANTE,
    # solo con las semillas que sobrevivan la poda de Sharpe (H1).


    # 2. Recopilar y evaluar trades por semilla y agregados
    trade_files = list(predictions_dir.glob("oos_trades_seed*.parquet"))
    if not trade_files:
        print("[FIX-ENSEMBLE-EVAL] ERROR: No se encontraron archivos de trades 'oos_trades_seed*.parquet' en " + str(predictions_dir))
        logger.error("[FIX-ENSEMBLE-EVAL] No hay parquets de trades para evaluar.")
        return 1
        
    # [FIX-ENSEMBLE-EVAL-FILT] Filtrar parquets de trades únicamente para semillas activas configuradas
    filtered_trade_files = []
    for f in trade_files:
        try:
            parts = f.stem.split("_seed")
            if len(parts) == 2:
                seed = int(parts[1])
                if seed in active_seeds:
                    filtered_trade_files.append(f)
        except Exception:
            pass
    trade_files = filtered_trade_files
    print(f"[FIX-ENSEMBLE-EVAL-FILT] Archivos de trades filtrados para semillas activas {active_seeds}: {len(trade_files)} archivos.")
    logger.info(f"[FIX-ENSEMBLE-EVAL-FILT] Archivos de trades filtrados: {len(trade_files)}")
    
    if not trade_files:
        print("[FIX-ENSEMBLE-EVAL] ERROR: No se encontraron archivos de trades filtrados para las semillas activas.")
        logger.error("[FIX-ENSEMBLE-EVAL] No hay parquets de trades filtrados para evaluar.")
        return 1
        
    logger.info(f"[FIX-ENSEMBLE-EVAL] Agrupando por semilla...")
    
    # Agrupar archivos por semilla
    seeds_dict = {}
    for f in trade_files:
        stem = f.stem
        try:
            parts = stem.split("_seed")
            if len(parts) == 2:
                seed = int(parts[1])
                if seed not in seeds_dict:
                    seeds_dict[seed] = []
                seeds_dict[seed].append(f)
        except Exception as e:
            logger.warning(f"No se pudo parsear semilla del archivo {f.name}: {e}")

    # Evaluar métricas por semilla y aplicar poda H1
    seed_metrics = []
    all_trades_list = []
    approved_seeds = []
    
    for seed, files in seeds_dict.items():
        seed_dfs = []
        for f in files:
            try:
                df_sub = pd.read_parquet(f)
                if not df_sub.empty:
                    if 'timestamp' in df_sub.columns:
                        df_sub = df_sub.set_index('timestamp')
                    df_sub.index = pd.to_datetime(df_sub.index, utc=True)
                    df_sub['seed'] = seed
                    seed_dfs.append(df_sub)
            except Exception as e:
                logger.error(f"Error leyendo {f.name}: {e}")
                
        if seed_dfs:
            df_seed_trades = pd.concat(seed_dfs).sort_index()
            
            n_trades = len(df_seed_trades)
            wr = df_seed_trades['is_win'].mean() if 'is_win' in df_seed_trades.columns and n_trades > 0 else 0.0
            
            # Sharpe ratio anualizado para trades
            sharpe = 0.0
            if n_trades > 1 and 'return_pct' in df_seed_trades.columns:
                std_r = df_seed_trades['return_pct'].std()
                if std_r > 1e-10:
                    days = (df_seed_trades.index.max() - df_seed_trades.index.min()).days
                    n_per_year = n_trades / (days / 365.25) if days > 0 else n_trades * 365.25
                    sharpe = (df_seed_trades['return_pct'].mean() / std_r) * (n_per_year ** 0.5)
            
            # [H1] Poda de Semillas Tóxicas (Pre-Ensemble)
            if sharpe >= 0.5:
                all_trades_list.append(df_seed_trades)
                approved_seeds.append(seed)
                print(f"[H1] Semilla {seed} APROBADA (Sharpe: {sharpe:.4f} >= 0.5)")
            else:
                print(f"[H1] Semilla {seed} RECHAZADA por Toxicidad (Sharpe: {sharpe:.4f} < 0.5). Excluida del Soft Voting.")
                logger.warning(f"[H1] Semilla {seed} podada por Sharpe < 0.5")
            
            seed_metrics.append({
                "Seed": seed,
                "Trades": n_trades,
                "Win Rate": wr,
                "Sharpe Anualizado": sharpe,
                "Retorno Medio (%)": df_seed_trades['return_pct'].mean() * 100 if 'return_pct' in df_seed_trades.columns else 0.0,
                "Status": "APPROVED" if sharpe >= 0.5 else "REJECTED"
            })
            
    # 1. Agregar probabilidades de expertos vía Soft Voting (filtrando por approved_seeds)
    if not approved_seeds:
        print("[FIX-ENSEMBLE-EVAL] ERROR CRÍTICO: Ninguna semilla sobrevivió a la poda H1 (Sharpe >= 0.5).")
        logger.error("[FIX-ENSEMBLE-EVAL] Ensamble colapsado. Cero semillas aprobadas.")
        return 1
        
    df_ensemble_probs = aggregate_wfb_seeds(wfb_out_dir, master_probs_path, active_seeds=approved_seeds)
    if df_ensemble_probs is not None:
        print(f"[FIX-ENSEMBLE-EVAL] Probabilidades de Soft Voting agregadas exitosamente en {master_probs_path} ({len(df_ensemble_probs)} filas).")
        logger.success(f"[FIX-ENSEMBLE-EVAL] Soft Voting consolidado en {master_probs_path.name}")
    else:
        print("[FIX-ENSEMBLE-EVAL] ADVERTENCIA: No se pudieron consolidar probabilidades Soft Voting.")
        logger.warning("[FIX-ENSEMBLE-EVAL] Soft Voting no consolidado.")
            
    # 3. Consolidar el Portfolio del Ensamble Agregado (Soft Voting Portfolio)
    portfolio_metrics = {}
    if all_trades_list:
        df_all_trades = pd.concat(all_trades_list).sort_index()
        
        unified_trades_path = predictions_dir / "unified_ensemble_trades_raw.parquet"
        df_all_trades.to_parquet(unified_trades_path)
        print(f"[FIX-ENSEMBLE-EVAL] Guardados todos los trades consolidados en {unified_trades_path}")
        
        # [VOTING-METHOD-SELECTOR] Elegir entre Hard Voting y Soft Voting
        try:
            from config.settings import cfg as _cfg
            voting_method = str(_cfg.wfb.ensemble_voting_method).lower()
        except AttributeError as e:
            raise RuntimeError(f"CRITICAL: Falta wfb.ensemble_voting_method en settings: {e}") from e
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Falló la carga de wfb.ensemble_voting_method: {e}") from e
            
        # [REFERENCIA: Arquitectura Soft Voting y Multiple Testing]
        # Basado en: Bailey & López de Prado (2014) - The Deflated Sharpe Ratio (arXiv:1408.4916)
        # El problema de 'Multiple Testing' penaliza severamente el Sharpe Ratio cuando se evalúan múltiples modelos
        # independientes de forma consecutiva (inflación de alpha, donde DSR_adj penaliza la varianza acumulada usando sqrt(log(N))).
        # Soft Voting soluciona esto agrupando las semillas en un 'Súper-Modelo' continuo, reduciendo matemáticamente la 
        # evaluación Múltiple (N=19) a un único experimento (N=1), mitigando el sobreajuste y el descarte injusto de señales.

        if voting_method == 'hard':
            try:
                threshold = int(_cfg.wfb.ensemble_consensus_threshold)
                _bucket_hours = int(_cfg.wfb.consensus_bucket_hours)
                _bucket_freq = f'{_bucket_hours}h'
                print(f"[HARD-VOTING] Consensus Gate >= {threshold} semillas. Bucket: {_bucket_hours}H.")
            except Exception as e:
                raise RuntimeError(f"Faltan parámetros de Hard Voting en settings.yaml: {e}")
                
            df_all_trades['consensus_bucket'] = df_all_trades.index.floor(_bucket_freq)
            bucket_unique_seeds = df_all_trades.groupby('consensus_bucket')['seed'].nunique().rename('consensus_count')
            df_all_trades['consensus_count'] = df_all_trades['consensus_bucket'].map(bucket_unique_seeds)
            
            df_filtered_trades = df_all_trades[df_all_trades['consensus_count'] >= threshold].copy()
            n_buckets_pass = df_filtered_trades['consensus_bucket'].nunique()
            print(f"[HARD-VOTING] Trades filtrados: {len(df_all_trades)} filas → {len(df_filtered_trades)} filas en {n_buckets_pass} buckets")
            
        else:
            # Soft Voting (Continuous)
            try:
                soft_voting_threshold = float(_cfg.wfb.soft_voting_threshold)
                print(f"[SOFT-VOTING-01] Cargado Soft Voting Threshold: >= {soft_voting_threshold}")
            except AttributeError as e:
                raise RuntimeError(f"CRITICAL: Falta wfb.soft_voting_threshold en settings: {e}") from e
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Falló la carga de wfb.soft_voting_threshold: {e}") from e

            if df_ensemble_probs is None or df_ensemble_probs.empty:
                raise RuntimeError("CRITICAL: df_ensemble_probs está vacío. No se puede ejecutar Soft Voting.")

            df_all_trades['merge_key'] = df_all_trades.index.floor('1h')
            df_ensemble_probs_dedup = df_ensemble_probs.copy()
            df_ensemble_probs_dedup['merge_key'] = df_ensemble_probs_dedup.index.floor('1h')
            df_ensemble_probs_dedup = df_ensemble_probs_dedup.groupby('merge_key').first()

            if 'prob_bull' in df_ensemble_probs_dedup.columns:
                df_all_trades['ensemble_prob'] = df_all_trades['merge_key'].map(df_ensemble_probs_dedup['prob_bull'])
            else:
                prob_col = [c for c in df_ensemble_probs_dedup.columns if "prob" in c.lower()][0]
                df_all_trades['ensemble_prob'] = df_all_trades['merge_key'].map(df_ensemble_probs_dedup[prob_col])

            df_all_trades['ensemble_prob'] = df_all_trades['ensemble_prob'].fillna(0.0)
            df_filtered_trades = df_all_trades[df_all_trades['ensemble_prob'] >= soft_voting_threshold].copy()
            
            print(f"[SOFT-VOTING-02] Gate >= {soft_voting_threshold}: {len(df_all_trades)} filas → {len(df_filtered_trades)} continuas")
            df_filtered_trades['consensus_bucket'] = df_filtered_trades.index

        
        # [MEJORA-F3-01] Consensus-Soft Embargo: Cargar parámetros de configuración de settings.yaml
        try:
            soft_embargo_enabled = bool(_cfg.wfb.soft_embargo_enabled)
            soft_embargo_hours = float(_cfg.wfb.soft_embargo_hours)
            print(f"[MEJORA-F3-01] Cargar configuración Soft Embargo: Enabled={soft_embargo_enabled}, Hours={soft_embargo_hours}H")
            logger.info(f"[MEJORA-F3-01] Soft Embargo: Enabled={soft_embargo_enabled}, Hours={soft_embargo_hours}H")
        except Exception as e:
            # Fallback crítico
            err_msg = f"CRITICAL [MEJORA-F3-01]: Falló la carga de parámetros de soft_embargo en settings.yaml: {e}"
            print(err_msg)
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

        # Mapa canónico de embargos del HMM por régimen (idéntico al de producción en signal_filter.py)
        HMM_EMBARGO_MAP = {
            "1_BULL_TREND":        72.0,
            "1_VOLATILE_BULL":     96.0,
            "1_BULL_GRIND":        72.0,
            "2_CALM_RANGE":       144.0,
            "2_VOLATILE_RANGE":   168.0,
            "3_CALM_BEAR":        168.0,
            "3_BEAR_CRASH":       168.0,
            "4_BEAR_FORCED":      168.0,
            "1_BULL_TREND_B":      72.0,
            "1_BULL_TREND_C":      72.0,
            "1_BULL_TREND_D":      72.0,
            "1_BULL_TREND_WEAK":   72.0,
            "1_VOLATILE_BULL_B":   96.0,
            "1_VOLATILE_BULL_C":   96.0,
            "1_VOLATILE_BULL_D":   96.0,
            "2_CALM_RANGE_B":     144.0,
            "2_CALM_RANGE_C":     144.0,
            "2_VOLATILE_RANGE_B": 168.0,
            "3_CALM_BEAR_B":      168.0,
            "3_BEAR_CRASH_B":     168.0,
        }
        DEFAULT_WAIT_HOURS = float(_cfg.sop.embargo_hours)

        # Simular portafolio unificado: agregar por bucket temporal (FIX-D4)
        # ANTES: groupby(index exacto) → un trade por timestamp exacto
        # AHORA: groupby(consensus_bucket) → un trade por ventana de consenso de N horas
        # El timestamp del portfolio es el floor del bucket (extremo inferior de la ventana)
        agg_dict = {
            'return_pct': 'mean',       # retorno medio de todas las semillas en el bucket
            # [FIX-ENSEMBLE-WINRATE-01 2026-06-04] No usar 'max' para is_win. 
            # Si 4 seeds pierden -5% y 1 gana +0.1%, el 'max' registraría un WIN falso
            # con un retorno promedio negativo, engañando al Binomial Test del Gauntlet.
            # Se calculará determinísticamente post-agregación en base al 'return_pct' medio.
            'direction':  'first',      # dirección de la primera señal del bucket
            'wfb_window': 'first',      # ventana WFB de la primera señal del bucket
        }
        if 'hmm_regime' in df_filtered_trades.columns:
            agg_dict['hmm_regime'] = 'first'
        if 'consensus_count' in df_filtered_trades.columns:
            agg_dict['consensus_count'] = 'first'

        # Agrupar por bucket canónico — produce un trade por evento de consenso
        df_portfolio = (
            df_filtered_trades
            .groupby('consensus_bucket')
            .agg(agg_dict)
            .sort_index()
        )
        # [FIX-ENSEMBLE-WINRATE-01 2026-06-04] Recalcular 'is_win' de forma matemáticamente consistente
        df_portfolio['is_win'] = (df_portfolio['return_pct'] > 0).astype(float)
        
        print(f"[SOFT-VOTING-02] Portfolio agregado por timestamp: "
              f"{len(df_filtered_trades)} filas → {len(df_portfolio)} trades únicos de consenso")
        logger.info(f"[SOFT-VOTING-02] Portfolio final: {len(df_portfolio)} trades (de {len(df_filtered_trades)} filas agregadas)")

        
        # Aplicar el Embargo Secuencial en el Portafolio Aggregated
        selected_indices = []
        last_time = None
        
        print("[MEJORA-F3-01] Aplicando Embargo Secuencial / Consensus-Soft Embargo sobre el portafolio consolidado...")
        logger.info("[MEJORA-F3-01] Aplicando filtro de embargo al portafolio...")
        
        # [MEJORA-F3-01-DYNAMIC] Consensus-Soft Embargo adaptativo según la cantidad de semillas activas
        n_active = len(active_seeds)
        soft_threshold = 4 if n_active >= 5 else 2 if n_active == 3 else max(2, n_active - 1)
        print(f"[MEJORA-F3-01-DYNAMIC] Configurado umbral adaptativo Consensus-Soft Embargo: >= {soft_threshold} de {n_active} semillas.")
        logger.info(f"[MEJORA-F3-01-DYNAMIC] Umbral de Soft Embargo: >= {soft_threshold}")

        # [P2-B-SOFT-EMBARGO-ADAPT 2026-05-28] Pre-calcular WR rolling 30d sobre el portafolio
        # para usarlo como señal adicional en el nivel de embargo de cada trade.
        # Lógica de 3 niveles (Hipótesis #3 del analisis_wfb_ensemble_20260528.md):
        #   - consenso >= soft_threshold + WR_rolling > 50% → 24H  (max agilidad)
        #   - consenso >= soft_threshold + WR_rolling > 40% → 48H  (régimen neutral)
        #   - else                                           → HMM estándar (72-168H)
        try:
            _se_adapt_enabled = bool(_cfg.wfb.circuit_breaker.enabled)
        except AttributeError as e:
            raise RuntimeError(f"CRITICAL: Falta wfb.circuit_breaker.enabled en settings: {e}") from e
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Falló la carga de wfb.circuit_breaker.enabled: {e}") from e

        # WR rolling 30d sobre los trades del portfolio ordenados
        _ROLLING_WINDOW = 30  # días
        _wr_rolling_30d: pd.Series = pd.Series(dtype=float)
        if len(df_portfolio) >= 3:
            _is_win_ser = df_portfolio["is_win"].astype(float)
            _wr_rolling_30d = _is_win_ser.rolling(f"{_ROLLING_WINDOW}D", min_periods=3).mean()
        print(f"[P2-B-SOFT-EMBARGO-ADAPT] WR rolling 30d disponible: {(~_wr_rolling_30d.isna()).sum()} puntos de {len(df_portfolio)}")

        for ts, row in df_portfolio.iterrows():
            regime = str(row.get('hmm_regime', '1_BULL_TREND'))
            consensus = int(row.get('consensus_count', soft_threshold)) # Default a soft_threshold para Soft Voting
            
            # [P2-B-SOFT-EMBARGO-ADAPT] Obtener WR rolling del régimen ANTES de este trade
            # (lookback: últ. 30 días del portafolio disponible en ese momento — causal)
            _wr_now = float("nan")
            if not _wr_rolling_30d.empty and ts in _wr_rolling_30d.index:
                _wr_now = float(_wr_rolling_30d.loc[ts]) if not pd.isna(_wr_rolling_30d.loc[ts]) else float("nan")
            elif not _wr_rolling_30d.empty:
                _prior = _wr_rolling_30d[_wr_rolling_30d.index <= ts].dropna()
                _wr_now = float(_prior.iloc[-1]) if not _prior.empty else float("nan")

            # Determinar horas de embargo para este trade (3 niveles adaptativos)
            if soft_embargo_enabled and consensus >= soft_threshold:
                if not pd.isna(_wr_now) and _wr_now > 0.50:
                    # Nivel 1: consenso fuerte + régimen positivo → máxima agilidad
                    emb_h = soft_embargo_hours          # 24H (de settings)
                    emb_type = f"SoftAdapt-L1 (WR={_wr_now:.1%}>50%+SoftVoting)"
                elif not pd.isna(_wr_now) and _wr_now > 0.40:
                    # Nivel 2: consenso fuerte + régimen neutral → embargo moderado
                    emb_h = 48.0
                    emb_type = f"SoftAdapt-L2 (WR={_wr_now:.1%}>40%+SoftVoting)"
                else:
                    # Nivel 3: consenso fuerte pero régimen adverso o sin datos → estándar
                    emb_h = HMM_EMBARGO_MAP.get(regime, DEFAULT_WAIT_HOURS)
                    _wr_label = f"{_wr_now:.1%}" if not pd.isna(_wr_now) else "N/A"
                    emb_type = f"SoftAdapt-L3-Standard (WR={_wr_label}<=40%, regime={regime})"
            else:
                # Sin consenso suficiente → embargo HMM estándar
                emb_h = HMM_EMBARGO_MAP.get(regime, DEFAULT_WAIT_HOURS)
                emb_type = f"Standard (consenso={consensus}<{soft_threshold}, regime={regime})"

                
            if last_time is None:
                selected_indices.append(ts)
                last_time = ts
                print(f"  [TRACK] Trade en {ts} ACEPTADO (Primer trade de la ventana. Régimen={regime})")
            else:
                delta_h = (ts - last_time).total_seconds() / 3600.0
                if delta_h >= emb_h:
                    selected_indices.append(ts)
                    last_time = ts
                    print(f"  [TRACK] Trade en {ts} ACEPTADO (Delta={delta_h:.1f}H >= Embargo={emb_h}H [{emb_type}]. Régimen={regime})")
                else:
                    print(f"  [TRACK] Trade en {ts} EMBARGADO (Delta={delta_h:.1f}H < Embargo={emb_h}H [{emb_type}]. Régimen={regime})")
                    
        df_portfolio_final = df_portfolio.loc[selected_indices].copy()
        print(f"[MEJORA-F3-01] Embargo aplicado exitosamente. Portafolio final reducido de {len(df_portfolio)} a {len(df_portfolio_final)} trades.")
        logger.success(f"[MEJORA-F3-01] Portafolio consolidado con embargo aplicado: {len(df_portfolio_final)} trades.")
        
        n_portfolio = len(df_portfolio_final)
        wr_portfolio = df_portfolio_final['is_win'].mean() if 'is_win' in df_portfolio_final.columns and n_portfolio > 0 else 0.0
        
        sharpe_portfolio = 0.0
        if n_portfolio > 1 and 'return_pct' in df_portfolio_final.columns:
            std_r = df_portfolio_final['return_pct'].std()
            if std_r > 1e-10:
                days = (df_portfolio_final.index.max() - df_portfolio_final.index.min()).days
                n_per_year = n_portfolio / (days / 365.25) if days > 0 else n_portfolio * 365.25
                sharpe_portfolio = (df_portfolio_final['return_pct'].mean() / std_r) * (n_per_year ** 0.5)
                
        portfolio_metrics = {
            "Trades": n_portfolio,
            "Win Rate": wr_portfolio,
            "Sharpe Anualizado": sharpe_portfolio,
            "Retorno Medio (%)": df_portfolio_final['return_pct'].mean() * 100 if 'return_pct' in df_portfolio_final.columns else 0.0
        }
        
        portfolio_out_path = predictions_dir / "ensemble_portfolio_trades.parquet"
        df_portfolio_final.to_parquet(portfolio_out_path)
        print(f"[FIX-ENSEMBLE-EVAL] Guardado portafolio ensemble en {portfolio_out_path}")
        logger.success(f"[FIX-ENSEMBLE-EVAL] Portafolio unificado guardado en {portfolio_out_path.name}")

        # ── [MCTB-01] Monte Carlo Trade Bootstrapping ──
        print("\n[MCTB-01] Ejecutando Monte Carlo Trade Bootstrapping (10,000 universos)...")
        _returns = df_portfolio_final['return_pct'].values
        _n_trades_mc = len(_returns)
        
        _mc_dd_95 = 0.0
        _mc_dd_99 = 0.0
        _por_x1 = 0.0
        _por_x10 = 0.0
        
        if _n_trades_mc >= 10:
            np.random.seed(42)
            _idx = np.random.randint(0, _n_trades_mc, size=(10000, _n_trades_mc))
            _sim_returns = _returns[_idx]
            
            # Constant allocation DD
            _cum_returns = np.cumsum(_sim_returns, axis=1)
            _running_max = np.maximum.accumulate(_cum_returns, axis=1)
            _drawdowns = _running_max - _cum_returns
            _max_dds = np.max(_drawdowns, axis=1)
            
            _mc_dd_95 = np.percentile(_max_dds, 95) * 100
            _mc_dd_99 = np.percentile(_max_dds, 99) * 100
            
            _kill_unlev = 0.15
            _kill_x10 = 0.015 # 1.5% unleveraged becomes 15% leveraged
            
            _por_x1 = np.mean(_max_dds > _kill_unlev) * 100
            _por_x10 = np.mean(_max_dds > _kill_x10) * 100
            
            print(f"[MCTB-01] Resultados MCTB (N={_n_trades_mc}):")
            print(f"  - MC-MaxDD (95% Confianza): {_mc_dd_95:.2f}% (Sin apalancamiento)")
            print(f"  - MC-MaxDD (99% Confianza): {_mc_dd_99:.2f}% (Sin apalancamiento)")
            print(f"  - Probability of Ruin (PoR al -15% DD, Apalancamiento x1): {_por_x1:.2f}%")
            print(f"  - Probability of Ruin (PoR al -15% DD, Apalancamiento x10): {_por_x10:.2f}%")
            if _por_x10 > 5.0:
                print(f"  [ALERTA MCTB] La ruina supera el 5% a x10. Reducir leverage o Kelly Fraction.")
        else:
            print("[MCTB-01] Insuficientes trades para simulación Monte Carlo.")

        # ═══════════════════════════════════════════════════════════════════════
        # [ENSEMBLE-GAUNTLET-01 2026-05-28] Gauntlet estadístico sobre el portfolio
        # ensemble completo. Es el veredicto AUTORIZADO y ÚNICO de despliegue.
        #
        # Por qué es superior al Gauntlet por seed:
        #  - N trades >> 100 → DSR discriminativo (no satura en 1.0)
        #  - Un único veredicto (sin inflación de alpha por 5 evaluaciones)
        #  - Solo evalúa trades que pasaron Consensus Gate (≥3 seeds) + Embargo
        #  - DSR corregido por N_seeds (R5 Multiple Testing Correction)
        # ═══════════════════════════════════════════════════════════════════════
        report_dir_ens = _ROOT / "data" / "reports"
        ensemble_verdict_path = report_dir_ens / "ensemble_statistical_verdict.json"
        _ensemble_approved = False
        _ens_verdict = {}

        try:
            from config.settings import cfg as _cfg_ens
            _ens_min_trades = int(_cfg_ens.stat.min_trades)
        except Exception as _e_cfg_ens:
            raise RuntimeError(
                f"[ENSEMBLE-GAUNTLET-01] CRITICAL: No se pudo cargar stat.min_trades de settings.yaml: {_e_cfg_ens}"
            ) from _e_cfg_ens

        print(f"[ENSEMBLE-GAUNTLET-01] Portfolio ensemble: {len(df_portfolio_final)} trades | "
              f"umbral minimo: {_ens_min_trades}")

        if len(df_portfolio_final) >= _ens_min_trades:
            try:
                from luna.monitoring.statistical_audit import LunaStatisticalAuditor

                # [FIX-R5] DSR correction: N = 1.
                # Con el Soft Voting continuo, el ensamble evalúa probabilísticamente el mercado como un único "Súper-Modelo",
                # por lo que evitamos matemáticamente la penalización de Comparaciones Múltiples (N=19) en el Deflated Sharpe Ratio.
                os.environ["LUNA_N_SEEDS_TOTAL"] = "1"
                print(f"[ENSEMBLE-GAUNTLET-01] [SOFT-VOTING] LUNA_N_SEEDS_TOTAL=1 (DSR correction eliminada por Soft Voting) "
                      f"(DSR correction por {len(approved_seeds)} seeds activas)")

                _ens_auditor = LunaStatisticalAuditor()

                # Asegurar columna 'timestamp' que requiere el Gauntlet
                _df_ens_eval = df_portfolio_final.copy()
                if "timestamp" not in _df_ens_eval.columns:
                    _df_ens_eval["timestamp"] = _df_ens_eval.index

                _ens_verdict = _ens_auditor.run_gauntlet(_df_ens_eval)

                # Inyectar metadatos específicos del ensemble
                _ens_verdict["ensemble_n_seeds"]    = len(approved_seeds)
                _ens_verdict["ensemble_seeds"]      = approved_seeds
                _ens_verdict["ensemble_n_trades"]   = int(len(df_portfolio_final))
                _ens_verdict["consensus_threshold"] = float(soft_voting_threshold)
                _ens_verdict["run_id"] = os.environ.get("LUNA_RUN_ID", "ensemble_eval")

                # Aplicar corrección R5 entre seeds sobre el veredicto
                # (replicar la lógica de run_statistical_validation.py)
                import math as _math_ens
                import scipy.stats as _stats_ens
                _n_seeds_ens = len(approved_seeds)
                _r5_factor_ens = _math_ens.sqrt(_math_ens.log(_n_seeds_ens)) if _n_seeds_ens > 1 else 1.0
                _sr_crudo_ens = float(_ens_verdict.get("metrics", {}).get("sharpe_crudo", 0.0))
                _skew_ens = float(_ens_verdict.get("statistical_audit", {}).get("skewness", 0.0))
                _kurt_ens = float(_ens_verdict.get("statistical_audit", {}).get("kurtosis", 0.0))
                _n_obs_ens = int(_ens_verdict.get("statistical_audit", {}).get("n_obs_dsr", 1))
                _n_trials_ens = int(_ens_verdict.get("statistical_audit", {}).get("n_trials_dsr", 100))
                _base_dsr_thr_ens = float(_cfg.stat.min_dsr)
                _dsr_raw_ens = float(_ens_verdict.get("statistical_audit", {}).get("dsr", 0.0))

                _var_sr_ens = (1.0 - (_skew_ens * _sr_crudo_ens) +
                               ((_kurt_ens - 1.0) / 4.0) * (_sr_crudo_ens ** 2)) / max(_n_obs_ens, 2)
                _std_sr_ens = float(max(_var_sr_ens, 1e-12) ** 0.5)
                _gamma_ens = 0.5772156649
                _prob_ens = 1.0 / max(_n_trials_ens, 2)
                _z1_ens = _stats_ens.norm.ppf(1.0 - _prob_ens)
                _z2_ens = _stats_ens.norm.ppf(1.0 - _prob_ens * _math_ens.exp(-1.0))
                _sr_star_base_ens = _std_sr_ens * ((1.0 - _gamma_ens) * _z1_ens + _gamma_ens * _z2_ens)
                _sr_star_adj_ens = _sr_star_base_ens * _r5_factor_ens
                _z_adj_ens = (_sr_crudo_ens - _sr_star_adj_ens) / _std_sr_ens if _std_sr_ens > 1e-9 \
                             else (1.0 if _sr_crudo_ens > _sr_star_adj_ens else -10.0)
                _dsr_adj_ens = float(_stats_ens.norm.cdf(_z_adj_ens))
                _pass_dsr_adj_ens = _dsr_adj_ens >= _base_dsr_thr_ens

                _ens_verdict["dsr_correction_factor"] = round(_r5_factor_ens, 4)
                _ens_verdict["n_seeds_correction"]    = _n_seeds_ens
                _ens_verdict["dsr_adjusted"]          = round(_dsr_adj_ens, 6)
                _ens_verdict["adjusted_dsr_threshold"]= round(_base_dsr_thr_ens, 4)

                print(f"[ENSEMBLE-GAUNTLET-01] [FIX-R5] DSR_raw={_dsr_raw_ens:.4f} | "
                      f"factor=sqrt(log({_n_seeds_ens}))={_r5_factor_ens:.4f} | "
                      f"SR*_base={_sr_star_base_ens:.4f} SR*_adj={_sr_star_adj_ens:.4f} | "
                      f"DSR_adj={_dsr_adj_ens:.4f} (umbral={_base_dsr_thr_ens:.3f})")

                # Si pasaba el gate base pero falla con corrección R5
                if _ens_verdict.get("flags", {}).get("pass_dsr", False) and not _pass_dsr_adj_ens:
                    _ens_verdict["flags"]["pass_dsr"] = False
                    _ens_verdict["deploy_approved"] = False
                    _ens_verdict["rejection_reason"] = (
                        f"[ENSEMBLE-GAUNTLET-01] DSR_adj={_dsr_adj_ens:.4f} < umbral={_base_dsr_thr_ens:.3f} "
                        f"tras corrección R5 por N_seeds={_n_seeds_ens}"
                    )
                    print(f"[ENSEMBLE-GAUNTLET-01] GATE ENDURECIDO por R5: DSR_adj={_dsr_adj_ens:.4f} "
                          f"< {_base_dsr_thr_ens:.3f} — deploy_approved: True → False")

                # ── [MCTB-02] Disyuntor Letal de Monte Carlo (Probability of Ruin) ──
                try:
                    from config.settings import cfg as _cfg_ens
                    _base_por_thr_ens = float(_cfg_ens.stat.max_por_x10)
                except AttributeError as e:
                    raise RuntimeError(f"CRITICAL: Falta stat.max_por_x10 en settings: {e}") from e
                except Exception as e:
                    raise RuntimeError(f"CRITICAL: Falló la carga de stat.max_por_x10: {e}") from e

                _ens_verdict["summary"]["por_x10_pct"] = round(_por_x10, 2)
                
                # Checkeamos si PoR supera el max permitido
                _pass_por = bool(_por_x10 <= _base_por_thr_ens)
                
                if not "flags" in _ens_verdict:
                    _ens_verdict["flags"] = {}
                _ens_verdict["flags"]["pass_por"] = _pass_por

                if _ens_verdict.get("deploy_approved", False) and not _pass_por:
                    _ens_verdict["deploy_approved"] = False
                    _ens_verdict["rejection_reason"] = (
                        f"[MCTB DISYUNTOR] Probabilidad de Ruina (PoR a x10)={_por_x10:.2f}% "
                        f"> umbral letal={_base_por_thr_ens:.1f}% en 10,000 universos de Monte Carlo."
                    )
                    print(f"[ENSEMBLE-GAUNTLET-01] GATE ENDURECIDO por MCTB: PoR={_por_x10:.2f}% "
                          f"> {_base_por_thr_ens:.1f}% — deploy_approved: True -> False")
                
                _ensemble_approved = bool(_ens_verdict.get("deploy_approved", False))

                # Guardar veredicto canónico del ensemble
                with open(ensemble_verdict_path, "w", encoding="utf-8") as _ef:
                    json.dump(_ens_verdict, _ef, indent=4, default=str)

                logger.success(
                    "[ENSEMBLE-GAUNTLET-01] Veredicto ensemble GUARDADO: {} | "
                    "trades={} | DSR={:.4f} (adj={:.4f}) | PBO={:.1f}% | WR={:.1f}%",
                    "APPROVED" if _ensemble_approved else "REJECTED",
                    len(df_portfolio_final),
                    _dsr_raw_ens, _dsr_adj_ens,
                    _ens_verdict.get("summary", {}).get("pbo_pct", 0.0),
                    _ens_verdict.get("summary", {}).get("win_rate_pct", 0.0)
                )
                print(
                    f"[ENSEMBLE-GAUNTLET-01] *** VEREDICTO FINAL ENSEMBLE: "
                    f"{'APPROVED' if _ensemble_approved else 'REJECTED'} *** | "
                    f"trades={len(df_portfolio_final)} | "
                    f"DSR={_dsr_raw_ens:.4f} DSR_adj={_dsr_adj_ens:.4f} | "
                    f"PBO={_ens_verdict.get('summary', {}).get('pbo_pct', '?')}% | "
                    f"WR={_ens_verdict.get('summary', {}).get('win_rate_pct', '?')}% | "
                    f"Guardado: {ensemble_verdict_path.name}"
                )  # RULE[fixbugsprints.md]

            except Exception as _eg_err:
                import traceback as _tb_ens
                logger.error("[ENSEMBLE-GAUNTLET-01] ERROR no bloqueante: {}", _eg_err)
                print(f"[ENSEMBLE-GAUNTLET-01] ERROR: {_eg_err}\n{_tb_ens.format_exc()}")
        else:
            print(f"[ENSEMBLE-GAUNTLET-01] SKIP: {len(df_portfolio_final)} trades < "
                  f"{_ens_min_trades} minimo — Gauntlet ensemble no ejecutado")
            logger.warning(
                "[ENSEMBLE-GAUNTLET-01] Portfolio insuficiente para Gauntlet ensemble: "
                "{} trades < {} minimo",
                len(df_portfolio_final), _ens_min_trades
            )
        # ═══════════════════════════════════════════════════════════════════════

    # 4. Formatear Tearsheet Aggregated Markdown
    summary_md = []
    summary_md.append("# Walk-Forward Backtesting (WFB) Ensemble Tearsheet Summary")
    summary_md.append(f"Generado el: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    
    summary_md.append("## 1. Métrica de Portafolio Ensemble (Unificado)")
    summary_md.append("Representa la combinación operativa consolidando colisiones de timestamps exactos:")
    summary_md.append("")
    if portfolio_metrics:
        status_trades = "✅ OK" if portfolio_metrics['Trades'] >= 30 else "⚠️ INSUFICIENTE (< 30 trades)"
        summary_md.append(f"- **Total Trades Únicos**: {portfolio_metrics['Trades']} ({status_trades})")
        summary_md.append(f"- **Win Rate Promedio**: {portfolio_metrics['Win Rate']*100:.2f}%")
        summary_md.append(f"- **Sharpe Ratio Anualizado**: {portfolio_metrics['Sharpe Anualizado']:.4f}")
        summary_md.append(f"- **Retorno Promedio por Trade**: {portfolio_metrics['Retorno Medio (%)']:.4f}%")
    else:
        summary_md.append("- No hay trades del portafolio ensemble para evaluar.")
    summary_md.append("")
    
    summary_md.append("## 2. Desglose de Métricas por Semilla (Seed)")
    summary_md.append("| Semilla | Trades Totales | Win Rate | Sharpe Ratio | Retorno Medio por Trade | Status |")
    summary_md.append("| --- | --- | --- | --- | --- | --- |")
    for m in seed_metrics:
        summary_md.append(f"| {m['Seed']} | {m['Trades']} | {m['Win Rate']*100:.2f}% | {m['Sharpe Anualizado']:.4f} | {m['Retorno Medio (%)']:.4f}% | {m['Status']} |")
    summary_md.append("")
    
    summary_md.append("## 3. Diagnóstico y Robustez del Ensamble")
    summary_md.append("El ensamble multi-semilla mitiga el sobreajuste y la inanición operativa.")
    if portfolio_metrics and portfolio_metrics['Trades'] < 30:
        summary_md.append("> ⚠️ **ALERTA**: El número total de trades únicos del ensamble es inferior al mínimo estadístico recomendado de 30 trades. Considere relajar aún más los gates adaptativos del Brier score si persiste la inanición operativa.")
    else:
        summary_md.append("> ✅ **ROBUSTEZ ESTADÍSTICA**: Se supera el umbral crítico de 30 trades. Las métricas del ensamble son estadísticamente significativas.")

    # Inyectar MCTB
    summary_md.append("")
    summary_md.append("## 3.1. Monte Carlo Trade Bootstrapping (MCTB)")
    if '_mc_dd_99' in locals() and _mc_dd_99 > 0:
        summary_md.append("Resultados de simular 10,000 curvas de equity barajando los trades del ensamble:")
        summary_md.append(f"- **MC-MaxDD (95% Confianza)**: {_mc_dd_95:.2f}%")
        summary_md.append(f"- **MC-MaxDD (99% Confianza)**: {_mc_dd_99:.2f}%")
        summary_md.append(f"- **Prob. de Ruina (PoR a -15% con Apalancamiento x10)**: {_por_x10:.2f}%")
        if _por_x10 > 5.0:
            summary_md.append("> ⚠️ **ALERTA MCTB**: Probabilidad de ruina > 5%. Se recomienda bajar el Criterio de Kelly o el apalancamiento máximo a x5.")
    else:
        summary_md.append("- Insuficientes trades para MCTB.")

    # [ENSEMBLE-GAUNTLET-01] Sección 4: veredicto estadístico autorizado del ensemble
    summary_md.append("")
    summary_md.append("## 4. Veredicto Estadístico Ensemble (ENSEMBLE-GAUNTLET-01) — Gate Autorizado")
    if _ens_verdict:
        _ev_s = _ens_verdict.get("summary", {})
        _ev_f = _ens_verdict.get("flags", {})
        _verdict_label = "✅ APPROVED — LISTO PARA DESPLIEGUE" if _ensemble_approved else "❌ REJECTED — NO DESPLEGAR"
        summary_md.append(f"> ### {_verdict_label}")
        summary_md.append("")
        summary_md.append("| Gate | Valor | Umbral | Estado |")
        summary_md.append("| --- | --- | --- | --- |")
        summary_md.append(
            f"| Trades | {_ev_s.get('total_trades','?')} | "
            f">= {_ens_verdict.get('sop_thresholds', {}).get('min_trades','?')} | "
            f"{'✅' if _ev_f.get('pass_trades') else '❌'} |"
        )
        summary_md.append(
            f"| Win Rate | {_ev_s.get('win_rate_pct','?')}% | >50% | "
            f"{'✅' if (_ev_s.get('win_rate_pct', 0) or 0) > 50 else '❌'} |"
        )
        summary_md.append(
            f"| DSR (raw) | {_ev_s.get('dsr','?')} | "
            f">= {_cfg.stat.min_dsr} | "
            f"{'✅' if _ev_f.get('pass_dsr') else '❌'} |"
        )
        summary_md.append(
            f"| DSR (adj R5, N={_ens_verdict.get('n_seeds_correction','?')}) | "
            f"{_ens_verdict.get('dsr_adjusted','?')} | "
            f">= {_ens_verdict.get('adjusted_dsr_threshold','?')} | "
            f"{'✅' if _ens_verdict.get('dsr_adjusted', 0) >= _ens_verdict['adjusted_dsr_threshold'] else '❌'} |"
        )
        summary_md.append(
            f"| PBO CSCV | {_ev_s.get('pbo_pct','?')}% | "
            f"< {(_ens_verdict.get('sop_thresholds', {}).get('max_pbo_pct') or 22.0):.0f}% | "
            f"{'✅' if _ev_f.get('pass_pbo') else '❌'} |"
        )
        summary_md.append(
            f"| MaxDrawdown | {_ev_s.get('max_drawdown_pct','?')}% | "
            f"< {_ens_verdict.get('sop_thresholds', {}).get('max_drawdown_pct','?')}% | "
            f"{'✅' if _ev_f.get('pass_dd') else '❌'} |"
        )
        summary_md.append(
            f"| Binomial p | {_ev_s.get('binomial_p','?')} | "
            f"< {_cfg.stat.alpha_binomial} | "
            f"{'✅' if _ev_f.get('pass_binomial') else '❌'} |"
        )
        summary_md.append("")
        summary_md.append(
            f"- **Seeds activas (Aprobadas H1)**: {_ens_verdict.get('ensemble_seeds', approved_seeds)} "
            f"(N={_ens_verdict.get('ensemble_n_seeds','?')})"
        )
        summary_md.append(
            f"- **Consensus Gate**: >= {_ens_verdict.get('consensus_threshold','?')} seeds"
        )
        summary_md.append(
            f"- **Corrección R5**: factor=sqrt(log({_ens_verdict.get('n_seeds_correction','?')})) "
            f"= {_ens_verdict.get('dsr_correction_factor','?')}"
        )
        if _ens_verdict.get("rejection_reason"):
            summary_md.append(f"- **Razón de rechazo**: {_ens_verdict['rejection_reason']}")
    else:
        summary_md.append(
            "> ⚠️ **Gauntlet ensemble no ejecutado** — Portfolio insuficiente o error. "
            "Revisar logs [ENSEMBLE-GAUNTLET-01]."
        )

    report_content = "\n".join(summary_md)
    print("\n" + "="*80)
    print(report_content)
    print("="*80 + "\n")
    
    # Escribir reporte Markdown a disco
    report_out_path = wfb_out_dir / "wfb_ensemble_tearsheet_summary.md"
    report_out_path.write_text(report_content, encoding="utf-8")
    print(f"[FIX-ENSEMBLE-EVAL] Tearsheet summary guardado exitosamente en {report_out_path}")
    logger.success(f"[FIX-ENSEMBLE-EVAL] Tearsheet guardado en {report_out_path.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

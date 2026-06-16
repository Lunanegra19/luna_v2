import pandas as pd
import numpy as np
import glob
from pathlib import Path

def metricas_fin(subset: pd.DataFrame, label: str = "") -> dict:
    n = len(subset)
    if n < 5:
        return {"n": n, "wr": np.nan, "ret": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    wr  = subset["is_win"].mean() * 100
    ret = subset["ret100"].sum()
    rets = subset["ret100"].values
    mean_r, std_r = np.mean(rets), np.std(rets)
    sharpe = (mean_r / std_r) * np.sqrt(52) if std_r > 1e-10 else 0.0
    cumret = (1 + subset["return_raw"].values).cumprod()
    running_max = np.maximum.accumulate(cumret)
    dd = (cumret - running_max) / np.maximum(running_max, 1e-10)
    maxdd = abs(dd.min()) * 100
    if label:
        print(f"{label:<35} | N={n:>4} | WR={wr:>5.1f}% | Ret={ret:>6.1f}pp | Sharpe={sharpe:>5.2f} | MaxDD={maxdd:>5.1f}%")
    return {"n": n, "wr": wr, "ret": ret, "sharpe": sharpe, "maxdd": maxdd}

def run_tests():
    print("\n" + "="*80)
    print("VALIDACION EMPIRICA DE HIPOTESIS OOS (Sin Overfitting)")
    print("="*80)
    
    # Load data
    dfs = []
    for f in glob.glob("data/predictions/oos_trades_seed*.parquet"):
        d = pd.read_parquet(f)
        if d.index.name == 'entry_time' or 'entry_time' not in d.columns:
            d = d.reset_index(names='entry_time') if d.index.name else d.reset_index()
        d['_seed'] = int(Path(f).stem.split('seed')[1])
        dfs.append(d)
    
    if not dfs:
        print("No se encontraron parquets.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    df['ret100'] = df['return_raw'] * 100
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    
    # ---------------------------------------------------------
    # TEST 1: DESACTIVAR OOD GUARD (Counterfactual Simulation)
    # ---------------------------------------------------------
    print("\n[TEST 1] SIMULACION: ¿Qué pasa si DESACTIVAMOS el OOD Guard?")
    print("Metodología: Comparar los trades ejecutados (Kelly > 0) vs TODOS los teóricos (Kelly ignorado)")
    
    # Filtro actual (Gauntlet)
    active_trades = df[df.get('kelly_fraction_used', 1.0) > 0.0]
    # Filtro simulado: desactivar OOD y PSI Drift (todos operan)
    all_theoretical = df.copy()
    
    metricas_fin(active_trades, "BASELINE (OOD Activado - Castiga)")
    metricas_fin(all_theoretical, "SIMULACION (OOD Desactivado - Opera Todo)")
    
    # ---------------------------------------------------------
    # TEST 2: ACORTAR VENTANA WFB (Decaimiento del Alpha)
    # ---------------------------------------------------------
    print("\n[TEST 2] SIMULACION: Retrain Mensual vs Trimestral (Alpha Decay)")
    print("Metodología: Partir cada ventana WFB en Mes 1, Mes 2, Mes 3. Evaluar performance.")
    
    df = df.sort_values(['_seed', 'wfb_window', 'entry_time'])
    
    def get_month_group(subdf):
        subdf = subdf.sort_values('entry_time')
        # Partimos en 3 cuantiles temporales iguales
        subdf['time_q'] = pd.qcut(np.arange(len(subdf)), 3, labels=['Mes 1', 'Mes 2', 'Mes 3'])
        return subdf
        
    df_time = df.groupby(['_seed', 'wfb_window'], group_keys=False).apply(get_month_group)
    
    mes1 = df_time[df_time['time_q'] == 'Mes 1']
    mes2 = df_time[df_time['time_q'] == 'Mes 2']
    mes3 = df_time[df_time['time_q'] == 'Mes 3']
    
    metricas_fin(mes1, "MES 1 (Recién entrenado)")
    metricas_fin(mes2, "MES 2 (Mitad de ventana)")
    metricas_fin(mes3, "MES 3 (Final de ventana)")
    
    # ---------------------------------------------------------
    # TEST 3: TOXICIDAD DE LAGS LARGOS (SHAP Values)
    # ---------------------------------------------------------
    print("\n[TEST 3] TOXICIDAD DE LAGS > 200h")
    print("Metodología: Analizar si los trades donde las features 'milag500h/336h' son Top SHAP drivers fallan más.")
    
    if 'shap_drivers' in df.columns:
        # Convertir a minúsculas
        df['shap_str'] = df['shap_drivers'].fillna('').astype(str).str.lower()
        
        # Trades conducidos por lags largos
        lags_largos = df[df['shap_str'].str.contains('336h|500h|240h', regex=True)]
        # Trades conducidos por lags cortos (12h, 48h, etc) sin lags largos
        lags_cortos = df[~df['shap_str'].str.contains('336h|500h|240h', regex=True) & df['shap_str'].str.contains('12h|24h|48h', regex=True)]
        
        metricas_fin(lags_cortos, "SHAP Drivers: Lags Cortos (<200h)")
        metricas_fin(lags_largos, "SHAP Drivers: Lags Largos (>200h)")
    else:
        print("shap_drivers no disponible en parquet.")

if __name__ == '__main__':
    run_tests()

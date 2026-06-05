import pandas as pd
import json
import os
import glob

print("==== INVESTIGACIÓN FORENSE PROFUNDA: W2 ====")

# 1. Analizar por qué W2 falló Brier en la run actual
w2_cache_dir = r"g:\Mi unidad\ia\luna_v2\data\wfb_cache\W2\models"
g2_json = r"g:\Mi unidad\ia\luna_v2\data\reports\wfb\gate_G2_W2_seed42.json"

if os.path.exists(g2_json):
    with open(g2_json, "r") as f:
        g2 = json.load(f)
        print("\n[A] REPORTE G2 ACTUAL (W2):")
        print(json.dumps(g2.get("metrics", {}).get("brier_by_agent", {}), indent=2))
        
# Buscar firmas de XGBoost en W2 cache
sig_files = glob.glob(os.path.join(w2_cache_dir, "*_signature.json"))
print("\n[B] ANÁLISIS DE CALIBRACIÓN DE AGENTES EN W2 (ACTUAL):")
for sf in sig_files:
    if "calibrator" in sf or "ood" in sf or "metalabeler" in sf:
        continue
    with open(sf, "r") as f:
        data = json.load(f)
        metrics = data.get("metrics", {})
        agent = data.get("config", {}).get("agent_name", "unknown")
        print(f"  Agent {agent.upper()}:")
        print(f"    - Brier_OOS: {metrics.get('brier_score_oos', 'N/A')}")
        print(f"    - Brier_Naive: {metrics.get('brier_score_naive', 'N/A')}")
        print(f"    - ROC_AUC_OOS: {metrics.get('roc_auc_oos', 'N/A')}")
        print(f"    - EV (Expected Value): {metrics.get('expected_value_oos', 'N/A')}")

# 2. Analizar el rendimiento histórico de W2 en runs antiguas
print("\n[C] RENDIMIENTO HISTÓRICO DE W2 EN RUNS ANTERIORES:")
hist_parquets = glob.glob(r"g:\Mi unidad\ia\luna_v2\data\predictions\oos_trades_seed*.parquet")
df_hist = pd.DataFrame()
dfs = []
for hp in hist_parquets:
    try:
        _df = pd.read_parquet(hp)
        if 'window_id' in _df.columns or 'exit_time' in _df.columns:
            # Add seed info
            seed_name = os.path.basename(hp).split('_seed')[1].split('.parquet')[0]
            _df['seed'] = seed_name
            dfs.append(_df)
    except:
        pass

if dfs:
    df_hist = pd.concat(dfs, ignore_index=True)
    # Identificar trades de W2 (si no hay window_id explícito, usamos fechas estimadas)
    # W1 era Jan 2025 - Mar 2025. W2 podría ser Feb 2025 - Apr 2025 o algo así.
    if 'window_id' in df_hist.columns:
        w2_hist = df_hist[df_hist['window_id'] == 'W2']
    else:
        # HACK: Filter by date roughly if window_id doesn't exist
        print("Aviso: 'window_id' no encontrado, usando todos los trades para ver la fecha de inicio.")
        w2_hist = df_hist

    if len(w2_hist) > 0:
        winrate = w2_hist['is_win'].mean()
        ret_tot = w2_hist['return_pct'].sum()
        print(f"  Total Trades en W2 (histórico global): {len(w2_hist)}")
        print(f"  WinRate Histórico W2: {winrate:.2%}")
        print(f"  Retorno Total Histórico W2: {ret_tot:.4%}")
        
        print("\n  Desglose por Semilla en W2:")
        if 'seed' in w2_hist.columns:
             grouped = w2_hist.groupby('seed').agg(
                 trades=('is_win', 'count'),
                 winrate=('is_win', 'mean'),
                 retorno=('return_pct', 'sum')
             )
             print(grouped.to_string())
    else:
        print("  No se encontraron trades históricos identificados como W2.")
else:
    print("  No se pudieron cargar trades históricos.")


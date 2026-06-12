"""
Analisis definitivo de la causa raiz de la inversion de señal del MetaLabeler.
Testa si la correlacion negativa es real o un artefacto estadistico.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

reports_dir = Path("data/reports/wfb")
parquets = sorted(reports_dir.glob("oos_trades_W*_seed*.parquet"))
all_trades = []
for p in parquets:
    df = pd.read_parquet(p)
    parts = p.stem.split("_")
    df["window"] = parts[2]
    df["seed"] = int(parts[3].replace("seed", ""))
    all_trades.append(df)
df_all = pd.concat(all_trades, ignore_index=True)
df_all["is_win"] = (df_all["return_raw"] > 0).astype(int)

APPROVED = {100, 777, 1337, 39395, 70519, 61865, 44793}

print("="*80)
print("ANALISIS CRITICO: W3 domina la estadistica global")
print("="*80)
df_w3 = df_all[df_all["window"]=="W3"]
total = len(df_all)
w3n = len(df_w3)
print(f"W3 representa {w3n}/{total} trades ({w3n/total*100:.1f}%) del total")
w3_wr = df_w3["is_win"].mean()*100
print(f"W3 WR global: {w3_wr:.1f}%")
r_m3, _ = stats.spearmanr(df_w3["meta_v2_prob"], df_w3["return_raw"])
print(f"W3 Spearman meta vs return: r={r_m3:.4f}")
print()
print("W3 Regimenes HMM:")
print(df_w3["hmm_regime"].value_counts().to_dict())
print()

print("--- Distribucion meta_v2_prob por ventana ---")
for w in ["W1","W2","W3","W4","W5"]:
    dw = df_all[df_all["window"]==w]["meta_v2_prob"].dropna()
    if len(dw) < 10:
        continue
    print(f"  {w}: mean={dw.mean():.4f} std={dw.std():.4f} min={dw.min():.4f} max={dw.max():.4f} n={len(dw)}")

print()
print("="*80)
print("ANALISIS CRITICO: W4 Aprobadas (WR=90.7%, MetaLabeler INVERTIDO)")
print("="*80)
dw4_apr = df_all[(df_all["window"]=="W4") & df_all["seed"].isin(APPROVED)]
n4 = len(dw4_apr)
wr4 = dw4_apr["is_win"].mean()*100
print(f"W4 Aprobadas: n={n4}, WR={wr4:.1f}%")
if n4 > 5:
    r_m4, _ = stats.spearmanr(dw4_apr["meta_v2_prob"], dw4_apr["return_raw"])
    print(f"W4 Aprobadas Spearman meta: r={r_m4:.4f}")
    meta_m4 = dw4_apr["meta_v2_prob"].mean()
    meta_s4 = dw4_apr["meta_v2_prob"].std()
    print(f"W4 Aprobadas meta_v2_prob: mean={meta_m4:.4f} std={meta_s4:.4f}")
print()
print("Con WR=90.7%, casi todo gana. En ese rango la correlacion es estadisticamente")
print("inestable — un solo batch de perdidas con meta_prob alto domina el r de Spearman.")

print()
print("="*80)
print("TEST DEFINITIVO: ¿Artefacto de truncacion o inversion real de señal?")
print("="*80)
print()
print("Los parquets OOS contienen los trades que PASARON el filtro MetaLabeler.")
print("La pregunta es: ¿dentro de ese rango truncado, la correlacion es estable?")
print()

for w in ["W2","W3","W4","W5"]:
    dw = df_all[df_all["window"]==w].copy()
    if len(dw) < 20:
        continue
    meta_col = dw["meta_v2_prob"].dropna()
    p10 = np.percentile(meta_col, 10)
    p90 = np.percentile(meta_col, 90)
    rango = p90 - p10
    r_m, p_m = stats.spearmanr(meta_col, dw.loc[meta_col.index, "return_raw"])
    print(f"  {w}: rango[p10={p10:.4f}, p90={p90:.4f}] delta={rango:.4f} | Spearman r={r_m:.4f} (p={p_m:.4f})")

print()
print("Si delta (rango de meta_v2_prob en OOS) es MUY PEQUEÑO (<0.05),")
print("la correlacion de Spearman sera INESTABLE y no interpretable.")
print("Si el rango es amplio (>0.10), la correlacion negativa ES REAL.")

print()
print("="*80)
print("ANALISIS METALABELER: ¿Qué predice realmente?")
print("Comparacion de meta_v2_prob en seeds con señal positiva vs negativa")
print("="*80)

# Seeds donde el MetaLabeler SI funciona (r positivo): 38581
# Seeds donde el MetaLabeler esta invertido: la mayoria
print()
for seed, label in [(38581, "POSITIVO r=+0.42"), (777, "INVERTIDO r=-0.36"), (39395, "INVERTIDO r=-0.36")]:
    ds = df_all[df_all["seed"]==seed]
    if len(ds) < 10:
        print(f"  Seed {seed}: no hay datos suficientes")
        continue
    meta_v = ds["meta_v2_prob"]
    ret_v = ds["return_raw"]
    r, p = stats.spearmanr(meta_v, ret_v)
    print(f"  Seed {seed} ({label}):")
    print(f"    n={len(ds)}, WR={ds['is_win'].mean()*100:.1f}%")
    print(f"    meta_v2_prob: mean={meta_v.mean():.4f} std={meta_v.std():.4f} min={meta_v.min():.4f} max={meta_v.max():.4f}")
    print(f"    Spearman confirmado: r={r:.4f} (p={p:.4f})")
    # Cuartiles dentro de esta semilla
    try:
        ds2 = ds.copy()
        ds2["meta_q"] = pd.qcut(ds2["meta_v2_prob"], 4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        cq = ds2.groupby("meta_q", observed=True).agg(n=("is_win","count"), wr=("is_win","mean")).reset_index()
        for _, row in cq.iterrows():
            print(f"      {row['meta_q']}: n={int(row['n'])}, WR={row['wr']*100:.1f}%")
    except:
        pass
    print()

print()
print("="*80)
print("DIAGNOSTICO DEFINITIVO DEL OOD GUARD: Contexto temporal")
print("El OOD Guard entrena con datos HISTORICOS. OOS = 2025-2026 (post-ETF, post-halving)")
print("="*80)
print()

# Analisis OOD por ventana y su relation con el periodo temporal
print("OOD kl_distance invertido en TODAS las ventanas o solo algunas?")
print()
for w in ["W1","W2","W3","W4","W5"]:
    dw = df_all[df_all["window"]==w].copy()
    if len(dw) < 40:
        continue
    try:
        dw["ood_q"] = pd.qcut(dw["ood_kl_distance"], 4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        wr_q1 = dw[dw["ood_q"]=="Q1"]["is_win"].mean()*100
        wr_q4 = dw[dw["ood_q"]=="Q4"]["is_win"].mean()*100
        delta = wr_q1 - wr_q4
        r_o, p_o = stats.spearmanr(dw["ood_kl_distance"], dw["return_raw"])
        ood_m = dw["ood_kl_distance"].mean()
        ood_s = dw["ood_kl_distance"].std()
        direction = "INVERTIDO" if delta > 5 else ("NORMAL" if delta < -5 else "PLANO")
        print(f"  {w}: ood_mean={ood_m:.5f} std={ood_s:.5f} | WR_Q1={wr_q1:.1f}% WR_Q4={wr_q4:.1f}% delta={delta:+.1f}pp | Spearman r={r_o:.4f} -> {direction}")
    except Exception as e:
        print(f"  {w}: error {e}")

print()
print("="*80)
print("HIPOTESIS MAS PROBABLE PARA OOD GUARD:")
print()
print("1. OOS 2025-2026 es estructuralmente DIFERENTE del training 2022-2024")
print("   (ETF spot, halvings, mayor liquidez) -> el IsolationForest siempre ve OOS como anomalo")
print("2. Las 'anomalias' (KL bajo) son breakouts/rallies de alta velocidad")
print("   que el XGBoost captura bien pero el IsolationForest llama 'raro'")
print("3. El OOD Guard esta CENSURANDO las mejores oportunidades del sistema")
print()
print("CONCLUSION: El OOD Guard deberia ser DESACTIVADO o reconfigurado para OOS 2025+")
print("  Opcion A: contamination=0.20 (muy permisivo, apenas penaliza)")
print("  Opcion B: skip_ood_kelly_penalty=True (anular la penalizacion Kelly del OOD)")
print("  Opcion C: Re-entrenar el IsolationForest incluyendo datos 2025 en el train set")

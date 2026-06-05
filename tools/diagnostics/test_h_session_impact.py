import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 65)
print("  H-SESSION-IMPACT — FASE 1-3: Distribucion horaria real W3")
print("=" * 65)

# Cargar raw probs de W3 (2208 barras horarias Jul-Sep 2025)
pq = Path(r"g:\Mi unidad\ia\luna_v2\data\runs\WFB_20260602_002420_seed42\seed42\W3\oos_raw_probs.parquet")
df = pd.read_parquet(pq)
df["hour_utc"] = df.index.hour

print(f"Barras totales W3: {len(df)}")
print(f"Periodo: {df.index.min().date()} -> {df.index.max().date()}")
dias = (df.index.max() - df.index.min()).days + 1
print(f"Dias cubiertos: {dias}")
print()

# Distribucion horaria
hour_counts = df["hour_utc"].value_counts().sort_index()
print("Barras por hora UTC:")
for h, n in hour_counts.items():
    bar = "#" * (n // 3)
    en_713 = "[actual]" if 7 <= h <= 13 else ""
    en_620 = "[propuesto]" if 6 <= h <= 20 else ""
    print(f"  {h:02d}h: {n:4d}  {en_713} {en_620}")

print()

# Comparativa de gates
gates = {
    "[7-13] actual":     list(range(7, 14)),
    "[6-20] propuesto":  list(range(6, 21)),
    "[0-23] sin gate":   list(range(0, 24)),
}

base_n = df[df["hour_utc"].isin(range(7, 14))].shape[0]
print("Comparativa de Session Gates:")
print(f"  {'Gate':<22} {'N barras':<10} {'% total':<10} Factor_vs_actual")
print("  " + "-" * 56)
for name, hours in gates.items():
    mask = df["hour_utc"].isin(hours)
    n = mask.sum()
    pct = n / len(df) * 100
    factor = n / base_n if base_n > 0 else 0
    print(f"  {name:<22} {n:<10} {pct:<10.1f}% x{factor:.2f}")

print()

# Calidad de senal por hora — prob_range como proxy
print("Prob_range media por hora (proxy de densidad de senal RANGE):")
hour_quality = df.groupby("hour_utc")["prob_range"].mean()
print(f"  Media TOTAL:          {df['prob_range'].mean():.4f}")
inside_713 = df[df["hour_utc"].isin(range(7, 14))]["prob_range"]
inside_620 = df[df["hour_utc"].isin(range(6, 21))]["prob_range"]
outside_713 = df[~df["hour_utc"].isin(range(7, 14))]["prob_range"]
print(f"  Media [7-13] actual:  {inside_713.mean():.4f}  (N={len(inside_713)})")
print(f"  Media [6-20] prop.:   {inside_620.mean():.4f}  (N={len(inside_620)})")
print(f"  Media fuera [7-13]:   {outside_713.mean():.4f}  (N={len(outside_713)})")

# Test estadistico
t, p = stats.ttest_ind(inside_713, outside_713)
print(f"\n  t-test prob_range: [7-13] vs fuera-de-[7-13]: t={t:.3f}, p={p:.4f}")
ks, pks = stats.ks_2samp(inside_713, outside_713)
print(f"  KS-test:  KS={ks:.4f}, p={pks:.4f}")

print()
top5 = hour_quality.sort_values(ascending=False).head(8)
bottom5 = hour_quality.sort_values(ascending=False).tail(5)
print("  Horas con MAYOR prob_range (top 8):")
for h, q in top5.items():
    flag = "[actual 7-13]" if 7 <= h <= 13 else ("[propuesto 6-20]" if 6 <= h <= 20 else "[excluido]")
    print(f"    {int(h):02d}h: {q:.4f}  {flag}")
print("  Horas con MENOR prob_range (bottom 5):")
for h, q in bottom5.items():
    flag = "[actual 7-13]" if 7 <= h <= 13 else ("[propuesto 6-20]" if 6 <= h <= 20 else "[excluido]")
    print(f"    {int(h):02d}h: {q:.4f}  {flag}")

print()
# Proyeccion de N trades ajustada por calidad de senal
n_actual = 1  # 1 trade observado en W3/seed42 con gate [7-13]
factor_barras_620 = len(inside_620) / len(inside_713)
factor_calidad_620 = inside_620.mean() / inside_713.mean()
proj_620 = n_actual * factor_barras_620 * factor_calidad_620

factor_barras_nogat = len(df) / len(inside_713)
factor_calidad_nogat = df["prob_range"].mean() / inside_713.mean()
proj_nogat = n_actual * factor_barras_nogat * factor_calidad_nogat

print("Proyeccion N trades ajustada por calidad de senal (prob_range):")
print(f"  [7-13] actual:    {n_actual} trade por seed (observado)")
print(f"  [6-20] propuesto: {proj_620:.2f} trades por seed")
print(f"     Factor barras: x{factor_barras_620:.2f}  |  Factor calidad: x{factor_calidad_620:.3f}")
print(f"  Sin gate:         {proj_nogat:.2f} trades por seed")
print()

# Cuantas seeds para N=30 con cada gate
seeds_for_30_actual = 30 / n_actual
seeds_for_30_620 = 30 / proj_620 if proj_620 > 0 else 999
print(f"  Seeds necesarias para N=30:")
print(f"    [7-13] actual:    ~{seeds_for_30_actual:.0f} seeds")
print(f"    [6-20] propuesto: ~{seeds_for_30_620:.0f} seeds")
print()

# CONCLUSION
print("=" * 65)
print("  RESULTADO H-SESSION-IMPACT")
print("=" * 65)
print(f"  Barras disponibles [7-13]: {len(inside_713)} ({100*len(inside_713)/len(df):.1f}% del periodo)")
print(f"  Barras disponibles [6-20]: {len(inside_620)} ({100*len(inside_620)/len(df):.1f}% del periodo)")
print(f"  Factor multiplicador bruto: x{factor_barras_620:.2f}")
print(f"  Factor ajustado por calidad senal: x{proj_620:.2f}")
print(f"  p-value (t-test [7-13] vs fuera): {p:.4f}")
if p > 0.05:
    print("  => Las horas fuera de [7-13] tienen prob_range SIMILAR (p>0.05)")
    print("  => La densidad de senal es UNIFORME por hora")
    print("  => La proyeccion geometrica (x{:.2f}) es valida".format(factor_barras_620))
    confirmed = True
else:
    print("  => Las horas fuera de [7-13] tienen prob_range DISTINTO (p<0.05)")
    print("  => Usar proyeccion ajustada por calidad: x{:.2f}".format(proj_620))
    confirmed = proj_620 > 1.5

print()
if confirmed:
    print("  >>> H-SESSION-IMPACT: CONFIRMADA")
    print(f"  >>> Ampliar gate [7-13]->[6-20] multiplicaria N por x{proj_620:.1f}")
    print(f"  >>> De ~30 seeds necesarias -> ~{seeds_for_30_620:.0f} seeds para N=30")
else:
    print("  >>> H-SESSION-IMPACT: DESCARTADA o EFECTO MINIMO")

print()
print("[FIX-DIAG-H-SESSION-01] Test completado: Session Gate analizado sobre 2208 barras W3 reales.")

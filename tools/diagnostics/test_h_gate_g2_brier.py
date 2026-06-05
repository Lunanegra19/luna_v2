"""
test_h_gate_g2_brier.py
HIPOTESIS H-GATE-G2-BRIER: El umbral Brier adaptativo (0.2686) esta en el borde
de la distribucion de Brier CALM_BEAR — el modelo se descarta por varianza entre
seeds, no por incompetencia real del agente.
Protocolo 5 fases — diagnostico_cuantitativo.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import re, glob

BASE  = Path(r"g:\Mi unidad\ia\luna_v2")
LOGS  = BASE / "logs"

print("=" * 65)
print("  H-GATE-G2-BRIER — Distribucion Brier CALM_BEAR en 28 seeds")
print("=" * 65)

# FASE 1: Extraer valores Brier de CALM_BEAR de todos los worker logs
worker_logs_all = sorted(LOGS.glob("wfb_worker_*.log"), key=lambda x: x.stat().st_mtime)
# Solo logs de la run nocturna (despues de las 22:00 del 2026-06-01)
import time, datetime
cutoff = datetime.datetime(2026, 6, 1, 22, 0, 0).timestamp()
worker_logs = [l for l in worker_logs_all if l.stat().st_mtime >= cutoff]
print(f"\nWorker logs de la run nocturna: {len(worker_logs)}")

# Patrones de extraccion Brier
patterns = [
    # Brier score en Gate-G2
    r"GATE-G2.*?calm_bear.*?brier[_score]*[=\s:]+([0-9]\.[0-9]{3,6})",
    r"calm_bear.*?GATE-G2.*?brier[_score]*[=\s:]+([0-9]\.[0-9]{3,6})",
    r"brier[_score]*[=\s:]+([0-9]\.[0-9]{3,6}).*?calm_bear",
    # Forma generica en Gate-G2
    r"G2.*?brier.*?([0-9]\.[02][0-9]{2,5})",
    # Forma del validador estadistico
    r"Brier.*?calm.*?([0-9]\.[0-9]{3,5})",
    r"calm.*?Brier.*?([0-9]\.[0-9]{3,5})",
    # DEGRADED con valor
    r"DEGRADED.*?calm.*?([0-9]\.[02][0-9]{2,5})",
]

brier_data = []  # lista de (seed, brier_val, degraded)
thresholds = []

for log in worker_logs:
    # Extraer seed del nombre
    seed_match = re.search(r"wfb_worker_\d+_\d+_(\d+)\.log", log.name)
    seed = seed_match.group(1) if seed_match else "UNK"
    is_degraded = None
    brier_val = None

    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        continue

    # Intentar todos los patrones
    for pat in patterns:
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            for m in matches:
                v = float(m)
                if 0.15 < v < 0.40:  # rango plausible Brier
                    brier_val = v
                    break
        if brier_val:
            break

    # Determinar si fue DEGRADED
    if "GATE-G2" in content and "calm_bear" in content.lower():
        if "DEGRADED" in content and "calm_bear" in content.lower():
            is_degraded = True
        elif "OPERABLE" in content and "calm_bear" in content.lower():
            is_degraded = False

    # Extraer threshold si aparece
    thr_match = re.search(r"threshold[=\s:]+([0-9]\.[0-9]{3,5}).*?brier|brier.*?threshold[=\s:]+([0-9]\.[0-9]{3,5})",
                           content, re.IGNORECASE)
    if thr_match:
        thr = thr_match.group(1) or thr_match.group(2)
        if thr:
            thresholds.append(float(thr))

    if brier_val is not None or is_degraded is not None:
        brier_data.append({
            "seed": seed,
            "brier": brier_val,
            "degraded": is_degraded,
            "log": log.name[:50]
        })

df_brier = pd.DataFrame(brier_data)
print(f"\nRegistros extraidos: {len(df_brier)}")
print(f"Con valor Brier numerico: {df_brier['brier'].notna().sum()}")
print(f"Con estado DEGRADED:      {df_brier['degraded'].notna().sum()}")

# Mostrar lo que se extrajo
if not df_brier.empty:
    print("\nDetalle por seed:")
    for _, row in df_brier.iterrows():
        bstr = f"Brier={row['brier']:.4f}" if pd.notna(row['brier']) else "Brier=N/E"
        dstr = f"DEGRADED={row['degraded']}" if pd.notna(row['degraded']) else "DEGRADED=N/E"
        print(f"  seed{row['seed']}: {bstr}  {dstr}")

brier_vals = df_brier["brier"].dropna().values
threshold_val = np.mean(thresholds) if thresholds else None
print(f"\nUmbral extraido de logs: {threshold_val:.4f}" if threshold_val else "\nUmbral no encontrado en logs.")

# FALLBACK: si no se extraen Briers numericos, buscar en logs estadisticos
if len(brier_vals) < 3:
    print("\n[FALLBACK] Buscando en logs de statistical_validation...")
    stat_logs = sorted(LOGS.glob("run_statistical_validation_*.log"), key=lambda x: x.stat().st_mtime)
    stat_logs = [l for l in stat_logs if l.stat().st_mtime >= cutoff]
    print(f"  Logs de validacion encontrados: {len(stat_logs)}")
    for log in stat_logs[:5]:
        try:
            with open(log, encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Buscar Brier en logs de validacion
            for pat in patterns:
                matches = re.findall(pat, content, re.IGNORECASE)
                for m in matches:
                    v = float(m)
                    if 0.15 < v < 0.40:
                        brier_vals = np.append(brier_vals, v)
                        print(f"  {log.name[:50]}: Brier={v:.4f}")
        except:
            pass

# FASE 2: Analisis estadistico de la distribucion Brier
print()
print("=" * 65)
print("  FASE 2 — Analisis estadistico distribucion Brier CALM_BEAR")
print("=" * 65)

# Analisis de DEGRADED vs NO-DEGRADED
n_degraded_known = df_brier["degraded"].sum() if "degraded" in df_brier.columns else 0
n_total_known = df_brier["degraded"].notna().sum()

print(f"\n  Seeds con estado GATE-G2 conocido: {n_total_known}")
print(f"  Seeds DEGRADED:                    {n_degraded_known}")
print(f"  Seeds NO-DEGRADED:                 {n_total_known - n_degraded_known}")
if n_total_known > 0:
    pct_deg = 100 * n_degraded_known / n_total_known
    print(f"  Tasa DEGRADED:                     {pct_deg:.1f}%")
    print()

    # Test binomial: H0 = siempre incompetente (tasa = 100%)
    n_deg_int = int(n_degraded_known)
    p_always_bad = stats.binom_test(n_deg_int, n_total_known, 1.0, alternative="less")
    # H0 = varianza pura (tasa = 50%)
    p_variance   = stats.binom_test(n_deg_int, n_total_known, 0.5, alternative="two-sided")
    print(f"  binom_test H0='siempre_incompetente' (P_true=1.0): p={p_always_bad:.6f}")
    print(f"  binom_test H0='pura_varianza' (P_true=0.5):        p={p_variance:.4f}")
    if p_always_bad < 0.001:
        print("  => RECHAZAMOS que el modelo sea siempre incompetente (p<0.001)")
    if p_variance > 0.05:
        print(f"  => NO rechazamos que sea varianza entre seeds (p={p_variance:.4f} > 0.05)")
    else:
        print(f"  => La tasa {pct_deg:.0f}% DEGRADED es SIGNIFICATIVAMENTE distinta de 50%")

if len(brier_vals) >= 3:
    print()
    print("=== DISTRIBUCION BRIER NUMERICA ===")
    brier_arr = np.array(brier_vals)
    media = brier_arr.mean()
    std   = brier_arr.std()
    print(f"  N valores:   {len(brier_arr)}")
    print(f"  Media:       {media:.4f}")
    print(f"  Std:         {std:.4f}")
    print(f"  Min/Max:     {brier_arr.min():.4f} / {brier_arr.max():.4f}")
    print(f"  p25/p75:     {np.percentile(brier_arr,25):.4f} / {np.percentile(brier_arr,75):.4f}")

    if threshold_val:
        distancia_sigma = abs(threshold_val - media) / std if std > 0 else 0
        n_above = (brier_arr > threshold_val).sum()
        print(f"\n  Umbral adaptativo: {threshold_val:.4f}")
        print(f"  Distancia umbral a la media: {distancia_sigma:.2f} sigma")
        print(f"  Seeds con Brier > umbral: {n_above}/{len(brier_arr)} ({100*n_above/len(brier_arr):.0f}%)")
        if distancia_sigma < 0.5:
            print("  => Umbral dentro de 0.5 sigma -> ALTA VARIANZA (hipotesis confirmada)")
        elif distancia_sigma < 1.0:
            print("  => Umbral dentro de 1 sigma -> varianza moderada (hipotesis probable)")
        elif distancia_sigma < 1.5:
            print("  => Umbral entre 1-1.5 sigma -> efecto parcial")
        else:
            print("  => Umbral >1.5 sigma -> el modelo tiene Brier alto de verdad")

# RESULTADO FINAL
print()
print("=" * 65)
print("  RESULTADO H-GATE-G2-BRIER")
print("=" * 65)
if n_total_known >= 5:
    pct_deg_final = 100 * n_degraded_known / n_total_known
    p_always = stats.binom_test(int(n_degraded_known), n_total_known, 1.0, alternative="less")
    if p_always < 0.001 and pct_deg_final < 80:
        print("  >>> CONFIRMADA")
        print(f"  >>> {pct_deg_final:.0f}% seeds DEGRADED, no el 100% esperado si el modelo fuera incompetente")
        print("  >>> El umbral esta en el borde de la distribucion — es varianza, no incompetencia")
        print()
        print("  IMPLICACION: El umbral Brier adaptativo puede ser demasiado bajo.")
        print("  INSPECCION REQUERIDA: leer codigo Gate-G2 para entender como se calcula 0.2686.")
    else:
        print("  >>> DESCARTADA o INCIERTA")
        print(f"  >>> Con {pct_deg_final:.0f}% DEGRADED no podemos descartar incompetencia del modelo")
else:
    print("  >>> DATOS INSUFICIENTES para test estadistico concluyente")
    print("  >>> Se necesita leer el codigo Gate-G2 directamente")

print()
print("[FIX-DIAG-H-G2-01] Test H-GATE-G2-BRIER completado.")

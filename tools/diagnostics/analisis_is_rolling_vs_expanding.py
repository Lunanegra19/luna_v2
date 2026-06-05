# -*- coding: utf-8 -*-
"""
Analisis de IS real por ventana WFB: rolling vs expanding
y herramientas existentes para aumentar muestreo
"""
import pandas as pd
from pathlib import Path

BASE = Path("g:/Mi unidad/ia/luna_v2")

windows = [
    {"id":"W1", "train_end":"2024-10-31"},
    {"id":"W2", "train_end":"2025-01-31"},
    {"id":"W3", "train_end":"2025-04-30"},
    {"id":"W4", "train_end":"2025-07-31"},
    {"id":"W5", "train_end":"2025-10-31"},
]
holdout = [
    {"id":"W1", "hs":"2025-01-01", "he":"2025-03-31"},
    {"id":"W2", "hs":"2025-04-01", "he":"2025-06-30"},
    {"id":"W3", "hs":"2025-07-01", "he":"2025-09-30"},
    {"id":"W4", "hs":"2025-10-01", "he":"2025-12-31"},
    {"id":"W5", "hs":"2026-01-01", "he":"2026-03-31"},
]
GLOBAL_START = pd.Timestamp("2017-08-17", tz="UTC")
ROLLING_YEARS = 3

print("=== IS REAL POR VENTANA WFB ===")
print()
print(f"{'VT':<4} {'IS_start (rolling)':<22} {'IS_end':<12} {'IS_meses_rolling':<18} {'IS_meses_expanding':<20}")
print("-" * 80)
for w in windows:
    te = pd.Timestamp(w["train_end"], tz="UTC")
    rolling_start = max(te - pd.DateOffset(years=ROLLING_YEARS), GLOBAL_START)
    m_rolling   = (te.year - rolling_start.year)*12 + (te.month - rolling_start.month)
    m_expanding = (te.year - GLOBAL_START.year)*12 + (te.month - GLOBAL_START.month)
    bars_rolling   = m_rolling * 730
    bars_expanding = m_expanding * 730
    print(f"{w['id']:<4} {str(rolling_start.date()):<22} {str(te.date()):<12} {m_rolling:<6}m (~{bars_rolling:>7} barras)  {m_expanding:<5}m (~{bars_expanding:>7} barras)")

print()
print("=== RATIO EXPANDING vs ROLLING ===")
print("  Con rolling=3y: ~36 meses IS por ventana")
print("  Con expanding:  86-98 meses IS (2.4x mas datos)")
print()

# Cuanto RANGE hay en IS rolling vs expanding (usando HMM real)
print("=== N BARRAS RANGE EN IS ROLLING vs EXPANDING ===")
hmm_paths = list(BASE.glob("**/hmm_regime_labels.parquet"))
if hmm_paths:
    hmm = pd.read_parquet(hmm_paths[0])
    hmm.index = pd.to_datetime(hmm.index, utc=True, errors="coerce")
    col = "HMM_Semantic" if "HMM_Semantic" in hmm.columns else "HMM_Regime"

    for w in windows:
        te = pd.Timestamp(w["train_end"], tz="UTC")
        rolling_start = max(te - pd.DateOffset(years=ROLLING_YEARS), GLOBAL_START)

        # Rolling
        hmm_roll = hmm[(hmm.index >= rolling_start) & (hmm.index <= te)]
        n_range_roll = hmm_roll[col].astype(str).str.contains("RANGE").sum()
        n_total_roll = len(hmm_roll)

        # Expanding
        hmm_exp = hmm[hmm.index <= te]
        n_range_exp = hmm_exp[col].astype(str).str.contains("RANGE").sum()
        n_total_exp = len(hmm_exp)

        pct_roll = n_range_roll/n_total_roll*100 if n_total_roll > 0 else 0
        pct_exp  = n_range_exp/n_total_exp*100  if n_total_exp > 0 else 0

        print(f"  {w['id']}: RANGE rolling={n_range_roll:>6} ({pct_roll:.1f}%) | expanding={n_range_exp:>6} ({pct_exp:.1f}%) | ratio={n_range_exp/max(n_range_roll,1):.1f}x")
else:
    print("  hmm_regime_labels.parquet no disponible")

print()
print("=== HERRAMIENTAS EXISTENTES PARA AUMENTAR MUESTREO ===")
print()

# Verificar que existe training_mode en settings
print("1. training_mode (IMPLEMENTADO en train_xgboost_v2.py):")
print("   Actual: training_mode=rolling | rolling_window_years=3")
print("   Alternativa: training_mode=expanding -> usa TODO el IS disponible (2017-hoy)")
print("   Cambio: 1 linea en settings.yaml")
print()

# Verificar event_sampling_hours
from pathlib import Path
sf_path = BASE / "config" / "settings.yaml"
import re
content = sf_path.read_text(encoding="utf-8")
match_sampling = re.search(r"event_sampling_hours:\s*(\S+)", content)
match_mode = re.search(r"training_mode:\s*(\S+)", content)
match_rwy = re.search(r"rolling_window_years:\s*(\S+)", content)
match_cpcv = re.search(r"n_purged_splits:\s*(\S+)", content)

print(f"   training_mode actual: {match_mode.group(1) if match_mode else 'NO ENCONTRADO'}")
print(f"   rolling_window_years: {match_rwy.group(1) if match_rwy else 'NO ENCONTRADO'}")
print()

print("2. event_sampling_hours (IMPLEMENTADO en train_xgboost_v2.py):")
print(f"   Actual: event_sampling_hours={match_sampling.group(1) if match_sampling else '1 (default, sin sampling)'}")
print("   Funcion: reduce solapamiento de etiquetas TBM muestreando cada N horas")
print("   Efecto: REDUCE barras entrenamiento pero mejora independencia de muestras")
print()

print("3. CPCV (Combinatorial Purged Cross-Validation) — IMPLEMENTADO:")
print(f"   n_purged_splits={match_cpcv.group(1) if match_cpcv else '?'} -> genera combinaciones de splits")
print("   Aumenta el numero de paths de evaluacion sin aumentar N de datos")
print("   Ya activo — no mejora el N del dataset IS, sino la robustez de la CV")
print()

print("4. WFB expanding windows — YA CONFIGURADO (W1-W5):")
print("   El sistema tiene 5 ventanas WFB. Cada ventana amplia el IS en 3 meses.")
print("   Pero con rolling=3y, el inicio del IS tambien avanza -> N constante ~36m")
print()

print("5. Anchor Window (NO IMPLEMENTADO):")
print("   Fijar train_start siempre en 2017-08-17 y solo avanzar train_end")
print("   Seria equivalente a expanding pero con control explicito del anchor")
print("   Implementacion: cambiar training_mode=expanding en settings.yaml")
print()

print("=== CONCLUSION ===")
print()
print("HERRAMIENTA CLAVE YA IMPLEMENTADA pero mal configurada:")
print("  training_mode=rolling (ACTUAL) -> descarta datos pre-2022 en cada ventana")
print("  training_mode=expanding (DISPONIBLE) -> usa TODO el IS 2017-hasta-train_end")
print()
print("Impacto de cambiar a expanding:")
print("  RANGE IS W3 rolling:   ~5.500 barras  (36 meses)")
print("  RANGE IS W3 expanding: ~18.316 barras (ver H2 - dato real del parquet)")
print("  Ratio: ~3.3x mas barras RANGE disponibles para entrenar")
print()
print("Por que se usa rolling=3y actualmente:")
print("  Diseño intencional: datos recientes son mas relevantes para predecir OOS reciente")
print("  Problema: con 3 anos, el IS 2022-2025 tiene MENOS RANGE que el IS 2017-2025")
print("  porque 2022-2025 fue predominantemente BULL/CRASH, no RANGE")
print("  El rolling descarta precisamente el RANGE historico mas representativo (2018-2021)")

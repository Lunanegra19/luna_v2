"""arch18_calm_bear_training_audit.py
Verifica con que datos fue entrenado el agente calm_bear.
El modelo existe (xgboost_meta_calm_bear_long.model) pero la etiqueta 3_CALM_BEAR
tiene 0 barras en el IS. Esto implica que el agente se entrenó con 0 muestras positivas.
"""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-18] CALM_BEAR TRAINING DATA AUDIT")
print("="*70)

# ── 1. Ver firma del agente calm_bear ─────────────────────────────────────────
print("\n[1] FIRMA DEL AGENTE calm_bear")
print("-"*60)
sig_path = ROOT / "data" / "models" / "xgboost_meta_calm_bear_long_signature.json"
if sig_path.exists():
    sig = json.loads(sig_path.read_text("utf-8"))
    print(f"  Threshold: {sig.get('optimal_threshold')}")
    print(f"  Features ({len(sig.get('features', []))}): {sig.get('features', [])[:5]}...")
    print(f"  DSR: {sig.get('dsr_cpcv', sig.get('dsr', 'NO ENCONTRADO'))}")
    print(f"  Trades IS: {sig.get('n_trades_is', sig.get('n_pos', 'NO ENCONTRADO'))}")
    print(f"  Base rate IS: {sig.get('base_rate', sig.get('win_rate_is', 'NO ENCONTRADO'))}")
    # Mostrar todos los campos
    for k, v in sig.items():
        if k != "features":
            print(f"  {k}: {v}")
else:
    print("  ARCHIVO NO ENCONTRADO")

# ── 2. Verificar barras calm_bear en features_train ───────────────────────────
print("\n[2] BARRAS 3_CALM_BEAR EN FEATURES_TRAIN.PARQUET")
print("-"*60)
train_path = ROOT / "data" / "features" / "features_train.parquet"
if train_path.exists():
    # Leer solo la columna HMM_Semantic
    df_cols = pd.read_parquet(train_path, columns=["HMM_Semantic"])
    cb_count = (df_cols["HMM_Semantic"] == "3_CALM_BEAR").sum()
    bf_count = (df_cols["HMM_Semantic"] == "4_BEAR_FORCED").sum()
    bc_count = (df_cols["HMM_Semantic"] == "3_BEAR_CRASH").sum()
    print(f"  3_CALM_BEAR en IS:  {cb_count} barras")
    print(f"  4_BEAR_FORCED en IS: {bf_count} barras")
    print(f"  3_BEAR_CRASH en IS:  {bc_count} barras")
    print(f"\n  CONCLUSION: El agente calm_bear entrenó con {cb_count} muestras positivas.")
    if cb_count == 0:
        print("  --> El agente calm_bear es estadisticamente invalido (N=0 positivos en IS)")

# ── 3. Evaluar las dos opciones de fix ────────────────────────────────────────
print("\n[3] OPCIONES DE FIX PARA ARCH-18")
print("-"*60)
print("""
OPCION A — Mover 4_BEAR_FORCED a calm_bear en regime_mapping:
  calm_bear: ['4_BEAR_FORCED', '4_BEAR_FORCED_B', ...]   (4.152 barras IS)
  bear: ['3_BEAR_CRASH', '3_BEAR_CRASH_B', ...]          (1.349 barras IS)

  PRO: El agente calm_bear recibe datos reales y puede aprender
  CON: El agente calm_bear EXISTENTE fue entrenado con 0 muestras de 4_BEAR_FORCED
       -> Habria que REENTRENAR calm_bear con los datos correctos

OPCION B — Fusionar calm_bear con bear en regime_mapping:
  calm_bear: eliminado del mapping
  bear: ['3_BEAR_CRASH', '3_BEAR_CRASH_B', '4_BEAR_FORCED', ...]  (5.501 barras IS)

  PRO: El agente bear ya tiene el modelo entrenado con ambas etiquetas
       No requiere reentrenamiento inmediato
       Consolida N IS: 5.501 barras vs 220 (rolling_5y W5)
  CON: Se pierde la granularidad CALM_BEAR vs BEAR_CRASH

OPCION C — Eliminar calm_bear del regime_mapping (sin reentrenamiento):
  El router ignoraria el agente calm_bear
  Las barras 4_BEAR_FORCED ya van al agente 'bear' (que las incluye en su lista)
  -> Solo ajuste de settings.yaml, efecto inmediato

  ESTO YA ES LO QUE PASA:
  4_BEAR_FORCED esta en AMBAS listas: en 'bear' Y... esperar.
""")

# Verificar si 4_BEAR_FORCED esta en bear o solo en calm_bear
import yaml
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
rm = cfg.get("fase2", {}).get("regime_mapping", {})
calm_bear_labels = rm.get("calm_bear", [])
bear_labels = rm.get("bear", [])
print(f"  calm_bear labels: {calm_bear_labels}")
print(f"  bear labels: {bear_labels}")
print(f"  4_BEAR_FORCED en calm_bear: {'4_BEAR_FORCED' in calm_bear_labels}")
print(f"  4_BEAR_FORCED en bear: {'4_BEAR_FORCED' in bear_labels}")
print(f"\n  VEREDICTO:")
if "4_BEAR_FORCED" in bear_labels and "3_CALM_BEAR" not in bear_labels:
    print("  Las 4.152 barras de 4_BEAR_FORCED YA van al agente 'bear'.")
    print("  El agente 'calm_bear' recibe solo 3_CALM_BEAR* (0 barras en IS).")
    print("  FIX RECOMENDADO: Eliminar el agente calm_bear del mapping y mover")
    print("  sus labels al agente 'bear'. Efecto: barras clasificadas como")
    print("  3_CALM_BEAR en OOS runtime iran al agente bear en lugar de calm_bear.")
    print("  IMPORTANTE: calm_bear puede aparecer en OOS aunque tenga 0 barras IS")
    print("  por la asincronía HMM IS vs OOS documentada en ARCH-18.")

print("\n[ARCH-18] Audit completado.")

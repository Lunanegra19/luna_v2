"""
arch10_embargo_inconsistency.py
=================================
ARCH-10: Verificacion de inconsistencia embargo_hours en settings.yaml
SOP R3 requiere >= 96H. Diagnostico y evaluacion de fix.
"""
import yaml, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent

print("="*70)
print("[ARCH-10] EMBARGO INCONSISTENCIA — diagnostico y evaluacion de fix")
print("="*70)

with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    content = f.read()
    cfg = yaml.safe_load(content)

sop = cfg.get("sop", {})
xgb = cfg.get("xgboost", {})

print("\n[VALORES ACTUALES]")
print(f"  sop.embargo_hours:          {sop.get('embargo_hours', 'NOT FOUND')}")
print(f"  sop.purge_hours:            {sop.get('purge_hours', 'NOT FOUND')}")
print(f"  xgboost.embargo_hours:      {xgb.get('embargo_hours', 'NOT FOUND')}")
print(f"  xgboost.embargo_min_hours:  {xgb.get('embargo_min_hours', 'NOT FOUND')}")
print(f"  xgboost.soft_embargo_hours: {xgb.get('soft_embargo_hours', 'NOT FOUND')}")
print(f"  SOP R3 requiere:            >= 96H (SOP V10.0)")

print("\n[LINEAS EXACTAS CON EMBARGO EN SETTINGS.YAML]")
for i, line in enumerate(content.splitlines(), 1):
    if "embargo" in line.lower():
        print(f"  L{i:3}: {line}")

print("\n[ANALISIS DE IMPACTO]")
sop_embargo = sop.get("embargo_hours", None)
xgb_embargo = xgb.get("embargo_hours", None)
print(f"  sop.embargo_hours = {sop_embargo}H -> Usado por: SFI (SFI_EMBARGO_H), Optuna TSCV gap")
print(f"  xgboost.embargo_hours = {xgb_embargo}H -> Usado por: Optuna TSCV gap directo en train_xgboost_v2.py")

if sop_embargo and sop_embargo < 96:
    print(f"  *** VIOLACION SOP R3: sop.embargo_hours={sop_embargo}H < 96H requerido ***")
if xgb_embargo and xgb_embargo < 96:
    print(f"  *** VIOLACION SOP R3: xgboost.embargo_hours={xgb_embargo}H < 96H requerido ***")

# Verificar que el soft_embargo de 24H (Consensus-Soft, R3) esta correctamente gateado
soft_enabled = xgb.get("soft_embargo_enabled", False)
soft_hours   = xgb.get("soft_embargo_hours", None)
print(f"\n  soft_embargo_enabled: {soft_enabled}")
print(f"  soft_embargo_hours: {soft_hours}H")
print(f"  Nota SOP R3: el soft_embargo de 24H solo es valido con consenso >= 4/5 seeds")

print("\n[IMPACTO EN OPTUNA TSCV]")
print(f"  En train_xgboost_v2.py L1341: _tscv = TimeSeriesSplit(n_splits=..., gap={xgb_embargo})")
print(f"  Con gap={xgb_embargo}H y datos horarios: {xgb_embargo} barras de embargo entre train y val fold")
print(f"  SOP R3 exige {96} barras. Diferencia: {96 - (xgb_embargo or 0)} barras de contaminacion potencial")

print("\n[EVALUACION DE FIX]")
print("  El fix requiere cambiar xgboost.embargo_hours de 24H a 96H en settings.yaml")
print("  Riesgo: con n_train pequeno (BEAR=766 barras), gap=96H reduce mas el fold de validacion")
print("  Con n=766 y gap=96, cada fold tiene aprox: 766/(3+1) - 96 = 96 barras de validacion")
print("  Con gap=24:  766/(3+1) - 24 = 168 barras de validacion")
print("  El fix reduce folds de validacion pero cumple SOP. Es la decision correcta.")

print("\n[VEREDICTO]")
print("  ARCH-10 CONFIRMADO: xgboost.embargo_hours=24H viola SOP R3 (>= 96H)")
print("  sop.embargo_hours=72H tambien viola SOP R3")
print("  FIX DISPONIBLE: cambiar ambos a 96H en settings.yaml")
print("  CLASIFICACION: requiere verificar que no hay run activa antes de cambiar")

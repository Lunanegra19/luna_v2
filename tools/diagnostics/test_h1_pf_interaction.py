"""
DIAGNÓSTICO: test_h1_pf_interaction.py
Bug: hmm_dyn_min_pf_bull_low_n=1.0 excluye regímenes BULL con pocas señales IS
     → cobertura cero en OOS BULL (W2: 0 trades)

Hipótesis: Con n<3 señales IS, el PF no es estadísticamente discriminante.
           La exclusión explícita (hmm_volatile_bull_exclude) ya cubre los destructores.
           hmm_dyn_min_pf_bull_low_n debe bajar a 0.0 (sin restricción adicional).
"""
import sys
sys.path.insert(0, 'c:/Users/Usuario/Desktop/ia/luna_v2')
import pandas as pd
import numpy as np

print("=" * 65)
print("TEST: Interacción hmm_dyn_min_pf_bull_low_n vs Cobertura OOS")
print("=" * 65)

# Simular escenario W2:
# IS (val 2025-02→03): mercado bajista en transición
# OOS (2025-Q2): mercado BULL recovery

# Regímenes en IS con sus señales y PF simulados (aproximados de logs W2)
is_regimes = {
    '2_CALM_RANGE':    {'n': 8,  'pf': 1.15, 'bull': False},  # gana → INCLUIDO
    '3_BEAR_CRASH':    {'n': 15, 'pf': 0.72, 'bull': False},  # pierde → pero BEAR-LONG añade
    '1_VOLATILE_BULL': {'n': 2,  'pf': 0.85, 'bull': True},   # excluido por H1 explícito
    '1_BULL_TREND':    {'n': 2,  'pf': 0.92, 'bull': True},   # n<3, PF<1.0 → ¿EXCLUIDO?
    '1_VOLATILE_BULL_B': {'n': 1, 'pf': 0.80, 'bull': True},  # n<3, PF<1.0 → ¿EXCLUIDO?
}

excluded_explicit = ['1_VOLATILE_BULL', '1_VOLATILE_BULL_C']

# Regímenes presentes en OOS W2
oos_regimes = ['3_BEAR_CRASH', '2_CALM_RANGE', '1_BULL_TREND', '1_VOLATILE_BULL_B', '1_VOLATILE_BULL']

print("\n--- ESCENARIO ACTUAL (hmm_dyn_min_pf_bull_low_n=1.0) ---")
allowed_strict = []
for regime, data in is_regimes.items():
    # Exclusión explícita
    if regime in excluded_explicit:
        print(f"  {regime:25s}: EXCLUIDO explícito (H1-FIX)")
        continue
    n, pf, is_bull = data['n'], data['pf'], data['bull']
    if n >= 3:
        if pf > 1.05 and True:  # total_ret > 0 simplificado
            allowed_strict.append(regime)
            print(f"  {regime:25s}: INCLUIDO (n={n}>=3, PF={pf:.2f}>1.05)")
        elif is_bull and pf > 0.95:
            allowed_strict.append(regime)
            print(f"  {regime:25s}: INCLUIDO BULL (n={n}>=3, PF={pf:.2f}>0.95)")
        else:
            print(f"  {regime:25s}: EXCLUIDO (n={n}>=3 pero PF={pf:.2f}<=1.05)")
    else:  # n < 3
        if is_bull and pf > 1.0:  # hmm_dyn_min_pf_bull_low_n=1.0
            allowed_strict.append(regime)
            print(f"  {regime:25s}: INCLUIDO BULL-LOWN (n={n}<3, PF={pf:.2f}>1.0)")
        else:
            print(f"  {regime:25s}: EXCLUIDO (n={n}<3, PF={pf:.2f}<=1.0, min_pf=1.0)")

# BEAR-LONG añade todos los bear
bear_regimes = ['3_BEAR_CRASH', '3_CALM_BEAR', '3_BEAR_CRASH_B', '3_BEAR_CRASH_C', '3_BEAR_CRASH_D']
allowed_strict += [b for b in bear_regimes if b not in allowed_strict]

print(f"\n  Lista final permitida: {allowed_strict}")
covered = [r for r in oos_regimes if r in allowed_strict]
uncovered = [r for r in oos_regimes if r not in allowed_strict and r not in excluded_explicit]
print(f"  Regímenes OOS cubiertos:   {covered}")
print(f"  Regímenes OOS SIN cubrir:  {uncovered}")
if uncovered:
    print(f"  ⚠️  COBERTURA INCOMPLETA → trades = 0 en regímenes: {uncovered}")

print("\n--- ESCENARIO PROPUESTO (hmm_dyn_min_pf_bull_low_n=0.0) ---")
print("  (n<3 bull: incluir por defecto, la exclusión la hace hmm_volatile_bull_exclude)")
allowed_relaxed = []
for regime, data in is_regimes.items():
    if regime in excluded_explicit:
        print(f"  {regime:25s}: EXCLUIDO explícito (H1-FIX)")
        continue
    n, pf, is_bull = data['n'], data['pf'], data['bull']
    if n >= 3:
        if pf > 1.05:
            allowed_relaxed.append(regime)
            print(f"  {regime:25s}: INCLUIDO (n={n}>=3, PF={pf:.2f}>1.05)")
        elif is_bull and pf > 0.95:
            allowed_relaxed.append(regime)
            print(f"  {regime:25s}: INCLUIDO BULL (n={n}>=3, PF={pf:.2f}>0.95)")
        else:
            print(f"  {regime:25s}: EXCLUIDO (n={n}>=3 pero PF={pf:.2f}<=0.95)")
    else:  # n < 3, min_pf=0.0 → incluir BULL por defecto
        if is_bull:
            allowed_relaxed.append(regime)
            print(f"  {regime:25s}: INCLUIDO BULL-DEFAULT (n={n}<3, min_pf=0.0)")

allowed_relaxed += [b for b in bear_regimes if b not in allowed_relaxed]
print(f"\n  Lista final permitida: {allowed_relaxed}")
covered2 = [r for r in oos_regimes if r in allowed_relaxed]
uncovered2 = [r for r in oos_regimes if r not in allowed_relaxed and r not in excluded_explicit]
print(f"  Regímenes OOS cubiertos:   {covered2}")
print(f"  Regímenes OOS SIN cubrir:  {uncovered2}")
if not uncovered2:
    print(f"  ✅ COBERTURA COMPLETA — excluidos solo los destructores explícitos")

print("\n--- CONCLUSIÓN ---")
print(f"  Con min_pf=1.0: {len(covered)}/{len(oos_regimes)} regímenes OOS cubiertos → COLAPSO")
print(f"  Con min_pf=0.0: {len(covered2)}/{len(oos_regimes)} regímenes OOS cubiertos → CORRECTO")
print()
print("  FIX: Bajar hmm_dyn_min_pf_bull_low_n a 0.0 en settings.yaml")
print("  Los regímenes destructores ya están cubiertos por hmm_volatile_bull_exclude")
print("  No se necesita doble restricción en n<3 con PF>1.0")

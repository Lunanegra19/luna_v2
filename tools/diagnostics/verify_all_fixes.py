"""
verify_all_fixes.py — Verifica que todos los fixes post-audit estan correctamente aplicados.
Checks: OBS-03, CRIT-03, CRIT-02, CRIT-01, SIGN-03.
"""
import ast, sys

RESULTS = []

def check(label, condition, detail=""):
    status = "OK" if condition else "FAIL"
    RESULTS.append((status, label, detail))
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))

# ---- OBS-03: HMM_Regime dtype fix en feature_pipeline.py ----
fp_fp = r'g:\Mi unidad\ia\luna_v2\luna\features\feature_pipeline.py'
with open(fp_fp, encoding='utf-8') as f:
    fp_content = f.read()

print("\n=== OBS-03: HMM_Regime dtype float64->int ===")
check("FIX-OBS-03-HMM-DTYPE tag presente", 'FIX-OBS-03-HMM-DTYPE' in fp_content)
check("Conversion fillna(-1).astype(int) presente", 'fillna(-1).astype(int)' in fp_content)
check("Protegido dentro del loop datasets.items()", 'for _hmm_dtype_col in ["HMM_Regime"]' in fp_content)
try:
    ast.parse(fp_content)
    check("feature_pipeline.py sintaxis OK", True)
except SyntaxError as e:
    check("feature_pipeline.py sintaxis OK", False, str(e))

# ---- CRIT-03: OOD Guard whitelist ----
fp_ood = r'g:\Mi unidad\ia\luna_v2\luna\utils\ood_feature_guard.py'
with open(fp_ood, encoding='utf-8') as f:
    ood_content = f.read()

print("\n=== CRIT-03: OOD Guard structural_features whitelist ===")
check("structural_features: list = [] en __init__", 'structural_features: list = []' in ood_content)
check("_load_from_settings carga whitelist", 'structural_features = list(getattr' in ood_content)
check("_analyze_feature tiene STRUCTURAL_EXEMPT", 'STRUCTURAL_EXEMPT' in ood_content)
check("FIX-CRIT-03 tag presente", 'FIX-CRIT-03' in ood_content)
try:
    ast.parse(ood_content)
    check("ood_feature_guard.py sintaxis OK", True)
except SyntaxError as e:
    check("ood_feature_guard.py sintaxis OK", False, str(e))

# ---- CRIT-03: settings.yaml whitelist ----
fp_cfg = r'g:\Mi unidad\ia\luna_v2\config\settings.yaml'
with open(fp_cfg, encoding='utf-8') as f:
    cfg_content = f.read()

print("\n=== CRIT-03: settings.yaml ood_guard.structural_features ===")
check("structural_features en settings.yaml", 'structural_features:' in cfg_content)
check("HMM_Regime en whitelist", '- HMM_Regime' in cfg_content)
check("HMM_Semantic en whitelist", '- HMM_Semantic' in cfg_content)

# ---- CRIT-02: MCW adaptive en train_xgboost_v2.py ----
fp_xgb = r'g:\Mi unidad\ia\luna_v2\luna\models\train_xgboost_v2.py'
with open(fp_xgb, encoding='utf-8') as f:
    xgb_content = f.read()

print("\n=== CRIT-02: MCW adaptive para n_train pequeno ===")
check("FIX-CRIT-02-MCW-ADAPTIVE tag presente", 'FIX-CRIT-02-MCW-ADAPTIVE' in xgb_content)
check("_mcw_max_adaptive calculado", '_mcw_max_adaptive = min(_mcw_max' in xgb_content)
check("_n_train_agent = len(self.X) presente", '_n_train_agent = len(self.X)' in xgb_content)

# ---- CRIT-01: n_splits adaptativo ----
print("\n=== CRIT-01: n_splits_is adaptativo al tamanio del agente ===")
check("FIX-CRIT-01-NSPLITS tag presente", 'FIX-CRIT-01-NSPLITS' in xgb_content)
check("Logica adaptativa n<2000/5000 presente", '_n_train_for_splits < 2000' in xgb_content)
check("Ya no usa formula n_months//4320 erronea", '_n_months = max(1, len(self.X)' not in xgb_content)

try:
    ast.parse(xgb_content)
    check("train_xgboost_v2.py sintaxis OK", True)
except SyntaxError as e:
    check("train_xgboost_v2.py sintaxis OK", False, str(e))

# ---- SIGN-03: n_estimators_min_floor documentado ----
print("\n=== SIGN-03: n_estimators_min_floor documentado ===")
check("FIX-XGB-NEST-FLOOR comentario en settings.yaml", 'FIX-XGB-NEST-FLOOR' in cfg_content)
check("n_estimators_min_floor: 100 presente", 'n_estimators_min_floor: 100' in cfg_content)

# ---- Resumen final ----
print("\n" + "="*60)
n_ok   = sum(1 for s, *_ in RESULTS if s == "OK")
n_fail = sum(1 for s, *_ in RESULTS if s == "FAIL")
print(f"RESULTADO FINAL: {n_ok} OK / {n_fail} FAIL")
if n_fail > 0:
    print("FIXES INCOMPLETOS:")
    for s, label, detail in RESULTS:
        if s == "FAIL":
            print(f"  - {label}: {detail}")
    sys.exit(1)
else:
    print("TODOS LOS FIXES VERIFICADOS -- LISTO PARA RELANZAR RUN")

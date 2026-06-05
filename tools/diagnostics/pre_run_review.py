"""
PRE-RUN REVIEW SCRIPT — Verifica todas las implementaciones de la sesion.
[2026-06-03]
Implementaciones a verificar:
  1. SFI-BALANCE-01   : Cuotas categoricas macro/onchain/calendar en feature_selection_e.py
  2. DXY-HMM-01       : DXY condicional al regimen HMM en feature_pipeline.py
  3. EXCHANGE-FLOW-01 : Exchange Net Flows en fetch_onchain.py
  4. LTH-SUPPLY-01    : LTH Supply proxy en fetch_onchain.py
  5. SHAP-AUDIT-01    : Auditor de importancia en luna/monitoring/ + pipeline_executor.py
  6. Settings sync    : Todos los parametros registrados en settings.yaml
"""
import sys, os, inspect, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'g:\Mi unidad\ia\luna_v2')

ROOT = Path(r'g:\Mi unidad\ia\luna_v2')
PASS = []
FAIL = []
WARN = []

def ok(msg):
    PASS.append(msg)
    print(f"  [OK]  {msg}")

def fail(msg):
    FAIL.append(msg)
    print(f"  [FAIL] {msg}")

def warn(msg):
    WARN.append(msg)
    print(f"  [WARN] {msg}")

print("=" * 70)
print("[PRE-RUN REVIEW] Verificacion de todas las implementaciones")
print("=" * 70)

# ─── 1. SFI-BALANCE-01 ────────────────────────────────────────────────────────
print()
print("-- 1. SFI-BALANCE-01 (feature_selection_e.py) ---------------------")
try:
    sfi_src = (ROOT / 'luna' / 'features' / 'feature_selection_e.py').read_text(encoding='utf-8')
    checks = [
        ('SFI-BALANCE-01', 'Tag de implementacion presente'),
        ('sfi_macro_features', 'Lista macro registrada'),
        ('sfi_onchain_features', 'Lista onchain registrada'),
        ('sfi_calendar_features', 'Lista calendar registrada'),
        ('sfi_macro_min_slots', 'Cuota macro activa'),
        ('sfi_onchain_min_slots', 'Cuota onchain activa'),
        ('sfi_calendar_min_slots', 'Cuota calendar activa'),
        ('_apply_category_quota', 'Funcion de cuota por categoria'),
        ('SFI_MACRO_MIN_SLOTS', 'Variable cuota macro'),
        ('SFI_ONCHAIN_MIN_SLOTS', 'Variable cuota onchain'),
    ]
    for tag, desc in checks:
        if tag in sfi_src:
            ok(f"SFI: {desc}")
        else:
            fail(f"SFI: {desc} — '{tag}' no encontrado")
except Exception as e:
    fail(f"SFI: No se pudo leer feature_selection_e.py: {e}")

# ─── 2. DXY-HMM-01 ────────────────────────────────────────────────────────────
print()
print("-- 2. DXY-HMM-01 (feature_pipeline.py) ----------------------------")
try:
    fp_src = (ROOT / 'luna' / 'features' / 'feature_pipeline.py').read_text(encoding='utf-8')
    checks = [
        ('DXY-HMM-01', 'Tag de implementacion'),
        ('DXY_HMM_cond', 'Feature DXY_HMM_cond'),
        ('DXY_HMM_bull_neg', 'Feature DXY_HMM_bull_neg'),
        ('DXY_HMM_interact', 'Feature DXY_HMM_interact'),
        ('apply_derived_features', 'En la funcion correcta'),
        ('HMM_Regime', 'Usa HMM_Regime como condicion'),
    ]
    for tag, desc in checks:
        if tag in fp_src:
            ok(f"DXY-HMM: {desc}")
        else:
            fail(f"DXY-HMM: {desc} — '{tag}' no encontrado")
except Exception as e:
    fail(f"DXY-HMM: No se pudo leer feature_pipeline.py: {e}")

# ─── 3. EXCHANGE-FLOW-01 ─────────────────────────────────────────────────────
print()
print("-- 3. EXCHANGE-FLOW-01 (fetch_onchain.py) -------------------------")
try:
    from luna.data.fetch_onchain import OnchainFetcher
    f = OnchainFetcher()
    if hasattr(f, 'fetch_cryptoquant_netflow'):
        ok("EXCHANGE-FLOW: fetch_cryptoquant_netflow existe")
    else:
        fail("EXCHANGE-FLOW: fetch_cryptoquant_netflow NO existe")

    oc_src = inspect.getsource(f.build_onchain_dataset)
    checks = [
        ('fetch_cryptoquant_netflow', 'Integrado en build_onchain_dataset'),
        ('EXCHANGE-FLOW-01', 'Tag presente en builder'),
    ]
    for tag, desc in checks:
        if tag in oc_src:
            ok(f"EXCHANGE-FLOW: {desc}")
        else:
            fail(f"EXCHANGE-FLOW: {desc} — '{tag}' no en builder")

    # Verificar las 5 features que genera
    nf_src = inspect.getsource(f.fetch_cryptoquant_netflow)
    for feat in ['Exchange_NetFlow', 'Exchange_NetFlow_7dEMA', 'Exchange_NetFlow_z30d',
                 'Exchange_NetFlow_Accum30d', 'Exchange_Outflow_Signal']:
        if feat in nf_src:
            ok(f"EXCHANGE-FLOW: Feature '{feat}' definida")
        else:
            fail(f"EXCHANGE-FLOW: Feature '{feat}' NO definida")
except Exception as e:
    fail(f"EXCHANGE-FLOW: Error al verificar: {e}")

# ─── 4. LTH-SUPPLY-01 ─────────────────────────────────────────────────────────
print()
print("-- 4. LTH-SUPPLY-01 (fetch_onchain.py) ----------------------------")
try:
    from luna.data.fetch_onchain import OnchainFetcher
    f = OnchainFetcher()
    if hasattr(f, 'fetch_lth_supply_proxy'):
        ok("LTH-SUPPLY: fetch_lth_supply_proxy existe")
    else:
        fail("LTH-SUPPLY: fetch_lth_supply_proxy NO existe")

    oc_src = inspect.getsource(f.build_onchain_dataset)
    if 'fetch_lth_supply_proxy' in oc_src:
        ok("LTH-SUPPLY: Integrado en build_onchain_dataset")
    else:
        fail("LTH-SUPPLY: NO integrado en build_onchain_dataset")

    lth_src = inspect.getsource(f.fetch_lth_supply_proxy)
    for feat in ['NonEx_Supply', 'Exchange_Supply_Pct', 'LTH_Supply_Change_30d',
                 'LTH_Accum_Signal', 'NonEx_Supply_z90d']:
        if feat in lth_src:
            ok(f"LTH-SUPPLY: Feature '{feat}' definida")
        else:
            fail(f"LTH-SUPPLY: Feature '{feat}' NO definida")

    for src_check in ['CoinMetrics', 'SplyExNtv', 'LTH-SUPPLY-01']:
        if src_check in lth_src:
            ok(f"LTH-SUPPLY: '{src_check}' presente")
        else:
            fail(f"LTH-SUPPLY: '{src_check}' NO encontrado")
except Exception as e:
    fail(f"LTH-SUPPLY: Error al verificar: {e}")

# ─── 5. SHAP-AUDIT-01 ─────────────────────────────────────────────────────────
print()
print("-- 5. SHAP-AUDIT-01 (shap_feature_auditor.py + pipeline_executor.py) --")
try:
    audit_path = ROOT / 'luna' / 'monitoring' / 'shap_feature_auditor.py'
    if audit_path.exists():
        ok("SHAP-AUDIT: shap_feature_auditor.py existe en luna/monitoring/")
        audit_src = audit_path.read_text(encoding='utf-8')
        for tag in ['run_shap_audit', '_extract_importance_by_category',
                    '_detect_alerts', 'SHAP-AUDIT-01', 'consecutive_windows_alert',
                    'audit_history.json']:
            if tag in audit_src:
                ok(f"SHAP-AUDIT: '{tag}' presente")
            else:
                fail(f"SHAP-AUDIT: '{tag}' NO encontrado")
    else:
        fail("SHAP-AUDIT: shap_feature_auditor.py NO existe")

    exec_src = (ROOT / 'luna' / 'pipeline_executor.py').read_text(encoding='utf-8')
    if 'shap_feature_auditor' in exec_src and 'SHAP-AUDIT-01' in exec_src:
        ok("SHAP-AUDIT: Hook en pipeline_executor.py")
    else:
        fail("SHAP-AUDIT: Hook NO encontrado en pipeline_executor.py")
except Exception as e:
    fail(f"SHAP-AUDIT: Error al verificar: {e}")

# ─── 6. Settings.yaml sync ────────────────────────────────────────────────────
print()
print("-- 6. settings.yaml — sincronizacion completa ---------------------")
try:
    from config.settings import cfg

    # Macro features
    wl_macro = list(getattr(cfg.features, 'sfi_macro_features', []) or [])
    for f_name in ['DXY_HMM_cond', 'DXY_HMM_bull_neg', 'DXY_HMM_interact']:
        if f_name in wl_macro:
            ok(f"settings: '{f_name}' en sfi_macro_features")
        else:
            fail(f"settings: '{f_name}' NO en sfi_macro_features")

    # Onchain features
    wl_onchain = list(getattr(cfg.features, 'sfi_onchain_features', []) or [])
    onchain_expected = [
        'Exchange_NetFlow_z30d', 'Exchange_NetFlow_Accum30d', 'Exchange_Outflow_Signal',
        'LTH_Supply_Change_30d', 'LTH_Accum_Signal', 'NonEx_Supply_z90d', 'Exchange_Supply_Pct'
    ]
    for f_name in onchain_expected:
        if f_name in wl_onchain:
            ok(f"settings: '{f_name}' en sfi_onchain_features")
        else:
            fail(f"settings: '{f_name}' NO en sfi_onchain_features")

    # Unified boost list
    wl_stable = list(getattr(cfg.features, 'sfi_macro_stable_features', []) or [])
    boost_expected = [
        'DXY_HMM_cond', 'Exchange_NetFlow_z30d', 'LTH_Supply_Change_30d',
        'LTH_Accum_Signal', 'Exchange_Supply_Pct'
    ]
    for f_name in boost_expected:
        if f_name in wl_stable:
            ok(f"settings: '{f_name}' en sfi_macro_stable_features (boost)")
        else:
            fail(f"settings: '{f_name}' NO en boost list")

    # Cuotas minimas
    macr_min = getattr(cfg.features, 'sfi_macro_min_slots', None)
    onch_min = getattr(cfg.features, 'sfi_onchain_min_slots', None)
    cal_min  = getattr(cfg.features, 'sfi_calendar_min_slots', None)
    if macr_min is not None: ok(f"settings: sfi_macro_min_slots={macr_min}")
    else: fail("settings: sfi_macro_min_slots NO encontrado")
    if onch_min is not None: ok(f"settings: sfi_onchain_min_slots={onch_min}")
    else: fail("settings: sfi_onchain_min_slots NO encontrado")
    if cal_min is not None: ok(f"settings: sfi_calendar_min_slots={cal_min}")
    else: fail("settings: sfi_calendar_min_slots NO encontrado")

    # SHAP audit params
    shap_cfg = getattr(cfg, 'shap_audit', None)
    if shap_cfg is not None:
        ok(f"settings: shap_audit presente | CUTOFF = {getattr(shap_cfg,'min_importance_threshold','?')}")
    else:
        fail("settings: shap_audit NO encontrado")

    print(f"  [INFO] sfi_macro_features: {len(wl_macro)} | sfi_onchain_features: {len(wl_onchain)} | boost: {len(wl_stable)}")

except Exception as e:
    fail(f"settings: Error al verificar: {e}")

# ─── 7. Imports críticos ──────────────────────────────────────────────────────
print()
print("-- 7. Imports criticos sin errores ---------------------------------")
critical_imports = [
    ('luna.data.fetch_onchain', 'OnchainFetcher'),
    ('luna.features.feature_pipeline', 'FeaturePipeline'),
    ('luna.features.feature_selection_e', None),
    ('luna.monitoring.shap_feature_auditor', 'run_shap_audit'),
    ('config.settings', 'cfg'),
]
for mod_name, attr in critical_imports:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        if attr and not hasattr(mod, attr):
            fail(f"Import '{mod_name}': '{attr}' no exportado")
        else:
            ok(f"Import '{mod_name}'" + (f" -> {attr}" if attr else ""))
    except Exception as e:
        fail(f"Import '{mod_name}': {e}")

# ─── RESUMEN ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"[RESUMEN PRE-RUN]")
print(f"  PASS:  {len(PASS)}")
print(f"  WARN:  {len(WARN)}")
print(f"  FAIL:  {len(FAIL)}")
if FAIL:
    print()
    print("  FALLOS A RESOLVER:")
    for f in FAIL:
        print(f"    - {f}")
if WARN:
    print()
    print("  ADVERTENCIAS:")
    for w in WARN:
        print(f"    - {w}")
print("=" * 70)
if not FAIL:
    print("  LISTO PARA ACTUALIZAR DATA LAKE Y LANZAR RUN")
else:
    print("  !! RESOLVER FALLOS ANTES DE CONTINUAR !!")

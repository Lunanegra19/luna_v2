"""
arch22_29_code_inspection.py
================================
Diagnostico rapido de ARCH-22, 24, 26, 27, 28, 29 mediante inspeccion de codigo.

ARCH-22: MetaLabelerV2 activo con N<50
ARCH-24: HMM state map no determinista
ARCH-26: lags estaticos en feature_pipeline
ARCH-27: rolling normalization IS->OOS shift
ARCH-28: MockXGB como generador de 22 trades
ARCH-29: LightGBM activo/zombie

USO: python tools/diagnostics/arch22_29_code_inspection.py
"""
import sys, re
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent.parent
LUNA = ROOT / "luna"

def grep_file(path, pattern, max_hits=5):
    results = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                results.append((i+1, line.strip()))
    except: pass
    return results[:max_hits]

print("="*70)
print("[ARCH-22-29] DIAGNOSTICO BATCH RAPIDO")
print("="*70)

# ── ARCH-22: MetaLabelerV2 activo con N<50 ────────────────────────────────────
print("\n[ARCH-22] METALABELERV2: ¿activo o skip_metalabeler=True?")
print("-"*60)
with open(ROOT/"config"/"settings.yaml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
skip_meta = cfg.get("xgboost", {}).get("skip_metalabeler", "NO ENCONTRADO")
print(f"  skip_metalabeler: {skip_meta}")
# Verificar el codigo del passthrough
calib_path = LUNA / "models" / "calibrate_probabilities.py"
hits = grep_file(calib_path, r"CALIB.PASSTHROUGH|n_xgb_pass|skip_metalabeler")
print(f"  calibrate_probabilities.py PASSTHROUGH hits:")
for lno, line in hits:
    print(f"    L{lno:4}: {line[:120]}")

# ── ARCH-24: HMM state map determinism ───────────────────────────────────────
print("\n[ARCH-24] HMM STATE MAP: ¿n_init fijo? ¿random_state fijo?")
print("-"*60)
hmm_path = LUNA / "models" / "hmm_regime.py"
hits_rs = grep_file(hmm_path, r"random_state|n_init|covariance_type")
for lno, line in hits_rs:
    print(f"  L{lno:4}: {line[:120]}")
# Verificar si el modelo es reproducible
hits_seed = grep_file(hmm_path, r"np\.random\.seed|torch\.manual_seed")
print(f"  Global seed fixes: {len(hits_seed)} hits")

# ── ARCH-26: Safety lags estaticos ─────────────────────────────────────────────
print("\n[ARCH-26] LAGS ESTATICOS EN FEATURE_PIPELINE")
print("-"*60)
fp_path = LUNA / "features" / "feature_pipeline.py"
lag_hits = grep_file(fp_path, r"shift\(|\.shift|_milag|lag_days|safety.lag|SAFETY_LAG", max_hits=10)
print(f"  Hits de lag/shift en feature_pipeline.py:")
for lno, line in lag_hits:
    print(f"  L{lno:4}: {line[:120]}")
# Verificar KNOWN_GRANGER_LAGS en SFI
sfi_path = LUNA / "features" / "feature_selection_e.py"
granger_hits = grep_file(sfi_path, r"KNOWN_GRANGER_LAGS|AutoLagDiscovery|lag_override")
print(f"\n  KNOWN_GRANGER_LAGS en feature_selection_e.py:")
for lno, line in granger_hits:
    print(f"  L{lno:4}: {line[:120]}")

# ── ARCH-27: Rolling normalization ─────────────────────────────────────────────
print("\n[ARCH-27] ROLLING NORMALIZATION: ¿aplica stats IS en OOS?")
print("-"*60)
rn_path = LUNA / "features" / "rolling_normalization.py"
if rn_path.exists():
    print(f"  rolling_normalization.py: EXISTE ({rn_path.stat().st_size} bytes)")
    hits_rn = grep_file(rn_path, r"fit_transform|transform|mean_|std_|fit\(|fitted")
    for lno, line in hits_rn:
        print(f"  L{lno:4}: {line[:120]}")
    # Verificar si guarda stats IS
    hits_save = grep_file(rn_path, r"self\.mean|self\.std|self\._stats|scaler\.")
    print(f"\n  Stats persistence hits: {len(hits_save)}")
    for lno, line in hits_save[:3]:
        print(f"  L{lno:4}: {line[:120]}")
else:
    print("  rolling_normalization.py: NO EXISTE")
    # Buscar en feature_pipeline.py
    hits_fp = grep_file(fp_path, r"rolling.*norm|RollingZ|z90d|z30d")
    for lno, line in hits_fp[:5]:
        print(f"  feature_pipeline.py L{lno:4}: {line[:120]}")

# ── ARCH-28: MockXGB como generador de 22 trades ───────────────────────────────
print("\n[ARCH-28] MOCKXGB: ¿se activo en la run que genero 22 trades RANGE?")
print("-"*60)
# Verificar si hay logs de la run con [MOCK] o MockXGB
log_dir = ROOT / "logs"
mock_in_logs = []
if log_dir.exists():
    for log in sorted(log_dir.glob("*.log"))[-3:]:  # ultimos 3 logs
        hits = grep_file(log, r"MockXGB|mock.*classifier|MOCK")
        if hits:
            mock_in_logs.append((log.name, hits))

print(f"  Logs con MockXGB en ultimas 3 corridas: {len(mock_in_logs)}")
for lname, hits in mock_in_logs:
    print(f"  {lname}:")
    for lno, line in hits[:3]:
        print(f"    L{lno}: {line[:100]}")

# Verificar como carga el modelo el router
router_path = LUNA / "models" / "regime_router.py"
load_hits = grep_file(router_path, r"load_model|xgb\.Booster|model_path|\.model|open.*rb")
print(f"\n  Model loading en regime_router.py: {len(load_hits)} hits")
for lno, line in load_hits[:5]:
    print(f"  L{lno:4}: {line[:120]}")

# ── ARCH-29: LightGBM zombie ───────────────────────────────────────────────────
print("\n[ARCH-29] LIGHTGBM: ¿activo en wfb_worker o pipeline_executor?")
print("-"*60)
ensemble_lgbm = LUNA / "models" / "ensemble_lgbm.py"
print(f"  ensemble_lgbm.py: {'EXISTE' if ensemble_lgbm.exists() else 'NO EXISTE'} ({ensemble_lgbm.stat().st_size if ensemble_lgbm.exists() else 0} bytes)")

# Buscar invocaciones en wfb_worker.py y pipeline_executor.py
scripts = ROOT / "scripts"
for fname in ["wfb_worker.py", "run_wfb_orchestrator.py", "train_production_model.py"]:
    fpath = scripts / fname
    hits = grep_file(fpath, r"lgbm|LightGBM|ensemble_lgbm|lightgbm")
    if hits:
        print(f"\n  {fname}: {len(hits)} hits LightGBM")
        for lno, line in hits:
            print(f"    L{lno}: {line[:120]}")
    else:
        print(f"  {fname}: 0 hits LightGBM")

# Buscar en regime_router.py
lgbm_router = grep_file(router_path, r"lightgbm|lgbm|agent_type.*light")
print(f"\n  regime_router.py LightGBM refs: {len(lgbm_router)}")
for lno, line in lgbm_router:
    print(f"  L{lno:4}: {line[:120]}")

print("\n[ARCH-22-29] Diagnostico completado.")

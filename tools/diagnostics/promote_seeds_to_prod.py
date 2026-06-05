"""
promote_seeds_to_prod.py
[DEPLOY-12SEEDS 2026-05-30] Promueve modelos de wfb_cache/seedXXX/W_last/models/
hacia data/models/prod/seedXXX/ para las 9 seeds nuevas del ensemble aprobado.

Seeds ya en prod/ (skip): 1337, 2025
Seeds a promover: 42, 100, 777, 29611, 85199, 43812, 28559, 76576, 62815, 60075
"""
import shutil
from pathlib import Path

BASE = Path("G:/Mi unidad/ia/luna_v2")
CACHE = BASE / "data" / "wfb_cache"
PROD = BASE / "data" / "models" / "prod"

# Mapeo seed → última ventana válida
SEED_WINDOW_MAP = {
    "42":    "W4",
    "100":   "W5",
    "777":   "W5",
    "29611": "W4",
    "85199": "W5",
    "43812": "W5",
    "28559": "W5",
    "76576": "W5",
    "62815": "W5",
    "60075": "W5",
}

# Seeds ya en prod (skip)
ALREADY_IN_PROD = {"1337", "2025"}
REQUIRED_FILES = [
    "xgboost_meta_bear_long.model",
    "xgboost_meta_bull_long.model",
    "xgboost_meta_range_long.model",
    "metalabeler_v2_long_lstm.pt",
    "metalabeler_v2_long_rf.joblib",
    "metalabeler_v2_long_config.json",
    "hmm_regime.pkl",
    "ood_guard.pkl",
]

print("[PROMOTE] === INICIO Promoción de Seeds a prod/ ===")
promoted = []
failed = []

for seed, window in SEED_WINDOW_MAP.items():
    if seed in ALREADY_IN_PROD:
        print(f"[PROMOTE] seed{seed}: ya en prod/ — SKIP")
        continue

    src = CACHE / f"seed{seed}" / window / "models"
    dst = PROD / f"seed{seed}"

    if not src.exists():
        print(f"[PROMOTE] seed{seed}: FALTA cache {src} — ERROR")
        failed.append(seed)
        continue

    # Verificar archivos requeridos en src
    missing = [f for f in REQUIRED_FILES if not (src / f).exists()]
    if missing:
        print(f"[PROMOTE] seed{seed}: archivos faltantes en cache: {missing} — ERROR")
        failed.append(seed)
        continue

    # Copiar
    if dst.exists():
        print(f"[PROMOTE] seed{seed}: dst existe ({dst.name}) — sobrescribiendo")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    print(f"[PROMOTE] seed{seed}: OK | {src} → {dst}")

    # Verificar copia
    copied_files = list(dst.iterdir())
    print(f"  Copiados {len(copied_files)} archivos | xgb_bull={('xgboost_meta_bull_long.model' in [f.name for f in copied_files])}")
    promoted.append(seed)

print(f"\n[PROMOTE] RESULTADO: {len(promoted)} promovidas | {len(failed)} fallidas")
print(f"[PROMOTE] Promovidas: {promoted}")
if failed:
    print(f"[PROMOTE] FALLIDAS: {failed}")

# Verificar state final de prod/
print("\n[PROMOTE] Estado final data/models/prod/:")
for d in sorted(PROD.iterdir()):
    if d.is_dir() and d.name.startswith("seed"):
        has_xgb = (d / "xgboost_meta_bull_long.model").exists()
        has_meta = (d / "metalabeler_v2_long_config.json").exists()
        has_hmm = (d / "hmm_regime.pkl").exists()
        status = "✅" if (has_xgb and has_meta and has_hmm) else "❌"
        print(f"  {d.name}: {status} xgb={has_xgb} meta_cfg={has_meta} hmm={has_hmm}")

print("[PROMOTE] === FIN ===")

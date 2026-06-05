"""
audit_prod_seeds_full.py
[AUDIT-SEEDS-LIVE 2026-05-30] Auditoría completa de seeds en /data/models/prod del VPS.
Verifica:
1. Qué seeds existen en prod/
2. Para cada seed: cuáles modelos XGBoost long son binarios reales (no mock ni faltantes)
3. MetaLabelers disponibles (long LSTM + RF)
4. HMM disponible
5. Reporte de integridad: seed COMPLETA = todos los modelos presentes y reales
"""
from pathlib import Path

PROD = Path("/root/luna_v2/data/models/prod")
REGIMES = ["bear", "bull", "range"]

def is_real_binary(path):
    """True si el archivo es un binario XGBoost (header 7b4c) o LGBM/joblib."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(4)
        # XGBoost binario: 7b4c0000
        if hdr[:2] == b'{L':
            return True
        # LGBM .model: empieza con 'tree\n' o similar ASCII
        if hdr[:4] == b'tree' or hdr[:4] == b'Tree':
            return True
        # Joblib pickle: 8003 (pickle v3) o 800x
        if hdr[:2] in [b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05']:
            return True
        # PyTorch .pt: 504b (zip) o 8003
        if hdr[:2] == b'PK' or hdr[:2] in [b'\x80\x02', b'\x80\x03']:
            return True
        # JSON mock: comienza con '{'  y segundo byte no es 'L'
        if hdr[:1] == b'{' and hdr[1:2] != b'L':
            return False
        # Si no es ningun formato conocido pero tampoco es mock: asumir real
        return True
    except Exception:
        return False

print("=" * 60)
print("AUDITORIA COMPLETA SEEDS EN PROD")
print("=" * 60)

seed_dirs = sorted([d for d in PROD.iterdir() if d.is_dir() and d.name.startswith("seed")])
print("Seeds en prod/: " + str([d.name for d in seed_dirs]))
print()

results = {}
for sd in seed_dirs:
    seed_id = sd.name
    report = {"seed": seed_id, "xgb": {}, "lgbm": {}, "meta": {}, "hmm": False, "complete": False}
    issues = []

    # 1. XGBoost agentes LONG (requeridos para direction_mode=long)
    for r in REGIMES:
        mp = sd / f"xgboost_meta_{r}_long.model"
        sig = sd / f"xgboost_meta_{r}_long_signature.json"
        if mp.exists():
            real = is_real_binary(mp)
            sz = mp.stat().st_size
            report["xgb"][r] = {"exists": True, "real": real, "size_kb": sz // 1024}
            if not real:
                issues.append(f"xgb_{r}_long: MOCK")
        else:
            report["xgb"][r] = {"exists": False, "real": False, "size_kb": 0}
            issues.append(f"xgb_{r}_long: FALTA")

    # 2. LGBM meta agentes LONG (opcionales pero deseados)
    for r in REGIMES:
        mp = sd / f"lgbm_meta_{r}_long.model"
        if mp.exists():
            sz = mp.stat().st_size
            report["lgbm"][r] = {"exists": True, "size_kb": sz // 1024}
        else:
            report["lgbm"][r] = {"exists": False, "size_kb": 0}

    # 3. MetaLabeler V2 LONG (LSTM + RF requeridos)
    lstm_long = sd / "metalabeler_v2_long_lstm.pt"
    rf_long = sd / "metalabeler_v2_long_rf.joblib"
    cfg_long = sd / "metalabeler_v2_long_config.json"
    meta_ok = lstm_long.exists() and rf_long.exists() and cfg_long.exists()
    report["meta"]["long_lstm"] = lstm_long.exists()
    report["meta"]["long_rf"] = rf_long.exists()
    report["meta"]["long_cfg"] = cfg_long.exists()
    if not meta_ok:
        issues.append("MetaLabeler_LONG: INCOMPLETO")

    # 4. HMM
    hmm = sd / "hmm_regime.pkl"
    report["hmm"] = hmm.exists()
    if not hmm.exists():
        issues.append("HMM: FALTA")

    # 5. Autoencoder
    ae = sd / "autoencoder_state.pt"
    report["ae"] = ae.exists()
    # No critico, pero lo reportamos

    # 6. OOD Guard
    ood = sd / "ood_guard.pkl"
    report["ood"] = ood.exists()

    # Determinar si seed esta COMPLETA para operar
    xgb_all_ok = all(report["xgb"][r]["exists"] and report["xgb"][r]["real"] for r in REGIMES)
    report["complete"] = xgb_all_ok and meta_ok and report["hmm"]
    report["issues"] = issues

    results[seed_id] = report

# Imprimir resumen
print("RESULTADO POR SEED:")
print("-" * 60)
complete_seeds = []
incomplete_seeds = []

for seed_id, r in results.items():
    status = "✅ COMPLETA" if r["complete"] else "❌ INCOMPLETA"
    print(seed_id + ": " + status)
    
    # XGBoost
    xgb_line = "  XGB_LONG: "
    for regime in REGIMES:
        info = r["xgb"][regime]
        if info["exists"] and info["real"]:
            xgb_line += regime + "=OK(" + str(info["size_kb"]) + "KB) "
        elif info["exists"] and not info["real"]:
            xgb_line += regime + "=MOCK "
        else:
            xgb_line += regime + "=FALTA "
    print(xgb_line)
    
    # Meta
    meta_line = "  META_LONG: lstm=" + str(r["meta"]["long_lstm"]) + " rf=" + str(r["meta"]["long_rf"]) + " cfg=" + str(r["meta"]["long_cfg"])
    print(meta_line)
    
    # Otros
    print("  HMM=" + str(r["hmm"]) + " AE=" + str(r["ae"]) + " OOD=" + str(r["ood"]))
    
    if r["issues"]:
        print("  ISSUES: " + str(r["issues"]))
    
    if r["complete"]:
        sid = int(seed_id.replace("seed", ""))
        complete_seeds.append(sid)
    else:
        incomplete_seeds.append(seed_id)
    print()

print("=" * 60)
print("SEEDS COMPLETAS (listas para live): " + str(sorted(complete_seeds)))
print("SEEDS INCOMPLETAS: " + str(incomplete_seeds))
print("TOTAL COMPLETAS: " + str(len(complete_seeds)) + "/" + str(len(results)))

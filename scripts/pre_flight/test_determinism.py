"""
[FIX-AE-DETERMINISM-01 2026-06-26] PRE-FLIGHT DE DETERMINISMO del pipeline WFB.
================================================================================
BASE DETERMINISTA conseguida el 2026-06-26: dos runs frescas `--seeds 42 --nocache`
producen artefactos BYTE-IDENTICOS end-to-end (oos_raw_probs, oos_trades, selected_features).
Ver docs/hallazgos_run_baseline_20260626.md.

Este test verifica ESTATICAMENTE que TODOS los puntos del fix de determinismo siguen en
su sitio. Si un fix futuro desalinea alguno, este test FALLA al inicio del WFB y alerta,
evitando que se construyan pruebas sobre un baseline no reproducible.

Puntos cubiertos (las 8 fuentes de no-determinismo que se cazaron y arreglaron):
  DET-1  Helper central seed_everything (torch/cuda/cudnn/numpy/random).
  DET-2  AutoEncoder de features (DeepFeatureAutoEncoder): manual_seed + cudnn.deterministic + shuffle seedeado.
  DET-3  OOD AutoEncoder (train_autoencoder): seed_everything + shuffle seedeado.
  DET-4  MetaLabeler (train_metalabeler_v2): seed_everything + SIN device="cuda" (GPU no-determinista).
  DET-5  SFI (feature_selection_e): binning con RNG seedeado + LightGBM deterministic/1-hilo + XGBoost scoring 1-hilo.
  DET-6  Orquestador limpia los caches CROSS-RUN (_dsr_cache.json / _lag_cache.json) al inicio de runs frescas.

Uso:
  python scripts/pre_flight/test_determinism.py            # standalone (exit 1 si roto)
  from scripts.pre_flight.test_determinism import check_determinism  # all_ok, results
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    p = _ROOT / rel
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def check_determinism():
    """Devuelve (all_ok, [(id, ok, msg), ...]) — verificacion estatica del determinismo."""
    res = []

    def chk(cid, ok, msg):
        res.append((cid, bool(ok), msg))

    # DET-1: helper central
    det = _read("luna/utils/determinism.py")
    chk("DET-1", ("def seed_everything" in det and "manual_seed" in det and "cudnn.deterministic" in det),
        "luna/utils/determinism.py::seed_everything cubre torch/cuda/cudnn")

    # DET-2: AE de features
    ae = _read("luna/features/autoencoder_features.py")
    chk("DET-2", ("manual_seed" in ae and "cudnn.deterministic" in ae and "generator=" in ae
                  and "[FIX-AE-DETERMINISM-01]" in ae),
        "AE features: manual_seed + cudnn.deterministic + DataLoader generator seedeado")

    # DET-3: OOD AE
    tae = _read("luna/models/train_autoencoder.py")
    chk("DET-3", ("seed_everything" in tae and "generator=" in tae),
        "train_autoencoder (OOD AE): seed_everything + shuffle seedeado")

    # DET-4: MetaLabeler — seedeado y SIN GPU (device=cuda es no-determinista)
    tml = _read("luna/models/train_metalabeler_v2.py")
    _no_cuda = ('device="cuda"' not in tml) and ("device='cuda'" not in tml)
    chk("DET-4", ("seed_everything" in tml and _no_cuda),
        "MetaLabeler: seed_everything + SIN device=cuda (XGBoost/torch en CPU = determinista)")

    # DET-5a: SFI binning C-MI con RNG seedeado (no np.random.normal global)
    fse = _read("luna/features/feature_selection_e.py")
    chk("DET-5a", ("np.random.normal" not in fse and "default_rng" in fse),
        "SFI binning C-MI: RNG seedeado (sin np.random.normal global)")

    # DET-5b: LightGBM SHAP-RFE determinista + 1 hilo (multi-hilo no es 100% reproducible)
    chk("DET-5b", ("deterministic=True" in fse and "force_row_wise=True" in fse
                   and "n_jobs=-1, verbose=-1" not in fse),
        "SFI LightGBM: deterministic=True + force_row_wise + n_jobs=1")

    # DET-5c: XGBoost de scoring DSR a 1 hilo (multi-hilo wobblea importancias -> flip de features)
    chk("DET-5c", ("n_jobs=-1, verbosity=0" not in fse),
        "SFI XGBoost scoring: n_jobs=1 (no multi-hilo)")

    # DET-6: orquestador limpia caches cross-run al inicio de runs frescas
    orch = _read("scripts/run_wfb_orchestrator.py")
    chk("DET-6", ("_dsr_cache.json" in orch and "_lag_cache.json" in orch and "os.remove" in orch),
        "Orquestador: limpia _dsr_cache/_lag_cache al inicio (estado cross-run -> reproducible)")

    # DET-7: el SFI NO reusa los caches adaptativos (DSR/lag) en --nocache (estado cross-run).
    chk("DET-7", ("LUNA_NOCACHE" in fse and "lag cache NO reusado" in fse),
        "SFI: guards LUNA_NOCACHE en DSR/lag cache (run fresca = sin estado cross-run)")

    all_ok = all(ok for _, ok, _ in res)
    return all_ok, res


def assert_determinism(loud: bool = True) -> bool:
    """Imprime el resultado. Devuelve True si OK. Pensado para llamarse al inicio del WFB."""
    ok, res = check_determinism()
    if loud:
        print("=" * 74)
        print("[PRE-FLIGHT DETERMINISMO] base determinista 2026-06-26 — verificando invariantes")
        for cid, passed, msg in res:
            print(f"  [{'PASS' if passed else 'FAIL'}] {cid}  {msg}")
        if ok:
            print("[PRE-FLIGHT DETERMINISMO] OK — los 8 puntos del determinismo siguen intactos.")
        else:
            print("[PRE-FLIGHT DETERMINISMO] *** CRITICAL: DETERMINISMO ROTO ***")
            print("  Un cambio desalineo el determinismo base. Dos runs frescas YA NO seran")
            print("  byte-identicas -> cualquier comparacion de palancas estara contaminada por ruido.")
            print("  Revisar los FAIL de arriba y docs/hallazgos_run_baseline_20260626.md antes de seguir.")
        print("=" * 74)
        import sys as _sys_det
        _sys_det.stdout.flush()  # visible inmediatamente al inicio del WFB (evita buffering)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if assert_determinism(loud=True) else 1)

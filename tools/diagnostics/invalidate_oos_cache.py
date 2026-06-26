#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
invalidate_oos_cache.py — Invalida SOLO los pasos OOS+Gauntlet del cache por-seed
para forzar su re-cálculo con --resume, reutilizando TODO el entrenamiento (features,
HMM, SFI, XGBoost, MetaLabeler, Calibrador).

Caso de uso: tests de palancas DOWNSTREAM (F1 banda DVOL, F4 veto de régimen) que solo
afectan la inferencia OOS (luna/models/predict_oos.py → SignalFilter). Verificado:
predict_oos re-aplica el filtro fresco en cada corrida; reusar los modelos es válido
porque NO dependen de la banda DVOL.

NO toca:
  - executor_state_wfb_{window}_data.json   (SHARED: features/AI-mining/SFI/HMM)  → se conservan
  - los modelos entrenados                                                        → se conservan

Toca SOLO:
  - executor_state_wfb_s{seed}_{window}_models.json  → quita los marcadores OOS+Gauntlet

Uso:
    python tools/diagnostics/invalidate_oos_cache.py --dry-run
    python tools/diagnostics/invalidate_oos_cache.py --seeds 42 100 777 1337 2025 19519 22539 22971 24197 25680 27644 28019
    python tools/diagnostics/invalidate_oos_cache.py            # todas las seeds en cache

Luego: lanzar el run con --resume (NO --nocache):
    python scripts/run_wfb_orchestrator.py --seeds <...> --resume
"""
import argparse, glob, json, os, re, sys, io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data", "wfb_cache")

# Pasos a invalidar (re-correr). Nombres EXACTOS verificados en executor_state.
DEFAULT_STEPS = [
    "Generador de Predicciones OOS",   # predict_oos.py — aquí corre SignalFilter (banda DVOL)
    "Validación Estadística",          # run_statistical_validation.py — gauntlet (si está cacheado)
]

# Patrón per-seed: executor_state_wfb_s{seed}_{window}_models.json  (NUNCA *_data.json compartido)
PAT = re.compile(r"executor_state_wfb_s(\d+)_(W\d+)_models\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="seeds a invalidar (default: todas las presentes en cache)")
    ap.add_argument("--steps", nargs="*", default=DEFAULT_STEPS,
                    help="nombres de paso a quitar de completed_steps")
    ap.add_argument("--dry-run", action="store_true", help="solo mostrar, no escribir")
    args = ap.parse_args()

    if not os.path.isdir(CACHE):
        print(f"[!] No existe {CACHE} — no hay cache que invalidar.")
        return

    files = sorted(glob.glob(os.path.join(CACHE, "executor_state_wfb_s*_W*_models.json")))
    if not files:
        print(f"[!] Sin archivos executor_state per-seed en {CACHE}.")
        return

    steps_set = set(args.steps)
    seeds_filter = set(args.seeds) if args.seeds else None

    scanned = modified = total_removed = 0
    touched_seeds = set()
    skipped_no_marker = 0

    for f in files:
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        seed = int(m.group(1)); window = m.group(2)
        if seeds_filter is not None and seed not in seeds_filter:
            continue
        scanned += 1
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] no se pudo leer {os.path.basename(f)}: {e}")
            continue
        steps = data.get("completed_steps", [])
        present = [s for s in steps if s in steps_set]
        if not present:
            skipped_no_marker += 1
            continue
        new_steps = [s for s in steps if s not in steps_set]
        total_removed += len(present)
        modified += 1
        touched_seeds.add(seed)
        tag = "[DRY]" if args.dry_run else "[OK ]"
        print(f"  {tag} s{seed} {window}: quita {present}")
        if not args.dry_run:
            data["completed_steps"] = new_steps
            json.dump(data, open(f, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

    print("\n" + ("=== DRY-RUN (no se escribió nada) ===" if args.dry_run else "=== HECHO ==="))
    print(f"  Archivos escaneados : {scanned}")
    print(f"  Archivos modificados: {modified}")
    print(f"  Marcadores quitados : {total_removed}")
    print(f"  Sin marcador (skip) : {skipped_no_marker}")
    print(f"  Seeds afectadas     : {sorted(touched_seeds)}")
    if modified and not args.dry_run:
        print("\n  Siguiente paso: lanzar con --resume (NO --nocache):")
        seeds_str = " ".join(str(s) for s in sorted(touched_seeds))
        print(f"    python scripts/run_wfb_orchestrator.py --seeds {seeds_str} --resume")
    print("\n  NOTA: el guard [CACHE-INTEGRITY-01] recalculará igualmente si los datos cambiaron")
    print("        (fingerprint distinto). Para F7 sign-off usar --nocache (R12 limpio).")


if __name__ == "__main__":
    main()

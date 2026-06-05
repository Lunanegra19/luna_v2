"""
test_funnel_accum.py — Test standalone del FIX-FUNNEL-ACCUM-01
Verifica que export_funnel_json acumula conteos entre ventanas WFB
en lugar de sobrescribir el JSON en cada ventana.
"""
import json
import tempfile
import pathlib


def simulate_export_funnel_accum(funnel_stats: dict, output_dir: str, run_id: str) -> dict:
    """Replica exacta de la lógica implementada en SignalFilter.export_funnel_json."""
    target_path = pathlib.Path(output_dir) / "signal_funnel.json"
    _ACCUM_KEYS = [
        "raw_oos_bars", "after_xgb", "after_lgbm", "after_ood", "after_cvd",
        "after_hmm", "after_meta", "after_cash_shield", "after_momentum", "after_embargo"
    ]
    existing = {}
    if target_path.exists() and run_id:
        with open(target_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("run_id", "") != run_id:
            print(f"  [RESET] run_id cambio: {existing.get('run_id')} -> {run_id}")
            existing = {}

    merged = dict(funnel_stats)
    merged["run_id"] = run_id
    merged["n_windows_accumulated"] = existing.get("n_windows_accumulated", 0) + 1
    for key in _ACCUM_KEYS:
        if key in funnel_stats:
            merged[key] = existing.get(key, 0) + funnel_stats[key]
    merged["filter_fallback_level"] = max(
        existing.get("filter_fallback_level", 0),
        funnel_stats.get("filter_fallback_level", 0)
    )
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4)
    return merged


def main():
    # Simular exactamente las 5 ventanas WFB de seed42
    # W1=EMPTY, W2=10 trades, W3=12 trades, W4=EMPTY, W5=1 trade
    windows = [
        {"name": "W1-EMPTY", "raw_oos_bars": 480, "after_xgb": 0,  "after_lgbm": 0,  "after_ood": 0,  "after_cvd": 0,  "after_hmm": 0,  "after_meta": 0,  "after_cash_shield": 0,  "after_momentum": 0,  "after_embargo": 0,  "filter_fallback_level": 0},
        {"name": "W2",       "raw_oos_bars": 480, "after_xgb": 13, "after_lgbm": 13, "after_ood": 13, "after_cvd": 13, "after_hmm": 13, "after_meta": 12, "after_cash_shield": 12, "after_momentum": 12, "after_embargo": 10, "filter_fallback_level": 0},
        {"name": "W3",       "raw_oos_bars": 480, "after_xgb": 14, "after_lgbm": 14, "after_ood": 14, "after_cvd": 14, "after_hmm": 14, "after_meta": 13, "after_cash_shield": 13, "after_momentum": 13, "after_embargo": 12, "filter_fallback_level": 0},
        {"name": "W4-EMPTY", "raw_oos_bars": 480, "after_xgb": 0,  "after_lgbm": 0,  "after_ood": 0,  "after_cvd": 0,  "after_hmm": 0,  "after_meta": 0,  "after_cash_shield": 0,  "after_momentum": 0,  "after_embargo": 0,  "filter_fallback_level": 0},
        {"name": "W5",       "raw_oos_bars": 505, "after_xgb": 7,  "after_lgbm": 7,  "after_ood": 7,  "after_cvd": 7,  "after_hmm": 7,  "after_meta": 7,  "after_cash_shield": 7,  "after_momentum": 7,  "after_embargo": 1,  "filter_fallback_level": 0},
    ]

    RUN_ID = "WFB_20260518_213330_9736_seed42_FINAL"

    errors = []
    with tempfile.TemporaryDirectory() as tmpdir:
        final = None
        print("=== TEST 1: Acumulacion de 5 ventanas WFB (seed42) ===")
        for w in windows:
            name = w.pop("name")
            final = simulate_export_funnel_accum(w, tmpdir, RUN_ID)
            print(f"  {name}: after_embargo acum={final['after_embargo']} | n_windows={final['n_windows_accumulated']}")

        # Verificaciones
        assert final["raw_oos_bars"] == 2425, f"raw_oos_bars: got {final['raw_oos_bars']}, expected 2425"
        assert final["after_embargo"] == 23, f"after_embargo: got {final['after_embargo']}, expected 23 (10+12+1)"
        assert final["n_windows_accumulated"] == 5, f"n_windows: got {final['n_windows_accumulated']}, expected 5"
        assert final["filter_fallback_level"] == 0
        print()
        print(f"  RESULTADO: raw_oos_bars={final['raw_oos_bars']} | after_embargo={final['after_embargo']} | n_windows={final['n_windows_accumulated']}")
        print("  [OK] Acumulacion correcta")

        print()
        print("=== TEST 2: Reset automatico con nuevo run_id ===")
        nueva_semilla = {"raw_oos_bars": 100, "after_xgb": 5, "after_lgbm": 5,
                         "after_ood": 5, "after_cvd": 5, "after_hmm": 5,
                         "after_meta": 4, "after_cash_shield": 4,
                         "after_momentum": 4, "after_embargo": 3, "filter_fallback_level": 0}
        final2 = simulate_export_funnel_accum(nueva_semilla, tmpdir, "WFB_NEW_SEED_53929")
        assert final2["after_embargo"] == 3, f"Reset fallo: got {final2['after_embargo']}, expected 3"
        assert final2["n_windows_accumulated"] == 1
        print(f"  RESULTADO: after_embargo={final2['after_embargo']} | n_windows={final2['n_windows_accumulated']}")
        print("  [OK] Reset por cambio de run_id correcto")

        print()
        print("=== TEST 3: filter_fallback_level se propaga correctamente ===")
        w_clean   = {"raw_oos_bars": 200, "after_xgb": 5, "after_lgbm": 5, "after_ood": 5, "after_cvd": 5,
                     "after_hmm": 5, "after_meta": 5, "after_cash_shield": 5, "after_momentum": 5, "after_embargo": 3,
                     "filter_fallback_level": 0}
        w_fallback = {"raw_oos_bars": 200, "after_xgb": 3, "after_lgbm": 3, "after_ood": 3, "after_cvd": 3,
                      "after_hmm": 3, "after_meta": 3, "after_cash_shield": 3, "after_momentum": 3, "after_embargo": 2,
                      "filter_fallback_level": 1}  # Ventana con fallback activado
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmpdir2:
            simulate_export_funnel_accum(w_clean, tmpdir2, "TEST_FALLBACK_RUN")
            final3 = simulate_export_funnel_accum(w_fallback, tmpdir2, "TEST_FALLBACK_RUN")
        assert final3["filter_fallback_level"] == 1, f"filter_fallback_level: got {final3['filter_fallback_level']}, expected 1"
        print(f"  RESULTADO: filter_fallback_level={final3['filter_fallback_level']} (max de todas las ventanas)")
        print("  [OK] Propagacion de fallback_level correcta")

    print()
    print("=" * 55)
    print("[OK] FIX-FUNNEL-ACCUM-01 — Todos los tests pasaron")
    print("     El validador estadistico recibira el funnel")
    print("     acumulado de TODAS las ventanas WFB, no solo W5.")
    print("=" * 55)


if __name__ == "__main__":
    main()

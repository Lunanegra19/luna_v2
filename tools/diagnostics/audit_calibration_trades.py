"""
[FIX-CALIB-BINARY-01 DETECTION-4] Auditor post-trade de calibración isotónica.

Detecta si los trades guardados en un WFB run tienen xgb_prob_cal == xgb_prob_raw,
señal definitiva de que el calibrador no fue aplicado durante la inferencia.

Uso:
    python tools/diagnostics/audit_calibration_trades.py
    python tools/diagnostics/audit_calibration_trades.py --window W1
    python tools/diagnostics/audit_calibration_trades.py --window W1 W2 W3

Escenarios detectados:
  A) xgb_prob_cal == xgb_prob_raw  →  calibrador nunca aplicado (bug FIX-CALIB-BINARY-01)
  B) xgb_prob_cal constante        →  calibrador colapsó (OOB clip en todos los valores)
  C) xgb_prob_cal sano             →  calibración correcta
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

WFB_DIR = Path('g:/Mi unidad/ia/luna_v2/data/reports/wfb')
CACHE_DIR = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache')

DIFF_THRESHOLD_PCT = 1.0   # % de barras con cal != raw para considerar calibracion "aplicada"
CONST_STD_CUTOFF = 1e-4 # std por debajo del cual se considera "constante"


def audit_window(window: str) -> dict:
    """Audita todos los trades de una ventana y retorna el diagnóstico."""
    files = list(WFB_DIR.glob(f'oos_trades_{window}_seed*.parquet'))
    if not files:
        return {'window': window, 'status': 'NO_TRADES', 'n_trades': 0}

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                df['_seed_file'] = f.name
                dfs.append(df)
        except Exception as e:
            print(f"[AUDIT] Warning: no se pudo leer {f.name}: {e}")

    if not dfs:
        return {'window': window, 'status': 'NO_VALID_TRADES', 'n_trades': 0}

    trades = pd.concat(dfs, ignore_index=False)
    n = len(trades)

    if 'xgb_prob' not in trades.columns or 'xgb_prob_cal' not in trades.columns:
        return {'window': window, 'status': 'MISSING_COLUMNS', 'n_trades': n}

    raw = trades['xgb_prob'].values
    cal = trades['xgb_prob_cal'].values

    diff = np.abs(cal - raw)
    n_equal    = (diff < 1e-6).sum()
    n_modified = (diff >= 1e-6).sum()
    pct_equal  = n_equal / max(n, 1) * 100
    pct_mod    = n_modified / max(n, 1) * 100
    std_cal    = float(np.std(cal))
    mean_diff  = float(np.mean(diff))
    max_diff   = float(np.max(diff))

    # Diagnostico
    if pct_equal >= 99.0:
        scenario = 'A_NO_CALIBRATION'   # Bug FIX-CALIB-BINARY-01
    elif std_cal < CONST_STD_THRESHOLD:
        scenario = 'B_CALIBRATOR_COLLAPSED'  # OOB clip en todos los valores
    elif pct_mod >= 50.0:
        scenario = 'C_HEALTHY'
    else:
        scenario = 'D_PARTIAL'  # Calibracion parcial (algunos agentes sin calibrador)

    # Verificar calibradores en caché
    cal_dir = CACHE_DIR / window / 'models'
    n_cal_files = len(list(cal_dir.glob('xgboost_isotonic_calibrator_*.joblib'))) if cal_dir.exists() else 0
    n_model_files = len(list(cal_dir.glob('xgboost_meta_*_long.model'))) if cal_dir.exists() else 0

    return {
        'window':        window,
        'status':        scenario,
        'n_trades':      n,
        'n_equal':       int(n_equal),
        'pct_equal':     round(pct_equal, 1),
        'n_modified':    int(n_modified),
        'pct_modified':  round(pct_mod, 1),
        'std_cal':       round(std_cal, 6),
        'mean_diff':     round(mean_diff, 6),
        'max_diff':      round(max_diff, 6),
        'cal_files':     n_cal_files,
        'model_files':   n_model_files,
        'cal_coverage':  f'{n_cal_files}/{n_model_files}' if n_model_files > 0 else 'N/A',
    }


def print_report(results: list[dict]) -> None:
    print()
    print('=' * 75)
    print('[FIX-CALIB-BINARY-01 DETECTION-4] AUDIT POST-TRADE DE CALIBRACION')
    print('=' * 75)
    print()

    icons = {
        'A_NO_CALIBRATION':      '🔴 BUG',
        'B_CALIBRATOR_COLLAPSED':'🟡 WARN',
        'C_HEALTHY':             '🟢 OK',
        'D_PARTIAL':             '🟠 PARCIAL',
        'NO_TRADES':             '⬜ N/A',
        'NO_VALID_TRADES':       '⬜ N/A',
        'MISSING_COLUMNS':       '⚪ ERROR',
    }

    for r in results:
        icon    = icons.get(r['status'], '?')
        n       = r.get('n_trades', 0)
        pct_eq  = r.get('pct_equal', 0)
        pct_mod = r.get('pct_modified', 0)
        std_cal = r.get('std_cal', 0)
        cal_cov = r.get('cal_coverage', 'N/A')
        print(f"  {r['window']}  {icon}  status={r['status']}")
        print(f"         trades={n} | cal==raw={pct_eq}% | cal!=raw={pct_mod}% | std_cal={std_cal} | cal_files={cal_cov}")

        if r['status'] == 'A_NO_CALIBRATION':
            print(f"         ⚠ DIAGNOSTICO: El calibrador NO fue aplicado durante la inferencia.")
            print(f"           Causa probable: FIX-CALIB-BINARY-01 (apertura en modo texto 'r' en lugar de 'rb').")
            print(f"           Solucion: Verificar regime_router.py L178 — debe ser open(path, 'rb').")
        elif r['status'] == 'B_CALIBRATOR_COLLAPSED':
            print(f"         ⚠ DIAGNOSTICO: Calibrador aplicado pero prob_cal constante (std={std_cal:.2e}).")
            print(f"           Causa: probs OOS fuera del rango de entrenamiento → out_of_bounds='clip' aplana todo.")
            print(f"           Solucion: Investigar rango IS del calibrador vs distribucion OOS actual.")
        elif r['status'] == 'D_PARTIAL':
            print(f"         ⚠ DIAGNOSTICO: Calibracion parcial (algunos agentes sin calibrador).")
            print(f"           Causa: n_cal_files({r.get('cal_files','?')}) < n_model_files({r.get('model_files','?')}).")
        print()

    # Resumen ejecutivo
    bugs    = [r for r in results if r['status'] == 'A_NO_CALIBRATION']
    warns   = [r for r in results if r['status'] in ('B_CALIBRATOR_COLLAPSED', 'D_PARTIAL')]
    healthy = [r for r in results if r['status'] == 'C_HEALTHY']

    print('-' * 75)
    print(f"RESUMEN: {len(healthy)} ventanas OK | {len(warns)} con advertencia | {len(bugs)} con BUG critico")
    if bugs:
        print()
        print("*** ACCION INMEDIATA REQUERIDA ***")
        print(f"    Ventanas afectadas por BUG FIX-CALIB-BINARY-01: {[r['window'] for r in bugs]}")
        print("    Estos trades fueron ejecutados SIN calibracion isotonica.")
        print("    WR real puede diferir significativamente del WR esperado calibrado.")
        print("    Re-ejecutar WFB con el fix aplicado antes de tomar decisiones sobre la run.")
    print('=' * 75)
    print()


def main():
    parser = argparse.ArgumentParser(description='Audit de calibración isotónica post-trade WFB')
    parser.add_argument('--window', nargs='+', default=None, help='Ventanas a auditar (ej: W1 W2 W3). Default: todas.')
    args = parser.parse_args()

    # Detectar ventanas disponibles
    if args.window:
        windows = args.window
    else:
        all_files = list(WFB_DIR.glob('oos_trades_W*.parquet'))
        windows   = sorted(set(f.name.split('_')[2] for f in all_files if '_' in f.name))
        if not windows:
            print('[AUDIT] No se encontraron archivos oos_trades_W*.parquet en ' + str(WFB_DIR))
            sys.exit(0)

    print(f'[AUDIT] Auditando ventanas: {windows}')
    results = [audit_window(w) for w in windows]
    print_report(results)

    # Exit code: 1 si hay bugs criticos
    has_bug = any(r['status'] == 'A_NO_CALIBRATION' for r in results)
    sys.exit(1 if has_bug else 0)


if __name__ == '__main__':
    main()

"""patch_arch04.py — aplica ARCH-04-FIX-A en train_xgboost_v2.py"""
from pathlib import Path

src_path = Path("luna/models/train_xgboost_v2.py")

# Leer en binario para preservar encoding original
raw = src_path.read_bytes()
content = raw.decode("utf-8", errors="replace")

# La línea objetivo tiene encoding garbled — buscar en el contenido raw
OLD_PATTERN = b"            # Calcular Sharpe para telemetr\xeda DSR\r\n            _preds_bin = np.where(_proba_is > 0.5, 1.0, 0.0)\r\n            if len(np.unique(_preds_bin)) == 1:"

# Probar también con windows line endings mezclados
import re

# Buscar la secuencia exacta en bytes
idx = raw.find(b"_preds_bin = np.where(_proba_is > 0.5, 1.0, 0.0)")
if idx < 0:
    print("ERROR: patron no encontrado — revisar encoding")
else:
    # Encontrar inicio de la línea del comentario anterior
    line_start = raw.rfind(b"\n", 0, idx) + 1
    # Encontrar fin de la línea 'if len(np.unique(_preds_bin)) == 1:'
    next_idx = raw.find(b"\n", idx + 50) + 1  # tras _preds_bin line
    if_line_end = raw.find(b"\n", next_idx) + 1

    OLD_BLOCK = raw[line_start:if_line_end]
    print(f"Bloque a reemplazar ({len(OLD_BLOCK)} bytes):")
    print(repr(OLD_BLOCK))

    NEW_BLOCK = (
        b"            # Calcular Sharpe para telemetria DSR\r\n"
        b"            # [ARCH-04-FIX-A 2026-06-02] Threshold alineado con deployment (sweep_min).\r\n"
        b"            # ANTES: 0.5 hardcoded - Optuna optimizaba para umbral que nunca se usa en prod.\r\n"
        b"            # AHORA: threshold_sweep_min - DSR telemetria refleja rendimiento OOS real.\r\n"
        b"            # El Brier principal no cambia (no usa threshold).\r\n"
        b"            try:\r\n"
        b"                from config.settings import cfg as _cfg_04a\r\n"
        b"                _optuna_deploy_thr = float(getattr(_cfg_04a.xgboost, 'threshold_sweep_min', 0.45))\r\n"
        b"            except Exception:\r\n"
        b"                _optuna_deploy_thr = 0.45  # fallback conservador\r\n"
        b"            if not getattr(self, '_arch04_printed', False):\r\n"
        b"                print(  # RULE[fixbugsprints.md]\r\n"
        b"                    f'[ARCH-04-FIX-A] Optuna telemetria DSR CUTOFF = {_optuna_deploy_thr:.3f} '\r\n"
        b"                    f'(alineado con sweep_min={_optuna_deploy_thr:.3f}, antes=0.5)'\r\n"
        b"                )\r\n"
        b"                self._arch04_printed = True\r\n"
        b"            _preds_bin = np.where(_proba_is > _optuna_deploy_thr, 1.0, 0.0)\r\n"
        b"            if len(np.unique(_preds_bin)) == 1:\r\n"
    )

    new_raw = raw[:line_start] + NEW_BLOCK + raw[if_line_end:]
    src_path.write_bytes(new_raw)
    print(f"\nFix aplicado. Verificando syntax...")

    import ast
    new_content = new_raw.decode("utf-8", errors="replace")
    try:
        ast.parse(new_content)
        print("[OK] Syntax valida post-ARCH-04-FIX-A")
    except SyntaxError as e:
        print(f"[ERROR] SyntaxError: L{e.lineno}: {e.msg}")
        # Revertir
        src_path.write_bytes(raw)
        print("[ROLLBACK] Revertido al original")

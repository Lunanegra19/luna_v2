"""patch_fix_hmm_shield_w2.py — FIX-HMM-SHIELD-01 parte 2: MI-guard activo en post_ath_bear"""
import sys, ast
sys.path.insert(0, '.')
from pathlib import Path

src = Path("luna/models/hmm_regime.py")
raw = src.read_bytes()

# Encontrar el bloque de combinacion de overrides (L1256 aprox)
TARGET = b"            is_bear_forced = macro_bear | panic_bear | dist_bear | post_ath_bear\r\n"
idx = raw.find(TARGET)
if idx < 0:
    print("ERROR: bloque is_bear_forced no encontrado")
    sys.exit(1)

print(f"Bloque encontrado en offset {idx}")

# Nuevo bloque: reemplazar la línea de is_bear_forced + añadir MI-guard
NEW_BLOCK  = b"            # [FIX-HMM-SHIELD-01 2026-06-02] MI-guard activo: si post_ath_bear empeora la MI\r\n"
NEW_BLOCK += b"            # por debajo del umbral SOP-R9 (min_mi=0.005), se desactiva post_ath_bear.\r\n"
NEW_BLOCK += b"            # Primero combinamos sin post_ath, medimos MI; si baja, lo excluimos.\r\n"
NEW_BLOCK += b"            _min_mi_shield = float(_cfg_hmm.hmm.min_mi)\r\n"
NEW_BLOCK += b"            _post_ath_enabled = True\r\n"
NEW_BLOCK += b"            try:\r\n"
NEW_BLOCK += b"                from sklearn.metrics import mutual_info_score as _mis\r\n"
NEW_BLOCK += b"                if 'close' in df_input.columns:\r\n"
NEW_BLOCK += b"                    _fwd_sign = (df_input['close'].pct_change(24).shift(-24) > 0).astype(int)\r\n"
NEW_BLOCK += b"                    _lbl_no_ath = labels.copy()\r\n"
NEW_BLOCK += b"                    _is_forced_no_ath = macro_bear | panic_bear | dist_bear\r\n"
NEW_BLOCK += b"                    _lbl_no_ath.loc[_is_forced_no_ath] = '4_BEAR_FORCED'\r\n"
NEW_BLOCK += b"                    _cat_no_ath = _lbl_no_ath.astype('category').cat.codes\r\n"
NEW_BLOCK += b"                    _df_mi = pd.DataFrame({'s': _cat_no_ath, 't': _fwd_sign}).dropna()\r\n"
NEW_BLOCK += b"                    _mi_no_ath = _mis(_df_mi['s'], _df_mi['t']) if len(_df_mi) > 100 else 0.0\r\n"
NEW_BLOCK += b"                    # Medir MI con post_ath_bear incluido\r\n"
NEW_BLOCK += b"                    _lbl_with_ath = labels.copy()\r\n"
NEW_BLOCK += b"                    _is_forced_with_ath = macro_bear | panic_bear | dist_bear | post_ath_bear\r\n"
NEW_BLOCK += b"                    _lbl_with_ath.loc[_is_forced_with_ath] = '4_BEAR_FORCED'\r\n"
NEW_BLOCK += b"                    _cat_with_ath = _lbl_with_ath.astype('category').cat.codes\r\n"
NEW_BLOCK += b"                    _df_mi2 = pd.DataFrame({'s': _cat_with_ath, 't': _fwd_sign}).dropna()\r\n"
NEW_BLOCK += b"                    _mi_with_ath = _mis(_df_mi2['s'], _df_mi2['t']) if len(_df_mi2) > 100 else 0.0\r\n"
NEW_BLOCK += b"                    # Decision: si post_ath empeora MI por debajo del umbral, desactivarlo\r\n"
NEW_BLOCK += b"                    if _mi_with_ath < _mi_no_ath and _mi_with_ath < _min_mi_shield:\r\n"
NEW_BLOCK += b"                        _post_ath_enabled = False\r\n"
NEW_BLOCK += b"                        print(  # RULE[fixbugsprints.md]\r\n"
NEW_BLOCK += b"                            f'[FIX-HMM-SHIELD-01] post_ath_bear DESACTIVADO: '\r\n"
NEW_BLOCK += b"                            f'MI sin post_ath={_mi_no_ath:.5f} vs con={_mi_with_ath:.5f} '\r\n"
NEW_BLOCK += b"                            f'(ambas < SOP-R9 min={_min_mi_shield}). '\r\n"
NEW_BLOCK += b"                            f'post_ath forzaba {post_ath_bear.sum()} barras innecesariamente.'\r\n"
NEW_BLOCK += b"                        )\r\n"
NEW_BLOCK += b"                    else:\r\n"
NEW_BLOCK += b"                        print(  # RULE[fixbugsprints.md]\r\n"
NEW_BLOCK += b"                            f'[FIX-HMM-SHIELD-01] post_ath_bear ACTIVO: '\r\n"
NEW_BLOCK += b"                            f'MI sin={_mi_no_ath:.5f} con={_mi_with_ath:.5f} '\r\n"
NEW_BLOCK += b"                            f'(no empeora o supera umbral SOP-R9={_min_mi_shield}). '\r\n"
NEW_BLOCK += b"                            f'{post_ath_bear.sum()} barras forzadas a BEAR.'\r\n"
NEW_BLOCK += b"                        )\r\n"
NEW_BLOCK += b"            except Exception as _e_mi_guard:\r\n"
NEW_BLOCK += b"                print(f'[FIX-HMM-SHIELD-01] MI-guard error (post_ath conservado): {_e_mi_guard}')\r\n"
NEW_BLOCK += b"            # Aplicar post_ath solo si el MI-guard lo permite\r\n"
NEW_BLOCK += b"            if _post_ath_enabled:\r\n"
NEW_BLOCK += b"                is_bear_forced = macro_bear | panic_bear | dist_bear | post_ath_bear\r\n"
NEW_BLOCK += b"            else:\r\n"
NEW_BLOCK += b"                is_bear_forced = macro_bear | panic_bear | dist_bear\r\n"

new_raw = raw[:idx] + NEW_BLOCK + raw[idx + len(TARGET):]
src.write_bytes(new_raw)
print(f"\n[FIX-HMM-SHIELD-01] Part 2 aplicado ({len(raw)} -> {len(new_raw)} bytes)")

try:
    ast.parse(new_raw.decode('utf-8', 'replace'))
    print("[OK] Syntax valida")
except SyntaxError as e:
    print(f"[ERROR] L{e.lineno}: {e.msg} -- ROLLBACK")
    src.write_bytes(raw)
    sys.exit(1)

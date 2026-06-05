"""patch_arch25.py - ARCH-25-FIX-A: OOD Feature Guard usa IS propio del agente"""
import sys
sys.path.insert(0, '.')
from pathlib import Path

src_path = Path("luna/models/train_xgboost_v2.py")
raw = src_path.read_bytes()

TARGET_MARKER = b"from luna.utils.ood_feature_guard import filter_ood_features as _ood_filter"
idx = raw.find(TARGET_MARKER)
if idx < 0:
    print("ERROR: marcador no encontrado")
    sys.exit(1)

try_idx = raw.rfind(b"        try:", 0, idx)
if try_idx < 0:
    print("ERROR: bloque try: no encontrado")
    sys.exit(1)

EXCEPT_END_MARKER = b"        except Exception as _ood_err:"
except_idx = raw.find(EXCEPT_END_MARKER, idx)
if except_idx < 0:
    print("ERROR: except no encontrado")
    sys.exit(1)

line_end1 = raw.find(b"\n", except_idx) + 1
line_end2 = raw.find(b"\n", line_end1) + 1

OLD_BLOCK = raw[try_idx:line_end2]
print(f"Bloque a reemplazar: {len(OLD_BLOCK)} bytes")
print(OLD_BLOCK.decode("utf-8", "replace")[:400])
print("---")

# Todo en ASCII puro para evitar errores de encoding en b-strings
NEW_BLOCK = b"        try:\r\n"
NEW_BLOCK += b"            from luna.utils.ood_feature_guard import filter_ood_features as _ood_filter\r\n"
NEW_BLOCK += b"            # [ARCH-25-FIX-A 2026-06-02] OOD Feature Guard usa IS propio del agente\r\n"
NEW_BLOCK += b"            # como X_oos en lugar de features_validation.parquet (100% BULL, KL=8.33).\r\n"
NEW_BLOCK += b"            # PROBLEMA: features RANGE/BEAR tienen alta varianza en su IS pero baja en\r\n"
NEW_BLOCK += b"            # BULL-only validation -> falsos positivos OOD -> features validas bloqueadas.\r\n"
NEW_BLOCK += b"            # SOLUCION: ultimo 20% IS del agente como X_oos. Fallback: val.parquet (<200 barras).\r\n"
NEW_BLOCK += b"            _regime_name_ood = str(getattr(self, 'regime_name', '') or '')\r\n"
NEW_BLOCK += b"            _df_train_ood = df_final\r\n"
NEW_BLOCK += b"            _df_oos_ood   = None\r\n"
NEW_BLOCK += b"            _oos_source   = 'none'\r\n"
NEW_BLOCK += b"\r\n"
NEW_BLOCK += b"            # Ultimo 20% del IS como X_oos (mismo regimen que el agente)\r\n"
NEW_BLOCK += b"            _n_is = len(_df_train_ood)\r\n"
NEW_BLOCK += b"            _split_20pct = int(_n_is * 0.8)\r\n"
NEW_BLOCK += b"            _df_is_recent = _df_train_ood.iloc[_split_20pct:]\r\n"
NEW_BLOCK += b"            if len(_df_is_recent) >= 200:\r\n"
NEW_BLOCK += b"                _df_oos_ood = _df_is_recent\r\n"
NEW_BLOCK += b"                _oos_source = f'IS_reciente_20pct_{_regime_name_ood}_N{len(_df_is_recent)}'\r\n"
NEW_BLOCK += b"                print(  # RULE[fixbugsprints.md]\r\n"
NEW_BLOCK += b"                    f'[ARCH-25-FIX-A] OOD Guard X_oos=IS reciente propio '\r\n"
NEW_BLOCK += b"                    f'N={len(_df_is_recent)} regimen={_regime_name_ood}'\r\n"
NEW_BLOCK += b"                )\r\n"
NEW_BLOCK += b"            else:\r\n"
NEW_BLOCK += b"                _val_path = self.root / 'data' / 'features' / 'features_validation.parquet'\r\n"
NEW_BLOCK += b"                if _val_path.exists():\r\n"
NEW_BLOCK += b"                    _df_oos_ood = pd.read_parquet(_val_path)\r\n"
NEW_BLOCK += b"                    _oos_source = 'features_validation_parquet_fallback'\r\n"
NEW_BLOCK += b"                    print(  # RULE[fixbugsprints.md]\r\n"
NEW_BLOCK += b"                        f'[ARCH-25-FIX-A] OOD Guard FALLBACK a validation.parquet '\r\n"
NEW_BLOCK += b"                        f'(IS reciente insuf: {len(_df_is_recent)} < 200)'\r\n"
NEW_BLOCK += b"                    )\r\n"
NEW_BLOCK += b"\r\n"
NEW_BLOCK += b"            if _df_oos_ood is not None:\r\n"
NEW_BLOCK += b"                _feats_to_check = [f for f in features_list\r\n"
NEW_BLOCK += b"                                   if f in _df_train_ood.columns and f in _df_oos_ood.columns]\r\n"
NEW_BLOCK += b"                _agent_ctx = str(getattr(self, 'regime_name', 'XGBoost') or 'XGBoost')\r\n"
NEW_BLOCK += b"                _valid_feats, _ood_reports = _ood_filter(\r\n"
NEW_BLOCK += b"                    X_train=_df_train_ood[_feats_to_check],\r\n"
NEW_BLOCK += b"                    X_oos=_df_oos_ood[_feats_to_check],\r\n"
NEW_BLOCK += b"                    context=f'XGBoost/{_agent_ctx}[{_oos_source}]',\r\n"
NEW_BLOCK += b"                )\r\n"
NEW_BLOCK += b"                _not_checked = [f for f in features_list if f not in _feats_to_check]\r\n"
NEW_BLOCK += b"                features_list = _valid_feats + _not_checked\r\n"
NEW_BLOCK += b"                _n_blocked = sum(1 for r in _ood_reports if r.blocked)\r\n"
NEW_BLOCK += b"                if _n_blocked > 0:\r\n"
NEW_BLOCK += b"                    _blocked_names = [r.feature for r in _ood_reports if r.blocked]\r\n"
NEW_BLOCK += b"                    logger.warning(\r\n"
NEW_BLOCK += b"                        f'[OOD-GUARD] {_n_blocked} features bloqueadas [{_oos_source}]: '\r\n"
NEW_BLOCK += b"                        f'{_blocked_names} -> features_list={len(features_list)}'\r\n"
NEW_BLOCK += b"                    )\r\n"
NEW_BLOCK += b"            else:\r\n"
NEW_BLOCK += b"                logger.debug('[OOD-GUARD] sin fuente OOS disponible -- guard omitido.')\r\n"
NEW_BLOCK += b"        except Exception as _ood_err:\r\n"
NEW_BLOCK += b"            logger.warning(f'[OOD-GUARD] Error: {_ood_err}')\r\n"

new_raw = raw[:try_idx] + NEW_BLOCK + raw[line_end2:]
src_path.write_bytes(new_raw)
print(f"\n[ARCH-25-FIX-A] Aplicado ({len(raw)} -> {len(new_raw)} bytes)")

import ast
try:
    ast.parse(new_raw.decode("utf-8", "replace"))
    print("[OK] Syntax valida")
except SyntaxError as e:
    print(f"[ERROR] L{e.lineno}: {e.msg} -- ROLLBACK")
    src_path.write_bytes(raw)
    print("[ROLLBACK] original restaurado")

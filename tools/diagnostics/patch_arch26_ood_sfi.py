"""
patch_arch26_ood_sfi.py — ARCH-26-FIX-A
Reemplaza OOD-GUARD-SFI para usar el ultimo 20% del IS propio
en lugar de features_validation.parquet (100% BULL_TREND 2025).
"""
import sys, ast
sys.path.insert(0, '.')
from pathlib import Path

src = Path("luna/features/feature_selection_e.py")
raw = src.read_bytes()
text = raw.decode('utf-8', 'replace')

# Buscar la funcion exacta a reemplazar
START_TAG = "_val_path_ood = DATA_DIR"
END_TAG = "        # [Paso B] Clustering"

idx_start_raw = text.find(START_TAG)
if idx_start_raw < 0:
    print("ERROR: marcador START no encontrado")
    sys.exit(1)

# Retroceder al inicio de la linea del comentario OOD-GUARD-SFI (2 lineas antes)
# que es el bloque de separadores y comentario
line_start = text.rfind('\n', 0, idx_start_raw) + 1
# Buscar la linea de comentario que precede (# -- [OOD-GUARD-SFI] o # ----- )
prev_comment = text.rfind('\n        # ', 0, line_start)
if prev_comment >= 0:
    # Verificar que es la linea de separadores/comentario del OOD block
    snippet = text[prev_comment:line_start]
    if 'OOD-GUARD-SFI' in snippet or '---' in snippet:
        line_start = prev_comment + 1

idx_end = text.find(END_TAG, idx_start_raw)
if idx_end < 0:
    # Buscar por X_raw = df[raw_cols]
    idx_end = text.find("        X_raw = df[raw_cols]", idx_start_raw)
    if idx_end < 0:
        print("ERROR: fin del bloque no encontrado")
        sys.exit(1)

# Ajustar a inicio de linea
idx_end = text.rfind('\n', 0, idx_end) + 1

OLD_BLOCK = text[line_start:idx_end]
print(f"Bloque encontrado ({len(OLD_BLOCK)} chars, {OLD_BLOCK.count(chr(10))} lineas)")
print(f"Primeras 200 chars: {repr(OLD_BLOCK[:200])}")
print("---")

NEW_BLOCK = """\
        # -- [ARCH-26-FIX-A 2026-06-02] OOD-GUARD-SFI con IS reciente (MISMO PATRON QUE ARCH-25) --
        # PROBLEMA: OOD-GUARD-SFI usaba features_validation.parquet = Ene-Abr 2025 (100% BULL extremo).
        # Features macro (M2, T10Y2Y, CPI, yield_curve) tienen baja varianza en bull puro ->
        # OOD las bloqueaba por LOW_STD -> 79/93 candidatas eliminadas (85% del pool SFI).
        # SOLUCION: usar ultimo 20% del IS como referencia OOD (mismo regimen, sin sesgo de distribucion).
        # Consistente con ARCH-25-FIX-A (XGBoost OOD Guard usa IS reciente).
        _ood_split_pct = 0.20   # ultimo 20% del IS como referencia OOD
        _ood_n_ref   = max(int(len(df) * _ood_split_pct), 200)
        _ood_n_train = len(df) - _ood_n_ref
        _ood_is_train = df.iloc[:_ood_n_train]
        _ood_is_ref   = df.iloc[_ood_n_train:]
        print(   # RULE[fixbugsprints.md]
            f"[ARCH-26-FIX-A] OOD-GUARD-SFI: IS reciente como ref OOD "
            f"(train={_ood_n_train} filas, ref={_ood_n_ref} filas, split={_ood_split_pct:.0%})"
        )
        logger.info(
            f"[ARCH-26-FIX-A] OOD-GUARD-SFI: referencia OOD = ultimas {_ood_n_ref} filas IS "
            f"({_ood_split_pct:.0%}). ANTES usaba features_validation.parquet (regimen BULL 2025)."
        )
        try:
            from luna.utils.ood_feature_guard import filter_ood_features as _ood_sfi_filter
            _ood_checkable = [c for c in raw_cols
                              if c in _ood_is_ref.columns and c in _ood_is_train.columns]
            if _ood_checkable:
                _valid_raw, _ood_rpts = _ood_sfi_filter(
                    X_train=_ood_is_train[_ood_checkable],
                    X_oos=_ood_is_ref[_ood_checkable],
                    context="SFI/CandidatePool-IS",
                )
                _n_ood_blocked = sum(1 for r in _ood_rpts if r.blocked)
                if _n_ood_blocked > 0:
                    _ood_blocked_names = [r.feature for r in _ood_rpts if r.blocked]
                    logger.warning(
                        f"[ARCH-26-FIX-A][OOD-GUARD-SFI] {_n_ood_blocked} features bloqueadas "
                        f"por distribucion degenerada en IS reciente: {_ood_blocked_names}. "
                        f"Estas features colapsan dentro del IS -> eliminar es correcto."
                    )
                    print(   # RULE[fixbugsprints.md]
                        f"[ARCH-26-FIX-A] OOD-GUARD-SFI bloqueo {_n_ood_blocked} features "
                        f"(degeneradas en IS reciente): {_ood_blocked_names}"
                    )
                    _ood_not_checked = [c for c in raw_cols if c not in _ood_checkable]
                    raw_cols = _valid_raw + _ood_not_checked
                else:
                    logger.info(
                        f"[ARCH-26-FIX-A][OOD-GUARD-SFI] OK: {len(_ood_checkable)} candidatos "
                        f"pasan el control OOD (IS reciente). Pool SFI intacto."
                    )
                    print(   # RULE[fixbugsprints.md]
                        f"[ARCH-26-FIX-A] OOD-GUARD-SFI: {len(_ood_checkable)} candidatos OK "
                        f"(sin degeneracion en IS reciente)"
                    )
            else:
                logger.debug("[ARCH-26-FIX-A][OOD-GUARD-SFI] No hay columnas evaluables -> guard omitido.")
        except Exception as _ood_sfi_err:
            logger.warning(
                f"[ARCH-26-FIX-A][OOD-GUARD-SFI] Error en guard OOD -> pool sin filtrar: {_ood_sfi_err}"
            )
            print(f"[ARCH-26-FIX-A] OOD-GUARD-SFI ERROR (pool conservado): {_ood_sfi_err}")
        # -----------------------------------------------------------------------------------------

"""

new_text = text[:line_start] + NEW_BLOCK + text[idx_end:]
src.write_bytes(new_text.encode('utf-8'))
print(f"\n[ARCH-26-FIX-A] Aplicado ({len(text)} -> {len(new_text)} chars)")

# Validar syntax
try:
    ast.parse(new_text)
    print("[OK] Syntax valida")
except SyntaxError as e:
    print(f"[ERROR] L{e.lineno}: {e.msg} -- ROLLBACK")
    src.write_bytes(raw)
    print("[ROLLBACK] original restaurado")
    sys.exit(1)

# Verificar que el marcador esta en el nuevo texto
assert 'ARCH-26-FIX-A' in new_text, "Marcador no encontrado post-patch"
assert '_ood_is_ref' in new_text, "_ood_is_ref no encontrado"
assert 'features_validation.parquet' not in new_text[new_text.find('ARCH-26-FIX-A')-500:new_text.find('ARCH-26-FIX-A')+3000], \
    "Aun referencia features_validation.parquet en bloque OOD"
print("[OK] Verificaciones de contenido: ARCH-26-FIX-A activo, features_validation.parquet eliminado")

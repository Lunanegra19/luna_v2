"""
OP-01: Utilidad compartida para el bloque PASS-THROUGH de reglas de Mining.

Antes este bloque de 10 lineas estaba duplicado literalmente en:
  - core/models/train_xgboost.py (L61-78)
  - core/models/train_metalabeler.py (L235-247)

Centralizado aqui para mantener coherencia. Si cambia la logica de seleccion
de reglas, solo hay que tocar este archivo.

FIX-PASSTHROUGH-VAR-01 (2026-03-21):
  El filtro anterior (sum() > 0) solo verificaba activaciones absolutas en IS
  pero no garantizaba varianza suficiente. Reglas con 1-2 hits en IS de 35,000
  filas son efectivamente constantes (var ~ 0), contaminando el modelo con ruido.

  Nuevo criterio:
    - min_hits: al menos N activaciones absolutas (default=10)
    - min_var_pct: al menos P% de filas con activacion (default=0.1%)
  Una regla que falla ambos criterios es DESCARTADA del pass-through.

  Raiz del bug: golden_rule_0..9 tenian 0 hits en features_validation y
  features_holdout porque sus condiciones nunca se activan en mercado alcista
  2025 (DXY<99, funding positivo, etc.). Se calculaban correcto sobre IS
  (2020-2023) pero en OOS eran siempre 0 -> noise puro para el modelo.
"""

try:
    from loguru import logger as _logger
except ImportError:
    import logging as _logging
    _logger = _logging.getLogger(__name__)


def get_active_passthrough_rules(
    df,
    features_list: list,
    min_hits: int = 10,
    min_var_pct: float = 0.001,
) -> list:
    """
    Devuelve las columnas de reglas Mining (golden_rule_N / genetic_rule_N)
    que tienen suficiente varianza para ser utiles al modelo.

    FIX-PASSTHROUGH-VAR-01: ademas de verificar activaciones absolutas,
    ahora exige un minimo dinamico de hits basado en el tamano del dataset:
      - min_hits: floor absoluto (default=10 activaciones)
      - min_var_pct: floor relativo (default=0.1% de filas activas)
      El threshold efectivo = max(min_hits, n_rows * min_var_pct)

    Esto elimina reglas quasi-constantes que el SFI no evalua por ser
    pass-through, pero que degradan silenciosamente el modelo en OOS.

    Args:
        df: DataFrame con las features completas (debe incluir las rule cols).
        features_list: Lista de features ya seleccionadas (por SFI u otro).
        min_hits: Numero minimo de activaciones absolutas para incluir la regla.
        min_var_pct: Fraccion minima de filas con activacion (0.001 = 0.1%).

    Returns:
        Lista de nuevas rule columns a inyectar (PASS-THROUGH).
    """
    n_rows = max(len(df), 1)
    # Threshold dinamico: el mayor de los dos criterios
    min_hits_effective = max(min_hits, int(n_rows * min_var_pct))

    candidate_cols = sorted([
        c for c in df.columns
        if (c.startswith("golden_rule_") or c.startswith("genetic_rule_"))
    ])

    accepted = []
    rejected = []

    for c in candidate_cols:
        if c in features_list:
            continue  # ya seleccionada por SFI, no duplicar

        n_hits = int(df[c].sum()) if c in df.columns else 0

        if n_hits >= min_hits_effective:
            accepted.append(c)
        else:
            rejected.append((c, n_hits))

    if rejected:
        _logger.warning(
            "[PASSTHROUGH] FIX-PASSTHROUGH-VAR-01: {} reglas descartadas "
            "(min_hits_effective={}): {}",
            len(rejected),
            min_hits_effective,
            [(c, h) for c, h in rejected[:10]]
        )
    if accepted:
        _logger.info(
            "[PASSTHROUGH] {} reglas aceptadas (>={} hits): {}",
            len(accepted), min_hits_effective, accepted[:10]
        )

    return accepted

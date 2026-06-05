"""
Export Alpha Rules — Luna V1 AI Mining (Engine 7/7)
====================================================================
Propósito: Lee los reportes generados por los 6 engines y los consolida
en un módulo Python nativo `core/features/alpha_rules.py`.

REGLA SOP R13: CERO JSON BRIDGES.
  El módulo generado es Python puro — no JSON, no pickle, no CSV.
  El feature_pipeline.py importa directamente:
      from luna.features.alpha_rules import get_alpha_features

Estructura del alpha_rules.py generado:
  - GOLDEN_RULES: list[dict]   — del master_pattern_report.md
  - GENETIC_RULES: list[dict]  — del deep_discovery_report.md (AG)
  - CAUSAL_VARS: list[str]     — del advanced_engine_report.md (evidence_score >= 3)
  - DTW_BULL_PROB: float        — del deep_discovery_report.md
  - TRIBE_BIAS: dict[int, str] — bias por tribu del cluster_pattern_report.md
  - get_alpha_features(df)     — función que calcula las señales alpha

Este módulo se regenera SEMANALMENTE offline.
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger

# ==============================================================================
# RUTAS CONSTANTES
# ==============================================================================
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR   = PROJECT_ROOT / "data" / "ai_mining" / "reports"
print("[BUGFIX-ALPHA-EXPORTER] Ruta de reportes corregida de 'data/reports/mining' a 'data/ai_mining/reports' para evitar 0 reglas en exportación.")
OUTPUT_FILE   = PROJECT_ROOT / "luna" / "features" / "alpha_rules.py"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers de reportes
# ─────────────────────────────────────────────────────────────────────────────

def parse_golden_rules(report_path: Path) -> list[dict]:
    """
    Extrae Golden Rules de master_pattern_report.md.
    Tabla esperada: | Condicion Exacta | Régimen | Win Rate | Expected Value | N |
    """
    if not report_path.exists():
        logger.warning(f"No encontrado: {report_path}")
        return []

    content = report_path.read_text(encoding="utf-8")
    table_match = re.search(
        r"\|\s*Condici[oó]n(?:\s+Exacta)?.*?\n\|.*?\n((?:\|.*?\n)+)", content, re.DOTALL
    )
    if not table_match:
        logger.warning("Golden Rules: tabla no encontrada en reporte")
        return []

    rules = []
    for row in table_match.group(1).strip().split("\n"):
        cols = [c.strip() for c in row.split("|") if c.strip()]
        if len(cols) < 4:
            continue
        raw_cond = cols[0]

        # Extraer condiciones entre backticks: `VAR >= 0.12`
        cond_parts = re.findall(r"`([^`]+)`", raw_cond)
        if not cond_parts:
            # Intentar sin backticks (condiciones directas)
            cond_parts = re.findall(r"([A-Za-z0-9_]+ [<>=]+ [\d.+-]+)", raw_cond)

        if not cond_parts:
            continue

        pandas_eval = " & ".join([f"({c})" for c in cond_parts])
        try:
            wr_raw = float(re.sub(r"[*%]", "", cols[2]).strip())
            # Fix M-05: guard contra doble normalización. El reporte debe generar
            # valores como "61.6" (porcentaje). Si genera "0.616" (fracción) → escalar.
            wr = wr_raw if wr_raw > 1.0 else wr_raw * 100.0
            ev = float(re.sub(r"[*%+]", "", cols[3]).strip())
        except (ValueError, IndexError):
            wr, ev = 0.0, 0.0

        # Fix A-07: validar que pandas_eval no contiene expresiones arbitrarias
        # antes de persistirlo en alpha_rules.py (se ejecuta con df.eval engine='python').
        _SAFE_EVAL = re.compile(r'^[\w\s\d\.\(\)&|<>=!+\-\*\/\.]+$')
        if not _SAFE_EVAL.match(pandas_eval):
            logger.warning(f"A-07: expresion pandas_eval unsafe descartada: {pandas_eval[:80]}")
            continue

        rules.append({
            "type":         "golden_storm",
            "pandas_eval":  pandas_eval,
            "win_rate":     wr,
            "ev_pct":       ev,
            "description":  raw_cond.replace("**", "").replace("<br>", " ").replace("`", ""),
        })

    rules.sort(key=lambda x: x["win_rate"], reverse=True)
    logger.info(f"Golden Rules: {len(rules)} reglas extraídas")
    return rules


def parse_genetic_rules(report_path: Path) -> list[dict]:
    """
    Extrae Genetic Rules de deep_discovery_report.md.
    Tabla: | Regla Logica | Win Rate | EV% | N |
    Formato condición: 'VAR op Pt (v=VALUE) AND ...'
    """
    if not report_path.exists():
        logger.warning(f"No encontrado: {report_path}")
        # Intentar cargar desde CSV si existe
        csv_path = report_path.parent / "deep_discovery_rules.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rules = []
            for _, row in df.iterrows():
                rules.append({
                    "type":        "genetic_rule",
                    "pandas_eval": row.get("pandas_eval", ""),
                    "win_rate":    float(row.get("win_rate", 0)),
                    "ev_pct":      float(row.get("ev_pct", 0)),
                    "description": row.get("conditions", ""),
                })
            logger.info(f"Genetic Rules: {len(rules)} desde CSV")
            return rules
        return []

    content = report_path.read_text(encoding="utf-8")
    # Soportar tanto el header antiguo '| Regla Logica' como el nuevo '| Condiciones (IF/AND)'
    table_match = re.search(r"\|\s*(?:Regla Logica|Condiciones \(IF/AND\)).*?\|\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
    if not table_match:
        logger.warning("Genetic Rules: tabla no encontrada")
        return []

    rules = []
    rows_raw = re.split(r"\n\|", "\n" + table_match.group(1).strip())

    for row in rows_raw:
        # El separador de condiciones en la celda es `<br>**AND**`
        # NO hacer replace(<br>, " ") antes de split — rompe el split por AND
        row_no_newline = row.replace("\n", " ")
        cols = [c.strip() for c in row_no_newline.split("|")]
        if len(cols) < 3:
            continue
        raw_cond = cols[0]
        if not raw_cond or raw_cond.startswith("---"):
            continue

        # Separar condiciones: split por <br>**AND** (nuevo formato) O por ' AND ' (simple)
        if "<br>" in raw_cond:
            cond_blocks_raw = re.split(r"<br>\s*\*{0,2}AND\*{0,2}\s*", raw_cond)
        else:
            cond_blocks_raw = raw_cond.split(" AND ")

        pandas_exprs = []
        for block in cond_blocks_raw:
            # Limpiar markdown: backticks, **, AND residual
            block_clean = block.replace("`", "").replace("**AND**", "").replace("**", "").strip()
            if not block_clean:
                continue
            # Nuevo formato directo: 'NASDAQ_Ret >= 0.0094'
            m_direct = re.search(r"([A-Za-z0-9_]+)\s*([<>=]+)\s*([-\d.]+)", block_clean)
            if m_direct:
                pandas_exprs.append(f"({m_direct.group(1)} {m_direct.group(2)} {m_direct.group(3)})")
            else:
                # Antiguo formato: 'VAR <= Pt (v=VALUE)'
                m = re.search(r"([A-Za-z0-9_]+)\s*([<>=]+).*?\(v=([-\d.]+)\)", block_clean)
                if m:
                    pandas_exprs.append(f"({m.group(1)} {m.group(2)} {m.group(3)})")

        if pandas_exprs:
            pandas_eval = " & ".join(pandas_exprs)
            try:
                wr = float(re.sub(r"[^0-9.-]", "", cols[1]))
                ev = float(re.sub(r"[^0-9.-]", "", cols[2]))
            except (ValueError, IndexError):
                wr, ev = 0.0, 0.0

            rules.append({
                "type":        "genetic_rule",
                "pandas_eval": pandas_eval,
                "win_rate":    wr,
                "ev_pct":      ev,
                "description": raw_cond.strip(),
            })

    rules.sort(key=lambda x: x["win_rate"], reverse=True)
    logger.info(f"Genetic Rules: {len(rules)} reglas extraídas")
    return rules


def parse_causal_vars(report_path: Path, min_score: int = 3) -> list[str]:
    """
    Extrae variables con evidence_score >= min_score de
    advanced_engine_results.csv o advanced_engine_report.md.
    """
    csv_path = REPORTS_DIR / "advanced_engine_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "evidence_score" in df.columns and "variable" in df.columns:
            sig = df[df["evidence_score"] >= min_score]["variable"].tolist()
            logger.info(f"Causal vars: {len(sig)} con score >= {min_score}")
            return sig

    if not report_path.exists():
        return []

    content = report_path.read_text(encoding="utf-8")
    # Buscar en la sección de variables significativas
    sig_match = re.search(r"## Variables Causales Significativas.*?\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
    if not sig_match:
        return []

    vars_found = re.findall(r"`([A-Za-z0-9_]+)`", sig_match.group(1))
    logger.info(f"Causal vars: {len(vars_found)} variables significativas")
    return list(set(vars_found))


def parse_dtw_bull_prob(report_path: Path) -> float:
    """Extrae DTW Bull Probability del deep_discovery_report.md."""
    if not report_path.exists():
        return 0.5
    content = report_path.read_text(encoding="utf-8")
    # Soportar formato nuevo español: "Probabilidad alcista DTW: 60.0% →"
    # Y formato antiguo inglés: "Bull Probability: 60.0%"
    m = re.search(r"(?:Bull Probability|Probabilidad alcista DTW).*?(\d+\.?\d*)%", content)
    if m:
        prob = float(m.group(1)) / 100.0
        logger.info(f"DTW Bull Probability: {prob:.1%}")
        return prob
    return 0.5


def parse_tribe_bias(report_path: Path) -> dict[int, str]:
    """Extrae el bias y WR de cada tribu del cluster_pattern_report.md.

    Returns:
        bias: dict[int, str] — {tribe_id: 'LARGA'/'NEUTRAL'/'CORTA'}
    """
    if not report_path.exists():
        return {}
    content = report_path.read_text(encoding="utf-8")

    # Tabla: | Tribu | N Barras | Win Rate | Ret 24H | Sharpe | Régimen |
    table_match = re.search(r"K-Means Tribus.*?\n\|.*?\n\|.*?\n((?:\|.*?\n)+)", content, re.DOTALL)
    if not table_match:
        return {}

    bias: dict[int, str] = {}
    for row in table_match.group(1).strip().split("\n"):
        cols = [c.strip() for c in row.split("|") if c.strip()]
        if len(cols) >= 6:
            try:
                tribe_cell = cols[0]
                tribe_m = re.search(r"(\d+)", tribe_cell)
                if not tribe_m:
                    continue
                tribe_id = int(tribe_m.group(1))
                # Última columna con LARGA/CORTA/NEUTRAL
                regime_m = re.search(r"\b(LARGA|CORTA|NEUTRAL)\b", " ".join(cols))
                if regime_m:
                    bias[tribe_id] = regime_m.group(1)
            except (ValueError, IndexError):
                continue
    logger.info(f"Tribe bias: {len(bias)} tribus parseadas")
    return bias


def parse_tribe_wr(report_path: Path) -> dict[int, float]:
    """Extrae el Win Rate numérico de cada tribu del cluster_pattern_report.md.

    Returns:
        wr_map: dict[int, float] — {tribe_id: win_rate_fraction}
    """
    if not report_path.exists():
        return {}
    content = report_path.read_text(encoding="utf-8")

    table_match = re.search(r"K-Means Tribus.*?\n\|.*?\n\|.*?\n((?:\|.*?\n)+)", content, re.DOTALL)
    if not table_match:
        return {}

    wr_map: dict[int, float] = {}
    for row in table_match.group(1).strip().split("\n"):
        cols = [c.strip() for c in row.split("|") if c.strip()]
        if len(cols) >= 3:
            try:
                tribe_cell = cols[0]
                tribe_m = re.search(r"(\d+)", tribe_cell)
                if not tribe_m:
                    continue
                tribe_id = int(tribe_m.group(1))
                # Win Rate suele estar en cols[2] — buscar patrón ##.#%
                wr_m = re.search(r"(\d+\.?\d*)%", " ".join(cols[1:4]))
                if wr_m:
                    wr_map[tribe_id] = round(float(wr_m.group(1)) / 100.0, 4)
            except (ValueError, IndexError):
                continue
    logger.info(f"Tribe WR map: {wr_map}")
    return wr_map



# ─────────────────────────────────────────────────────────────────────────────
# Generador del módulo Python nativo
# ─────────────────────────────────────────────────────────────────────────────

def generate_alpha_rules_module(
    golden_rules: list[dict],
    genetic_rules: list[dict],
    causal_vars:   list[str],
    dtw_bull_prob: float,
    tribe_bias:    dict[int, str],
    tribe_wr_map:  dict[int, float] | None = None,  # M3 nuevo
) -> str:
    """
    Genera el contenido del fichero alpha_rules.py como string Python.
    Incluye TRIBE_WR_MAP, LARGA_TRIBES, NEUTRAL_TRIBES y apply_tribe_features (M3).
    """
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        '"""',
        f"alpha_rules.py — GENERADO AUTOMÁTICAMENTE por export_alpha_rules.py",
        f"Timestamp: {now_str}",
        f"Golden Rules: {len(golden_rules)}  |  Genetic Rules: {len(genetic_rules)}",
        f"DO NOT EDIT MANUALLY — se sobreescribe semanalmente con run_weekly_mining.py",
        '"""',
        "",
        "from __future__ import annotations",
        "import pandas as pd",
        "import numpy as np",
        "",
        "# ──────────────────────────────────────────────────────────────────",
        "# GOLDEN STORM RULES (Master Pattern Engine)",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        "GOLDEN_RULES: list[dict] = [",
    ]

    for r in golden_rules[:15]:  # top-15 golden rules
        safe_desc = r["description"].replace("'", "\\'")[:120]
        lines.append(f"    {{")
        lines.append(f"        'type':        'golden_storm',")
        lines.append(f"        'pandas_eval': {repr(r['pandas_eval'])},")
        lines.append(f"        'win_rate':    {r['win_rate']},")
        lines.append(f"        'ev_pct':      {r['ev_pct']},")
        lines.append(f"        'description': {repr(safe_desc)},")
        lines.append(f"    }},")
    lines.append("]")
    lines.append("")

    lines += [
        "# ──────────────────────────────────────────────────────────────────",
        "# GENETIC RULES (Deep Discovery Engine — AG 20 generaciones)",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        "GENETIC_RULES: list[dict] = [",
    ]
    for r in genetic_rules[:20]:  # top-20 genetic rules
        safe_desc = r["description"].replace("'", "\\'")[:120]
        lines.append(f"    {{")
        lines.append(f"        'type':        'genetic_rule',")
        lines.append(f"        'pandas_eval': {repr(r['pandas_eval'])},")
        lines.append(f"        'win_rate':    {r['win_rate']},")
        lines.append(f"        'ev_pct':      {r['ev_pct']},")
        lines.append(f"        'description': {repr(safe_desc)},")
        lines.append(f"    }},")
    lines.append("]")
    lines.append("")

    lines += [
        "# ──────────────────────────────────────────────────────────────────",
        "# CAUSAL VARIABLES (Advanced Engine — Granger*** + TE_net > 0)",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        f"CAUSAL_VARS: list[str] = {repr(causal_vars)}",
        "",
        "# ──────────────────────────────────────────────────────────────────",
        "# DTW FRACTAL PROBABILITY",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        f"DTW_BULL_PROB: float = {dtw_bull_prob}  # P(BTC sube en 24H | análogos históricos)",
        "",
        "# ──────────────────────────────────────────────────────────────────",
        "# K-MEANS TRIBE BIAS",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        f"TRIBE_BIAS: dict[int, str] = {repr(tribe_bias)}",
        "",
        "# ──────────────────────────────────────────────────────────────────",
        "# K-MEANS TRIBE WIN-RATE MAP (M3 — actualizado semanalmente)",
        "# ──────────────────────────────────────────────────────────────────",
        "",
    ]

    # M3: construir TRIBE_WR_MAP desde el parse de WRs (fallback a valores por bias)
    default_wr_by_bias = {'LARGA': 0.56, 'NEUTRAL': 0.50, 'CORTA': 0.44}
    wr_map_final: dict[int, float] = {}
    for tid, bias_val in tribe_bias.items():
        if tribe_wr_map and tid in tribe_wr_map:
            wr_map_final[tid] = tribe_wr_map[tid]
        else:
            wr_map_final[tid] = default_wr_by_bias.get(bias_val, 0.50)

    larga_set   = {tid for tid, b in tribe_bias.items() if b == 'LARGA'}
    neutral_set = {tid for tid, b in tribe_bias.items() if b in ('NEUTRAL', 'CORTA')}

    lines += [
        f"TRIBE_WR_MAP: dict[int, float] = {repr(wr_map_final)}",
        "",
        f"LARGA_TRIBES   = frozenset({repr(larga_set)})",
        f"NEUTRAL_TRIBES = frozenset({repr(neutral_set)})",
        "",
        "ALL_RULES: list[dict] = GOLDEN_RULES + GENETIC_RULES",
        "",
    ]



    # ── Función get_alpha_features ────────────────────────────────────────
    lines += [
        "# ──────────────────────────────────────────────────────────────────",
        "# FUNCIÓN PRINCIPAL — Cálculo de señales alpha",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        "def get_alpha_features(df: pd.DataFrame) -> pd.DataFrame:",
        '    """',
        "    Calcula señales alpha a partir de las reglas descubiertas por AI Mining.",
        "    Llamada en el paso 9 de feature_pipeline.py.",
        "",
        "    Args:",
        "        df: DataFrame con todas las features crudas (post-lag, post-zscore)",
        "",
        "    Returns:",
        "        df con columnas adicionales:",
        "          - alpha_golden_score  : suma de Golden Rules activas (0 a N)",
        "          - alpha_genetic_score : suma de Genetic Rules activas (0 a N)",
        "          - alpha_combined      : señal combinada [-1, 1]",
        "          - alpha_dtw_signal    : señal DTW dinámica: dtw_direction × tanh(mom_24H × 20) ∈ [-1,1]",
        "          - alpha_tribe_bias    : sesgo de tribu (-1/0/1)",
        '    """',
        "    df = df.copy()",
        "",
        "    # ── Golden Storm Score ──────────────────────────────────────────",
        "    golden_score = pd.Series(0.0, index=df.index)",
        "    for rule in GOLDEN_RULES:",
        "        try:",
        "            mask = df.eval(rule['pandas_eval'], engine='python')",
        "            golden_score += mask.astype(float) * (rule['win_rate'] / 100.0)",
        "        except Exception:",
        "            pass",
        "    # Normalizar al número de reglas activas máximo",
        "    if GOLDEN_RULES:",
        "        golden_score /= len(GOLDEN_RULES)",
        "    df['alpha_golden_score'] = golden_score.clip(0, 1)",
        "",
        "    # ── Genetic Score ───────────────────────────────────────────────",
        "    genetic_score = pd.Series(0.0, index=df.index)",
        "    for rule in GENETIC_RULES:",
        "        try:",
        "            mask = df.eval(rule['pandas_eval'], engine='python')",
        "            genetic_score += mask.astype(float) * (rule['win_rate'] / 100.0)",
        "        except Exception:",
        "            pass",
        "    if GENETIC_RULES:",
        "        genetic_score /= len(GENETIC_RULES)",
        "    df['alpha_genetic_score'] = genetic_score.clip(0, 1)",
        "",
        "    # ── Combined Alpha ──────────────────────────────────────────────",
        "    # Ponderado: Golden 60% | Genetic 40%",
        "    combined = 0.6 * df['alpha_golden_score'] + 0.4 * df['alpha_genetic_score']",
        "    # Centrar en 0: [0,1] → [-1,1]",
        "    df['alpha_combined'] = (combined * 2 - 1).clip(-1, 1)",
        "",
        "    # ── DTW Fractal Signal (dinámica horaria) ────────────────────────",
        "    # 1. Dirección del bias DTW: +1 si bull, -1 si bear",
        "    # 2. Momentum a 24H del precio actual como amplitud dinámica",
        "    # tanh escala el momentum a [-1,1] de forma suave",
        "    dtw_direction = 1.0 if DTW_BULL_PROB >= 0.5 else -1.0",
        "    if 'close' in df.columns:",
        "        mom_24h = df['close'].pct_change(24).fillna(0.0)",
        "    else:",
        "        mom_24h = pd.Series(0.0, index=df.index)",
        "    df['alpha_dtw_signal'] = (dtw_direction * np.tanh(mom_24h * 20)).clip(-1, 1)",
        "",
        "    # ── Tribe Bias ──────────────────────────────────────────────────",
        "    # Soporta K_Shape_Cluster_ID (kshape_engine) y KMeans_Tribe_ID (cluster_pattern_engine)",
        "    bias_map = {",
        "        tid: (1 if bias == 'LARGA' else -1 if bias == 'CORTA' else 0)",
        "        for tid, bias in TRIBE_BIAS.items()",
        "    }",
        "    # P3-3-FIX: KMeans_Tribe_ID es la columna primaria (K_Shape decommisionado)",
        "    _tribe_col = None",
        "    if 'KMeans_Tribe_ID' in df.columns:",
        "        _tribe_col = 'KMeans_Tribe_ID'",
        "    elif 'K_Shape_Cluster_ID' in df.columns:  # legacy fallback solo",
        "        _tribe_col = 'K_Shape_Cluster_ID'",
        "    if _tribe_col:",
        "        df['alpha_tribe_bias'] = df[_tribe_col].map(bias_map).fillna(0).astype(float)",
        "    else:",
        "        df['alpha_tribe_bias'] = 0.0",
        "",
        "    # ── [PASS-THROUGH] Golden/Genetic Rule binaries ─────────────────",
        "    # Genera golden_rule_0..N y genetic_rule_0..M como columnas 0/1.",
        "    # Son pass-through: bypasan SFI y van directo a XGBoost.",
        "    # COL ALIASES: mapea nombres del Mining a nombres en features_train.",
        "    _COL_ALIASES = {",
        "        # Yield Curve: Mining usa nombre FRED, features_train usa nombre propio",
        "        'YieldCurve_10Y3M': 'yield_curve_spread',",
        "        'T10Y2Y':           'yield_curve_spread',",
        "        # Onchain: active_addresses_7d_ma -> ActiveAddresses_7d",
        "        'active_addresses_7d_ma': 'ActiveAddresses_7d',",
        "        # SSR: Mining usa 'SSR' (raw), features_train tiene 'SSR_ZScore'",
        "        'SSR':              'SSR_ZScore',",
        "    }",
        "    _df2 = df.copy()",
        "    for _orig, _alias in _COL_ALIASES.items():",
        "        if _orig not in _df2.columns and _alias in _df2.columns:",
        "            _df2[_orig] = _df2[_alias]",
        "    # FearGreed: Mining usa escala raw 0-100; pipeline lo normaliza a 0-1.",
        "    # Convertimos FearGreed_Normalized -> FearGreed (0-100) para que",
        "    # los thresholds del Mining (ej: 84) sean comparables.",
        "    if 'FearGreed' not in _df2.columns and 'FearGreed_Normalized' in _df2.columns:",
        "        _df2['FearGreed'] = _df2['FearGreed_Normalized'] * 100.0",
        "    # NASDAQ_Ret: Mining usa retorno porcentual de NASDAQ 1H.",
        "    # features_train tiene 'NASDAQ' como precio raw -> calculamos ret.",
        "    if 'NASDAQ_Ret' not in _df2.columns and 'NASDAQ' in _df2.columns:",
        "        _df2['NASDAQ_Ret'] = _df2['NASDAQ'].pct_change(1).fillna(0.0)",
        "    # Whale_Vol_ZScore: proxy basado en Tx_Volume (BTC en cadena) si disponible,",
        "    # o volume OHLCV como fallback.",
        "    if 'Whale_Vol_ZScore' not in _df2.columns:",
        "        _vsrc = _df2['Tx_Volume'] if 'Tx_Volume' in _df2.columns else _df2.get('volume', None)",
        "        if _vsrc is not None:",
        "            _vol = _vsrc.replace(0, float('nan'))",
        "            _roll = _vol.rolling(window=90*24, min_periods=720)",
        "            _df2['Whale_Vol_ZScore'] = ((_vol - _roll.mean()) / _roll.std().replace(0, 1)).ffill().fillna(0).clip(-4, 4)",
        "    for _i, _rule in enumerate(GOLDEN_RULES):",
        "        _col = f'golden_rule_{_i}'",
        "        try:",
        "            _mask = _df2.eval(_rule['pandas_eval'], engine='python')",
        "            df[_col] = _mask.astype(float).fillna(0.0)",
        "        except Exception:",
        "            df[_col] = 0.0",
        "    for _i, _rule in enumerate(GENETIC_RULES):",
        "        _col = f'genetic_rule_{_i}'",
        "        try:",
        "            _mask = _df2.eval(_rule['pandas_eval'], engine='python')",
        "            df[_col] = _mask.astype(float).fillna(0.0)",
        "        except Exception:",
        "            df[_col] = 0.0",
        "",
        "    return df",
        "",
        "",
        "def get_rule_summary() -> dict:",
        '    """Resumen de reglas disponibles para logging/monitoring."""',
        "    return {",
        "        'golden_rules':    len(GOLDEN_RULES),",
        "        'genetic_rules':   len(GENETIC_RULES),",
        "        'causal_vars':     len(CAUSAL_VARS),",
        "        'dtw_bull_prob':   DTW_BULL_PROB,",
        "        'tribe_bias_keys': list(TRIBE_BIAS.keys()),",
        "        'tribe_wr_map':    TRIBE_WR_MAP,",
        "    }",
        "",
        "",
        "# ──────────────────────────────────────────────────────────────────",
        "# MEJORA M3 — Features derivadas de Tribe para XGBoost",
        "# Llamada en feature_pipeline.py paso 9B",
        "# ──────────────────────────────────────────────────────────────────",
        "",
        "def apply_tribe_features(df: pd.DataFrame) -> pd.DataFrame:",
        '    """',
        "    M3: Convierte KMeans_Tribe_ID/K_Shape_Cluster_ID en features numéricas:",
        "      - tribe_wr_historical : WR histórico continuo (0.0-1.0)",
        "      - tribe_in_larga      : 1.0 si tribu LARGA, 0.0 si no",
        "      - tribe_wr_zscore     : Z-Score rolling 90d del WR",
        '    """',
        "    tribe_col = None",
        "    # P3-3-FIX: KMeans_Tribe_ID es la columna primaria (K_Shape decommisionado)",
        "    if 'KMeans_Tribe_ID' in df.columns:",
        "        tribe_col = 'KMeans_Tribe_ID'",
        "    elif 'K_Shape_Cluster_ID' in df.columns:  # legacy fallback solo",
        "        tribe_col = 'K_Shape_Cluster_ID'",
        "    if tribe_col is None:",
        "        return df",
        "    df = df.copy()",
        "    df['tribe_wr_historical'] = df[tribe_col].map(TRIBE_WR_MAP).fillna(0.5)",
        "    df['tribe_in_larga'] = df[tribe_col].isin(LARGA_TRIBES).astype(float)",
        "    _wr_roll = df['tribe_wr_historical'].rolling(window=90 * 24, min_periods=24)",
        "    df['tribe_wr_zscore'] = (",
        "        (df['tribe_wr_historical'] - _wr_roll.mean()) /",
        "        (_wr_roll.std().replace(0, 1))",
        "    ).clip(-3, 3)",
        "    return df",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Export Alpha Rules — INICIO")
    logger.info("=" * 60)

    # Leer reportes de todos los engines
    golden_rules  = parse_golden_rules(REPORTS_DIR / "master_pattern_report.md")
    genetic_rules = parse_genetic_rules(REPORTS_DIR / "deep_discovery_report.md")
    causal_vars   = parse_causal_vars(REPORTS_DIR / "advanced_engine_report.md")
    dtw_bull_prob = parse_dtw_bull_prob(REPORTS_DIR / "deep_discovery_report.md")
    tribe_bias    = parse_tribe_bias(REPORTS_DIR / "cluster_pattern_report.md")
    tribe_wr_map  = parse_tribe_wr(REPORTS_DIR / "cluster_pattern_report.md")  # M3

    logger.info(f"Golden Rules:  {len(golden_rules)}")
    logger.info(f"Genetic Rules: {len(genetic_rules)}")
    logger.info(f"Causal Vars:   {len(causal_vars)}")
    logger.info(f"DTW Bull Prob: {dtw_bull_prob:.1%}")
    logger.info(f"Tribe Bias:    {tribe_bias}")
    logger.info(f"Tribe WR Map:  {tribe_wr_map}")

    if not golden_rules and not genetic_rules:
        logger.error(
            "No hay reglas disponibles. Ejecutar primero:\n"
            "  1. master_pattern_engine.py\n"
            "  2. deep_discovery_engine.py"
        )
        return

    # Generar módulo Python nativo
    module_content = generate_alpha_rules_module(
        golden_rules, genetic_rules, causal_vars, dtw_bull_prob, tribe_bias,
        tribe_wr_map=tribe_wr_map  # M3: propagar WR mapa
    )

    # Escribir alpha_rules.py
    OUTPUT_FILE.write_text(module_content, encoding="utf-8")
    logger.success(f"[SUCCESS] alpha_rules.py generado en: {OUTPUT_FILE}")
    logger.info(
        f"   {len(golden_rules)} Golden Rules  +  "
        f"{len(genetic_rules)} Genetic Rules  →  "
        f"{len(golden_rules) + len(genetic_rules)} reglas totales"
    )

    # ── [FIX-COUPLING-03] Exportar Ledger de Tribus en JSON ──
    # Para evitar leak de capas, PositionSizer consumirá este JSON.
    try:
        metadata_dir = PROJECT_ROOT / "data" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = metadata_dir / "tribe_analytics.json"
        
        # Re-construir wr_map_final tal como lo hace generate_alpha_rules_module
        default_wr_by_bias = {'LARGA': 0.56, 'NEUTRAL': 0.50, 'CORTA': 0.44}
        wr_map_final: dict[int, float] = {}
        for tid, bias_val in tribe_bias.items():
            if tribe_wr_map and tid in tribe_wr_map:
                wr_map_final[tid] = tribe_wr_map[tid]
            else:
                wr_map_final[tid] = default_wr_by_bias.get(bias_val, 0.50)

        ledger_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "tribe_wr_map": wr_map_final,
            "tribe_bias": tribe_bias
        }
        
        with open(ledger_path, "w", encoding="utf-8") as f_ledger:
            json.dump(ledger_data, f_ledger, indent=4)
            
        logger.success(f"[SUCCESS] Ledger de analíticas de tribu exportado a: {ledger_path}")
        print(f"[FIX-COUPLING-03] [SUCCESS] Metadatos de tribu exportados exitosamente a {ledger_path}")
        print(f"  - Tribus mapeadas: {list(wr_map_final.keys())}")
    except Exception as e:
        logger.error(f"Error al escribir ledger de tribu JSON: {e}")
        print(f"[FIX-COUPLING-03/ERROR] Fallo al guardar tribe_analytics.json: {e}")

    # Verificar que el módulo es importable
    try:
        import importlib.util
        spec   = importlib.util.spec_from_file_location("alpha_rules_test", OUTPUT_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        summary = module.get_rule_summary()
        logger.success(f"[SUCCESS] Módulo importado correctamente: {summary}")
    except Exception as e:
        logger.error(f"Error al importar alpha_rules.py generado: {e}")

    logger.info("Export Alpha Rules — COMPLETADO")


if __name__ == "__main__":
    main()

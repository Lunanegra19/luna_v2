"""
rule_validator.py — Bootstrap WR Validation para señales raras
==============================================================
Valida estadísticamente las PASSTHROUGH_FEATURES (Golden Rules individuales
y Genetic Rules) que el SFI-CPCV no puede evaluar por N insuficiente.

Metodología:
  - Test binomial unilateral: WR observado > 50% (chance aleatoria)
  - Bootstrap 1000 muestras con reemplazo para intervalo de confianza 95%
  - Threshold de validez: p-value < 0.05 (WR estadísticamente > 50%)

Output: data/features/rule_validation_report.md

Uso:
    python core/features/rule_validator.py
    python core/features/rule_validator.py --parquet data/features/features_train.parquet
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR     = PROJECT_ROOT / "data" / "features"
REPORT_FILE  = DATA_DIR / "rule_validation_report.md"
SELECTED_FILE = DATA_DIR / "selected_features.json"

from luna.features.feature_selection_e import PASSTHROUGH_FEATURES


def _bootstrap_wr(hits: np.ndarray, n_bootstrap: int = 1000,
                  ci: float = 0.95) -> tuple[float, float, float]:
    """
    Bootstrap WR: media y IC 95% del Win Rate sobre los hits de la regla.
    
    Args:
        hits: Serie binaria del target en las filas donde la regla se activa (1=win, 0=loss)
        n_bootstrap: Número de muestras bootstrap
        ci: Nivel de confianza
        
    Returns:
        (wr_mean, wr_lower, wr_upper)
    """
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    boot_wrs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(hits, size=n, replace=True)
        boot_wrs.append(sample.mean())
    
    alpha = 1 - ci
    lower = np.percentile(boot_wrs, alpha / 2 * 100)
    upper = np.percentile(boot_wrs, (1 - alpha / 2) * 100)
    return float(np.mean(boot_wrs)), float(lower), float(upper)


def validate_rules(df: pd.DataFrame, target_col: str = "target") -> List[Dict]:
    """
    Valida cada PASSTHROUGH_FEATURE presente en df usando test binomial + Bootstrap.
    
    Args:
        df: features_train.parquet con columnas golden_rule_N, genetic_rule_N y target
        target_col: nombre de la columna target (binario 0/1)
        
    Returns:
        Lista de dicts con resultados por regla
    """
    if target_col not in df.columns:
        if "close" in df.columns:
            logger.info(f"Columna '{target_col}' no encontrada — derivando desde close (retorno 24H > 0)")
            df = df.copy()
            # FIX-LAB-01B: Retorno forward 24H y preservación de NaNs
            fwd_ret = df["close"].pct_change(24).shift(-24)
            y_float = (fwd_ret > 0).astype(float)
            y_float[fwd_ret.isna()] = np.nan
            df[target_col] = y_float
        else:
            logger.error(f"Columna '{target_col}' ni 'close' encontradas en df.")
            return []

    y = df[target_col]
    results = []

    for col in PASSTHROUGH_FEATURES:
        if col not in df.columns:
            logger.warning(f"  [{col}] No encontrada en features_train — ¿ejecutaste --only-features?")
            results.append({
                "feature": col,
                "n_hits": 0,
                "activation_pct": 0.0,
                "wr_observed": 0.0,
                "wr_bootstrap_mean": 0.0,
                "wr_lower_95ci": 0.0,
                "wr_upper_95ci": 0.0,
                "p_value": 1.0,
                "status": "❌ NOT_FOUND",
                "verdict": "NO_DATA"
            })
            continue

        mask = (df[col].fillna(0) > 0) & y.notna()
        n_hits = int(mask.sum())
        n_total = int(y.notna().sum())
        activation_pct = n_hits / n_total * 100 if n_total > 0 else 0.0

        if n_hits < 10:
            logger.warning(f"  [{col}] {n_hits} hits — N insuficiente para test estadístico")
            results.append({
                "feature": col,
                "n_hits": n_hits,
                "activation_pct": activation_pct,
                "wr_observed": 0.0,
                "wr_bootstrap_mean": 0.0,
                "wr_lower_95ci": 0.0,
                "wr_upper_95ci": 0.0,
                "p_value": 1.0,
                "status": "⚠️ TOO_FEW",
                "verdict": "WEAK"
            })
            continue

        hits_target = y[mask].astype(int).values
        wr_obs = float(hits_target.mean())
        n_wins = int(hits_target.sum())

        # Test binomial: ¿WR estadísticamente > 50%?
        try:
            result = binomtest(n_wins, n_hits, p=0.5, alternative='greater')
            p_value = float(result.pvalue)
        except Exception:
            p_value = 1.0

        # Bootstrap CI
        wr_boot, wr_lower, wr_upper = _bootstrap_wr(hits_target)

        valid = p_value < 0.05
        verdict = "VALID" if valid else "WEAK"
        status = "✅ VALID" if valid else "⚠️ WEAK"

        logger.info(
            f"  [{col}] N={n_hits} ({activation_pct:.1f}%) | "
            f"WR={wr_obs:.1%} (IC95: {wr_lower:.1%}-{wr_upper:.1%}) | "
            f"p={p_value:.4f} | {status}"
        )

        results.append({
            "feature": col,
            "n_hits": n_hits,
            "activation_pct": round(activation_pct, 2),
            "wr_observed": round(wr_obs, 4),
            "wr_bootstrap_mean": round(wr_boot, 4),
            "wr_lower_95ci": round(wr_lower, 4),
            "wr_upper_95ci": round(wr_upper, 4),
            "p_value": round(p_value, 6),
            "status": status,
            "verdict": verdict
        })

    return results


def write_report(results: List[Dict]) -> None:
    """Guarda el reporte de validación en markdown."""
    valid_rules = [r for r in results if r["verdict"] == "VALID"]
    weak_rules  = [r for r in results if r["verdict"] == "WEAK"]
    missing     = [r for r in results if r["verdict"] == "NO_DATA"]

    lines = [
        "# Rule Validation Report — Luna V1 Pass-Through Features",
        f"**Generado:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Total reglas:** {len(results)} | "
        f"**Válidas (p<0.05):** {len(valid_rules)} | "
        f"**Débiles:** {len(weak_rules)} | "
        f"**No encontradas:** {len(missing)}",
        "",
        "> Las reglas VALID tienen WR estadísticamente > 50% (test binomial p<0.05).",
        "> Todas las reglas, válidas o débiles, pasan a XGBoost — XGBoost decide internamente.",
        "> Las reglas VALID se priorizan como candidatas para el próximo Mining run.",
        "",
        "## Tabla de Resultados",
        "",
        "| Feature | N Hits | Activación | WR Obs. | IC 95% | p-value | Veredicto |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda x: -x["wr_observed"]):
        ic = f"{r['wr_lower_95ci']:.1%} – {r['wr_upper_95ci']:.1%}" if r["n_hits"] > 10 else "N/A"
        lines.append(
            f"| `{r['feature']}` | {r['n_hits']} | {r['activation_pct']:.1f}% | "
            f"{r['wr_observed']:.1%} | {ic} | {r['p_value']:.4f} | {r['status']} |"
        )

    lines += [
        "",
        "## Reglas Válidas para XGBoost",
        "",
    ]
    for r in valid_rules:
        lines.append(
            f"- **`{r['feature']}`** — WR {r['wr_observed']:.1%} "
            f"(IC95: {r['wr_lower_95ci']:.1%}–{r['wr_upper_95ci']:.1%}) "
            f"| N={r['n_hits']} | p={r['p_value']:.4f}"
        )

    if weak_rules:
        lines += ["", "## Reglas Débiles (WR no significativo)", ""]
        for r in weak_rules:
            wr_str = f"{r['wr_observed']:.1%}" if r["n_hits"] > 10 else "N/A"
            lines.append(f"- `{r['feature']}` — WR {wr_str} | N={r['n_hits']} | p={r['p_value']:.4f}")

    lines += [
        "",
        "---",
        "*Generado por Luna V1 rule_validator.py*"
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"✅ Reporte guardado: {REPORT_FILE}")

    # Actualizar selected_features.json con la lista de reglas válidas
    if SELECTED_FILE.exists():
        with open(SELECTED_FILE) as f:
            sel_data = json.load(f)
        sel_data["pass_through_validation"] = {
            r["feature"]: {
                "verdict": r["verdict"],
                "n_hits": r["n_hits"],
                "wr_observed": r["wr_observed"],
                "p_value": r["p_value"],
            }
            for r in results
        }
        with open(SELECTED_FILE, "w") as f:
            json.dump(sel_data, f, indent=2, default=str)
        logger.success("✅ selected_features.json actualizado con validación Bootstrap")


def main():
    parser = argparse.ArgumentParser(description="Valida pass-through features con Bootstrap WR")
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(DATA_DIR / "features_train.parquet"),
        help="Ruta al parquet de features_train"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="Nombre de la columna target"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Rule Validator — Bootstrap WR Validation")
    logger.info("=" * 60)
    logger.info(f"Parquet: {args.parquet}")
    logger.info(f"Pass-Through Features: {len(PASSTHROUGH_FEATURES)}")

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        logger.error(f"No encontrado: {parquet_path}. Ejecuta --only-features primero.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} cols")

    results = validate_rules(df, target_col=args.target)
    write_report(results)

    valid_count = sum(1 for r in results if r["verdict"] == "VALID")
    logger.info(f"\n{'='*60}")
    logger.info(f"Resumen: {valid_count}/{len(results)} reglas con señal estadística válida (p<0.05)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

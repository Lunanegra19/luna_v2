"""
guard_pipeline.py
=================
Luna V1 — Capa de Seguridad Anti-Leakage y Anti-Overfitting
Inspirado en mamba_guard.py de Luna v2 y las Reglas de Hierro del SOP Luna v2.

Detecta automáticamente:
  - columnas con información futura en el parquet (R1 — Causalidad Estricta)
  - correlaciones sospechosamente altas con el target (>0.5 en features)
  - patrones de look-ahead en código Python (.shift(-N), center=True, scaler.fit(X_all))
  - métricas estadísticas incoherentes (MeanSR < 0 pero DSR > 0)
  - features sin señal estadística real (Sharpe < umbral con costos incluidos)

Uso:
    python core/security/guard_pipeline.py data/features/features_train.parquet
    from luna.security.guard_pipeline import GuardPipeline
    GuardPipeline.run_full_audit("data/features/features_train.parquet", "target")
"""
from __future__ import annotations

import re
import ast
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).parent.parent.parent

# ── Columnas que NUNCA deben estar en el feature space (R1) ─────────────────
# NOTA: 'target' y 'close' son columnas lícitas en el parquet (son la variable
#        dependiente y el precio). Lo que se prohíbe es que 'future_ret_24h'
#        (retorno futuro crudo) aparezca como feature adicional.
LEAKAGE_COLS = [
    # Retornos futuros — derivados del target (PROHIBIDOS como features)
    "future_ret_24h", "future_ret_4h", "future_ret_1h", "future_ret_12h",
    "future_ret_48h", "future_ret_72h",
    "fwd_return", "forward_return", "next_ret", "next_return",
    # Labels intermedias (no el target final)
    "barrier_touch", "barrier_label", "tb_label",
    # Variantes de retorno futuro
    "ret_future", "future_close", "future_price",
]

# Prefijos/sufijos que indican look-ahead
LEAKAGE_PATTERNS = [
    r"future_",
    r"_future",
    r"forward_",
    r"fwd_",
    r"next_",
    r"_next",
    r"lead_",
    r"_lead",
]

# Umbral de correlación para alerta (feature vs target)
CORR_THRESHOLD_ALERT = 0.40    # WARNING
CORR_THRESHOLD_CRITICAL = 0.55  # ERROR — casi certeza de leakage

# ── Fix C-02: Whitelist de features que CONTIENEN 'lead'/'next' en su nombre ─
# pero son causales legítimos (lag hacia el pasado, no al futuro).
# La documentación Granger (Manual 3.0) valida ETH_Lead_1H como predictivo causal.
# Sin esta whitelist, el guard_pipeline purga incorrectamente features de alta calidad.
LEAKAGE_WHITELIST = [
    "ETH_Lead_1H",       # ETH close desplazado -1H (ETH lidera BTC Granger-causalmente)
    "ETH_Lead_4H",       # Variante 4H del mismo principio
    "eth_lead_1h",       # Alias lowercase
    "eth_lead_4h",
]


class GuardPipeline:
    """
    Auditor de seguridad del pipeline de Luna V1.
    Ejecuta todos los checks de las Reglas de Hierro aplicables al feature pipeline.
    """

    # ── 1. Leakage por nombre de columna ─────────────────────────────────────

    @staticmethod
    def check_leakage_col_names(df: pd.DataFrame) -> list[str]:
        """
        R1: Detecta columnas con nombres que indican información futura.
        Comprueba contra LEAKAGE_COLS exactos y contra LEAKAGE_PATTERNS regex.

        Returns:
            Lista de columnas problemáticas detectadas.
        """
        found = []
        for col in df.columns:
            # Fix C-02: excluir features de la whitelist antes de aplicar patrones
            # Modificado para soportar lags dinamicos (_milagXXX) basados en features en la whitelist
            is_whitelisted = False
            for w in LEAKAGE_WHITELIST:
                if col == w or col.startswith(w + "_"):
                    is_whitelisted = True
                    break
                    
            if is_whitelisted:
                continue
            
            # Match exacto
            if col in LEAKAGE_COLS:
                found.append(col)
                continue
            # Match por patrón regex
            for pattern in LEAKAGE_PATTERNS:
                if re.search(pattern, col.lower()):
                    found.append(col)
                    break
        return found

    # ── 2. Correlación sospechosa con target ─────────────────────────────────

    @staticmethod
    def check_high_target_correlation(
        df: pd.DataFrame,
        target_col: str = "target",
        alert_threshold: float = CORR_THRESHOLD_ALERT,
        critical_threshold: float = CORR_THRESHOLD_CRITICAL,
        exclude_cols: list = None,
    ) -> dict:
        """
        R1: Detecta features con correlación sospechosamente alta con el target.
        Correlación >0.40 en feature → WARNING.
        Correlación >0.55 en feature → CRITICAL (casi certeza de leakage).

        Note: El target binario (0/1) tiene correlación máxima teórica ~0.70-0.85
              con su propio retorno subyacente. Una feature legítima raramente
              supera 0.15. Cualquier cosa >0.30 merece investigación.
              'close' tampoco debe auditarse ya que es la serie base.

        Returns:
            Dict {col: corr} de columnas problemáticas.
        """
        if target_col not in df.columns:
            logger.warning(f"Columna target '{target_col}' no encontrada")
            return {}

        # Excluir columnas de infraestructura del parquet (no son features)
        default_exclude = {target_col, "close", "open", "high", "low", "volume"}
        if exclude_cols:
            default_exclude.update(exclude_cols)

        y = df[target_col]
        suspicious = {}

        for col in df.columns:
            if col in default_exclude:
                continue
            try:
                corr = abs(df[col].corr(y))
                if corr >= alert_threshold:
                    suspicious[col] = float(corr)
            except Exception:
                continue

        return dict(sorted(suspicious.items(), key=lambda x: x[1], reverse=True))

    # ── 3. Look-ahead en código Python ───────────────────────────────────────

    @staticmethod
    def check_lookahead_code(filepath: str | Path) -> list[str]:
        """
        R1: Escanea un archivo Python buscando patrones de look-ahead.
        Detecta:
          - .shift(-N)  → desplazamiento negativo = mirar al futuro
          - rolling(center=True) → ventana centrada = mira al futuro
          - scaler.fit(X) donde X no es X_train → leakage de scaling
          - KFold( sin PurgedKFold → validación incorrecta en series temporales

        Returns:
            Lista de violaciones encontradas (descripción + línea).
        """
        violations = []
        try:
            code = Path(filepath).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return [f"ERROR: No se pudo leer {filepath}: {e}"]

        lines = code.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Shift negativo (look-ahead)
            if re.search(r"\.shift\(\s*-\d+", line) and "target" not in line.lower():
                violations.append(
                    f"CRITICAL L{i}: .shift(-N) detectado — look-ahead bias. "
                    f"Línea: {stripped[:80]}"
                )

            # Rolling centrado
            if "center=True" in line.replace(" ", ""):
                violations.append(
                    f"CRITICAL L{i}: rolling(center=True) — usa datos futuros. "
                    f"Línea: {stripped[:80]}"
                )

            # Scaler fit en dataset completo
            if re.search(r"scaler\.fit\s*\(\s*(X|df|data|features)\s*\)", line):
                violations.append(
                    f"CRITICAL L{i}: scaler.fit() posiblemente en dataset completo. "
                    f"Usar scaler.fit(X_train). Línea: {stripped[:80]}"
                )

            # KFold sin purge
            if "KFold(" in line and "Purged" not in line and "TimeSeriesSplit" not in line:
                violations.append(
                    f"WARNING L{i}: KFold estándar detectado. "
                    f"Usar PurgedKFold/CPCV para series temporales. Línea: {stripped[:80]}"
                )

        return violations

    # ── 4. Incoherencia DSR / MeanSR ─────────────────────────────────────────

    @staticmethod
    def check_dsr_sanity(
        mean_sr: float,
        dsr: float,
        feature_name: str,
    ) -> Optional[str]:
        """
        R5: Detecta incoherencia estadística: DSR>0 con MeanSR<0.
        Esta combinación indica que la implementación del DSR es incorrecta
        o que la feature es estadísticamente inconsistente.

        DSR correcto (Bailey & LdP 2014): DSR = Φ[(SR - SR*) / σ_SR]
        Si MeanSR (= SR) < 0, entonces DSR debería ser << 0.5 siempre.

        Returns:
            Mensaje de warning si hay incoherencia, None si es consistente.
        """
        if mean_sr < 0 and dsr > 0.5:
            return (
                f"INCOHERENCIA DSR: '{feature_name}' tiene MeanSR={mean_sr:.3f} (<0) "
                f"pero DSR={dsr:.3f} (>0.5). La feature no tiene señal positiva real. "
                f"Verificar implementación DSR (usar fórmula Bailey & LdP 2014)."
            )
        if mean_sr < -0.5 and dsr > 0.0:
            return (
                f"WARNING DSR: '{feature_name}' tiene MeanSR={mean_sr:.3f} muy negativo "
                f"pero DSR={dsr:.3f}. Revisar implementación."
            )
        return None

    # ── 5. Sanity check métricas generales ───────────────────────────────────

    @staticmethod
    def sanity_check_metrics(metrics: dict) -> list[str]:
        """
        Detecta métricas sospechosas según las Red Flags del SOP Luna v2:

        Red Flags:
         - Sharpe > 4.0 → look-ahead casi seguro
         - WinRate > 70% → irrealismo / costos no aplicados
         - MaxDrawdown = 0.0 → imposible en trading real
         - Accuracy > 60% en 1H → muy sospechoso
        """
        alerts = []

        sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0))
        if sharpe > 4.0:
            alerts.append(
                f"CRITICAL: Sharpe={sharpe:.2f} > 4.0. "
                "Alta probabilidad de LOOK-AHEAD BIAS (SOP Red Flag 1)."
            )

        wr = metrics.get("win_rate", metrics.get("accuracy", 0))
        if wr > 0.70:
            alerts.append(
                f"WARNING: Win Rate/Accuracy={wr:.1%} > 70%. "
                "Improbable en datos horarios. Verificar costos ≥0.15% round-trip."
            )

        mdd = metrics.get("max_drawdown", metrics.get("maxdd", None))
        if mdd is not None and abs(mdd) < 0.001:
            alerts.append(
                f"CRITICAL: MaxDD={mdd:.4f} ≈ 0. "
                "Imposible en trading real. Posible leakage o backtest incorrecto."
            )

        return alerts

    # ── 6. Audit completo del parquet ─────────────────────────────────────────

    @classmethod
    def run_full_audit(
        cls,
        parquet_path: str | Path,
        target_col: str = "target",
        verbose: bool = True,
    ) -> dict:
        """
        Ejecuta todos los checks sobre un parquet de features.
        Genera reporte completo con pass/fail por regla.

        Args:
            parquet_path: Ruta al archivo .parquet
            target_col: Nombre de la columna target
            verbose: Imprimir resumen en consola

        Returns:
            Dict con resultados de cada check.
        """
        path = Path(parquet_path)
        if not path.exists():
            logger.error(f"Parquet no encontrado: {path}")
            return {"error": f"No encontrado: {path}"}

        logger.info(f"[Guard] Auditando: {path.name}")
        df = pd.read_parquet(path)
        logger.info(f"[Guard] Shape: {df.shape}")

        results = {
            "parquet": str(path),
            "shape": list(df.shape),
            "checks": {},
        }

        # Check 1: Nombres de columnas con leakage
        leakage_names = cls.check_leakage_col_names(df)
        results["checks"]["leakage_col_names"] = {
            "status": "FAIL" if leakage_names else "PASS",
            "violations": leakage_names,
            "n_violations": len(leakage_names),
        }
        if leakage_names:
            logger.error(f"[Guard] R1 FAIL: {len(leakage_names)} columnas de leakage: {leakage_names}")
        else:
            logger.success("[Guard] R1 PASS: Sin columnas de look-ahead por nombre")

        # Check 2: Correlación alta con target
        suspicious_corr = cls.check_high_target_correlation(df, target_col)
        critical_corr = {k: v for k, v in suspicious_corr.items() if v >= CORR_THRESHOLD_CRITICAL}
        results["checks"]["target_correlation"] = {
            "status": "FAIL" if critical_corr else ("WARNING" if suspicious_corr else "PASS"),
            "suspicious": suspicious_corr,
            "critical": critical_corr,
        }
        for col, corr in suspicious_corr.items():
            level = "CRITICAL" if corr >= CORR_THRESHOLD_CRITICAL else "WARNING"
            logger.warning(f"[Guard] {level}: {col} corr={corr:.4f} vs target")

        if not suspicious_corr:
            logger.success("[Guard] R1 PASS: Sin correlaciones sospechosas con target")

        # Check 3: Columnas LEAKAGE_COLS presentes
        explicit_leak = [c for c in df.columns if c in LEAKAGE_COLS]
        results["checks"]["explicit_leakage"] = {
            "status": "FAIL" if explicit_leak else "PASS",
            "found": explicit_leak,
        }
        if explicit_leak:
            logger.error(f"[Guard] R1 CRITICAL: Columnas explícitas de leakage en parquet: {explicit_leak}")

        # Resumen
        n_fail = sum(
            1 for c in results["checks"].values()
            if c.get("status") in ["FAIL"]
        )
        n_warn = sum(
            1 for c in results["checks"].values()
            if c.get("status") == "WARNING"
        )
        results["summary"] = {
            "total_checks": len(results["checks"]),
            "failed": n_fail,
            "warnings": n_warn,
            "passed": len(results["checks"]) - n_fail - n_warn,
            "verdict": "CLEAN" if n_fail == 0 else "COMPROMISED",
        }

        if verbose:
            verdict = results["summary"]["verdict"]
            logger.info(
                f"[Guard] AUDIT COMPLETO: {n_fail} FAIL | {n_warn} WARNING | "
                f"Veredicto: {verdict}"
            )

        return results


# ── Función de purga para usar en feature_pipeline.py ─────────────────────────

def purge_leakage_columns(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """
    Elimina del DataFrame cualquier columna que represente look-ahead bias.
    Debe llamarse ANTES de split_and_save() en feature_pipeline.py.

    Elimina:
      1. Columnas en LEAKAGE_COLS explícitas (lista blanca negativa)
      2. Columnas con patrón de nombre futuro (future_, fwd_, next_, etc.)
      3. El target_col se mantiene (es necesario para el split)

    Returns:
        DataFrame sin columnas de leakage.
    """
    guard = GuardPipeline()
    to_remove = guard.check_leakage_col_names(df)

    # El target_col en sí NO se elimina (se necesita para el split del pipeline)
    to_remove = [c for c in to_remove if c != target_col]

    if to_remove:
        logger.warning(
            f"[Guard] PURGA: Eliminando {len(to_remove)} columnas de leakage antes de guardar: "
            f"{to_remove}"
        )
        df = df.drop(columns=to_remove, errors="ignore")
    else:
        logger.success("[Guard] PURGA: Sin columnas de leakage detectadas — parquet limpio")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Guard Pipeline Luna V1 — Auditoría anti-leakage"
    )
    parser.add_argument(
        "parquet", nargs="?",
        default=str(_ROOT / "data" / "features" / "features_train.parquet"),
        help="Ruta al parquet de features (default: features_train.parquet)"
    )
    parser.add_argument("--target", default="target", help="Columna target")
    parser.add_argument(
        "--audit-code", type=str, default=None,
        help="Ruta a archivo Python para auditar look-ahead en código"
    )
    parser.add_argument("--output", type=str, default=None, help="Guardar reporte JSON")
    args = parser.parse_args()

    # Auditoría del parquet
    results = GuardPipeline.run_full_audit(args.parquet, args.target)

    # Auditoría de código si se especifica
    if args.audit_code:
        code_violations = GuardPipeline.check_lookahead_code(args.audit_code)
        results["code_audit"] = {
            "file": args.audit_code,
            "violations": code_violations,
            "status": "FAIL" if code_violations else "PASS",
        }
        if code_violations:
            print(f"\nViolaciones en {args.audit_code}:")
            for v in code_violations:
                print(f"  {v}")
        else:
            print(f"\nCódigo limpio: {args.audit_code}")

    # Output JSON
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReporte guardado: {args.output}")

    # Exit code 1 si hay fallos
    sys.exit(1 if results["summary"]["verdict"] == "COMPROMISED" else 0)

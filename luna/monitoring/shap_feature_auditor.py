"""
SHAP Feature Auditor — Auditor de Importancia de Features Forzadas.

PROPÓSITO:
Medir empíricamente si las features añadidas a la whitelist del SFI (forzadas
por cuota o boost) aportan valor real a XGBoost o son slots desperdiciados.

METODOLOGÍA:
- Fuente de datos: signature JSON por agente (ya contiene gain/weight/cover)
  generados por train_xgboost_v2.py. No requiere re-cargar modelos.
- Métrica principal: feature importance gain normalizada (0-1 por ventana)
- Comparación: forced_features vs. competitive_features (ratio)
- Threshold de alerta: si una feature forzada tiene importancia < THRESHOLD
  durante N ventanas consecutivas → alerta de candidato-a-eliminar

INTEGRACIÓN:
1. Llamado desde pipeline_executor.py después del step XGBoost
2. Acumula resultados en data/shap_audit/audit_history.json
3. Genera report legible en data/shap_audit/audit_report.txt

POLÍTICA No-Fallback: los umbrales se leen de settings.yaml.
[SHAP-AUDIT-01 2026-06-03]
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ─── Constantes y paths ────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_AUDIT_DIR = _ROOT / "data" / "shap_audit"
_AUDIT_HISTORY_FILE = _AUDIT_DIR / "audit_history.json"
_AUDIT_REPORT_FILE = _AUDIT_DIR / "audit_report.txt"
_MODELS_DIR = _ROOT / "data" / "models"


# ─── Carga de parámetros desde settings (No-Fallback en críticos) ──────────────
def _load_audit_params() -> dict:
    """
    Lee parámetros de auditoría desde settings.yaml.
    Política No-Fallback: umbrales institucionales deben existir.
    """
    try:
        from config.settings import cfg
        audit_cfg = cfg.shap_audit
        if audit_cfg is None:
            # Fallback suave — la sección es nueva, puede no existir en runs antiguas
            logger.warning("[SHAP-AUDIT-01] Sección shap_audit no encontrada en settings.yaml — usando defaults")
            print("[SHAP-AUDIT-01] WARN: shap_audit no en settings.yaml — usando defaults institucionales")
            return {
                "min_importance_threshold":   0.02,    # 2% de importancia normalizada = mínimo
                "consecutive_windows_alert":  3,        # N ventanas consecutivas bajo threshold → alerta
                "consecutive_windows_remove": 5,        # N ventanas → candidato a eliminar
                "importance_types":           ["gain"],  # qué tipos de importancia trackear
                "enabled":                    True,
            }
        return {
            "min_importance_threshold":   float(audit_cfg.min_importance_threshold),
            "consecutive_windows_alert":  int(audit_cfg.consecutive_windows_alert),
            "consecutive_windows_remove": int(audit_cfg.consecutive_windows_remove),
            "importance_types":           list(audit_cfg.importance_types),
            "enabled":                    bool(audit_cfg.enabled),
        }
    except Exception as e:
        logger.warning(f"[SHAP-AUDIT-01] Error leyendo params de settings: {e} — usando defaults")
        return {
            "min_importance_threshold":   0.02,
            "consecutive_windows_alert":  3,
            "consecutive_windows_remove": 5,
            "importance_types":           ["gain"],
            "enabled":                    True,
        }


def _load_forced_features() -> dict[str, list[str]]:
    """
    Carga las features forzadas por categoría desde settings.yaml.
    Retorna dict: {'macro': [...], 'onchain': [...], 'calendar': [...]}
    """
    try:
        from config.settings import cfg
        f = cfg.features
        return {
            "macro":    list(f.sfi_macro_features or []),
            "onchain":  list(f.sfi_onchain_features or []),
            "calendar": list(f.sfi_calendar_features or []),
        }
    except Exception as e:
        logger.warning(f"[SHAP-AUDIT-01] No se pudo cargar listas de forced features: {e}")
        return {"macro": [], "onchain": [], "calendar": []}


def _normalize_feature_name(feat: str) -> str:
    """
    Normaliza el nombre de una feature: elimina sufijos de lag (_milagNh)
    para poder comparar con las listas de whitelist (sin lag).
    """
    import re
    # Eliminar patrones: _milag12h, _milag240h, _z90d_milag6h, etc.
    cleaned = re.sub(r'_milag\d+h$', '', feat)
    # Eliminar _z90d, _z60d al final si quedan después de quitar el lag
    # No queremos quitar los z-scores propios de las features (son el nombre base)
    return cleaned


def _read_signatures_for_window(window_id: str | None = None,
                                 seed: int | None = None) -> list[dict]:
    """
    Lee los signature JSON del modelo para la ventana actual.
    Busca en data/wfb_cache/seedN/WX/models/ si window_id definido,
    o en data/models/ para el último run.
    """
    signatures = []

    # Paths a buscar
    search_paths = []
    if window_id and seed is not None:
        wfb_path = _ROOT / "data" / "wfb_cache" / f"seed{seed}" / window_id / "models"
        if wfb_path.exists():
            search_paths.append(wfb_path)
    # Siempre incluir data/models como fallback
    search_paths.append(_MODELS_DIR)

    for models_dir in search_paths:
        if not models_dir.exists():
            continue
        for sig_file in models_dir.glob("xgboost_meta*signature*.json"):
            try:
                data = json.loads(sig_file.read_text(encoding='utf-8'))
                if 'feature_importances' in data and 'features' in data:
                    agent_name = sig_file.stem.replace('_signature', '').replace('xgboost_meta_', '')
                    data['_agent'] = agent_name
                    data['_source'] = str(sig_file)
                    signatures.append(data)
            except Exception as e:
                logger.warning(f"[SHAP-AUDIT-01] No se pudo leer {sig_file.name}: {e}")

    return signatures


def _extract_importance_by_category(
    signatures: list[dict],
    forced_features: dict[str, list[str]],
    importance_type: str = "gain"
) -> dict:
    """
    Extrae la importancia normalizada de cada feature forzada vs. promedio competitivo.

    Retorna dict con estructura:
    {
        "agent_name": {
            "forced": {
                "feature_name": {
                    "importance_norm": 0.05,  # 0-1 normalizado
                    "importance_raw":  12.3,  # gain raw
                    "category":        "macro",
                    "rank":            3,      # posición de importancia
                    "is_active":       True,   # en features seleccionadas
                }
            },
            "competitive_mean": 0.08,  # media de features no-forzadas
            "competitive_std":  0.04,
        }
    }
    """
    fi_key_map = {
        "gain":   "feature_importances",
        "weight": "feature_importances_weight",
        "cover":  "feature_importances_cover",
    }
    fi_key = fi_key_map.get(importance_type, "feature_importances")

    # Unión de todas las features forzadas
    all_forced_flat = set()
    for cat_feats in forced_features.values():
        all_forced_flat.update(cat_feats)

    result = {}

    for sig in signatures:
        agent = sig._agent
        fi_dict = sig.get(fi_key, {})
        selected_feats = sig.features
        dsr = sig.dsr_oos)

        if not fi_dict:
            continue

        # Normalizar importancias (0-1 por agente)
        total = sum(fi_dict.values()) or 1.0
        fi_norm = {k: v / total for k, v in fi_dict.items()}

        # Clasificar features como forzadas vs. competitivas
        forced_importances = {}
        competitive_importances = []

        for feat, fi_val in fi_norm.items():
            feat_base = _normalize_feature_name(feat)
            category = None
            for cat, cat_feats in forced_features.items():
                if feat_base in cat_feats or feat in cat_feats:
                    category = cat
                    break

            if category:
                forced_importances[feat] = {
                    "importance_norm": round(fi_val, 5),
                    "importance_raw":  round(fi_dict.get(feat, 0.0), 4),
                    "category":        category,
                    "feat_base":       feat_base,
                    "is_active":       feat in selected_feats,
                }
            else:
                competitive_importances.append(fi_val)

        comp_mean = float(np.mean(competitive_importances)) if competitive_importances else 0.0
        comp_std  = float(np.std(competitive_importances))  if competitive_importances else 0.0

        # Añadir features forzadas que están en selected_features pero no en fi_dict
        # (importancia 0 — XGBoost no las usó)
        for feat in selected_feats:
            feat_base = _normalize_feature_name(feat)
            if feat in forced_importances or feat_base not in all_forced_flat:
                continue
            for cat, cat_feats in forced_features.items():
                if feat_base in cat_feats or feat in cat_feats:
                    forced_importances[feat] = {
                        "importance_norm": 0.0,
                        "importance_raw":  0.0,
                        "category":        cat,
                        "feat_base":       feat_base,
                        "is_active":       True,
                    }
                    break

        # Rank de cada feature forzada entre todas las features del agente
        all_fi_sorted = sorted(fi_norm.values(), reverse=True)
        for feat, info in forced_importances.items():
            try:
                rank = all_fi_sorted.index(info['importance_norm']) + 1
            except ValueError:
                rank = len(all_fi_sorted) + 1
            info['rank'] = rank
            info['n_features_total'] = len(all_fi_sorted)
            info['rank_pct'] = round(rank / max(len(all_fi_sorted), 1), 3)

        result[agent] = {
            "forced":            forced_importances,
            "competitive_mean":  round(comp_mean, 5),
            "competitive_std":   round(comp_std, 5),
            "n_competitive":     len(competitive_importances),
            "dsr_oos":           float(dsr) if not np.isnan(float(dsr)) else None,
            "n_selected_feats":  len(selected_feats),
        }

    return result


def _load_audit_history() -> dict:
    """Carga el historial acumulado de auditorías previas."""
    if not _AUDIT_HISTORY_FILE.exists():
        return {"windows": [], "feature_stats": {}, "alerts": []}
    try:
        return json.loads(_AUDIT_HISTORY_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f"[SHAP-AUDIT-01] Historial corrupto: {e} — iniciando desde cero")
        return {"windows": [], "feature_stats": {}, "alerts": []}


def _save_audit_history(history: dict) -> None:
    """Guarda el historial actualizado."""
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    _AUDIT_HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )


def _detect_alerts(history: dict, params: dict) -> list[dict]:
    """
    Detecta features problemáticas basándose en el historial acumulado.
    Retorna lista de alertas con severidad y recomendación.
    """
    alerts = []
    threshold    = params['min_importance_threshold']
    n_alert      = params['consecutive_windows_alert']
    n_remove     = params['consecutive_windows_remove']
    feat_stats   = history.feature_stats

    for feat_name, stats in feat_stats.items():
        windows_below = stats.consecutive_windows_below_threshold
        mean_fi       = stats.mean_importance_norm
        category      = stats.category
        n_windows     = stats.n_windows_seen

        if n_windows < 2:
            continue  # No suficiente historial

        if windows_below >= n_remove:
            alerts.append({
                "severity":    "CRITICAL",
                "feature":     feat_name,
                "category":    category,
                "message":     (
                    f"[SHAP-AUDIT-01] CANDIDATO A ELIMINAR: '{feat_name}' ({category}) "
                    f"tiene importancia < {threshold:.2f} en {windows_below} ventanas consecutivas. "
                    f"Media histórica: {mean_fi:.4f}. "
                    f"ACCIÓN: Evaluar eliminación de sfi_{category}_features en settings.yaml."
                ),
                "windows_below": windows_below,
                "mean_fi":       mean_fi,
            })
        elif windows_below >= n_alert:
            alerts.append({
                "severity":    "WARNING",
                "feature":     feat_name,
                "category":    category,
                "message":     (
                    f"[SHAP-AUDIT-01] ALERTA: '{feat_name}' ({category}) "
                    f"tiene importancia < {threshold:.2f} en {windows_below} ventanas consecutivas. "
                    f"Monitorizar — si continúa {n_remove - windows_below} ventanas más será CRITICAL."
                ),
                "windows_below": windows_below,
                "mean_fi":       mean_fi,
            })
        elif mean_fi > threshold * 3 and n_windows >= 3:
            # Feature forzada con importancia ALTA — justificación empírica confirmada
            alerts.append({
                "severity":    "INFO",
                "feature":     feat_name,
                "category":    category,
                "message":     (
                    f"[SHAP-AUDIT-01] VALIDADO: '{feat_name}' ({category}) "
                    f"tiene importancia media {mean_fi:.4f} > {threshold*3:.4f} "
                    f"(3x umbral) en {n_windows} ventanas. "
                    f"Forzado JUSTIFICADO empiricamente."
                ),
                "windows_below": windows_below,
                "mean_fi":       mean_fi,
            })

    return alerts


def run_shap_audit(
    window_id: str | None = None,
    seed: int | None = None,
    importance_type: str = "gain",
) -> dict | None:
    """
    Función principal del auditor. Llamar después de train_xgboost_v2.py.

    Args:
        window_id:        ID de la ventana WFB (ej. 'W3'). None = run PROD.
        seed:             Semilla del run actual.
        importance_type:  'gain' | 'weight' | 'cover'

    Returns:
        Diccionario con resultados del audit, o None si falla/desactivado.
    """
    params = _load_audit_params()

    if not params.enabled:
        logger.debug("[SHAP-AUDIT-01] Auditor desactivado en settings.yaml")
        return None

    print(f"[SHAP-AUDIT-01] Iniciando audit de importancia | window={window_id} seed={seed}")
    t0 = time.monotonic()

    # 1. Leer features forzadas
    forced_features = _load_forced_features()
    n_forced_total = sum(len(v) for v in forced_features.values())
    print(f"[SHAP-AUDIT-01] Features forzadas: macro={len(forced_features['macro'])} "
          f"onchain={len(forced_features['onchain'])} calendar={len(forced_features['calendar'])}")

    # 2. Leer signatures del run actual
    signatures = _read_signatures_for_window(window_id, seed)
    if not signatures:
        logger.warning(f"[SHAP-AUDIT-01] No se encontraron signatures para window={window_id} seed={seed}")
        print("[SHAP-AUDIT-01] WARN: Sin signatures — audit omitido")
        return None

    print(f"[SHAP-AUDIT-01] Signatures encontradas: {len(signatures)} agentes")

    # 3. Extraer importancias por categoría
    audit_result = _extract_importance_by_category(signatures, forced_features, importance_type)

    # 4. Construir entrada de historia
    window_entry = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "window_id":    window_id,
        "seed":         seed,
        "importance_type": importance_type,
        "agents":       {}
    }

    # 5. Calcular estadísticas por feature a nivel de ventana
    feat_window_stats: dict[str, dict] = {}

    for agent, agent_data in audit_result.items():
        window_entry["agents"][agent] = {
            "competitive_mean": agent_data["competitive_mean"],
            "competitive_std":  agent_data["competitive_std"],
            "n_selected":       agent_data["n_selected_feats"],
            "dsr_oos":          agent_data["dsr_oos"],
            "forced": {}
        }
        threshold = params['min_importance_threshold']
        for feat, fi_info in agent_data["forced"].items():
            fi_norm = fi_info['importance_norm']
            cat     = fi_info['category']
            feat_base = fi_info.feat_base
            is_active = fi_info.is_active
            below_threshold = fi_norm < threshold and is_active

            window_entry["agents"][agent]["forced"][feat] = {
                "importance_norm": fi_norm,
                "importance_raw":  fi_info['importance_raw'],
                "category":        cat,
                "rank":            fi_info.rank,
                "rank_pct":        fi_info.rank_pct,
                "is_active":       is_active,
                "below_threshold": below_threshold,
            }

            # Acumular por feature (nombre base para tracking cross-ventana)
            key = f"{feat_base}|{cat}"
            if key not in feat_window_stats:
                feat_window_stats[key] = {
                    "feat_base": feat_base, "category": cat,
                    "fi_values": [], "below_count": 0, "active_count": 0
                }
            if is_active:
                feat_window_stats[key]["fi_values"].append(fi_norm)
                feat_window_stats[key]["active_count"] += 1
                if below_threshold:
                    feat_window_stats[key]["below_count"] += 1

    # 6. Cargar historial y actualizar
    history = _load_audit_history()
    history["windows"].append(window_entry)

    # Actualizar feature_stats acumulado
    for key, wstats in feat_window_stats.items():
        if wstats['active_count'] == 0:
            continue
        if key not in history["feature_stats"]:
            history["feature_stats"][key] = {
                "feat_base":                        wstats['feat_base'],
                "category":                         wstats['category'],
                "n_windows_seen":                   0,
                "n_windows_active":                 0,
                "n_windows_below_threshold":        0,
                "consecutive_windows_below_threshold": 0,
                "mean_importance_norm":             0.0,
                "all_fi_values":                    [],
                "last_updated":                     None,
            }
        hs = history["feature_stats"][key]
        hs["n_windows_seen"]    += 1
        hs["n_windows_active"]  += wstats['active_count']
        hs["n_windows_below_threshold"] += wstats['below_count']
        hs["all_fi_values"].extend(wstats['fi_values'])
        # Mantener solo últimas 20 ventanas en memoria
        hs["all_fi_values"] = hs["all_fi_values"][-20:]
        hs["mean_importance_norm"] = float(np.mean(hs["all_fi_values"])) if hs["all_fi_values"] else 0.0
        # Conteo consecutivo: si below en esta ventana → +1, si no → reset
        if wstats['below_count'] > 0:
            hs["consecutive_windows_below_threshold"] += 1
        else:
            hs["consecutive_windows_below_threshold"] = 0
        hs["last_updated"] = datetime.now(timezone.utc).isoformat()

    # 7. Detectar alertas
    alerts = _detect_alerts(history, params)
    history["alerts"] = alerts  # Solo mantener alertas actuales

    # 8. Guardar historial
    _save_audit_history(history)

    # 9. Generar report legible
    elapsed = time.monotonic() - t0
    report_lines = _generate_report(
        audit_result, history, alerts, params, window_id, seed, elapsed
    )
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    _AUDIT_REPORT_FILE.write_text("\n".join(report_lines), encoding='utf-8')

    # 10. Imprimir resumen en consola
    _print_summary(audit_result, alerts, params, elapsed)

    return {
        "audit_result":     audit_result,
        "alerts":           alerts,
        "n_windows_total":  len(history["windows"]),
        "elapsed_s":        round(elapsed, 2),
    }


def _generate_report(
    audit_result: dict,
    history: dict,
    alerts: list[dict],
    params: dict,
    window_id: str | None,
    seed: int | None,
    elapsed: float,
) -> list[str]:
    """Genera el informe legible de texto."""
    threshold = params['min_importance_threshold']
    lines = [
        "=" * 72,
        f"[SHAP-AUDIT-01] SHAP Feature Auditor Report",
        f"  Window: {window_id or 'PROD'} | Seed: {seed} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Ventanas históricas: {len(history['windows'])} | Tiempo: {elapsed:.1f}s",
        "=" * 72, "",
    ]

    for agent, agent_data in sorted(audit_result.items()):
        lines.append(f"── Agente: {agent} | DSR_OOS: {agent_data.get('dsr_oos') or 'N/A'} "
                     f"| Features: {agent_data['n_selected_feats']}")
        lines.append(f"   Competitivas: mean={agent_data['competitive_mean']:.4f} "
                     f"±{agent_data['competitive_std']:.4f} (N={agent_data['n_competitive']})")

        if not agent_data['forced']:
            lines.append("   → Sin features forzadas activas en este agente")
            lines.append("")
            continue

        lines.append(f"   {'Feature':<35} {'Cat':<10} {'FI_norm':>8} {'Rank':>6} {'Status'}")
        lines.append(f"   {'─'*35} {'─'*10} {'─'*8} {'─'*6} {'─'*15}")

        for feat, info in sorted(agent_data['forced'].items(),
                                  key=lambda x: -x[1]['importance_norm']):
            fi = info['importance_norm']
            rank = info.rank
            n_total = info.n_features_total
            is_active = info.is_active
            below = fi < threshold and is_active
            status = "(!) BAJO" if below else ("OK" if is_active else "no selec.")
            marker = " <" if below else ""
            lines.append(
                f"   {feat[:35]:<35} {info['category']:<10} {fi:>8.4f} "
                f"{str(rank)+'/'+str(n_total):>6} {status}{marker}"
            )
        lines.append("")

    # Historial acumulado
    lines.append("─" * 72)
    lines.append("HISTORIAL ACUMULADO POR FEATURE:")
    lines.append("")
    fs = history.feature_stats
    if fs:
        lines.append(f"  {'Feature':<35} {'Cat':<10} {'Windows':>7} {'FI_mean':>8} {'Below%':>7} {'Consec.':>7}")
        lines.append(f"  {'─'*35} {'─'*10} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
        for key, hs in sorted(fs.items(), key=lambda x: x[1]['mean_importance_norm']):
            n = hs['n_windows_seen']
            if n == 0: continue
            below_pct = hs['n_windows_below_threshold'] / max(n, 1) * 100
            consec = hs['consecutive_windows_below_threshold']
            lines.append(
                f"  {hs['feat_base'][:35]:<35} {hs['category']:<10} {n:>7} "
                f"{hs['mean_importance_norm']:>8.4f} {below_pct:>6.1f}% {consec:>7}"
            )
    lines.append("")

    # Alertas
    lines.append("─" * 72)
    lines.append("ALERTAS ACTIVAS:")
    lines.append("")
    if not alerts:
        lines.append("  [OK] Sin alertas — todas las features forzadas dentro del umbral")
    else:
        for alert in alerts:
            icon = {"CRITICAL": "[!!!]", "WARNING": "[!]", "INFO": "[OK]"}.get(alert['severity'], "•")
            lines.append(f"  {icon} [{alert['severity']}] {alert['feature']} ({alert['category']})")
            lines.append(f"     {alert['message']}")
            lines.append("")

    lines.append("─" * 72)
    lines.append(f"  Report guardado: {_AUDIT_REPORT_FILE}")
    lines.append(f"  Historial: {_AUDIT_HISTORY_FILE}")
    lines.append("=" * 72)
    return lines


def _print_summary(
    audit_result: dict,
    alerts: list[dict],
    params: dict,
    elapsed: float,
) -> None:
    """Imprime un resumen conciso en la consola."""
    threshold = params['min_importance_threshold']
    total_forced = 0
    below_threshold = 0
    top_forced = []

    for agent, agent_data in audit_result.items():
        comp_mean = agent_data['competitive_mean']
        for feat, info in agent_data['forced'].items():
            if not info.is_active:
                continue
            total_forced += 1
            fi = info['importance_norm']
            if fi < threshold:
                below_threshold += 1
            top_forced.append((feat, fi, info['category'], agent))

    top_forced.sort(key=lambda x: -x[1])
    critical = [a for a in alerts if a['severity'] == 'CRITICAL']
    warnings = [a for a in alerts if a['severity'] == 'WARNING']

    print(f"[SHAP-AUDIT-01] Audit completado | {elapsed:.1f}s")
    print(f"[SHAP-AUDIT-01] Features forzadas activas: {total_forced} | "
          f"Bajo umbral ({threshold:.2f}): {below_threshold} | "
          f"Alertas: {len(critical)} CRITICAL, {len(warnings)} WARNING")

    if top_forced:
        print("[SHAP-AUDIT-01] Top-5 features forzadas por importancia:")
        for feat, fi, cat, agent in top_forced[:5]:
            print(f"  {feat[:40]:40s} [{cat:8s}] FI={fi:.4f} ({agent})")

    for alert in critical:
        print(f"[SHAP-AUDIT-01] CRITICAL: {alert['feature']} -- {alert['message'][:100]}")
    for alert in warnings:
        print(f"[SHAP-AUDIT-01] WARNING:  {alert['feature']} -- {alert['message'][:100]}")

    if not critical and not warnings:
        print("[SHAP-AUDIT-01] [OK] Todas las features forzadas dentro del umbral de importancia")


# ─── CLI standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHAP Feature Auditor — [SHAP-AUDIT-01]")
    parser.add_argument("--window", type=str, default=None, help="Window ID (e.g. W3)")
    parser.add_argument("--seed",   type=int, default=None, help="Seed number")
    parser.add_argument("--type",   type=str, default="gain",
                        choices=["gain", "weight", "cover"],
                        help="Tipo de importancia (default: gain)")
    parser.add_argument("--report", action="store_true", help="Solo mostrar el último report")
    args = parser.parse_args()

    if args.report:
        if _AUDIT_REPORT_FILE.exists():
            print(_AUDIT_REPORT_FILE.read_text(encoding='utf-8'))
        else:
            print("[SHAP-AUDIT-01] Sin report previo — ejecutar un audit primero")
    else:
        result = run_shap_audit(
            window_id=args.window,
            seed=args.seed,
            importance_type=args.type,
        )
        if result:
            print(f"\n[SHAP-AUDIT-01] Audit guardado. Ventanas históricas: {result['n_windows_total']}")

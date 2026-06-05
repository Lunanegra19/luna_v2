"""
dashboard_healthcheck.py — Monitor automático de salud del dashboard Luna V2

Detecta errores silenciosos que no aparecen en logs del servidor:
  - JSON corrupto en responses (ej. bug doble-header de 2026-05-27)
  - Endpoints con HTTP 200 pero JSON inválido
  - Campos requeridos ausentes en la respuesta
  - DB sin datos recientes (ciclo no guardó en las últimas N horas)
  - Dashboard caído o en timeout

Ejecutado:
  A) Automáticamente después de cada ciclo live (post_cycle_enabled=true en settings.yaml)
  B) Como PM2 cron independiente cada 30 min (luna-dashboard-healthcheck)

Alerta via Telegram si cualquier check falla.
Cumple: RULE[fixaplly.md], RULE[fixbugsprints.md], RULE[settingsyfallvack.md]
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
LUNA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LUNA_ROOT))

# ── Load config (No-Fallback policy: KeyError si falta sección crítica) ───────
try:
    import yaml
    _cfg_path = LUNA_ROOT / "config" / "settings.yaml"
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _settings = yaml.safe_load(_f)
    if "dashboard_healthcheck" not in _settings:
        raise KeyError("[HEALTHCHECK/CRITICAL] Sección 'dashboard_healthcheck' ausente en settings.yaml. "
                       "Añadirla es obligatorio para operar (RULE[settingsyfallvack.md]).")
    HC_CFG = _settings["dashboard_healthcheck"]
    print(f"[HEALTHCHECK/BOOT] Config cargada desde {_cfg_path}. enabled={HC_CFG.get('enabled')}")
except Exception as e:
    print(f"[HEALTHCHECK/FATAL] Error cargando settings.yaml: {e}")
    raise

# ── Telegram alert ────────────────────────────────────────────────────────────
def _send_telegram_alert(message: str) -> bool:
    """Envía alert Telegram usando las env vars del sistema live."""
    if not HC_CFG.get("telegram_alert_on_failure", True):
        print("[HEALTHCHECK/TELEGRAM] Alertas Telegram desactivadas en config.")
        return False
    try:
        # [BUGFIX-HEALTHCHECK-01] Clase corregida: TelegramAlerter → TelegramAlerts
        # (el módulo luna.live.telegram_alerts exporta TelegramAlerts, no TelegramAlerter)
        print("[HEALTHCHECK/TELEGRAM] Importando TelegramAlerts...")
        from luna.live.telegram_alerts import TelegramAlerts
        tg = TelegramAlerts()
        tg.send_alert(message, priority="critical")
        print(f"[HEALTHCHECK/TELEGRAM] ✅ Alert enviado: {message[:80]}...")
        return True
    except Exception as e:
        # Fallback: requests directo si el módulo luna no está disponible
        print(f"[HEALTHCHECK/TELEGRAM-WARN] TelegramAlerter falló ({e}), intentando API directa...")
        try:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if not bot_token or not chat_id:
                print("[HEALTHCHECK/TELEGRAM-WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no definidos.")
                return False
            payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = json.loads(resp.read()).get("ok", False)
                print(f"[HEALTHCHECK/TELEGRAM] API directa: ok={ok}")
                return ok
        except Exception as e2:
            print(f"[HEALTHCHECK/TELEGRAM-ERROR] Fallo total en Telegram: {e2}")
            return False


# ── Core check function ───────────────────────────────────────────────────────
def check_endpoint(base_url: str, path: str, required_fields: list,
                   allow_standby: bool, timeout_sec: int,
                   extra_params: str = "") -> dict:
    """
    Hace una petición HTTP al endpoint y valida:
    1. HTTP 200 (no 401, no 500, no timeout)
    2. JSON válido (no corrupto)
    3. Campo 'status' presente
    4. Si status == 'success': campos required_fields presentes
    5. Si allow_standby: status == 'standby' es aceptable

    Returns dict con: ok, error, raw[:200], response_time_ms, status_code
    """
    url = f"{base_url}{path}"
    if extra_params:
        url = f"{url}?{extra_params}"

    print(f"[HEALTHCHECK/CHECK] → {path}{('?' + extra_params[:40]) if extra_params else ''}")
    t0 = time.time()
    raw = ""
    status_code = 0

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Luna-HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status_code = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
        response_time_ms = int((time.time() - t0) * 1000)
        print(f"[HEALTHCHECK/CHECK]   HTTP {status_code} en {response_time_ms}ms | len={len(raw)}")
    except urllib.error.HTTPError as e:
        response_time_ms = int((time.time() - t0) * 1000)
        print(f"[HEALTHCHECK/CHECK]   ❌ HTTP Error {e.code}: {e.reason}")
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}", "raw": "", "response_time_ms": response_time_ms, "status_code": e.code}
    except Exception as e:
        response_time_ms = int((time.time() - t0) * 1000)
        print(f"[HEALTHCHECK/CHECK]   ❌ Error de red: {e}")
        return {"ok": False, "error": f"NetworkError: {e}", "raw": "", "response_time_ms": response_time_ms, "status_code": 0}

    # Check 1: JSON parseable
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        error_msg = (f"JSON CORRUPTO: {e}. "
                     f"Raw inicio: '{raw[:120].replace(chr(10), ' ')}...'")
        print(f"[HEALTHCHECK/CHECK]   ❌ {error_msg}")
        return {"ok": False, "error": error_msg, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}

    # Check 2: Campo 'status' presente (opcional para endpoints legacy sin campo status)
    resp_status = data.get("status", "__MISSING__")
    if resp_status == "__MISSING__":
        # Verificar directamente si required_fields están presentes (endpoint legacy sin campo status)
        if required_fields and all(f in data for f in required_fields):
            print(f"[HEALTHCHECK/CHECK]   ✅ OK (endpoint legacy sin campo 'status' — required_fields presentes: {required_fields})")
            return {"ok": True, "error": None, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}
        error_msg = f"Campo 'status' ausente en respuesta. Keys: {list(data.keys())[:5]}"
        print(f"[HEALTHCHECK/CHECK]   ❌ {error_msg}")
        return {"ok": False, "error": error_msg, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}

    # Check 3: Unauthorized
    if resp_status == "unauthorized":
        error_msg = f"401 Unauthorized — bypass localhost no funcionó o endpoint fuera de /api/"
        print(f"[HEALTHCHECK/CHECK]   ❌ {error_msg}")
        return {"ok": False, "error": error_msg, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": 401}

    # Check 4: Standby es aceptable si allow_standby=True
    if resp_status == "standby":
        if allow_standby:
            print(f"[HEALTHCHECK/CHECK]   ✅ standby (esperado para este endpoint)")
            return {"ok": True, "error": None, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}
        else:
            error_msg = f"status='standby' no esperado en este endpoint (allow_standby=False)"
            print(f"[HEALTHCHECK/CHECK]   ❌ {error_msg}")
            return {"ok": False, "error": error_msg, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}

    # Check 5: Error explícito
    if resp_status == "error":
        error_msg = f"Endpoint reporta error: {data.get('message', 'sin detalle')[:100]}"
        print(f"[HEALTHCHECK/CHECK]   ❌ {error_msg}")
        return {"ok": False, "error": error_msg, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}

    # Check 6: Si success, verificar required_fields en data
    if resp_status == "success" and required_fields:
        data_body = data.get("data", data)  # algunos endpoints ponen datos en "data", otros en root
        missing = [f for f in required_fields if f not in data_body and f not in data]
        if missing:
            error_msg = f"Campos requeridos ausentes en response: {missing}"
            print(f"[HEALTHCHECK/CHECK]   ❌ {error_msg}")
            return {"ok": False, "error": error_msg, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}

    print(f"[HEALTHCHECK/CHECK]   ✅ OK (status={resp_status})")
    return {"ok": True, "error": None, "raw": raw[:200], "response_time_ms": response_time_ms, "status_code": status_code}


# ── DB freshness check ────────────────────────────────────────────────────────
def check_db_freshness(max_age_hours: int) -> dict:
    """
    Verifica que audit_logs tenga un registro reciente (< max_age_hours).
    Si el último ciclo tiene más de max_age_hours, probablemente el live está caído.
    """
    print(f"[HEALTHCHECK/DB] Verificando frescura de audit_logs (max_age={max_age_hours}h)...")
    try:
        from luna.database.db_manager import DatabaseManager
        db = DatabaseManager()
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(timestamp), COUNT(*) FROM audit_logs")
                row = cur.fetchone()
                if not row or row[0] is None:
                    error_msg = "audit_logs está VACÍA — ningún ciclo ha guardado datos"
                    print(f"[HEALTHCHECK/DB]   ❌ {error_msg}")
                    return {"ok": False, "error": error_msg}
                last_ts, total = row
                # Calcular age
                now_utc = datetime.now(timezone.utc)
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                age_hours = (now_utc - last_ts).total_seconds() / 3600
                print(f"[HEALTHCHECK/DB]   Último registro: {last_ts} UTC ({age_hours:.1f}h atrás) | Total: {total}")
                if age_hours > max_age_hours:
                    error_msg = (f"audit_logs sin datos recientes: último registro hace {age_hours:.1f}h "
                                 f"(límite={max_age_hours}h). ¿El live está caído?")
                    print(f"[HEALTHCHECK/DB]   ❌ {error_msg}")
                    return {"ok": False, "error": error_msg}
                print(f"[HEALTHCHECK/DB]   ✅ DB fresca ({age_hours:.1f}h < {max_age_hours}h límite)")
                return {"ok": True, "error": None, "last_ts": str(last_ts), "age_hours": age_hours}
    except Exception as e:
        error_msg = f"Error conectando a DB: {e}"
        print(f"[HEALTHCHECK/DB]   ❌ {error_msg}")
        return {"ok": False, "error": error_msg}


# ── Main check runner ─────────────────────────────────────────────────────────
def run_all_checks(triggered_by: str = "manual") -> dict:
    """
    Ejecuta todos los health checks configurados en settings.yaml.
    Returns: dict con failures (list), passed (int), total (int), duration_sec (float)
    """
    if not HC_CFG.get("enabled", True):
        print("[HEALTHCHECK] Dashboard healthcheck desactivado en settings.yaml.")
        return {"failures": [], "passed": 0, "total": 0, "duration_sec": 0}

    t_start = time.time()
    base_url = HC_CFG.get("base_url")
    if not base_url:
        raise RuntimeError("[HEALTHCHECK/CRITICAL] 'base_url' ausente en dashboard_healthcheck config.")
    timeout = HC_CFG.get("max_response_time_sec", 10)
    max_age_hours = HC_CFG.get("max_data_age_hours", 2)
    endpoint_cfgs = HC_CFG.get("endpoints", [])

    print(f"\n[HEALTHCHECK] ═══ Iniciando health check ({triggered_by}) ═══")
    print(f"[HEALTHCHECK] base_url={base_url} | endpoints={len(endpoint_cfgs)} | triggered_by={triggered_by}")

    failures = []
    passed = 0
    total = 0

    # ── 1. Verificar cada endpoint configurado ────────────────────────────────
    for ep_cfg in endpoint_cfgs:
        ep_path = ep_cfg["path"]
        ep_desc = ep_cfg.get("description", ep_path)
        required = ep_cfg.get("required_fields", ["status"])
        allow_standby = ep_cfg.get("allow_standby", True)
        total += 1

        # Para hour-decision añadimos la hora actual como parámetro
        extra_params = ""
        if "hour-decision" in ep_path:
            now = datetime.now(timezone.utc)
            local_now = datetime.now()
            start_utc = now.replace(minute=0, second=0, microsecond=0)
            end_utc = now.replace(minute=59, second=59, microsecond=999000)
            local_date = local_now.strftime("%Y-%m-%d")
            local_hour = local_now.hour
            extra_params = (
                f"start_utc={urllib.parse.quote(start_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z'))}"
                f"&end_utc={urllib.parse.quote(end_utc.strftime('%Y-%m-%dT%H:%M:%S.999Z'))}"
                f"&local_date={local_date}&local_hour={local_hour}"
            )

        result = check_endpoint(base_url, ep_path, required, allow_standby, timeout, extra_params)
        if result["ok"]:
            passed += 1
        else:
            failures.append({
                "endpoint": ep_path,
                "description": ep_desc,
                "error": result["error"],
                "raw": result.get("raw", ""),
                "response_time_ms": result.get("response_time_ms", 0),
                "status_code": result.get("status_code", 0),
            })

    # ── 2. Verificar frescura de DB ───────────────────────────────────────────
    total += 1
    db_result = check_db_freshness(max_age_hours)
    if db_result["ok"]:
        passed += 1
    else:
        failures.append({
            "endpoint": "postgresql/audit_logs",
            "description": f"Frescura de audit_logs (<{max_age_hours}h)",
            "error": db_result["error"],
            "raw": "",
            "response_time_ms": 0,
            "status_code": 0,
        })

    duration = time.time() - t_start
    print(f"[HEALTHCHECK] ═══ Resultado: {passed}/{total} OK | Fallos: {len(failures)} | {duration:.1f}s ═══")

    # ── 3. Enviar alert Telegram si hay fallos ────────────────────────────────
    if failures:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        fail_lines = []
        for f in failures:
            fail_lines.append(
                f"❌ *{f['description']}*\n"
                f"   `{f['endpoint']}`\n"
                f"   Error: `{f['error'][:120]}`"
            )
        alert_msg = (
            f"🚨 *LUNA V2 — Dashboard Health-Check FALLIDO* ({now_str})\n"
            f"Trigger: `{triggered_by}`\n"
            f"Resultado: `{len(failures)}/{total}` checks fallaron\n\n"
            + "\n\n".join(fail_lines)
            + f"\n\n⏱️ Duración: {duration:.1f}s"
        )
        print(f"[HEALTHCHECK/ALERT] Enviando Telegram alert ({len(failures)} fallos)...")
        _send_telegram_alert(alert_msg)
    else:
        print(f"[HEALTHCHECK] ✅ Todos los checks OK ({passed}/{total}) | triggered_by={triggered_by}")

    return {
        "failures": failures,
        "passed": passed,
        "total": total,
        "duration_sec": duration,
        "triggered_by": triggered_by,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Luna V2 Dashboard Health-Check")
    parser.add_argument("--trigger", default="manual", help="Quien disparó el check (manual/post-cycle/cron)")
    args = parser.parse_args()

    print(f"[HEALTHCHECK/MAIN] Iniciando health check. trigger={args.trigger}")
    result = run_all_checks(triggered_by=args.trigger)

    if result["failures"]:
        print(f"\n[HEALTHCHECK/EXIT] ❌ {len(result['failures'])} fallos detectados. Ver Telegram.")
        sys.exit(1)
    else:
        print(f"\n[HEALTHCHECK/EXIT] ✅ Todo OK ({result['passed']}/{result['total']})")
        sys.exit(0)

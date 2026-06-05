"""Inject run_sop_health_checks() before class DashboardHTTPHandler"""

path = '/root/luna_v2/dashboard/server.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

inject_marker = 'class DashboardHTTPHandler(http.server.SimpleHTTPRequestHandler):'
if inject_marker not in content:
    print(f"ERROR: Marker not found!")
    exit(1)

if 'def run_sop_health_checks' in content:
    print("Función ya existe, saltando.")
    exit(0)

health_fn = '''
# ===========================================================================
# [SOH-CHECKS] Monitor de Integridad - Checks Horarios :30
# Verifica todos los fixes críticos implementados el 2026-05-25
# ===========================================================================
_SOH_CACHE = {"ts": 0, "results": None}
_SOH_CACHE_TTL = 20 * 60  # 20 minutos de cache

def run_sop_health_checks(force=False):
    """[SOH-CHECKS] Ejecuta 15 checks de integridad del sistema. Cache 20min."""
    import time as _time, subprocess as _sp, socket as _sock
    import pandas as _pd
    from pathlib import Path as _Path
    global _SOH_CACHE
    now_ts = _time.time()
    if not force and _SOH_CACHE["results"] and (now_ts - _SOH_CACHE["ts"]) < _SOH_CACHE_TTL:
        print(f"[SOH-CHECKS] Cache activo ({(now_ts - _SOH_CACHE['ts'])/60:.1f}min)")
        return _SOH_CACHE["results"]
    print("[SOH-CHECKS] Ejecutando 15 health checks...")
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    PARQUET_LIVE = _Path("/root/luna_v2/data/features/features_live.parquet")
    SETTINGS_PATH = _Path("/root/luna_v2/config/settings.yaml")
    NGINX_PATH = _Path("/etc/nginx/sites-enabled/luna-dashboard")
    SERVER_PATH = _Path("/root/luna_v2/dashboard/server.py")
    GRAPHIFY_OUT = _Path("/root/luna_v2/graphify/out")

    _parquet_df = None
    _parquet_err = None
    try:
        _parquet_df = _pd.read_parquet(PARQUET_LIVE)
        print(f"[SOH-CHECKS] Parquet: {_parquet_df.shape}")
    except Exception as _e:
        _parquet_err = str(_e)

    checks = []
    def _chk(id_, cat, name, fn):
        try:
            st, det = fn()
            print(f"[SOH-CHECKS] {id_}: {st} — {det}")
            checks.append({"id": id_, "cat": cat, "name": name, "status": st, "details": det})
        except Exception as _e:
            print(f"[SOH-CHECKS] {id_} EXCEPTION: {_e}")
            checks.append({"id": id_, "cat": cat, "name": name, "status": "FAIL", "details": f"Exception: {str(_e)[:120]}"})

    def chk01():
        if _parquet_err: return "FAIL", f"No se pudo leer: {_parquet_err[:80]}"
        age_h = (now_ts - PARQUET_LIVE.stat().st_mtime) / 3600
        if age_h > 2: return "FAIL", f"OBSOLETO: {age_h:.1f}h sin actualizar (max 2h)"
        elif age_h > 1: return "WARN", f"Actualizado hace {age_h*60:.0f}min (límite 120min)"
        return "PASS", f"Actualizado hace {age_h*60:.0f}min — última fila: {str(_parquet_df.index[-1])[:16]}"
    _chk("CHK-01", "LIVE DATA", "features_live.parquet reciente (<2h)", chk01)

    def chk02():
        ALIASES = ["FundingRate_EMA3", "FundingRate_Pct90d", "OI_Open_USD", "OI_High_USD", "OI_Low_USD", "ETF_Flow_Proxy"]
        if _parquet_df is None: return "FAIL", "Parquet no disponible"
        missing = [a for a in ALIASES if a not in _parquet_df.columns]
        if missing: return "FAIL", f"[FIX-SKEW-FINAL] Aliases FALTANTES: {missing}"
        last = _parquet_df[ALIASES].iloc[-1]
        nan_al = [a for a in ALIASES if _pd.isna(last[a])]
        if nan_al: return "WARN", f"NaN en última fila: {nan_al}"
        return "PASS", f"6/6 aliases OK. FR_EMA3={last['FundingRate_EMA3']:.5f}, OI_Open={last['OI_Open_USD']:.0f}"
    _chk("CHK-02", "FIX-SKEW-FINAL", "6 aliases canónicos presentes con datos (post-dropna bridge)", chk02)

    def chk03():
        OHLCV = ["Futures_Volume", "close_perps"]
        if _parquet_df is None: return "FAIL", "Parquet no disponible"
        missing = [c for c in OHLCV if c not in _parquet_df.columns]
        if missing: return "FAIL", f"[OHLCV-FIX] Columnas FALTANTES: {missing}"
        last = _parquet_df[OHLCV].iloc[-1]
        if any(_pd.isna(last[c]) for c in OHLCV): return "WARN", "Columnas OHLCV con NaN en última fila"
        return "PASS", f"Futures_Vol={last['Futures_Volume']:.0f}, close_perps={last['close_perps']:.2f}"
    _chk("CHK-03", "OHLCV-FIX", "Columnas OHLCV reales (Futures_Volume, close_perps)", chk03)

    def chk04():
        if _parquet_df is None: return "FAIL", "Parquet no disponible"
        n = _parquet_df.shape[1]
        if n < 512: return "FAIL", f"Solo {n} features (mínimo 512)"
        return "PASS", f"{n} features canónicas en features_live.parquet"
    _chk("CHK-04", "FEATURES", "Total features canónicas >= 512", chk04)

    def chk05():
        if _parquet_df is None: return "FAIL", "Parquet no disponible"
        last_row = _parquet_df.iloc[-1]
        null_pct = last_row.isna().mean() * 100
        complete_pct = 100 - null_pct
        n_null = int(last_row.isna().sum())
        if complete_pct < 85: return "FAIL", f"Solo {complete_pct:.1f}% completo ({n_null} NaN en última fila)"
        elif complete_pct < 95: return "WARN", f"{complete_pct:.1f}% completo ({n_null} NaN). Target: 95%"
        return "PASS", f"{complete_pct:.1f}% completo ({n_null} NaN en última fila)"
    _chk("CHK-05", "COMPLETENESS", "Completeness >= 95% en última fila", chk05)

    def chk06():
        r = _sp.run(["pm2", "list"], capture_output=True, text=True, timeout=5)
        out = r.stdout
        if "luna-v2-live-demo" not in out: return "FAIL", "luna-v2-live-demo NO en PM2"
        chunk = out.split("luna-v2-live-demo")[1][:120]
        if "online" not in chunk: return "FAIL", "luna-v2-live-demo NO está online"
        return "PASS", "luna-v2-live-demo online en PM2"
    _chk("CHK-06", "PM2-TRADER", "Bot luna-v2-live-demo está online", chk06)

    def chk07():
        r = _sp.run(["pm2", "list"], capture_output=True, text=True, timeout=5)
        out = r.stdout
        if "luna-dashboard" not in out: return "FAIL", "luna-dashboard NO en PM2"
        chunk = out.split("luna-dashboard")[1][:120]
        if "online" not in chunk: return "FAIL", "luna-dashboard NO está online"
        return "PASS", "luna-dashboard online en PM2"
    _chk("CHK-07", "PM2-DASHBOARD", "Dashboard luna-dashboard está online", chk07)

    def chk08():
        r = _sp.run(["grep", "-a", "HOLD\\|LONG\\|SHORT\\|SyntaxError", "/root/.pm2/logs/luna-dashboard-out.log"],
                    capture_output=True, text=True, timeout=5)
        lines = [l for l in r.stdout.strip().split("\\n") if l.strip()][-10:]
        if any("SyntaxError" in l or "Unexpected token" in l for l in lines):
            return "FAIL", "[DOUBLE-HEADERS] SyntaxError en logs — JSON corrompido detectado"
        last = next((l for l in reversed(lines) if any(x in l for x in ["HOLD","LONG","SHORT"])), None)
        if last: return "PASS", f"Última decisión válida: ...{last[-70:]}"
        return "WARN", "No hay decisiones recientes en logs (normal fuera del ciclo horario)"
    _chk("CHK-08", "HOUR-DECISION", "/api/hour-decision retorna JSON válido", chk08)

    def chk09():
        code = SERVER_PATH.read_text(encoding="utf-8")
        lines = code.split("\\n")
        suspects = []
        found_resp = False; found_end = False; resp_l = 0; end_l = 0
        in_options = False
        for i, line in enumerate(lines):
            s = line.strip()
            if "def do_OPTIONS" in s: in_options = True
            if in_options and ("def do_GET" in s or "def do_POST" in s or "def do_DELETE" in s): in_options = False
            if in_options: continue
            if "self.send_response(200)" in s: found_resp = True; found_end = False; resp_l = i+1
            if found_resp and "self.end_headers()" in s: found_end = True; end_l = i+1
            if found_end and "self._send_json(" in s:
                suspects.append(f"L{resp_l}-L{end_l}-L{i+1}")
                found_resp = False; found_end = False
        if suspects: return "FAIL", f"[DOUBLE-HEADERS] {len(suspects)} bugs: {suspects}"
        return "PASS", "0 endpoints con doble headers en server.py"
    _chk("CHK-09", "DOUBLE-HEADERS", "Cero endpoints con bug de doble headers", chk09)

    def chk10():
        if not NGINX_PATH.exists(): return "WARN", f"Nginx config no encontrado"
        cfg = NGINX_PATH.read_text()
        if "X-Frame-Options DENY" in cfg: return "FAIL", "[NGINX] X-Frame-Options DENY activo — iframes bloqueados!"
        if "X-Frame-Options SAMEORIGIN" in cfg: return "PASS", "X-Frame-Options SAMEORIGIN — iframes same-domain OK"
        return "WARN", "X-Frame-Options no encontrado en nginx config"
    _chk("CHK-10", "NGINX", "X-Frame-Options SAMEORIGIN (habilita iframes graphify)", chk10)

    def chk11():
        if not NGINX_PATH.exists(): return "WARN", "Nginx config no encontrado"
        cfg = NGINX_PATH.read_text()
        if "unpkg.com" in cfg: return "PASS", "CSP incluye unpkg.com — vis-network 3D puede cargar"
        return "FAIL", "[CSP] unpkg.com NO en Content-Security-Policy — vis-network bloqueado"
    _chk("CHK-11", "NGINX-CSP", "CSP permite unpkg.com para vis-network 3D (graphify)", chk11)

    def chk12():
        gj = GRAPHIFY_OUT / "graph.json"; gh = GRAPHIFY_OUT / "graph.html"
        m = [f for f in [gj, gh] if not f.exists()]
        if m: return "FAIL", f"[GRAPHIFY] Faltantes: {[str(x.name) for x in m]}"
        gj_mb = gj.stat().st_size / (1024*1024)
        if gj_mb < 0.1: return "WARN", f"graph.json muy pequeño ({gj_mb:.2f}MB)"
        return "PASS", f"graph.json={gj_mb:.2f}MB, graph.html={gh.stat().st_size/(1024*1024):.2f}MB ✓"
    _chk("CHK-12", "GRAPHIFY", "graph.json + graph.html presentes en /graphify/out/", chk12)

    def chk13():
        try:
            import time as _t; t0 = _t.time()
            s = _sock.create_connection(("localhost", 5432), timeout=2); s.close()
            lat = (_t.time() - t0) * 1000
            if lat > 500: return "WARN", f"PostgreSQL lento: {lat:.0f}ms"
            return "PASS", f"PostgreSQL responde en {lat:.0f}ms"
        except Exception as e: return "FAIL", f"PostgreSQL NO responde: {e}"
    _chk("CHK-13", "DATABASE", "PostgreSQL responde en localhost:5432 (<1s)", chk13)

    def chk14():
        if not SETTINGS_PATH.exists(): return "FAIL", "settings.yaml no encontrado"
        if _yaml is None: return "WARN", "yaml no disponible, saltando check"
        cfg = _yaml.safe_load(SETTINGS_PATH.read_text())
        embargo = None
        for section in cfg.values():
            if isinstance(section, dict):
                if "embargo_hours" in section: embargo = section["embargo_hours"]; break
                for sub in section.values():
                    if isinstance(sub, dict) and "embargo_hours" in sub: embargo = sub["embargo_hours"]; break
        if embargo is None: return "WARN", "embargo_hours no encontrado en settings.yaml"
        if embargo < 24: return "FAIL", f"[SOP-R3] embargo_hours={embargo}h — VIOLA mínimo 24H"
        return "PASS", f"embargo_hours={embargo}h >= 24H mínimo SOP V10.0 ✓"
    _chk("CHK-14", "SOP-R3 EMBARGO", "embargo_hours >= 24H en settings.yaml", chk14)

    def chk15():
        pp = _Path("/root/luna_v2/luna/features/feature_pipeline.py")
        if not pp.exists(): return "WARN", "feature_pipeline.py no encontrado"
        code = pp.read_text(encoding="utf-8")
        import re
        pats = [
            (r'\\.get\\("min_dsr"\\s*,\\s*[\\d\\.]+\\)', "min_dsr"),
            (r'\\.get\\("max_pbo"\\s*,\\s*[\\d\\.]+\\)', "max_pbo"),
            (r'\\.get\\("pbo_n_blocks"\\s*,\\s*\\d+\\)', "pbo_n_blocks"),
            (r'\\.get\\("embargo_hours"\\s*,\\s*\\d+\\)', "embargo_hours"),
        ]
        found = [desc for pat, desc in pats if re.search(pat, code)]
        if found: return "FAIL", f"[NO-FALLBACK] Fallbacks silenciosos: {found}"
        if "FIX-SKEW-FINAL" not in code: return "WARN", "FIX-SKEW-FINAL no encontrado en pipeline"
        return "PASS", "No-Fallback OK: 0 parámetros críticos con fallback silencioso"
    _chk("CHK-15", "NO-FALLBACK", "Sin valores hardcodeados críticos en pipeline (SOP R2/R3)", chk15)

    summary = {
        "pass": sum(1 for c in checks if c["status"] == "PASS"),
        "warn": sum(1 for c in checks if c["status"] == "WARN"),
        "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "total": len(checks), "ts": now_ts
    }
    result = {"checks": checks, "summary": summary}
    _SOH_CACHE["ts"] = now_ts
    _SOH_CACHE["results"] = result
    print(f"[SOH-CHECKS] OK: {summary['pass']} PASS | {summary['warn']} WARN | {summary['fail']} FAIL")
    return result

'''

content = content.replace(inject_marker, health_fn + inject_marker, 1)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"[SOH-BACKEND] run_sop_health_checks() inyectada. Verificando sintaxis...")

import subprocess
r = subprocess.run(['/root/miniconda3/envs/luna_env/bin/python', '-c', f'import ast; ast.parse(open("{path}").read()); print("SYNTAX OK")'],
                  capture_output=True, text=True)
print(r.stdout if r.stdout else r.stderr)

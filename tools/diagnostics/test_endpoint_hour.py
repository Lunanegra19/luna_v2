import urllib.request
import urllib.parse
import json
import base64
import http.cookiejar

# Auth via Basic first to get session cookie, then call API
BASE = "http://localhost:8080"
USER = "luna"
PASS = "E-Y03L5moyv3urXpn7fXa2dWpemymJ2c"

# Step 1: Login POST to get session cookie
cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

login_data = urllib.parse.urlencode({"username": USER, "password": PASS}).encode()
try:
    login_resp = opener.open(f"{BASE}/login", login_data, timeout=5)
    print(f"[AUTH] Login status: {login_resp.status}")
except Exception as e:
    print(f"[AUTH] Login error (puede ser redirect 302): {type(e).__name__}: {e}")

# Step 2: Call the API with the session cookie
url = (f"{BASE}/api/vps/hour-decision?"
       f"start_utc=2026-05-25T18:00:00&end_utc=2026-05-25T18:05:00"
       f"&local_date=2026-05-25&local_hour=18")
try:
    with opener.open(url, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
    d = json.loads(raw)
    data = d.get("data", {})
    print(f"\n{'='*60}")
    print(f"[ENDPOINT STATUS]   {d.get('status')}")
    print(f"{'='*60}")
    print(f"[ACTION]            {data.get('action')}")
    print(f"[XGB_PROB]          {data.get('xgb_prob')}")
    print(f"[CONFIDENCE]        {data.get('confidence')}")
    print(f"[HMM_REGIME]        {data.get('hmm_regime')}")
    print(f"[DURATION]          {data.get('duration')}")
    print(f"[TIMESTAMP]         {data.get('timestamp')}")
    print(f"\n--- CAMPOS OPERACIONALES (G3 FIX) ---")
    print(f"[CLOCK_DRIFT]       {data.get('clock_drift_minutes')} min  ({data.get('clock_drift_status')})")
    print(f"[LATENCY]           {data.get('execution_latency_sec')} s")
    print(f"[SLIPPAGE]          {data.get('slippage_pct')} %")
    print(f"[NAN_COLS]          {data.get('nan_inf_cols')}")
    print(f"[LEVERAGE]          {data.get('active_leverage')} x")
    print(f"[IS_APPROVED]       {data.get('is_approved')}")
    equity = data.get('api_liveness_equity')
    print(f"[EQUITY_OKX]        ${equity:,.2f}" if equity else "[EQUITY_OKX]        N/A")
    print(f"\n--- STEPS (CLASIFICADOR G1 FIX) ---")
    steps = data.get("steps", [])
    step_labels = ["Boot/Carga", "Heartbeat/Recon/Riesgo", "Data/Features", "Inferencia/Guards", "Exec OKX", "Duracion"]
    for i, (step, label) in enumerate(zip(steps, step_labels)):
        lines = [l for l in step.split("\n") if l.strip()]
        ok = "✅" if lines and "Ejecutado exitosamente" not in lines[0] else "⚠️ (sintetico)"
        print(f"  Paso {i+1} [{label}]: {len(lines)} lineas {ok}")
        for l in lines[:3]:
            print(f"    {l[:100]}")
        if len(lines) > 3:
            print(f"    ... (+{len(lines)-3} mas)")
except Exception as e:
    import traceback
    print(f"[API ERROR] {e}")
    traceback.print_exc()

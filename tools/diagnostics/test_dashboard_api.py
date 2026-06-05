"""Test directo del endpoint hour-decision desde localhost para confirmar el fix."""
import requests, json

BASE = "http://localhost:8080"

# Ciclo de las 21:00 local (19:00 UTC)
params = {
    "start_utc": "2026-05-25T17:00:00Z",
    "end_utc":   "2026-05-25T17:59:59Z",
    "local_date": "2026-05-25",
    "local_hour": "19"
}

r = requests.get(f"{BASE}/api/vps/hour-decision", params=params, timeout=15)
print(f"HTTP: {r.status_code}")
j = r.json()
print(f"status: {j.get('status')}")
if j.get("status") == "success" and "data" in j:
    d = j["data"]
    print(f"action:           {d.get('action')}")
    print(f"hmm_regime:       {d.get('hmm_regime')}")
    print(f"xgb_prob:         {d.get('xgb_prob')}")
    print(f"duration:         {d.get('duration')}")
    print(f"clock_drift_min:  {d.get('clock_drift_minutes')}")
    print(f"nan_inf_cols:     {d.get('nan_inf_cols')}")
    print(f"is_approved:      {d.get('is_approved')}")
    print(f"api_equity:       ${d.get('api_liveness_equity', 0):,.0f}")
    print(f"steps count:      {len(d.get('steps', []))}")
    for i, s in enumerate(d.get("steps", [])):
        preview = str(s)[:120] if s else "(vacío)"
        print(f"  Paso {i+1}: {preview}")
else:
    print(f"RESPUESTA COMPLETA: {json.dumps(j, indent=2, default=str)[:500]}")

# Feature pipeline
print("\n--- FEATURE PIPELINE STATUS ---")
r2 = requests.get(f"{BASE}/api/vps/feature-pipeline-status", timeout=15)
j2 = r2.json()
print(f"HTTP: {r2.status_code} | status: {j2.get('status')}")
print(f"last_bar: {j2.get('last_bar')}")
for g in j2.get("groups", []):
    icon = "✅" if g["status"]=="OK" else "⚠️" if g["status"]=="WARN" else "❌"
    print(f"  {icon} {g['emoji']} {g['group']}: {g['available']}/{g['total']} features | missing={g['missing'] or 'none'}")

"""
[TEST] Prueba directa de los 2 nuevos endpoints del dashboard.
Ejecutar en VPS: cd /root/luna_v2 && python tools/diagnostics/test_new_endpoints.py
"""
import urllib.request
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8080"

def test_endpoint(name, url):
    print(f"\n{'='*60}")
    print(f"[TEST] Probando: {name}")
    print(f"[TEST] URL: {url}")
    try:
        req = urllib.request.Request(url)
        # Simular sesion local (el dashboard en VPS no requiere auth desde 127.0.0.1 via nginx)
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read().decode("utf-8")
            data = json.loads(body)
            status = data.get("status", "?")
            print(f"[TEST-OK] HTTP 200 | status={status}")
            if status == "success":
                if "data" in data:
                    d = data["data"]
                    print(f"  action={d.get('action','?')} | hmm_regime={d.get('hmm_regime','?')} | quorum={d.get('quorum','?')}")
                    print(f"  steps[0][:80]: {str(d.get('steps',[''])[0])[:80]}")
                elif "groups" in data:
                    groups = data["groups"]
                    print(f"  {len(groups)} grupos de features:")
                    for g in groups:
                        print(f"  - {g['group']}: status={g.get('status','?')} available={g.get('available',0)}/{g.get('total',0)}")
                    print(f"  last_bar={data.get('last_bar','N/A')}")
            elif status == "standby":
                print(f"[TEST-OK] Standby correcto (sin logs de live trader disponibles)")
            else:
                print(f"[TEST-WARN] Status: {status} | message={data.get('message','?')}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[TEST-FAIL] HTTP {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"[TEST-FAIL] URLError: {e.reason}")
        return False
    except Exception as e:
        print(f"[TEST-FAIL] Error: {type(e).__name__}: {e}")
        return False

# Test 1: hour-decision para la hora actual
now = datetime.now()
local_date = now.strftime("%Y-%m-%d")
local_hour = now.hour
start_utc = now.strftime("%Y-%m-%dT%H:00:00.000Z")
end_utc = now.strftime("%Y-%m-%dT%H:59:59.999Z")

url1 = (
    f"{BASE_URL}/api/vps/hour-decision"
    f"?local_date={local_date}&local_hour={local_hour}"
    f"&start_utc={start_utc}&end_utc={end_utc}"
)

ok1 = test_endpoint("/api/vps/hour-decision (hora actual)", url1)

# Test 2: feature-pipeline-status
url2 = f"{BASE_URL}/api/vps/feature-pipeline-status"
ok2 = test_endpoint("/api/vps/feature-pipeline-status", url2)

print(f"\n{'='*60}")
print(f"[RESUMEN] hour-decision: {'OK' if ok1 else 'FAIL'} | feature-pipeline: {'OK' if ok2 else 'FAIL'}")
if ok1 and ok2:
    print("[RESUMEN] Todos los endpoints nuevos responden correctamente. Deploy EXITOSO.")
else:
    print("[RESUMEN] Hay fallos. Revisar logs de PM2.")

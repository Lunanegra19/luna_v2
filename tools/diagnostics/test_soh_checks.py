"""Test the SOH health checks endpoint directly"""
import sys
sys.path.insert(0, '/root/luna_v2/dashboard')

# Import the function directly
exec(open('/root/luna_v2/dashboard/server.py').read().split('class DashboardHTTPHandler')[0])
result = run_sop_health_checks(force=True)
print(f"\n{'='*60}")
print(f"SOH HEALTH CHECKS RESULT: {result['summary']['pass']} PASS | {result['summary']['warn']} WARN | {result['summary']['fail']} FAIL")
print('='*60)
for c in result['checks']:
    icon = {'PASS':'✅','WARN':'⚠️','FAIL':'❌'}.get(c['status'],'?')
    print(f"{icon} [{c['id']}] {c['name']}")
    print(f"   → {c['details']}")

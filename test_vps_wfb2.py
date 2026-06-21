import sys
sys.path.append("/root/luna_v2")
from dashboard.server import get_seed_metrics_from_verdict
import traceback

try:
    print(get_seed_metrics_from_verdict(42))
except Exception as e:
    traceback.print_exc()

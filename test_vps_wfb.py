import sys
sys.path.append("/root/luna_v2")
from dashboard.server import get_wfb_seeds_summary
active, hist = get_wfb_seeds_summary()
print(len(active["champions"]), [c["calmar"] for c in active["champions"]])

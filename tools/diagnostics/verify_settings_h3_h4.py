import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from config.settings import cfg

checks = [
    (cfg.gauntlet,   "max_pbo",                    0.45,  "H4: max_pbo"),
    (cfg.wfb,        "ensemble_consensus_threshold", 4,    "H4: consensus_threshold"),
    (cfg.wfb,        "max_seeds_to_explore",         12,   "H4: max_seeds_to_explore"),
    (cfg.wfb,        "min_seeds_to_approve",          3,   "H4: min_seeds_to_approve"),
    (cfg.wfb,        "prune_threshold",              0.80, "H4: prune_threshold"),
    (cfg.metalabeler,"meta_v2_thresh_bull_strong",   0.48, "H3: thresh_bull_strong"),
    (cfg.metalabeler,"meta_v2_thresh_bull_unstable", 0.57, "H3: thresh_bull_unstable"),
    (cfg.metalabeler,"meta_v2_rolling_percentile",   0.60, "H3: rolling_percentile"),
    (cfg.metalabeler,"simulate_online_recalibration",False,"H3: sim_online_off"),
]

all_ok = True
for section, param, expected, desc in checks:
    val = getattr(section, param, "MISSING")
    ok  = (val == expected)
    all_ok = all_ok and ok
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {desc:<38s} = {val!r}")

print()
print("RESULTADO:", "TODOS OK - listo para relanzar" if all_ok else "FALLOS - revisar settings.yaml")

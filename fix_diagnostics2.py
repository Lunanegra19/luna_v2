import os
import re

files_to_fix = [
    r"tools/diagnostics/simulate_embargo.py",
    r"config/settings.py",
    r"luna/monitoring/shap_feature_auditor.py",
    r"tools/diagnostics/analyze_early_stop.py",
    r"tools/diagnostics/arch10_verify_fix.py",
    r"tools/diagnostics/arch19_20_verify_fix.py",
    r"tools/diagnostics/find_33_signals.py",
    r"tools/diagnostics/find_optimal_seeds.py",
    r"tools/diagnostics/investigation_xgb_edge.py",
    r"tools/diagnostics/simulate_calibration_strategies.py",
    r"tools/diagnostics/simulate_levers_w1.py",
    r"tools/diagnostics/simulate_strategy_c_impact.py",
    r"tools/diagnostics/test_dynamic_metalabeler.py",
    r"tools/diagnostics/test_fase3_proposals.py"
]

pattern_get = re.compile(r'\.get\(\s*["\'](\w+)["\']\s*,\s*[^)]+\)')

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "config/settings.py" in file_path:
        new_content = pattern_get.sub(r'["\1"]', content)
    else:
        new_content = pattern_get.sub(r'.\1', content)
        
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed .get in {file_path}")

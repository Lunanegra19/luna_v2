import os
import re

files_to_fix = [
    r"tools/diagnostics/apply_fp_fixes.py",
    r"tools/diagnostics/apply_hmm_fix.py",
    r"tools/diagnostics/apply_phase_xgb_fixes.py",
    r"tools/diagnostics/arch_deep_diagnosis.py",
    r"tools/diagnostics/arch_structural_reanalysis.py",
    r"tools/diagnostics/fase4_ha_bear_colapso.py",
    r"tools/diagnostics/find_optimal_seeds.py",
    r"tools/diagnostics/fix03_purge_rows.py",
    r"tools/diagnostics/hmm_shield_analysis.py",
    r"tools/diagnostics/inspect_forced_features.py",
    r"tools/diagnostics/patch_arch04.py",
    r"tools/diagnostics/patch_arch2123.py",
    r"tools/diagnostics/patch_fix_hmm_shield_w2.py",
    r"tools/diagnostics/reproduce_fp_bugs.py",
    r"tools/diagnostics/simulate_calibration_strategies.py",
    r"tools/diagnostics/simulate_embargo.py",
    r"tools/diagnostics/test_fase3_proposals.py",
    r"config/settings.py",
    r"luna/monitoring/shap_feature_auditor.py",
    r"tools/diagnostics/analyze_early_stop.py",
    r"tools/diagnostics/arch10_verify_fix.py",
    r"tools/diagnostics/arch19_20_verify_fix.py",
    r"tools/diagnostics/find_33_signals.py",
    r"tools/diagnostics/investigation_xgb_edge.py",
    r"tools/diagnostics/simulate_levers_w1.py",
    r"tools/diagnostics/simulate_strategy_c_impact.py",
    r"tools/diagnostics/test_dynamic_metalabeler.py"
]

pattern_getattr = re.compile(r'getattr\(([^,]+),\s*["\'](\w+)["\']\s*,\s*[^)]+\)')

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Reemplazo de getattr(obj, 'prop', default) -> obj.prop
    new_content = pattern_getattr.sub(r'\1.\2', content)
    
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed getattr in {file_path}")

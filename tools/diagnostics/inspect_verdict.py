import json

for fname, label in [
    (r'G:\Mi unidad\ia\luna_v2\data\reports\2026-05-21_T0125_WFB_20260521_011023_7864_seed1337_FINAL_statistical_verdict.json', 'SFI18 T0125'),
    (r'G:\Mi unidad\ia\luna_v2\data\reports\2026-05-21_T0345_WFB_20260521_033115_26948_seed1337_FINAL_statistical_verdict.json', 'SFI16 T0345 APROBADO')
]:
    with open(fname) as f:
        v = json.load(f)
    sa = v.get('statistical_audit', {})
    print(f'=== {label} ===')
    print('  statistical_audit:', sa)
    print('  deploy_approved (top):', v.get('deploy_approved'))
    print()

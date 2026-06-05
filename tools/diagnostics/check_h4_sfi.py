import sys, json
sys.stdout.reconfigure(encoding='utf-8')
h4 = {'ath_dist_pct', 'ath_streak_h', 'ath_streak_z90d', 'price_z_score_252d',
      'realized_vol_ratio', 'realized_vol_ratio_z90d', 'realized_vol_ratio_milag48h'}

for w in ['W1', 'W2']:
    try:
        with open(f'data/reports/wfb/selected_features_{w}.json') as f:
            d = json.load(f)
        if isinstance(d, dict):
            feats = set(d.get('selected_features', []))
        else:
            feats = set(d)
        found   = h4 & feats
        missing = h4 - feats
        print(f"=== {w} ({len(feats)} features seleccionadas) ===")
        f_str = str(sorted(found)) if found else "NINGUNA"
        m_str = str(sorted(missing)) if missing else "NINGUNA"
        print(f"  H4 APROBADAS  ({len(found)}): {f_str}")
        print(f"  H4 RECHAZADAS ({len(missing)}): {m_str}")
        print(f"  Todas las seleccionadas: {sorted(feats)}")
        print()
    except Exception as e:
        print(f"{w}: ERROR {e}")

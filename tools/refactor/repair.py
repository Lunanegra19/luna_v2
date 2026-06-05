import codecs

with codecs.open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'r', encoding='latin-1', errors='ignore') as f:
    text = f.read()

start_str = '        total = len(X.columns)'
end_str = '        # Seleccionar las TOP N features rankeadas por adjusted_dsr + Alphas'

idx1 = text.find(start_str)
idx2 = text.find(end_str)

if idx1 != -1 and idx2 != -1:
    before = text[:idx1]
    after = text[idx2:]
    
    correct_block = '''        total = len(X.columns)
        from joblib import Parallel, delayed
        import time
        import pandas as pd

        def _process_feature(i, col):
            t0 = time.time()
            _nan_pct = float(X_a[col].isna().mean())
            _var = float(X_a[col].var(ddof=0))
            if _nan_pct > 0.85 or _var < 1e-6 or pd.isna(_var):
                return col, i, {"passed": False, "deflated_sharpe": 0.0, "mean_sharpe": 0.0, "n_folds": 0}, {}, t0, None, 0.0

            x1 = X_a[col].fillna(0).values.reshape(-1, 1)

            mask_cond = None
            if any(col.startswith(p) for p in LOW_FREQ_PREFIX):
                activation_rate = (X_a[col].fillna(0) > 0).mean()
                if activation_rate < LOW_FREQ_THRESHOLD:
                    mask_cond = (X_a[col].fillna(0) > 0).values

            res = self._eval_one(x1, y_a, p_a, cpcv, feature_name=col, mask=mask_cond)

            stab_info: dict = {}
            adjusted_dsr = res["deflated_sharpe"]
            if col not in ALPHA_SIGNALS and res["n_folds"] > 0:
                try:
                    stab_info = self._eval_temporal_stability(x1, y_a, p_a, ts_a, cpcv_small)
                    _stab_score = stab_info.get("stability_score", 1.0)
                    adjusted_dsr = res["deflated_sharpe"] * ((1 - _stab_penalty_w) + _stab_penalty_w * _stab_score)
                    
                    if _stab_score < 0.50:
                        adjusted_dsr = adjusted_dsr * (_stab_score / 0.50)**2

                    if adf_penalties and col in adf_penalties:
                        adjusted_dsr *= adf_penalties[col]
                    if adv_penalties and col in adv_penalties:
                        adv_pen = adv_penalties[col]
                        if adv_pen == 0.0:
                            res = {**res, "passed": False}
                            adjusted_dsr = -9999.0
                        else:
                            adjusted_dsr *= adv_pen
                        
                    stab_info["adjusted_dsr"] = round(adjusted_dsr, 4)
                except Exception as _se2:
                    logger.debug(f"  SFI-02: stability eval fallida para {col}: {_se2}")

            return col, i, res, stab_info, t0, mask_cond, adjusted_dsr

        logger.info(f"    [SFI] Iniciando procesamiento paralelo de {total} features con joblib...")
        _raw_results = Parallel(n_jobs=-1, batch_size=1, pre_dispatch='1.5*n_jobs')(
            delayed(_process_feature)(i, col) for i, col in enumerate(X.columns)
        )

        _raw_results.sort(key=lambda x: x[1])
        for col, i, res, stab_info, t0, mask_cond, adjusted_dsr in _raw_results:
            self.scores[col] = {**res, "stability": stab_info, "adjusted_dsr": round(adjusted_dsr, 4)}
            status = "[OK]" if res["passed"] else "[XX]"
            cond_tag = " [COND]" if mask_cond is not None else ""
            
            # Simple fallback for stab format
            stab_tag = ""
            if stab_info:
                p_years = stab_info.get("positive_years", "?")
                tot_years = len(stab_info.get("yearly_dsrs", [1,2,3,4,5]))
                s_score = stab_info.get("stability_score", 0.0)
                stab_tag = f" [STAB={s_score:.2f} {p_years}/{tot_years}]"

            elapsed = time.time() - t0
            logger.info(f"  [{i+1:3d}/{total}] {col:<40} DSR={res['deflated_sharpe']:+.3f} adjDSR={adjusted_dsr:+.3f} MeanSR={res['mean_sharpe']:+.3f} Folds={res['n_folds']} {status}{cond_tag}{stab_tag} ({elapsed:.1f}s)")

'''
    
    with codecs.open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'w', encoding='latin-1') as f:
        f.write(before + correct_block + after)
    print('File patched.')
else:
    print('Could not find markers.')

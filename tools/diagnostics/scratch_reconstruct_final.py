with open('c:/Users/Usuario/Downloads/feature_selection_e.py', encoding='utf-8', errors='replace') as f:
    lines_orig = f.readlines()

new_content = lines_orig[:2025] # Up to the end of OOD-GUARD

new_content.extend([
    '        # ──────────────────────────────────────────────────────────────────────────\n',
    '        # [Paso B] Clustering Jerárquico & Selección Representantes\n',
    '        # ──────────────────────────────────────────────────────────────────────────\n',
    '        X_raw = df[raw_cols].copy()\n',
    '        X_alpha = df[alpha_cols]\n',
    '        prices = df[\'close\'] if \'close\' in df.columns else y.cumsum()\n',
    '\n',
    '        chk_B = self._load_checkpoint(\'B\') if resume else None\n',
    '        if chk_B:\n',
    '            repr_features = chk_B[\'selected\']\n',
    '            logger.info(f\'[B] Retomando desde checkpoint: {len(repr_features)} features\')\n',
    '        else:\n',
    '            repr_features = self.clusterer.fit_transform(X_raw, y, prices=prices)\n',
    '            self._save_checkpoint(\'B\', {\'selected\': repr_features})\n',
    '\n',
    '        self.results[\'n_after_clustering\'] = len(repr_features)\n',
    '\n',
    '        # ──────────────────────────────────────────────────────────────────────────\n',
    '        # [Paso C] Automatic Lag Discovery & Alignment\n',
    '        # ──────────────────────────────────────────────────────────────────────────\n',
    '        _lag_cache_path = Path(DATA_DIR) / \'_lag_cache.json\'\n',
    '\n',
    '        def _data_fingerprint(df_: pd.DataFrame) -> str:\n',
    '            import pandas as pd\n',
    '            idx = df_.index\n',
    '            s = idx[0].isoformat() if hasattr(idx[0], \'isoformat\') else str(idx[0])\n',
    '            e = idx[-1].isoformat() if hasattr(idx[-1], \'isoformat\') else str(idx[-1])\n',
    '            return f\'{len(df_)}rows|{s}|{e}|{len(df_.columns)}cols\'\n',
    '\n',
    '        _effective_cache = None\n',
    '        _mi_lags_valid = {}\n',
    '        _dsr_verified_lags = {}\n',
    '        current_fp = _data_fingerprint(X_raw)\n',
    '\n',
    '        if _lag_cache_path.exists():\n',
    '            try:\n',
    '                import json\n',
    '                _cache_data = json.loads(_lag_cache_path.read_text(encoding=\'utf-8\'))\n',
    '                if current_fp == _cache_data.get(\'mi_data_fingerprint\'):\n',
    '                    _mi_lags_valid = _cache_data.get(\'mi_lags\', {})\n',
    '                    _effective_cache = {**_domain_lags, **_mi_lags_valid}\n',
    '            except Exception as e:\n',
    '                logger.warning(f\'[C] Error leyendo _lag_cache.json: {e}\')\n',
    '\n'
])

# For lines 2103 to 2459: strip EXACTLY 4 leading spaces (to preserve relative indentation but move the block left by one level)
for i in range(2103, 2460):
    l = lines_orig[i]
    if l.startswith('    ') and l.strip() != '':
        new_content.append(l[4:])
    else:
        new_content.append(l)

# For lines 2460 to end: copy EXACTLY as is
for i in range(2460, len(lines_orig)):
    new_content.append(lines_orig[i])

with open('g:/Mi unidad/ia/luna_v2/luna/features/feature_selection_e.py', 'w', encoding='utf-8') as f:
    f.writelines(new_content)

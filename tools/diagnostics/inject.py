import sys
path = r'g:\Mi unidad\ia\luna_v2\luna\features\feature_pipeline.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if 'FeaturePipeline.run() comenzando' in line:
        new_lines.append(lines[i+1])
        
        new_lines.append('''
        # [FAST-INJECT] Fase 4: Optimization de Pipeline Redundancy
        if fast_inject:
            logger.info("[FP] FAST-INJECT MODE ACTIVO: Saltando reconstruccion de variables base.")
            parts = []
            for p in ["features_train.parquet", "features_validation.parquet", "features_holdout.parquet"]:
                pth = _FEATURES_DIR / p
                if pth.exists():
                    try:
                        import pandas as pd
                        parts.append(pd.read_parquet(pth))
                    except Exception as e:
                        logger.warning(f"[FP] FAST-INJECT no pudo leer {p}: {e}")
            if parts:
                import pandas as pd
                df = pd.concat(parts).sort_index()
                cols_to_drop = ["KMeans_Tribe_ID", "Master_Causal_Signal", "HMM_Regime", "HMM_Semantic"]
                df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
                
                df = self.integrate_mining_outputs(df)
                
                logger.info(f"[FP] FAST-INJECT FINALIZADO | {_time.monotonic() - _t_total:.1f}s")
                if live_mode:
                    return {"live": df}
                return self.split_and_save(df)
            else:
                logger.warning("[FP] FAST-INJECT Fallo: No se encontraron parquets base. Fallback a pipeline completo.")
''')
        with open(path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(new_lines + lines[i+2:])
        print("INJECTED")
        sys.exit(0)
print("NOT FOUND")

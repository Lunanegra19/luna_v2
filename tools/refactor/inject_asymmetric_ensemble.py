import sys
from pathlib import Path
import re

file_path = Path("scripts/evaluate_ensemble_wfb.py")
content = file_path.read_text(encoding="utf-8")

# 1. We replace the hard voting block (lines 227 to 253) with Asymmetric logic
target_voting = """        if voting_method == 'hard':
            try:
                # [DUAL-CONSENSUS-FIX 2026-06-22] Umbral dinámico asimétrico
                _is_short = ('direction' in df_all_trades.columns) and len(df_all_trades) > 0 and (df_all_trades['direction'].iloc[0] == 'short')
                try:
                    base_threshold = int(getattr(_cfg.wfb, 'ensemble_consensus_threshold_short', _cfg.wfb.ensemble_consensus_threshold))
                except Exception:
                    base_threshold = int(_cfg.wfb.ensemble_consensus_threshold)
                    
                # [ENSAMBLE DEBE SER MAYOR A 1/3 HARDVOTING]
                min_required_threshold = int(len(approved_seeds) / 3) + 1
                threshold = max(base_threshold, min_required_threshold)
                
                _bucket_hours = int(_cfg.wfb.consensus_bucket_hours)
                _bucket_freq = f'{_bucket_hours}h'
                print(f"[HARD-VOTING] Consensus Gate >= {threshold} semillas (Mínimo requerido > 1/3 de {len(approved_seeds)} = {min_required_threshold}) ({'SHORT' if _is_short else 'LONG'}). Bucket: {_bucket_hours}H.")
            except Exception as e:
                raise RuntimeError(f"CRITICAL [DUAL-CONSENSUS-FIX]: Faltan parámetros de Hard Voting en settings.yaml. Política No-Fallback: {e}")
                
            df_all_trades['consensus_bucket'] = df_all_trades.index.floor(_bucket_freq)
            bucket_unique_seeds = df_all_trades.groupby('consensus_bucket')['seed'].nunique().rename('consensus_count')
            df_all_trades['consensus_count'] = df_all_trades['consensus_bucket'].map(bucket_unique_seeds)
            
            df_filtered_trades = df_all_trades[df_all_trades['consensus_count'] >= threshold].copy()
            n_buckets_pass = df_filtered_trades['consensus_bucket'].nunique()
            print(f"[HARD-VOTING] Trades filtrados: {len(df_all_trades)} filas → {len(df_filtered_trades)} filas en {n_buckets_pass} buckets")"""

replacement_voting = """        if voting_method == 'hard' or voting_method == 'asymmetric':
            # [ASYMMETRIC-ENSEMBLE-FIX 2026-06-24] Lógica Dual Asimétrica
            _is_short = ('direction' in df_all_trades.columns) and len(df_all_trades) > 0 and (df_all_trades['direction'].iloc[0] == 'short')
            try:
                _bucket_hours = int(_cfg.wfb.consensus_bucket_hours)
                _bucket_freq = f'{_bucket_hours}h'
                long_min_seeds = int(getattr(_cfg.wfb, 'ensemble_long_min_seeds', 2))
                long_min_prob = float(getattr(_cfg.wfb, 'ensemble_long_min_prob', 0.60))
                short_min_seeds = int(getattr(_cfg.wfb, 'ensemble_short_min_seeds', 2))
                short_min_wr = float(getattr(_cfg.wfb, 'ensemble_short_min_wr', 0.40))
                
                threshold = short_min_seeds if _is_short else long_min_seeds
            except Exception as e:
                raise RuntimeError(f"CRITICAL [ASYMMETRIC-ENSEMBLE-FIX]: Faltan parámetros Asimétricos en settings.yaml. Política No-Fallback: {e}")

            df_all_trades['consensus_bucket'] = df_all_trades.index.floor(_bucket_freq)
            
            # Filtrado local asimétrico
            if _is_short:
                print(f"[ASYMMETRIC-VOTING] BEAR MODE: Rolling WR >= {short_min_wr:.2f} | Seeds >= {threshold}")
                if 'rolling_win_rate' in df_all_trades.columns:
                    valid_votes = df_all_trades[df_all_trades['rolling_win_rate'] >= short_min_wr].copy()
                else:
                    print("[ASYMMETRIC-VOTING] ADVERTENCIA: no se encontró 'rolling_win_rate'. Se admiten todos los votos.")
                    valid_votes = df_all_trades.copy()
            else:
                print(f"[ASYMMETRIC-VOTING] BULL MODE: Strong Conviction (xgb_prob >= {long_min_prob:.2f}) | Seeds >= {threshold}")
                if 'xgb_prob' in df_all_trades.columns:
                    valid_votes = df_all_trades[df_all_trades['xgb_prob'] >= long_min_prob].copy()
                else:
                    print("[ASYMMETRIC-VOTING] ADVERTENCIA: no se encontró 'xgb_prob'. Se admiten todos los votos.")
                    valid_votes = df_all_trades.copy()
                    
            bucket_unique_seeds = valid_votes.groupby('consensus_bucket')['seed'].nunique().rename('consensus_count')
            valid_votes['consensus_count'] = valid_votes['consensus_bucket'].map(bucket_unique_seeds)
            
            df_filtered_trades = valid_votes[valid_votes['consensus_count'] >= threshold].copy()
            n_buckets_pass = df_filtered_trades['consensus_bucket'].nunique()
            print(f"[ASYMMETRIC-VOTING] Trades filtrados: {len(df_all_trades)} filas → {len(valid_votes)} votos válidos → {len(df_filtered_trades)} filas en {n_buckets_pass} buckets")"""

if target_voting in content:
    content = content.replace(target_voting, replacement_voting)
    print("Voting logic updated successfully.")
else:
    print("WARNING: Voting logic target not found.")

# 2. We inject the Schizophrenia Filter (Mutual Cancellation) right after df_portfolio is created
target_schizo = """        print(f"[SOFT-VOTING-02] Portfolio agregado por timestamp: "
              f"{len(df_filtered_trades)} filas → {len(df_portfolio)} trades únicos de consenso")
        logger.info(f"[SOFT-VOTING-02] Portfolio final: {len(df_portfolio)} trades (de {len(df_filtered_trades)} filas agregadas)")"""

replacement_schizo = """        print(f"[SOFT-VOTING-02] Portfolio agregado por timestamp: "
              f"{len(df_filtered_trades)} filas → {len(df_portfolio)} trades únicos de consenso")
        logger.info(f"[SOFT-VOTING-02] Portfolio final: {len(df_portfolio)} trades (de {len(df_filtered_trades)} filas agregadas)")

        # =====================================================================
        # [SCHIZOPHRENIA-FILTER 2026-06-24] Cancelación Mutua de Colisiones
        # =====================================================================
        try:
            from luna.validation.ensemble_voter import get_asymmetric_opponent_portfolio
            
            print("[SCHIZOPHRENIA-FILTER] Comprobando colisiones direccionales...")
            _opponent_dir = 'long' if _is_short else 'short'
            
            # Función auxiliar (definida ad-hoc o importada) para buscar las señales del oponente
            import os
            _temp_env = os.environ.get("LUNA_DIRECTION", "")
            
            # Extraer buckets del portafolio actual
            my_buckets = set(df_portfolio.index)
            
            # Cargar y aplicar las reglas del oponente para buscar colisiones EXACTAS (mismo consensus_bucket)
            opponent_files = list(predictions_dir.glob(f"oos_trades_seed*_{_opponent_dir}.parquet"))
            opponent_dfs = []
            for f in opponent_files:
                try:
                    df_opp = pd.read_parquet(f)
                    if not df_opp.empty:
                        if 'timestamp' in df_opp.columns:
                            df_opp = df_opp.set_index('timestamp')
                        df_opp.index = pd.to_datetime(df_opp.index, utc=True)
                        opponent_dfs.append(df_opp)
                except Exception:
                    pass
            
            if opponent_dfs:
                df_opp_all = pd.concat(opponent_dfs).sort_index()
                df_opp_all['consensus_bucket'] = df_opp_all.index.floor(_bucket_freq)
                
                # Reglas del oponente
                if _opponent_dir == 'long':
                    opp_min_prob = float(getattr(_cfg.wfb, 'ensemble_long_min_prob', 0.60))
                    opp_thr = int(getattr(_cfg.wfb, 'ensemble_long_min_seeds', 2))
                    if 'xgb_prob' in df_opp_all.columns:
                        opp_valid = df_opp_all[df_opp_all['xgb_prob'] >= opp_min_prob].copy()
                    else:
                        opp_valid = df_opp_all.copy()
                else:
                    opp_min_wr = float(getattr(_cfg.wfb, 'ensemble_short_min_wr', 0.40))
                    opp_thr = int(getattr(_cfg.wfb, 'ensemble_short_min_seeds', 2))
                    if 'rolling_win_rate' in df_opp_all.columns:
                        opp_valid = df_opp_all[df_opp_all['rolling_win_rate'] >= opp_min_wr].copy()
                    else:
                        opp_valid = df_opp_all.copy()
                        
                opp_counts = opp_valid.groupby('consensus_bucket')['direction'].count()
                opp_passed_buckets = set(opp_counts[opp_counts >= opp_thr].index)
                
                # Colisiones (Schizophrenia)
                collisions = my_buckets.intersection(opp_passed_buckets)
                
                if collisions:
                    print(f"[SCHIZOPHRENIA-FILTER] 💥 ADVERTENCIA: Se detectaron {len(collisions)} colisiones con el bot {_opponent_dir.upper()}.")
                    print(f"[SCHIZOPHRENIA-FILTER] 💥 Fechas de colisión: {[str(c) for c in collisions]}")
                    print(f"[SCHIZOPHRENIA-FILTER] 🛑 Cancelación Mutua (Skip Trade) activada para proteger el capital.")
                    # Drop de las colisiones
                    df_portfolio = df_portfolio.drop(list(collisions))
                    print(f"[SCHIZOPHRENIA-FILTER] Portafolio reducido a {len(df_portfolio)} trades seguros.")
                    logger.warning(f"[SCHIZOPHRENIA-FILTER] Eliminados {len(collisions)} trades por colisión con {_opponent_dir.upper()}.")
                else:
                    print(f"[SCHIZOPHRENIA-FILTER] ✅ No se detectaron colisiones con {_opponent_dir.upper()}.")
                    
        except Exception as e_schizo:
            print(f"[SCHIZOPHRENIA-FILTER] Error evaluando colisiones: {e_schizo}")
            logger.error(f"[SCHIZOPHRENIA-FILTER] Error evaluando colisiones: {e_schizo}")
        # ====================================================================="""

if target_schizo in content:
    content = content.replace(target_schizo, replacement_schizo)
    print("Schizophrenia filter updated successfully.")
else:
    print("WARNING: Schizophrenia filter target not found.")

file_path.write_text(content, encoding="utf-8")
print("evaluate_ensemble_wfb.py script modifications saved.")

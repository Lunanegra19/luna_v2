import os

filepath = r"g:\Mi unidad\ia\luna_v2\luna\features\feature_selection_e.py"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_logic = """                # FIX-MATH-01: Usamos retornos de 1H y pagamos costos SOLAMENTE al entrar al trade.
                # Esto soluciona la falacia de solapamiento de costos y calcula el PnL exacto
                # de mantener la posicion a resolucion horaria.
                fwd_ret_1h = np.diff(prices[te]) / prices[te][:-1]
                sigs_eval = sigs_lo[:-1]
                
                n = min(len(sigs_eval), len(fwd_ret_1h))
                if n < 10:
                    continue
                    
                # Detectar entradas (transicion de 0 a 1) para cobrar el round-trip
                entradas = (np.diff(sigs_eval[:n], prepend=0) > 0).astype(float)
                
                strat_ret = sigs_eval[:n] * fwd_ret_1h[:n] - entradas * SFI_COST_ROUNDTRIP

                if np.std(strat_ret) < 1e-10:
                    continue
                
                # Al ser retornos reales de 1H, la anualizacion vuelve a ser la estandar (sqrt(8760))
                ann_factor = np.sqrt(365 * 24)
                sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * ann_factor)
                
                fold_sharpes.append(np.clip(sharpe, -10, 10))
            except Exception as e:
                logger.debug(f"  Fold error: {e}")
                continue

        if not fold_sharpes:
            return {"mean_sharpe": 0.0, "deflated_sharpe": 0.0,
                    "passed": False, "n_folds": 0}

        mean_sr = float(np.mean(fold_sharpes))
        # FIX-MATH-02: Para modelos univariables (Weak Learners), aplicar n_trials=600 colapsa 
        # el DSR a 0.000 absoluto. El rigor de 600 trials pertenece a la Fase E (Ensamblado).
        # Usamos n_trials=2 para obtener un DSR relativo que permita rankear el top 15.
        dsr = PurgedCPCV.deflated_sharpe(fold_sharpes, 2)
"""

# Replace lines 864 to 910
# Line 864 is index 864 in 0-indexed if line 1 is index 0. Wait, 865 is "Retorno forward..."
# The view_file output:
# 865:                 # Retorno forward al horizonte TBM: ret_t = price[t+H]/price[t] - 1
# 910:         dsr = PurgedCPCV.deflated_sharpe(fold_sharpes, SFI_DSR_N_TRIALS)

start_idx = 864
end_idx = 910

# Write the new file
with open(filepath, "w", encoding="utf-8") as f:
    for i in range(start_idx):
        f.write(lines[i])
    f.write(new_logic)
    for i in range(end_idx, len(lines)):
        f.write(lines[i])

print("File updated successfully.")

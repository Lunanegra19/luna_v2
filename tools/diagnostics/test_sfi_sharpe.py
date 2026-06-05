import numpy as np

def simulate_eval_one_single_trade():
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.005, 1000)
    prices = np.exp(np.cumsum(returns)) * 50000
    
    sigs_lo = np.zeros(1000)
    sigs_lo[-97] = 1.0 # 1 at the very end just before cutoff, wait, _tbm_h=96
    
    _tbm_h = 96
    SFI_COST_ROUNDTRIP = 0.0015
        
    fwd_ret = prices[_tbm_h:] / prices[:-_tbm_h] - 1
    sigs_eval = sigs_lo[:-_tbm_h]
    
    n = min(len(sigs_eval), len(fwd_ret))
    strat_ret = sigs_eval[:n] * fwd_ret[:n] - sigs_eval[:n] * SFI_COST_ROUNDTRIP
    
    ann_factor = np.sqrt((365 * 24) / _tbm_h)
    sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * ann_factor)
    
    print(f"Number of trades: {sigs_eval.sum()}")
    print(f"Mean strat_ret: {np.mean(strat_ret):.6f}")
    print(f"Std strat_ret: {np.std(strat_ret):.6f}")
    print(f"Raw Sharpe: {sharpe:.4f}")
    print(f"Clipped Sharpe: {np.clip(sharpe, -10, 10):.4f}")

simulate_eval_one_single_trade()

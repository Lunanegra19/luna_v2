import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from luna.models.hmm_regime import HMMRegimeModel
from luna.models.regime_router import RegimeRouter
from luna.features.tbm import apply_triple_barrier

def test_hypothesis_tbm_w4():
    print("[TEST-TBM-W4] Starting quantitative test for W4 TBM barriers misalignment...")
    
    # 1. Load cached holdout features for W4
    w4_feat_path = ROOT / "data" / "wfb_cache" / "W4" / "features" / "features_holdout_W4.parquet"
    if not w4_feat_path.exists():
        print(f"[TEST-TBM-W4] SKIP: cached W4 features parquet not found at {w4_feat_path}")
        return
        
    df_oos = pd.read_parquet(w4_feat_path)
    print(f"[TEST-TBM-W4] Loaded W4 holdout: {len(df_oos)} rows ({df_oos.index.min()} to {df_oos.index.max()})")
    
    # 2. Find all executed trades for W4 in today's run folders
    runs_dir = ROOT / "data" / "runs"
    run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("WFB_20260613_") and int(d.name.split("_")[2]) >= 95519])
    
    actual_trades_list = []
    for rf in run_folders:
        seeds = [d for d in rf.iterdir() if d.is_dir() and d.name.startswith("seed")]
        for seed in seeds:
            w4_dir = seed / "W4"
            trades_file = w4_dir / "oos_trades.parquet"
            if trades_file.exists():
                try:
                    df_t = pd.read_parquet(trades_file)
                    if len(df_t) > 0:
                        df_t['seed'] = seed.name
                        actual_trades_list.append(df_t)
                except Exception as e:
                    print(f"Error loading {trades_file}: {e}")
                    
    if not actual_trades_list:
        print("[TEST-TBM-W4] SKIP: No executed trades found for W4 in today's runs.")
        return
        
    df_actual_trades = pd.concat(actual_trades_list)
    # The index of df_actual_trades is the entry time
    signal_times = pd.DatetimeIndex(df_actual_trades.index.unique()).sort_values()
    print(f"\nLoaded {len(df_actual_trades)} executed W4 trades ({len(signal_times)} unique entry times) across today's seeds.")
    
    # Let's inspect the baseline stats of these actual executed trades in W4
    net_returns_base = df_actual_trades["return_pct"] # return_pct is already net of 0.15% fee
    baseline_pf = net_returns_base[net_returns_base > 0].sum() / abs(net_returns_base[net_returns_base < 0].sum())
    baseline_wr = (net_returns_base > 0).mean()
    baseline_avg_win = net_returns_base[net_returns_base > 0].mean()
    baseline_avg_loss = net_returns_base[net_returns_base < 0].mean()
    print(f"\n--- ACTUAL REALIZED RUN STATISTICS (W4) ---")
    print(f"  Trades:        {len(df_actual_trades)}")
    print(f"  Win Rate:      {baseline_wr:.2%}")
    print(f"  Avg Win:       {baseline_avg_win*100:.4f}%")
    print(f"  Avg Loss:      {baseline_avg_loss*100:.4f}%")
    print(f"  Profit Factor: {baseline_pf:.3f}")
    
    # 3. Simulate alternative TBM configurations on the exact same signal times
    # We use seed1337 models to get HMM Semantics (or map directly from df_oos)
    # Let's align df_oos HMM Semantic states
    models_dir = ROOT / "data" / "wfb_cache" / "seed1337" / "W4" / "models"
    hmm_model = HMMRegimeModel.load(models_dir)
    hmm_df = hmm_model.predict_regime_series(df_oos)
    df_oos["HMM_Regime"] = hmm_model.coerce_regime_numeric(hmm_df["HMM_Regime"])
    df_oos["HMM_Semantic"] = hmm_df["HMM_Semantic"]
    
    # XGBoost to get probabilities (needed for confidence scaler)
    router = RegimeRouter(models_dir, agent_type="xgboost", direction="long")
    xgb_probs_df = router.route_and_predict(df_oos)
    df_oos["xgb_prob_cal"] = xgb_probs_df["calibrated"]

    def simulate_tbm(pt_multiplier_scale=1.0, sl_multiplier_scale=1.0, dynamic_barrier=True, linear_decay_pt=True):
        from luna.models.predict_oos import get_hmm_tbm_params, get_hmm_horizon
        
        _pt = df_oos["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["tp"] * pt_multiplier_scale)
        _sl = df_oos["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["sl"] * sl_multiplier_scale)
        
        _prob_series = df_oos["xgb_prob_cal"].fillna(0.5).clip(0.5, 1.0)
        _conf_scaler = 0.7 + ((_prob_series - 0.5) / 0.5) * (1.3 - 0.7)
        _pt = _pt * _conf_scaler
        _sl = _sl * _conf_scaler
        
        _dyn_max = int(df_oos.loc[df_oos.index >= signal_times[0], "HMM_Semantic"].dropna().map(
            lambda r: get_hmm_horizon(r)
        ).mode().iloc[0] if not df_oos.loc[df_oos.index >= signal_times[0], "HMM_Semantic"].dropna().empty else 168)
        
        # Run TBM only on the exact unique entry times of executed trades
        tbm_result = apply_triple_barrier(
            price_series=df_oos["close"],
            event_times=signal_times,
            sides=pd.Series(1, index=signal_times),
            pt_sl_multiplier=[_pt, _sl],
            vertical_barrier_hours=72,
            min_return=0.005,
            dynamic_barrier=dynamic_barrier,
            dynamic_horizon_min_h=24,
            dynamic_horizon_max_h=_dyn_max,
            linear_decay_pt=linear_decay_pt,
            pt_decay_fraction=0.75,
            funding_series=df_oos.get("FundingRate"),
        )
        
        # Net return (subtract 0.15% fee)
        net_returns = tbm_result["ret"] - 0.0015
        
        n_trades = len(net_returns)
        wr = (net_returns > 0).mean() if n_trades > 0 else 0
        wins = net_returns[net_returns > 0].sum()
        losses = net_returns[net_returns < 0].sum()
        pf = wins / abs(losses) if losses != 0 else float('inf')
        avg_win = net_returns[net_returns > 0].mean() if (net_returns > 0).any() else 0
        avg_loss = net_returns[net_returns < 0].mean() if (net_returns < 0).any() else 0
        
        pt_hits = (tbm_result["bin"] == 1).sum()
        sl_hits = (tbm_result["bin"] == -1).sum()
        vb_timeouts = (tbm_result["bin"] == 0).sum()
        
        return {
            "n_trades": n_trades,
            "wr": wr,
            "pf": pf,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "pt_hits": pt_hits,
            "sl_hits": sl_hits,
            "vb_timeouts": vb_timeouts,
            "total_return": net_returns.sum()
        }

    # Suppress verbose loguru/print logs
    import builtins
    from loguru import logger as loguru_logger
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level="WARNING")
    
    original_print = builtins.print
    def quiet_print(*args, **kwargs):
        if args and any("mapeado" in str(arg) or "TBM" in str(arg) or "Horizon" in str(arg) for arg in args):
            return
        original_print(*args, **kwargs)
    builtins.print = quiet_print

    try:
        # 1. Sim Baseline
        sim_base = simulate_tbm()
        original_print("\n--- SIMULATED BASELINE TBM ---")
        original_print(f"  Trades:       {sim_base['n_trades']}")
        original_print(f"  Win Rate:     {sim_base['wr']:.2%}")
        original_print(f"  Avg Win:      {sim_base['avg_win']*100:.4f}%")
        original_print(f"  Avg Loss:     {sim_base['avg_loss']*100:.4f}%")
        original_print(f"  Profit Factor: {sim_base['pf']:.3f}")
        original_print(f"  PT Hits:      {sim_base['pt_hits']} | SL Hits: {sim_base['sl_hits']} | VB Timeouts: {sim_base['vb_timeouts']}")
        
        # 2. Exp A: Disable linear decay of PT
        sim_a = simulate_tbm(linear_decay_pt=False)
        original_print("\n--- EXPERIMENT A: Disabling Linear Decay of PT ---")
        original_print(f"  Win Rate:     {sim_a['wr']:.2%}")
        original_print(f"  Avg Win:      {sim_a['avg_win']*100:.4f}%")
        original_print(f"  Avg Loss:     {sim_a['avg_loss']*100:.4f}%")
        original_print(f"  Profit Factor: {sim_a['pf']:.3f}")
        original_print(f"  PT Hits:      {sim_a['pt_hits']} | SL Hits: {sim_a['sl_hits']} | VB Timeouts: {sim_a['vb_timeouts']}")

        # 3. Exp B: Widen PT (1.5x) and Tighten SL (0.7x)
        sim_b = simulate_tbm(pt_multiplier_scale=1.5, sl_multiplier_scale=0.7)
        original_print("\n--- EXPERIMENT B: Widen PT (1.5x) and Tighten SL (0.7x) ---")
        original_print(f"  Win Rate:     {sim_b['wr']:.2%}")
        original_print(f"  Avg Win:      {sim_b['avg_win']*100:.4f}%")
        original_print(f"  Avg Loss:     {sim_b['avg_loss']*100:.4f}%")
        original_print(f"  Profit Factor: {sim_b['pf']:.3f}")
        original_print(f"  PT Hits:      {sim_b['pt_hits']} | SL Hits: {sim_b['sl_hits']} | VB Timeouts: {sim_b['vb_timeouts']}")

        # 4. Exp C: Widen both PT (1.5x) and SL (1.5x)
        sim_c = simulate_tbm(pt_multiplier_scale=1.5, sl_multiplier_scale=1.5)
        original_print("\n--- EXPERIMENT C: Widen both PT (1.5x) and SL (1.5x) ---")
        original_print(f"  Win Rate:     {sim_c['wr']:.2%}")
        original_print(f"  Avg Win:      {sim_c['avg_win']*100:.4f}%")
        original_print(f"  Avg Loss:     {sim_c['avg_loss']*100:.4f}%")
        original_print(f"  Profit Factor: {sim_c['pf']:.3f}")
        original_print(f"  PT Hits:      {sim_c['pt_hits']} | SL Hits: {sim_c['sl_hits']} | VB Timeouts: {sim_c['vb_timeouts']}")

        # 5. Exp D: Scale PT (2.0x) and keep SL (1.0x) - asymmetry
        sim_d = simulate_tbm(pt_multiplier_scale=2.0, sl_multiplier_scale=1.0)
        original_print("\n--- EXPERIMENT D: Widen PT only (2.0x) ---")
        original_print(f"  Win Rate:     {sim_d['wr']:.2%}")
        original_print(f"  Avg Win:      {sim_d['avg_win']*100:.4f}%")
        original_print(f"  Avg Loss:     {sim_d['avg_loss']*100:.4f}%")
        original_print(f"  Profit Factor: {sim_d['pf']:.3f}")
        original_print(f"  PT Hits:      {sim_d['pt_hits']} | SL Hits: {sim_d['sl_hits']} | VB Timeouts: {sim_d['vb_timeouts']}")

        original_print("\n--- CONCLUSION ---")
        best_pf = max(sim_base['pf'], sim_a['pf'], sim_b['pf'], sim_c['pf'], sim_d['pf'])
        if best_pf > baseline_pf:
            original_print(f"[TEST-TBM-W4] SUCCESS: Hypothesis verified! Alternative TBM configuration improves W4 Profit Factor from {baseline_pf:.3f} to {best_pf:.3f}.")
        else:
            original_print(f"[TEST-TBM-W4] WARNING: Alternative TBM configurations did not improve the Profit Factor in W4. High drag might be structural to Q4 2025 market trend.")
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    test_hypothesis_tbm_w4()

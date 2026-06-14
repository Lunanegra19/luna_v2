import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from luna.models.hmm_regime import HMMRegimeModel
from luna.models.signal_filter import SignalFilter
from luna.models.regime_router import RegimeRouter

def test_hypothesis_hmm_w1():
    print("[TEST-HMM-W1] Starting quantitative test for W1 HMM allowed regimes...")
    
    # 1. Load cached holdout features for W1
    w1_feat_path = ROOT / "data" / "wfb_cache" / "W1" / "features" / "features_holdout_W1.parquet"
    if not w1_feat_path.exists():
        print(f"[TEST-HMM-W1] SKIP: cached W1 features parquet not found at {w1_feat_path}")
        return
        
    df_oos = pd.read_parquet(w1_feat_path)
    print(f"[TEST-HMM-W1] Loaded W1 holdout: {len(df_oos)} rows ({df_oos.index.min()} to {df_oos.index.max()})")
    
    # We will use seed86454 as our test subject for W1 since it completed W1
    models_dir = ROOT / "data" / "wfb_cache" / "seed86454" / "W1" / "models"
    if not models_dir.exists():
        print(f"[TEST-HMM-W1] SKIP: cached W1 models for seed86454 not found at {models_dir}")
        return

    # 2. Load the HMM model and predict regimes
    hmm_model = HMMRegimeModel.load(models_dir)
    hmm_df = hmm_model.predict_regime_series(df_oos)
    
    df_oos["HMM_Regime"] = hmm_model.coerce_regime_numeric(hmm_df["HMM_Regime"])
    df_oos["HMM_Semantic"] = hmm_df["HMM_Semantic"]
    
    print("\n--- HMM Semantic State Distribution in W1 holdout ---")
    counts = df_oos["HMM_Semantic"].value_counts()
    for state, count in counts.items():
        print(f"  {state}: {count} hours ({count/len(df_oos):.1%})")
        
    # Check if 1_VOLATILE_BULL_B is present
    vbb_present = "1_VOLATILE_BULL_B" in counts.index
    print(f"\n1_VOLATILE_BULL_B present: {vbb_present}")
    
    # 3. Simulate XGBoost predictions
    # We use RegimeRouter to route and predict raw probabilities
    router = RegimeRouter(models_dir, agent_type="xgboost", direction="long")
    xgb_probs_df = router.route_and_predict(df_oos)
    df_oos["xgb_prob"] = xgb_probs_df["raw"]
    df_oos["xgb_prob_cal"] = xgb_probs_df["calibrated"]
    
    # 4. Initialize SignalFilter
    sf = SignalFilter(models_dir=models_dir)
    
    # 5. Get candidate signals (XGB prob > threshold)
    xgb_thresh = 0.48  # standard baseline threshold
    candidate_mask = df_oos["xgb_prob_cal"] >= xgb_thresh
    n_candidates = int(candidate_mask.sum())
    print(f"\nCandidate signals (XGB prob >= {xgb_thresh}): {n_candidates}")
    
    if n_candidates == 0:
        print("[TEST-HMM-W1] No candidate signals generated. Cannot test HMM regimes filter.")
        return
        
    # Let's count how many candidate signals fall under each HMM regime
    candidate_regimes = df_oos.loc[candidate_mask, "HMM_Semantic"].value_counts()
    print("\nCandidate signals by HMM regime:")
    for reg, cnt in candidate_regimes.items():
        print(f"  {reg}: {cnt} signals")
        
    # 6. Apply HMM allowed regimes filter with current settings (no 1_VOLATILE_BULL_B)
    from config.settings import cfg
    original_allowed = list(cfg.metalabeler.hmm_allowed_regimes)
    print(f"\nOriginal allowed regimes: {original_allowed}")
    
    # Simulate filtering
    import re
    def is_allowed(val, allowed_list):
        v = str(val).upper()
        if v in allowed_list:
            return True
        allowed_bases = {re.sub(r'_[A-D]$', '', str(lbl).upper()) for lbl in allowed_list}
        for base in allowed_bases:
            if v.startswith(base):
                return True
        return False
        
    orig_allowed_upper = [x.upper() for x in original_allowed]
    passed_orig_mask = candidate_mask & df_oos["HMM_Semantic"].apply(lambda x: is_allowed(x, orig_allowed_upper))
    n_passed_orig = int(passed_orig_mask.sum())
    print(f"Signals passing HMM filter (Original): {n_passed_orig}")
    
    # 7. Apply HMM allowed regimes filter with proposed settings (added 1_VOLATILE_BULL_B)
    proposed_allowed = original_allowed + ["1_VOLATILE_BULL_B", "1_VOLATILE_BULL"]
    proposed_allowed_upper = [x.upper() for x in proposed_allowed]
    passed_prop_mask = candidate_mask & df_oos["HMM_Semantic"].apply(lambda x: is_allowed(x, proposed_allowed_upper))
    n_passed_prop = int(passed_prop_mask.sum())
    print(f"Signals passing HMM filter (Proposed): {n_passed_prop}")
    
    unlocked = n_passed_prop - n_passed_orig
    print(f"  --> Unlocked signals: {unlocked} (+{unlocked / max(1, n_passed_orig):.1%} increase)")
    
    assert vbb_present, "Hypothesis failed: 1_VOLATILE_BULL_B was not present in the holdout!"
    assert unlocked > 0, "Hypothesis failed: Adding 1_VOLATILE_BULL_B did not unlock any signals!"
    print("[TEST-HMM-W1] Hypothesis VERIFIED: 1_VOLATILE_BULL_B was blocking W1 signals.")
    
    # 8. Evaluate hypothetical performance of unlocked signals
    # We apply TBM to the unlocked signals to see if they are profitable
    from luna.features.tbm import apply_triple_barrier
    
    signal_times = df_oos.index[passed_prop_mask]
    if len(signal_times) > 0:
        # Get TBM parameters
        from luna.models.predict_oos import get_hmm_tbm_params, get_hmm_horizon
        _pt = df_oos["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["tp"])
        _sl = df_oos["HMM_Semantic"].map(lambda r: get_hmm_tbm_params(r)["sl"])
        
        # Scaling by confidence
        _prob_series = df_oos["xgb_prob_cal"].fillna(0.5).clip(0.5, 1.0)
        _conf_scaler = 0.7 + ((_prob_series - 0.5) / 0.5) * (1.3 - 0.7)
        _pt = _pt * _conf_scaler
        _sl = _sl * _conf_scaler
        
        # Moda del horizonte
        _dyn_max = int(df_oos.loc[df_oos.index >= signal_times[0], "HMM_Semantic"].dropna().map(
            lambda r: get_hmm_horizon(r)
        ).mode().iloc[0] if not df_oos.loc[df_oos.index >= signal_times[0], "HMM_Semantic"].dropna().empty else 168)
        
        tbm_result = apply_triple_barrier(
            price_series=df_oos["close"],
            event_times=signal_times,
            sides=pd.Series(1, index=signal_times),
            pt_sl_multiplier=[_pt, _sl],
            vertical_barrier_hours=72,
            min_return=0.005,
            dynamic_barrier=True,
            dynamic_horizon_min_h=24,
            dynamic_horizon_max_h=_dyn_max,
            linear_decay_pt=True,
            pt_decay_fraction=0.75,
            funding_series=df_oos.get("FundingRate"),
        )
        
        # Calculate returns net of fees (0.15% standard taker round-trip or 0.25% SOP)
        # Note: return_raw in TBM is gross. We subtract 0.15% (0.0015) for net return.
        net_returns = tbm_result["ret"] - 0.0015
        wins = net_returns[net_returns > 0].sum()
        losses = net_returns[net_returns < 0].sum()
        pf = wins / abs(losses) if losses != 0 else float('inf')
        wr = (net_returns > 0).mean()
        
        print("\n--- Simulated Performance of Unlocked Signals in W1 (Net of 0.15% fee) ---")
        print(f"  Total Trades: {len(net_returns)}")
        print(f"  Win Rate:     {wr:.2%}")
        print(f"  Avg Win:      {net_returns[net_returns > 0].mean()*100:.4f}%")
        print(f"  Avg Loss:     {net_returns[net_returns < 0].mean()*100:.4f}%")
        print(f"  Profit Factor: {pf:.3f}")
        print(f"  Total Return:  {net_returns.sum()*100:.4f}%")

        
        # Verify if they are profitable
        print(f"[TEST-HMM-W1] Profit Factor is {pf:.3f} and WR is {wr:.2%}.")
        if pf > 1.0:
            print("[TEST-HMM-W1] SUCCESS: Unlocked signals are PROFITABLE (PF > 1.0).")
        else:
            print("[TEST-HMM-W1] WARNING: Unlocked signals are NOT profitable (PF <= 1.0). "
                  "Adding them increases signals but might not add edge in this specific window.")

if __name__ == "__main__":
    test_hypothesis_hmm_w1()

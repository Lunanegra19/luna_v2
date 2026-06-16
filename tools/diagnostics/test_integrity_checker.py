import pandas as pd
import logging
import sys

logger = logging.getLogger("luna.pipeline_integrity")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
logger.addHandler(sh)

_TAG = "[PIPELINE-INTEGRITY]"
_DIFF_EQ_TOLERANCE = 1e-6

def post_window_check(trades_df: pd.DataFrame, window_id: str) -> dict:
    print(f"{_TAG} â”€â–¶ POST-WINDOW CHECK: {window_id} â—€â”€")

    if trades_df is None or len(trades_df) == 0:
        return {"window_id": window_id, "n_trades": 0, "status": "CASH"}

    n = len(trades_df)
    metrics = {"window_id": window_id, "n_trades": n}

    if "xgb_prob" in trades_df.columns and "xgb_prob_cal" in trades_df.columns:
        diff = (trades_df["xgb_prob_cal"] - trades_df["xgb_prob"]).abs()
        pct_eq = float((diff < _DIFF_EQ_TOLERANCE).mean() * 100)
        metrics["pct_cal_eq_raw"] = round(pct_eq, 1)

        if pct_eq >= 99.0:
            _msg = (
                f"{_TAG} INFO [{window_id}] POST-WINDOW: "
                f"xgb_prob_cal == xgb_prob_raw en {pct_eq:.0f}% de los {n} trades. "
                f"Calibration Bypass Active (SPW=1.0 baseline esperado)."
            )
            print(_msg)
            logger.info(_msg)
            metrics["cal_bug"] = False
        elif pct_eq > 20:
            print(f"{_TAG}   [{window_id}] WARNING: {pct_eq:.0f}% trades con cal==raw (parcial).")
            logger.warning(f"{_TAG} [{window_id}] cal==raw parcial: {pct_eq:.0f}%")
            metrics["cal_bug"] = "PARTIAL"
        else:
            print(f"{_TAG}   [{window_id}] âœ“. CalibraciÃ³n activa: solo {pct_eq:.1f}% trades con cal==raw.")
            metrics["cal_bug"] = False

    if "is_win" in trades_df.columns:
        wr = float(trades_df["is_win"].mean() * 100)
        metrics["win_rate_pct"] = round(wr, 1)

        if n < 30:
            msg = f"{_TAG}   [{window_id}] INFO: WR={wr:.1f}% sobre {n} trades (Insignificante segÃºn SOP R8)."
            print(msg)
            logger.info(msg)
        else:
            if wr < 20:
                print(
                    f"{_TAG}   [{window_id}] âŒ CRITICAL: WR={wr:.1f}% < 20%. "
                    f"El modelo estÃ¡ prediciendo PEOR que azar."
                )
                logger.critical(f"{_TAG} [{window_id}] WR={wr:.1f}% â€” peor que azar.")
            elif wr < 35:
                print(f"{_TAG}   [{window_id}] âš  WARNING: WR={wr:.1f}% â€” bajo (azar=50%). Verificar pipeline.")
                logger.warning(f"{_TAG} [{window_id}] WR bajo: {wr:.1f}%")
            elif wr > 80:
                print(
                    f"{_TAG}   [{window_id}] âš  WARNING: WR={wr:.1f}% > 80% â€” sospechosamente alto. "
                )
                logger.warning(f"{_TAG} [{window_id}] WR sospechosamente alto: {wr:.1f}%")
            else:
                print(f"{_TAG}   [{window_id}] âœ“. WR={wr:.1f}% dentro del rango esperado.")

    if n < 5:
        print(f"{_TAG}   [{window_id}] âš  WARNING: Solo {n} trades.")
        logger.warning(f"{_TAG} [{window_id}] Muy pocos trades: {n}")
    else:
        print(f"{_TAG}   [{window_id}] âœ“. N trades: {n}")

    return metrics

# TEST 1: W3 Scenario (3 trades, 0 wins, cal == raw)
df_w3 = pd.DataFrame({
    "xgb_prob": [0.6, 0.7, 0.8],
    "xgb_prob_cal": [0.6, 0.7, 0.8],
    "is_win": [False, False, False],
    "signal_threshold": [0.55, 0.55, 0.55],
    "HMM_Semantic": ["1_BULL_TREND"]*3
})

print("--- RUNNING W3 SIMULATION ---")
post_window_check(df_w3, "W3_seed42")

# TEST 2: Valid large sample (50 trades, 60% win rate, cal != raw)
df_valid = pd.DataFrame({
    "xgb_prob": [0.6]*50,
    "xgb_prob_cal": [0.65]*50,
    "is_win": [True]*30 + [False]*20,
    "signal_threshold": [0.55]*50,
    "HMM_Semantic": ["1_BULL_TREND"]*50
})

print("\n--- RUNNING VALID SIMULATION ---")
post_window_check(df_valid, "W_VALID")

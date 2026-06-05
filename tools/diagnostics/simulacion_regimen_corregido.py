import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Configurar rutas y paths del sistema
ROOT = Path("g:/Mi unidad/ia/luna_v2")
sys.path.insert(0, str(ROOT))

from luna.features.tbm import apply_triple_barrier
from luna.models.predict_oos import HMM_TBM_PARAMS, _HMM_TBM_FALLBACK, HMM_HORIZON_MAP, _HMM_HORIZON_FALLBACK

# Mapeo corregido que resuelve el "Silent Regime Fallback" Bug
HMM_TBM_PARAMS_CORRECTED = {
    # Bull
    "1_VOLATILE_BULL":     {"sl": 2.5, "tp": 3.0},
    "1_VOLATILE_BULL_B":   {"sl": 2.5, "tp": 3.0},
    "1_BULL_TREND":        {"sl": 1.5, "tp": 2.0},
    "1_BULL_TREND_WEAK":   {"sl": 1.5, "tp": 2.0},
    "1_BULL_TREND_B":      {"sl": 1.5, "tp": 2.0},
    "1_BULL_GRIND":        {"sl": 1.5, "tp": 2.0},
    # Range
    "2_CALM_RANGE":        {"sl": 0.6, "tp": 1.2},
    "2_CALM_RANGE_B":      {"sl": 0.6, "tp": 1.2},
    "2_VOLATILE_RANGE":    {"sl": 1.2, "tp": 1.8},
    "2_VOLATILE_RANGE_B":  {"sl": 1.2, "tp": 1.8},
    # Bear
    "3_BEAR_CRASH":        {"sl": 1.5, "tp": 1.5},
    "3_CALM_BEAR":         {"sl": 1.5, "tp": 1.5},
    "4_BEAR_FORCED":       {"sl": 1.5, "tp": 1.5},
}

HMM_HORIZON_MAP_CORRECTED = {
    # Bull
    "1_VOLATILE_BULL":     240,
    "1_VOLATILE_BULL_B":   240,
    "1_BULL_TREND":        168,
    "1_BULL_TREND_WEAK":   168,
    "1_BULL_TREND_B":      168,
    "1_BULL_GRIND":        168,
    # Range
    "2_CALM_RANGE":        96,
    "2_CALM_RANGE_B":      96,
    "2_VOLATILE_RANGE":    120,
    "2_VOLATILE_RANGE_B":  120,
    # Bear
    "3_BEAR_CRASH":        168,
    "3_CALM_BEAR":         168,
    "4_BEAR_FORCED":       168,
}

def run_simulation():
    print("=========================================================================")
    print("  SIMULACIÓN DE CORRECCIÓN DE BUG RÉGIMENES Y ASIMETRÍA P&L - SEMILLA 42")
    print("  [FIX-BUGS-PRINTS] Test de impacto de correctores y mejoras")
    print("=========================================================================\n")

    wfb_dir = ROOT / "data" / "reports" / "wfb"
    features_dir = ROOT / "data" / "features"

    windows = [1, 2, 3]
    all_results = {}

    for w in windows:
        trades_path = wfb_dir / f"oos_trades_W{w}_seed42.parquet"
        holdout_path = features_dir / f"features_holdout_W{w}.parquet"

        if not trades_path.exists() or not holdout_path.exists():
            print(f"[ERROR] No se encontraron archivos para Ventana W{w}!")
            continue

        trades_df = pd.read_parquet(trades_path)
        holdout_df = pd.read_parquet(holdout_path)

        if trades_df.index.name == 'timestamp' or 'timestamp' not in trades_df.columns:
            trades_df = trades_df.reset_index()
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], utc=True)
        trades_df = trades_df.set_index('timestamp').sort_index()

        if holdout_df.index.name != 'timestamp' and 'timestamp' in holdout_df.columns:
            holdout_df = holdout_df.set_index('timestamp')
        holdout_df.index = pd.to_datetime(holdout_df.index, utc=True)
        holdout_df = holdout_df.sort_index()

        close_series = holdout_df['close']
        signal_times = trades_df.index

        hmm_semantic = trades_df['hmm_regime']
        xgb_prob_cal = trades_df['xgb_prob_cal']
        meta_v2_prob = trades_df['meta_v2_prob']

        prob_series = xgb_prob_cal.fillna(trades_df['xgb_prob']).fillna(meta_v2_prob).fillna(0.5).clip(0.5, 1.0)
        conf_scaler = 0.7 + ((prob_series - 0.5) / 0.5) * (1.3 - 0.7)

        all_results[f"W{w}"] = {
            "close": close_series,
            "signal_times": signal_times,
            "hmm_semantic": hmm_semantic,
            "conf_scaler": conf_scaler,
            "trades_df": trades_df,
            "meta_v2_prob": meta_v2_prob,
        }

    scenarios = {
        "0_Baseline": {
            "desc": "Línea Base: Con Linear TP Decay, y HMM Bug Silencioso (regímenes fallback)",
            "linear_decay_pt": True,
            "hmm_params": HMM_TBM_PARAMS,
            "hmm_horizons": HMM_HORIZON_MAP,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
        },
        "1_Fix_HMM_Regimes": {
            "desc": "Solo arreglar HMM Bug Silencioso (mapear todas las variantes de regímenes)",
            "linear_decay_pt": True,
            "hmm_params": HMM_TBM_PARAMS_CORRECTED,
            "hmm_horizons": HMM_HORIZON_MAP_CORRECTED,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
        },
        "2_No_Linear_Decay": {
            "desc": "Solo desactivar Linear TP Decay (manteniendo HMM Bug Silencioso)",
            "linear_decay_pt": False,
            "hmm_params": HMM_TBM_PARAMS,
            "hmm_horizons": HMM_HORIZON_MAP,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
        },
        "3_Fix_HMM_and_No_Decay": {
            "desc": "Combinado: Fix HMM Bug + TP Estático (Sin Linear TP Decay)",
            "linear_decay_pt": False,
            "hmm_params": HMM_TBM_PARAMS_CORRECTED,
            "hmm_horizons": HMM_HORIZON_MAP_CORRECTED,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
        },
        "4_Fix_HMM_No_Decay_and_Asymmetric": {
            "desc": "Premium: Fix HMM + TP Estático + Asimétrico BULL_TREND (TP=2.5x, SL=1.5x)",
            "linear_decay_pt": False,
            "hmm_params": {
                **HMM_TBM_PARAMS_CORRECTED,
                "1_BULL_TREND": {"sl": 1.5, "tp": 2.5},
                "1_BULL_TREND_WEAK": {"sl": 1.5, "tp": 2.5},
                "1_BULL_TREND_B": {"sl": 1.5, "tp": 2.5},
                "1_BULL_GRIND": {"sl": 1.5, "tp": 2.5},
            },
            "hmm_horizons": HMM_HORIZON_MAP_CORRECTED,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
        },
        "5_Fix_HMM_NoDecay_Asymmetric_Meta55": {
            "desc": "Combo Meta: Fix HMM + TP Estático + Asimétrico + MetaLabeler >= 0.55",
            "linear_decay_pt": False,
            "hmm_params": {
                **HMM_TBM_PARAMS_CORRECTED,
                "1_BULL_TREND": {"sl": 1.5, "tp": 2.5},
                "1_BULL_TREND_WEAK": {"sl": 1.5, "tp": 2.5},
                "1_BULL_TREND_B": {"sl": 1.5, "tp": 2.5},
                "1_BULL_GRIND": {"sl": 1.5, "tp": 2.5},
            },
            "hmm_horizons": HMM_HORIZON_MAP_CORRECTED,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
            "min_meta_prob": 0.55,
        },
        "6_Fix_HMM_NoDecay_Asymmetric_Meta55_NoVolBull": {
            "desc": "Combo Superior: Fix HMM + TP Estático + Asimétrico + Meta >= 0.55 + Excluir VOLATILE_BULL",
            "linear_decay_pt": False,
            "hmm_params": {
                **HMM_TBM_PARAMS_CORRECTED,
                "1_BULL_TREND": {"sl": 1.5, "tp": 2.5},
                "1_BULL_TREND_WEAK": {"sl": 1.5, "tp": 2.5},
                "1_BULL_TREND_B": {"sl": 1.5, "tp": 2.5},
                "1_BULL_GRIND": {"sl": 1.5, "tp": 2.5},
            },
            "hmm_horizons": HMM_HORIZON_MAP_CORRECTED,
            "hmm_fallback_p": _HMM_TBM_FALLBACK,
            "hmm_fallback_h": _HMM_HORIZON_FALLBACK,
            "min_meta_prob": 0.55,
            "exclude_regimes": ["1_VOLATILE_BULL", "1_VOLATILE_BULL_B"],
        }
    }

    scenario_metrics = {}

    for name, config in scenarios.items():
        all_sim_trades = []

        for w in windows:
            if f"W{w}" not in all_results:
                continue

            w_data = all_results[f"W{w}"]
            close = w_data["close"]
            signal_times = w_data["signal_times"]
            hmm_semantic = w_data["hmm_semantic"]
            conf_scaler = w_data["conf_scaler"]
            trades_df = w_data["trades_df"]
            meta_v2_prob_series = w_data["meta_v2_prob"]

            # Resolver params base usando el diccionario del escenario
            _pt_base = hmm_semantic.map(lambda r: config["hmm_params"].get(r, config["hmm_fallback_p"])["tp"])
            _sl_base = hmm_semantic.map(lambda r: config["hmm_params"].get(r, config["hmm_fallback_p"])["sl"])

            # Aplicar conf scaler
            _pt = _pt_base * conf_scaler
            _sl = _sl_base * conf_scaler

            # Configuración de horizontes dinámicos
            _vb_h = 72
            _dyn_min = 24
            
            try:
                _dyn_max = int(hmm_semantic.dropna().map(
                    lambda r: config["hmm_horizons"].get(r, config["hmm_fallback_h"])
                ).mode().iloc[0])
            except Exception:
                _dyn_max = config["hmm_fallback_h"]

            # Simular TBM usando apply_triple_barrier nativo
            tbm_res = apply_triple_barrier(
                price_series=close,
                event_times=signal_times,
                sides=pd.Series(1, index=signal_times),
                pt_sl_multiplier=[_pt, _sl],
                vertical_barrier_hours=_vb_h,
                min_return=0.003, 
                dynamic_barrier=True,
                dynamic_horizon_min_h=_dyn_min,
                dynamic_horizon_max_h=_dyn_max,
                linear_decay_pt=config["linear_decay_pt"],
                pt_decay_fraction=0.75,
            )

            # Compilar retorno
            COST_RT = 0.0015
            sim_trades = []
            for t in signal_times:
                if t not in tbm_res.index:
                    continue
                row = tbm_res.loc[t]
                ret_raw_tbm = float(row["ret"])
                if pd.isna(ret_raw_tbm):
                    continue

                ret_bruto = ret_raw_tbm - COST_RT
                original_mult = trades_df.loc[t, "kelly_fraction_used"]
                ret_compuesto = ret_bruto * original_mult

                sim_trades.append({
                    "timestamp": t,
                    "window": f"W{w}",
                    "ret_bruto": ret_bruto,
                    "ret_compuesto": ret_compuesto,
                    "is_win": ret_bruto > 0,
                    "hmm_regime": hmm_semantic.loc[t],
                    "meta_v2_prob": float(meta_v2_prob_series.loc[t]) if meta_v2_prob_series is not None and t in meta_v2_prob_series.index else np.nan,
                })

            sim_window_df = pd.DataFrame(sim_trades)
            if len(sim_window_df) > 0:
                all_sim_trades.append(sim_window_df)

        if not all_sim_trades:
            continue

        trades = pd.concat(all_sim_trades, ignore_index=True)
        
        # Aplicar filtros específicos del escenario
        trades_scenario = trades.copy()
        if config.get("min_meta_prob") is not None:
            trades_scenario = trades_scenario[trades_scenario["meta_v2_prob"] >= config["min_meta_prob"]]
        if config.get("exclude_regimes") is not None:
            trades_scenario = trades_scenario[~trades_scenario["hmm_regime"].isin(config["exclude_regimes"])]

        if len(trades_scenario) == 0:
            print(f"[WARN] Escenario {name} quedó con 0 trades.")
            continue

        # Calcular Métricas del Escenario
        n_total = len(trades_scenario)
        wr = (trades_scenario['ret_bruto'] > 0).mean() * 100
        ev_bruto = trades_scenario['ret_bruto'].mean() * 100
        std_bruto = trades_scenario['ret_bruto'].std() * 100

        gains = trades_scenario[trades_scenario['ret_bruto'] > 0]['ret_bruto'] * 100
        losses = trades_scenario[trades_scenario['ret_bruto'] < 0]['ret_bruto'] * 100

        avg_gain = gains.mean() if len(gains) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        rr_bruto = avg_gain / abs(avg_loss) if avg_loss != 0 else float('inf')

        # Sharpe anualizado
        days = (trades_scenario['timestamp'].max() - trades_scenario['timestamp'].min()).days
        if days <= 0:
            days = 270
        n_per_year = n_total / (days / 365.25)
        sharpe = (ev_bruto / std_bruto) * (n_per_year ** 0.5) if std_bruto > 1e-10 else 0.0

        ret_acum_bruto = (1 + trades_scenario['ret_bruto']).prod() - 1
        ret_acum_compuesto = (1 + trades_scenario['ret_compuesto']).prod() - 1

        scenario_metrics[name] = {
            "trades": n_total,
            "wr": wr,
            "ev": ev_bruto,
            "gain": avg_gain,
            "loss": avg_loss,
            "rr": rr_bruto,
            "sharpe": sharpe,
            "ret_comp": ret_acum_compuesto * 100,
            "ret_flat": ret_acum_bruto * 100
        }

    # Imprimir Tabla Comparativa en formato Markdown
    print("\n\n=========================================================================")
    print("  RESULTADOS COMPARATIVOS DE LOS CORRECTORES Y MEJORAS")
    print("=========================================================================\n")
    print(f"| Escenario | Trades | Win Rate | Ganancia Med | Pérdida Med | R:R Ratio | Sharpe | Ret Acum Comp |")
    print(f"|---|---|---|---|---|---|---|---|")
    for s_name, metrics in scenario_metrics.items():
        print(f"| **{s_name}** | {metrics['trades']} | {metrics['wr']:.2f}% | {metrics['gain']:+.4f}% | {metrics['loss']:.4f}% | {metrics['rr']:.4f} | {metrics['sharpe']:.3f} | {metrics['ret_comp']:.4f}% |")

if __name__ == "__main__":
    run_simulation()

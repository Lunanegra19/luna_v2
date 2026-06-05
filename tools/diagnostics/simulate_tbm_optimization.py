import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

# Configurar rutas y paths del sistema
ROOT = Path("g:/Mi unidad/ia/luna_v2")
sys.path.insert(0, str(ROOT))

from luna.features.tbm import apply_triple_barrier, get_daily_volatility
from luna.models.predict_oos import HMM_TBM_PARAMS, _HMM_TBM_FALLBACK, HMM_HORIZON_MAP, _HMM_HORIZON_FALLBACK

def run_simulation():
    print("=========================================================================")
    print("  SIMULACIÓN DE OPTIMIZACIÓN TBM Y ESTUDIO DE TRADES - SEMILLA 42")
    print("  [FIX-BUGS-PRINTS] Análisis profundo de asimetría P&L y exit barriers")
    print("=========================================================================\n")

    # 1. Cargar trades y holdouts
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

        # Cargar trades y holdout
        trades_df = pd.read_parquet(trades_path)
        holdout_df = pd.read_parquet(holdout_path)

        # Alinear zonas horarias a UTC
        if trades_df.index.name == 'timestamp' or 'timestamp' not in trades_df.columns:
            trades_df = trades_df.reset_index()
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], utc=True)
        trades_df = trades_df.set_index('timestamp').sort_index()

        if holdout_df.index.name != 'timestamp' and 'timestamp' in holdout_df.columns:
            holdout_df = holdout_df.set_index('timestamp')
        holdout_df.index = pd.to_datetime(holdout_df.index, utc=True)
        holdout_df = holdout_df.sort_index()

        # Extraer variables clave
        close_series = holdout_df['close']
        signal_times = trades_df.index

        print(f"--- VENTANA W{w} ---")
        print(f"  Signals cargadas: {len(signal_times)} | Precios de holdout: {len(close_series)} barras ({close_series.index.min()} -> {close_series.index.max()})")

        # Reconstruir las variables del trade que existían en el momento del entry
        # Mapeamos hmm_regime, xgb_prob_cal, meta_v2_prob
        hmm_semantic = trades_df['hmm_regime']
        xgb_prob_cal = trades_df['xgb_prob_cal']
        meta_v2_prob = trades_df['meta_v2_prob']

        # El conf_scaler utiliza xgb_prob_cal si está disponible, sino xgb_prob, sino meta_v2_prob.
        # En la run real, predict_oos.py prioriza xgb_prob_cal si existe.
        prob_series = xgb_prob_cal.fillna(trades_df['xgb_prob']).fillna(meta_v2_prob).fillna(0.5).clip(0.5, 1.0)
        conf_scaler = 0.7 + ((prob_series - 0.5) / 0.5) * (1.3 - 0.7)

        # Mediana del conf_scaler
        print(f"  Conf Scaler - Mín: {conf_scaler.min():.4f} | Máx: {conf_scaler.max():.4f} | Mediana: {conf_scaler.median():.4f}")

        # Guardar en diccionario para simular
        window_data = {
            "close": close_series,
            "signal_times": signal_times,
            "hmm_semantic": hmm_semantic,
            "prob_series": prob_series,
            "conf_scaler": conf_scaler,
            "trades_df": trades_df,
            "xgb_prob_cal": xgb_prob_cal,
            "meta_v2_prob": meta_v2_prob
        }
        all_results[f"W{w}"] = window_data

    # Simular escenarios
    scenarios = {
        "0_Baseline": {
            "desc": "Línea Base: Linear Decay en TP activo + Conf Scaler en SL activo",
            "linear_decay_pt": True,
            "pt_decay_frac": 0.75,
            "scale_sl": True,
            "hmm_params_override": None
        },
        "1_No_Linear_Decay": {
            "desc": "Desactivar Linear Profit Target Decay (TP estático)",
            "linear_decay_pt": False,
            "pt_decay_frac": 0.75,
            "scale_sl": True,
            "hmm_params_override": None
        },
        "2_Static_SL": {
            "desc": "Desactivar Conf Scaler en Stop Loss (SL no se ensancha en alta confianza)",
            "linear_decay_pt": True,
            "pt_decay_frac": 0.75,
            "scale_sl": False,
            "hmm_params_override": None
        },
        "3_Symmetric_TP_SL": {
            "desc": "Hacer TP y SL simétricos por régimen (tp: 1.5, sl: 1.5 para BULL_TREND / VOLATILE_BULL)",
            "linear_decay_pt": True,
            "pt_decay_frac": 0.75,
            "scale_sl": True,
            "hmm_params_override": {
                "1_VOLATILE_BULL": {"sl": 1.5, "tp": 1.5},
                "1_BULL_TREND":    {"sl": 1.5, "tp": 1.5},
                "2_CALM_RANGE":    {"sl": 0.6, "tp": 0.6},
                "2_VOLATILE_RANGE":{"sl": 1.2, "tp": 1.2},
                "3_BEAR_CRASH":    {"sl": 1.5, "tp": 1.5},
            }
        },
        "4_NoDecay_StaticSL": {
            "desc": "Combinada: TP sin decay + SL sin conf scaling",
            "linear_decay_pt": False,
            "pt_decay_frac": 0.75,
            "scale_sl": False,
            "hmm_params_override": None
        },
        "5_Combined_Optimized": {
            "desc": "Combinada Premium: TP sin decay + SL sin scaling + Asymmetric positive R:R en VOLATILE_BULL",
            "linear_decay_pt": False,
            "pt_decay_frac": 0.75,
            "scale_sl": False,
            "hmm_params_override": {
                "1_VOLATILE_BULL": {"sl": 1.5, "tp": 2.5}, # TP=2.5x, SL=1.5x (R:R positivo!)
                "1_BULL_TREND":    {"sl": 1.5, "tp": 2.0},
                "2_CALM_RANGE":    {"sl": 0.6, "tp": 1.2},
                "2_VOLATILE_RANGE":{"sl": 1.2, "tp": 1.8},
                "3_BEAR_CRASH":    {"sl": 1.2, "tp": 1.5},
            }
        }
    }

    # Resultados para cada escenario
    scenario_metrics = {}

    for name, config in scenarios.items():
        print(f"\n=========================================================================")
        print(f" Simulado Escenario: {name} - {config['desc']}")
        print(f"=========================================================================")

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
            xgb_prob_cal_w = w_data["xgb_prob_cal"]
            meta_v2_prob_w = w_data["meta_v2_prob"]

            # Resolver params base
            params_dict = config["hmm_params_override"] if config["hmm_params_override"] is not None else HMM_TBM_PARAMS

            # Construir PT y SL Series
            _pt_base = hmm_semantic.map(lambda r: params_dict.get(r, _HMM_TBM_FALLBACK)["tp"])
            _sl_base = hmm_semantic.map(lambda r: params_dict.get(r, _HMM_TBM_FALLBACK)["sl"])

            # Aplicar conf scaler
            _pt = _pt_base * conf_scaler
            if config["scale_sl"]:
                _sl = _sl_base * conf_scaler
            else:
                _sl = _sl_base # No escalado! Queda estático.

            # Configuración de horizontes dinámicos de settings.yaml
            _vb_h = 72
            _dyn_min = 24
            
            # Obtener max_horizon (moda de la ventana)
            # En la run real de predict_oos, es la moda del HMM_Semantic
            try:
                _dyn_max = int(hmm_semantic.dropna().map(
                    lambda r: HMM_HORIZON_MAP.get(r, _HMM_HORIZON_FALLBACK)
                ).mode().iloc[0])
            except Exception:
                _dyn_max = _HMM_HORIZON_FALLBACK

            # Simular TBM usando apply_triple_barrier nativo
            tbm_res = apply_triple_barrier(
                price_series=close,
                event_times=signal_times,
                sides=pd.Series(1, index=signal_times),
                pt_sl_multiplier=[_pt, _sl],
                vertical_barrier_hours=_vb_h,
                min_return=0.003, # de settings.yaml xgboost.tbm_min_return
                dynamic_barrier=True,
                dynamic_horizon_min_h=_dyn_min,
                dynamic_horizon_max_h=_dyn_max,
                linear_decay_pt=config["linear_decay_pt"],
                pt_decay_fraction=config["pt_decay_frac"],
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
                # Para P&L compuesto o proporcional, use el multiplicador original
                original_mult = trades_df.loc[t, "kelly_fraction_used"]
                ret_compuesto = ret_bruto * original_mult

                sim_trades.append({
                    "timestamp": t,
                    "window": f"W{w}",
                    "direction": "long",
                    "ret_bruto": ret_bruto,
                    "ret_compuesto": ret_compuesto,
                    "is_win": ret_bruto > 0,
                    "hmm_regime": hmm_semantic.loc[t],
                    "meta_v2_prob": meta_v2_prob_w.loc[t] if meta_v2_prob_w is not None and t in meta_v2_prob_w.index else np.nan,
                    "xgb_prob_cal": xgb_prob_cal_w.loc[t] if xgb_prob_cal_w is not None and t in xgb_prob_cal_w.index else np.nan
                })

            sim_window_df = pd.DataFrame(sim_trades)
            if len(sim_window_df) > 0:
                all_sim_trades.append(sim_window_df)

        if not all_sim_trades:
            print("[WARN] No se generaron trades simulados en este escenario.")
            continue

        trades = pd.concat(all_sim_trades, ignore_index=True)
        
        # Calcular Métricas del Escenario
        n_total = len(trades)
        wr = (trades['ret_bruto'] > 0).mean() * 100
        ev_bruto = trades['ret_bruto'].mean() * 100
        std_bruto = trades['ret_bruto'].std() * 100

        gains = trades[trades['ret_bruto'] > 0]['ret_bruto'] * 100
        losses = trades[trades['ret_bruto'] < 0]['ret_bruto'] * 100

        avg_gain = gains.mean() if len(gains) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        rr_bruto = avg_gain / abs(avg_loss) if avg_loss != 0 else float('inf')

        # Sharpe anualizado simplificado
        # Usamos duración de muestra real en días
        days = (trades['timestamp'].max() - trades['timestamp'].min()).days
        if days <= 0:
            days = 270
        n_per_year = n_total / (days / 365.25)
        sharpe = (ev_bruto / std_bruto) * (n_per_year ** 0.5) if std_bruto > 1e-10 else 0.0

        # Retorno compuesto acumulado bruto (flat risk del 100%) y escalado
        ret_acum_bruto = (1 + trades['ret_bruto']).prod() - 1
        ret_acum_compuesto = (1 + trades['ret_compuesto']).prod() - 1

        print(f"\n  --- RESULTADOS DEL ESCENARIO ---")
        print(f"  Total Trades        : {n_total}")
        print(f"  Win Rate            : {wr:.2f}% (Wins: {len(gains)} / Losses: {len(losses)})")
        print(f"  EV Medio Bruto (%)  : {ev_bruto:.4f}%")
        print(f"  Ganancia Promedio   : {avg_gain:.4f}%")
        print(f"  Pérdida Promedio    : {avg_loss:.4f}%")
        print(f"  Ratio R:R Bruto Real: {rr_bruto:.4f}")
        print(f"  Sharpe Anual        : {sharpe:.4f}")
        print(f"  Retorno Acum Comp.  : {ret_acum_compuesto*100:.4f}% (Flat: {ret_acum_bruto*100:.4f}%)")

        # Desglose por Ventana
        print(f"\n  --- DESGLOSE POR VENTANA ---")
        for w in sorted(trades['window'].unique()):
            w_df = trades[trades['window'] == w]
            w_g = w_df[w_df['ret_bruto'] > 0]['ret_bruto'] * 100
            w_l = w_df[w_df['ret_bruto'] < 0]['ret_bruto'] * 100
            w_wr = (w_df['ret_bruto'] > 0).mean() * 100
            w_ev = w_df['ret_bruto'].mean() * 100
            w_std = w_df['ret_bruto'].std() * 100
            w_days = (w_df['timestamp'].max() - w_df['timestamp'].min()).days
            if w_days <= 0: w_days = 90
            w_n_per_year = len(w_df) / (w_days / 365.25)
            w_sharpe = (w_ev / w_std) * (w_n_per_year ** 0.5) if w_std > 1e-10 else 0.0
            w_ret_comp = ((1 + w_df['ret_compuesto']).prod() - 1) * 100
            
            print(f"    {w} | Trades: {len(w_df):2d} | WR: {w_wr:.1f}% | Sharpe: {w_sharpe:.3f} | Ret Medio: {w_ev:.4f}% | Ret Acum: {w_ret_comp:.3f}% | Gain/Loss: {w_g.mean():.2f}%/{w_l.mean():.2f}%")

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

    # 4. Imprimir Tabla Comparativa en formato Markdown
    print("\n\n=========================================================================")
    print("  TABLA COMPARATIVA GLOBAL DE ESCENARIOS (SEMILLA 42)")
    print("=========================================================================\n")
    print(f"| Escenario | Trades | Win Rate | Ganancia Med | Pérdida Med | R:R Ratio | Sharpe | Ret Acum Comp |")
    print(f"|---|---|---|---|---|---|---|---|")
    for s_name, metrics in scenario_metrics.items():
        print(f"| **{s_name}** | {metrics['trades']} | {metrics['wr']:.2f}% | {metrics['gain']:+.4f}% | {metrics['loss']:.4f}% | {metrics['rr']:.4f} | {metrics['sharpe']:.3f} | {metrics['ret_comp']:.4f}% |")

    print("\n[FIX-BUGS-PRINTS] Simulación finalizada de manera profesional y exhaustiva.")

if __name__ == "__main__":
    run_simulation()

# INFORME WFB MASIVO Y EXHAUSTIVO (NIVEL OMEGA) - MEJORADO
> Generado el: 2026-06-08T06:19:02.881939+00:00

## 0. ANÁLISIS DE TELEMETRÍA Y DIAGNÓSTICO PROFUNDO
### 0.1 Distribución de Rendimiento por Semilla
- **Total semillas analizadas en esta ventana**: 10
- **Trades medios**: 19.30 (Max: 31, Min: 7)
- **Win Rate medio**: 52.68% (Max: 77.78%)
- **Sharpe medio**: 0.4822

### 0.2 Cuellos de Botella Estadísticos (Razones de Fallo)
- **DSR Fallido**: 9 semillas (90.0%)
- **PBO Fallido**: 10 semillas (100.0%)
- **Binomial Fallido**: 7 semillas (70.0%)
- **Trades Insuficientes**: 8 semillas (80.0%)

### 0.3 Diagnóstico Estructural del Pipeline
El análisis del embudo muestra que:
1. **Degradación DSR**: El 100% casi absoluto de semillas falla el DSR antes del ajuste por N=20.
2. **Inanición en Embargo**: Las señales generadas por XGBoost y MetaLabeler son fuertemente censuradas en la etapa de Embargo.
3. **Sensibilidad PBO**: Al requerirse 32 trades mínimos, cualquier semilla por debajo de ese límite fuerza un fallback conservador de PBO=0.50.

## PARTE 1: VEREDICTOS JSON CRUDOS

### Archivo: `2026-06-08_T0700_WFB_20260608_065705_23440_seed888_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 31,
        "win_rate": 0.4838709677419355,
        "total_return_pct": -0.07031104584040992,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.5058500250504482,
        "sharpe_crudo": -0.22418732745284417,
        "calmar_ratio": -44.31893176845964
    },
    "statistical_audit": {
        "dsr": 1.3831995977269267e-45,
        "binomial_p_value": 0.639949934091419,
        "estimated_pbo": 0.5,
        "skewness": 0.7599819812707242,
        "kurtosis": 1.7318660945603295,
        "n_obs_dsr": 31,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": true,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:00:55.311306+00:00",
    "summary": {
        "total_trades": 31,
        "win_rate_pct": 48.39,
        "max_drawdown_pct": 0.51,
        "total_return_pct": -0.0703,
        "sharpe_crudo": -0.2242,
        "calmar_ratio": -44.32,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.63995,
        "pass_trades": true,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_065705_23440_seed888_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 0.0,
    "wfv_results": {
        "W3": {
            "n_trades": 22,
            "win_rate": 0.3182,
            "start_date": "2025-07-03 00:00:00+00:00",
            "end_date": "2025-09-20 20:00:00+00:00",
            "trivial": false
        },
        "W4": {
            "n_trades": 5,
            "win_rate": 0.8,
            "start_date": "2025-10-01 00:00:00+00:00",
            "end_date": "2025-10-26 22:00:00+00:00",
            "trivial": false
        },
        "W5": {
            "n_trades": 4,
            "win_rate": 1.0,
            "start_date": "2026-01-14 06:00:00+00:00",
            "end_date": "2026-03-25 02:00:00+00:00",
            "trivial": true
        }
    },
    "latest_window_blind": false,
    "latest_window_blind_id": null,
    "n_windows_with_trades": 3,
    "signal_pipeline": {
        "raw_oos_bars": 7227,
        "after_xgb": 2541,
        "after_lgbm": 2541,
        "after_ood": 2541,
        "after_cvd": 2541,
        "after_hmm": 2333,
        "after_session_gate": 797,
        "after_macro_gate": 797,
        "after_meta": 1233,
        "after_cash_shield": 1233,
        "after_momentum": 1098,
        "filter_fallback_level": 0,
        "after_embargo": 45,
        "run_id": "WFB_seed888_funnel",
        "n_windows_accumulated": 3
    }
}
```

### Archivo: `2026-06-08_T0708_WFB_20260608_070305_36576_seed999_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 21,
        "win_rate": 0.6190476190476191,
        "total_return_pct": 0.03136314115854688,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.23954058614855137,
        "sharpe_crudo": 0.07719046286627401,
        "calmar_ratio": 32.224377550117644
    },
    "statistical_audit": {
        "dsr": 5.758246947594158e-28,
        "binomial_p_value": 0.19165515899658203,
        "estimated_pbo": 0.5,
        "skewness": -0.832928323496138,
        "kurtosis": 0.7367324028242903,
        "n_obs_dsr": 21,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": true,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:08:38.954015+00:00",
    "summary": {
        "total_trades": 21,
        "win_rate_pct": 61.9,
        "max_drawdown_pct": 0.24,
        "total_return_pct": 0.0314,
        "sharpe_crudo": 0.0772,
        "calmar_ratio": 32.22,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.191655,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": true,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_070305_36576_seed999_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 2.7e-05,
    "wfv_results": {
        "W1": {
            "n_trades": 1,
            "win_rate": 0.0,
            "start_date": "2025-01-06 22:00:00+00:00",
            "end_date": "2025-01-06 22:00:00+00:00",
            "trivial": true
        },
        "W2": {
            "n_trades": 6,
            "win_rate": 0.6667,
            "start_date": "2025-04-01 00:00:00+00:00",
            "end_date": "2025-06-20 20:00:00+00:00",
            "trivial": false
        },
        "W3": {
            "n_trades": 8,
            "win_rate": 0.5,
            "start_date": "2025-07-02 06:00:00+00:00",
            "end_date": "2025-08-11 07:00:00+00:00",
            "trivial": false
        },
        "W4": {
            "n_trades": 5,
            "win_rate": 0.8,
            "start_date": "2025-10-01 00:00:00+00:00",
            "end_date": "2025-10-09 21:00:00+00:00",
            "trivial": false
        },
        "W5": {
            "n_trades": 1,
            "win_rate": 1.0,
            "start_date": "2026-02-02 10:00:00+00:00",
            "end_date": "2026-02-02 10:00:00+00:00",
            "trivial": true
        }
    },
    "latest_window_blind": false,
    "latest_window_blind_id": null,
    "n_windows_with_trades": 5,
    "signal_pipeline": {
        "raw_oos_bars": 12005,
        "after_xgb": 3388,
        "after_lgbm": 3388,
        "after_ood": 3388,
        "after_cvd": 3388,
        "after_hmm": 2763,
        "after_session_gate": 634,
        "after_macro_gate": 634,
        "after_meta": 874,
        "after_cash_shield": 874,
        "after_momentum": 768,
        "filter_fallback_level": 0,
        "after_embargo": 41,
        "run_id": "WFB_seed999_funnel",
        "n_windows_accumulated": 5
    }
}
```

### Archivo: `2026-06-08_T0713_WFB_20260608_071105_11256_seed1111_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 7,
        "win_rate": 0.42857142857142855,
        "total_return_pct": 0.2019081220618446,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.15254450339969858,
        "sharpe_crudo": 0.5454856521755733,
        "calmar_ratio": 357.59115537993955
    },
    "statistical_audit": {
        "dsr": 2.9493364959005454e-12,
        "binomial_p_value": 0.7734375,
        "estimated_pbo": 0.5,
        "skewness": 0.4561123988571888,
        "kurtosis": -1.2689435587898055,
        "n_obs_dsr": 7,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:13:55.486514+00:00",
    "summary": {
        "total_trades": 7,
        "win_rate_pct": 42.86,
        "max_drawdown_pct": 0.15,
        "total_return_pct": 0.2019,
        "sharpe_crudo": 0.5455,
        "calmar_ratio": 357.59,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.773438,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_071105_11256_seed1111_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 0.006407,
    "signal_pipeline": {
        "raw_oos_bars": 4802,
        "after_xgb": 755,
        "after_lgbm": 755,
        "after_ood": 755,
        "after_cvd": 755,
        "after_hmm": 730,
        "after_session_gate": 557,
        "after_macro_gate": 557,
        "after_meta": 487,
        "after_cash_shield": 487,
        "after_momentum": 345,
        "filter_fallback_level": 0,
        "after_embargo": 17,
        "run_id": "WFB_seed1111_funnel",
        "n_windows_accumulated": 2
    }
}
```

### Archivo: `2026-06-08_T0732_WFB_20260608_072906_33248_seed4444_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 9,
        "win_rate": 0.7777777777777778,
        "total_return_pct": 0.5410477546776926,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.019009438738580492,
        "sharpe_crudo": 3.2516385677651147,
        "calmar_ratio": 17105.389656590814
    },
    "statistical_audit": {
        "dsr": 1.0,
        "binomial_p_value": 0.08984375,
        "estimated_pbo": 0.5,
        "skewness": 0.15821533417251676,
        "kurtosis": -0.9890280963842271,
        "n_obs_dsr": 9,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": true,
        "pass_binomial": true,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:32:14.340763+00:00",
    "summary": {
        "total_trades": 9,
        "win_rate_pct": 77.78,
        "max_drawdown_pct": 0.02,
        "total_return_pct": 0.541,
        "sharpe_crudo": 3.2516,
        "calmar_ratio": 17105.39,
        "dsr": 1.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.089844,
        "pass_trades": false,
        "pass_dsr": true,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": true,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_072906_33248_seed4444_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 1.0,
    "signal_pipeline": {
        "raw_oos_bars": 4802,
        "after_xgb": 2481,
        "after_lgbm": 2481,
        "after_ood": 2481,
        "after_cvd": 2481,
        "after_hmm": 2302,
        "after_session_gate": 1025,
        "after_macro_gate": 1025,
        "after_meta": 697,
        "after_cash_shield": 697,
        "after_momentum": 538,
        "filter_fallback_level": 0,
        "after_embargo": 33,
        "run_id": "WFB_seed4444_funnel",
        "n_windows_accumulated": 2
    }
}
```

### Archivo: `2026-06-08_T0739_WFB_20260608_073506_27972_seed5555_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 17,
        "win_rate": 0.6470588235294118,
        "total_return_pct": 0.3379116376480251,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.10028372551608139,
        "sharpe_crudo": 1.3983757391607012,
        "calmar_ratio": 1394.4194154777979
    },
    "statistical_audit": {
        "dsr": 1.4804959967373978e-302,
        "binomial_p_value": 0.1661529541015625,
        "estimated_pbo": 0.5,
        "skewness": 0.8296168912849567,
        "kurtosis": 1.359809422059116,
        "n_obs_dsr": 17,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": true,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:39:57.474673+00:00",
    "summary": {
        "total_trades": 17,
        "win_rate_pct": 64.71,
        "max_drawdown_pct": 0.1,
        "total_return_pct": 0.3379,
        "sharpe_crudo": 1.3984,
        "calmar_ratio": 1394.42,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.166153,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": true,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_073506_27972_seed5555_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 1.0,
    "signal_pipeline": {
        "raw_oos_bars": 7227,
        "after_xgb": 1533,
        "after_lgbm": 1533,
        "after_ood": 1533,
        "after_cvd": 1533,
        "after_hmm": 1474,
        "after_session_gate": 917,
        "after_macro_gate": 917,
        "after_meta": 921,
        "after_cash_shield": 921,
        "after_momentum": 792,
        "filter_fallback_level": 0,
        "after_embargo": 37,
        "run_id": "WFB_seed5555_funnel",
        "n_windows_accumulated": 3
    }
}
```

### Archivo: `2026-06-08_T0746_WFB_20260608_074306_26376_seed6666_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 18,
        "win_rate": 0.6111111111111112,
        "total_return_pct": 0.1075792551758381,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.20878938448977966,
        "sharpe_crudo": 0.3152461883239409,
        "calmar_ratio": 150.9876515486219
    },
    "statistical_audit": {
        "dsr": 2.3437866281712975e-20,
        "binomial_p_value": 0.2403411865234375,
        "estimated_pbo": 0.5,
        "skewness": -0.2063201412673689,
        "kurtosis": 0.4081552120364549,
        "n_obs_dsr": 18,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:46:50.764869+00:00",
    "summary": {
        "total_trades": 18,
        "win_rate_pct": 61.11,
        "max_drawdown_pct": 0.21,
        "total_return_pct": 0.1076,
        "sharpe_crudo": 0.3152,
        "calmar_ratio": 150.99,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.240341,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_074306_26376_seed6666_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 0.001053,
    "signal_pipeline": {
        "raw_oos_bars": 7179,
        "after_xgb": 1630,
        "after_lgbm": 1630,
        "after_ood": 1630,
        "after_cvd": 1630,
        "after_hmm": 1544,
        "after_session_gate": 959,
        "after_macro_gate": 959,
        "after_meta": 748,
        "after_cash_shield": 748,
        "after_momentum": 563,
        "filter_fallback_level": 0,
        "after_embargo": 36,
        "run_id": "WFB_seed6666_funnel",
        "n_windows_accumulated": 3
    }
}
```

### Archivo: `2026-06-08_T0752_WFB_20260608_074906_33224_seed7777_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 22,
        "win_rate": 0.4090909090909091,
        "total_return_pct": -0.010196152483432641,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.2419306819322175,
        "sharpe_crudo": -0.0366672293654947,
        "calmar_ratio": -15.15608895599603
    },
    "statistical_audit": {
        "dsr": 8.681910302470943e-33,
        "binomial_p_value": 0.8568606376647949,
        "estimated_pbo": 0.5,
        "skewness": 0.804100396196108,
        "kurtosis": 0.9547086951516937,
        "n_obs_dsr": 22,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T05:52:00.242437+00:00",
    "summary": {
        "total_trades": 22,
        "win_rate_pct": 40.91,
        "max_drawdown_pct": 0.24,
        "total_return_pct": -0.0102,
        "sharpe_crudo": -0.0367,
        "calmar_ratio": -15.16,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.856861,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_074906_33224_seed7777_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 3e-06,
    "wfv_results": {
        "W3": {
            "n_trades": 20,
            "win_rate": 0.4,
            "start_date": "2025-07-03 01:00:00+00:00",
            "end_date": "2025-09-21 01:00:00+00:00",
            "trivial": false
        },
        "W4": {
            "n_trades": 2,
            "win_rate": 0.5,
            "start_date": "2025-10-01 12:00:00+00:00",
            "end_date": "2025-10-26 22:00:00+00:00",
            "trivial": true
        }
    },
    "latest_window_blind": false,
    "latest_window_blind_id": null,
    "n_windows_with_trades": 2,
    "signal_pipeline": {
        "raw_oos_bars": 4850,
        "after_xgb": 1755,
        "after_lgbm": 1755,
        "after_ood": 1755,
        "after_cvd": 1755,
        "after_hmm": 1743,
        "after_session_gate": 195,
        "after_macro_gate": 195,
        "after_meta": 951,
        "after_cash_shield": 951,
        "after_momentum": 951,
        "filter_fallback_level": 0,
        "after_embargo": 39,
        "run_id": "WFB_seed7777_funnel",
        "n_windows_accumulated": 2
    }
}
```

### Archivo: `2026-06-08_T0800_WFB_20260608_075506_28952_seed8888_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 17,
        "win_rate": 0.35294117647058826,
        "total_return_pct": 0.025378082137916813,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.17168558806666404,
        "sharpe_crudo": 0.13192605527047213,
        "calmar_ratio": 76.84165966175703
    },
    "statistical_audit": {
        "dsr": 4.571333872405865e-30,
        "binomial_p_value": 0.9282684326171875,
        "estimated_pbo": 0.5,
        "skewness": 1.8732028465665251,
        "kurtosis": 3.035828012585073,
        "n_obs_dsr": 17,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T06:00:29.491330+00:00",
    "summary": {
        "total_trades": 17,
        "win_rate_pct": 35.29,
        "max_drawdown_pct": 0.17,
        "total_return_pct": 0.0254,
        "sharpe_crudo": 0.1319,
        "calmar_ratio": 76.84,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.928268,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_075506_28952_seed8888_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 8.6e-05,
    "signal_pipeline": {
        "status": "zero_signals",
        "n_trades": 0,
        "window_id": "W5",
        "seed": "8888",
        "reason": "0 senales pasaron todos los filtros (XGBoost + MetaLabeler + HMM + Embargo)",
        "after_xgb": 0,
        "after_meta": 0,
        "after_hmm": 0,
        "after_embargo": 0,
        "disabled_agents": "none"
    }
}
```

### Archivo: `2026-06-08_T0807_WFB_20260608_080306_34000_seed9999_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 21,
        "win_rate": 0.5714285714285714,
        "total_return_pct": 0.04960977564658631,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.18448480427515634,
        "sharpe_crudo": 0.27162491584946163,
        "calmar_ratio": 147.2343030726461
    },
    "statistical_audit": {
        "dsr": 2.3073596466040213e-19,
        "binomial_p_value": 0.33181190490722656,
        "estimated_pbo": 0.5,
        "skewness": -1.1948031102566412,
        "kurtosis": 2.1898620055775044,
        "n_obs_dsr": 21,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": false,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T06:07:02.206286+00:00",
    "summary": {
        "total_trades": 21,
        "win_rate_pct": 57.14,
        "max_drawdown_pct": 0.18,
        "total_return_pct": 0.0496,
        "sharpe_crudo": 0.2716,
        "calmar_ratio": 147.23,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.331812,
        "pass_trades": false,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_080306_34000_seed9999_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 0.000471,
    "wfv_results": {
        "W3": {
            "n_trades": 12,
            "win_rate": 0.4167,
            "start_date": "2025-07-23 01:00:00+00:00",
            "end_date": "2025-09-20 02:00:00+00:00",
            "trivial": false
        },
        "W4": {
            "n_trades": 3,
            "win_rate": 1.0,
            "start_date": "2025-10-01 15:00:00+00:00",
            "end_date": "2025-10-08 00:00:00+00:00",
            "trivial": true
        },
        "W5": {
            "n_trades": 6,
            "win_rate": 0.6667,
            "start_date": "2026-01-14 06:00:00+00:00",
            "end_date": "2026-03-22 01:00:00+00:00",
            "trivial": false
        }
    },
    "latest_window_blind": false,
    "latest_window_blind_id": null,
    "n_windows_with_trades": 3,
    "signal_pipeline": {
        "raw_oos_bars": 7227,
        "after_xgb": 2054,
        "after_lgbm": 2054,
        "after_ood": 2054,
        "after_cvd": 2054,
        "after_hmm": 1998,
        "after_session_gate": 1039,
        "after_macro_gate": 1039,
        "after_meta": 981,
        "after_cash_shield": 981,
        "after_momentum": 788,
        "filter_fallback_level": 0,
        "after_embargo": 43,
        "run_id": "WFB_seed9999_funnel",
        "n_windows_accumulated": 3
    }
}
```

### Archivo: `2026-06-08_T0815_WFB_20260608_081506_36344_seed12345_FINAL_statistical_verdict.json`
```json
{
    "deploy_approved": false,
    "metrics": {
        "total_trades": 30,
        "win_rate": 0.36666666666666664,
        "total_return_pct": -0.3545924708532322,
        "total_return_is_capped": false,
        "max_drawdown_pct": 0.4735335369475746,
        "sharpe_crudo": -0.9084340393656251,
        "calmar_ratio": -191.84154204186783
    },
    "statistical_audit": {
        "dsr": 2.1767245980401934e-66,
        "binomial_p_value": 0.9506314266473055,
        "estimated_pbo": 0.5,
        "skewness": 0.26278758854706413,
        "kurtosis": 0.8284719938840119,
        "n_obs_dsr": 30,
        "n_trials_dsr": 100
    },
    "flags": {
        "pass_dsr": false,
        "pass_binomial": false,
        "pass_trades": true,
        "pass_dd": true,
        "pass_pbo": false
    },
    "sop_thresholds": {
        "min_dsr": 0.75,
        "max_pbo_pct": 45.0,
        "min_trades": 30,
        "max_drawdown_pct": 60.0
    },
    "timestamp": "2026-06-08T06:15:08.299141+00:00",
    "summary": {
        "total_trades": 30,
        "win_rate_pct": 36.67,
        "max_drawdown_pct": 0.47,
        "total_return_pct": -0.3546,
        "sharpe_crudo": -0.9084,
        "calmar_ratio": -191.84,
        "dsr": 0.0,
        "pbo_pct": 50.0,
        "binomial_p": 0.950631,
        "pass_trades": true,
        "pass_dsr": false,
        "pass_dd": true,
        "pass_pbo": false,
        "pass_binomial": false,
        "deploy_approved": false
    },
    "signal_filter_fallback_level": 0,
    "run_id": "WFB_20260608_081506_36344_seed12345_FINAL",
    "adjusted_dsr_threshold": 0.75,
    "n_seeds_correction": 20,
    "dsr_correction_factor": 1.7308,
    "dsr_adjusted": 0.0,
    "wfv_results": {
        "W1": {
            "n_trades": 8,
            "win_rate": 0.375,
            "start_date": "2025-01-01 00:00:00+00:00",
            "end_date": "2025-02-20 19:00:00+00:00",
            "trivial": false
        },
        "W3": {
            "n_trades": 22,
            "win_rate": 0.3636,
            "start_date": "2025-07-03 00:00:00+00:00",
            "end_date": "2025-09-20 23:00:00+00:00",
            "trivial": false
        }
    },
    "latest_window_blind": false,
    "latest_window_blind_id": null,
    "n_windows_with_trades": 2,
    "signal_pipeline": {
        "raw_oos_bars": 2425,
        "after_xgb": 1277,
        "after_lgbm": 1277,
        "after_ood": 1277,
        "after_cvd": 1277,
        "after_hmm": 1277,
        "after_session_gate": 1277,
        "after_macro_gate": 1277,
        "after_meta": 692,
        "after_cash_shield": 692,
        "after_momentum": 692,
        "filter_fallback_level": 0,
        "after_embargo": 26,
        "run_id": "WFB_seed12345_funnel",
        "n_windows_accumulated": 1
    }
}
```


> (Nota: Para mantener este documento manejable en git, se incluyen los 10 primeros veredictos como muestra representativa. El análisis inicial resume la totalidad del dataset.)

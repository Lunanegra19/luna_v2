from .core import test, _cfg
from datetime import datetime

@test("TEST-70  SOP: Coherencia TBM Multipliers (pt_mult_min > sl_mult_min > 0)", section="consistency")
def t70_tbm_multipliers():
    cfg = _cfg()
    try:
        pt = float(cfg.xgboost.pt_mult_min)
        sl = float(cfg.xgboost.sl_mult_min)
    except Exception as e:
        assert False, f"pt_mult_min o sl_mult_min ausentes en cfg.xgboost: {e}"
        
    assert sl > 0, f"sl_mult_min={sl} no es > 0"
    assert pt > sl, f"pt_mult_min={pt} no es mayor a sl_mult_min={sl} (Riesgo Asimetrico R7/Kelly)"
    return f"pt={pt}, sl={sl} OK"

@test("TEST-71  SOP: Coherencia Horizontes Dinamicos (min <= VBH <= max)", section="consistency")
def t71_horizontes():
    cfg = _cfg()
    try:
        h_min = int(cfg.xgboost.dynamic_horizon_min_h)
        vbh = int(cfg.xgboost.vertical_barrier_hours)
        h_max = int(cfg.xgboost.dynamic_horizon_max_h)
    except Exception as e:
        assert False, f"Atributos de horizonte ausentes en cfg.xgboost: {e}"
        
    assert h_min > 0, f"dynamic_horizon_min_h={h_min} invalido"
    assert h_min <= vbh, f"dynamic_horizon_min_h={h_min} > vertical_barrier_hours={vbh}"
    assert vbh <= h_max, f"vertical_barrier_hours={vbh} > dynamic_horizon_max_h={h_max}"
    return f"min={h_min}h, VBH={vbh}h, max={h_max}h OK"

@test("TEST-72  SOP: Umbrales de Riesgo y DSR sin fallback en settings.yaml", section="consistency")
def t72_risk_gates():
    cfg = _cfg()
    try:
        stat_cfg = cfg.stat
    except Exception:
        assert False, "bloque 'stat' ausente en settings.yaml"
        
    required = ["min_trades", "min_dsr", "max_pbo", "alpha_binomial", "max_drawdown"]
    missing = []
    
    for r in required:
        try:
            val = getattr(stat_cfg, r)
        except Exception:
            missing.append(r)
            
    assert not missing, f"Umbrales de riesgo obligatorios ausentes en cfg.stat: {missing} (Peligro de Fallback Silencioso)"
    return "Todos los Gates de Riesgo explicitos OK"

@test("TEST-73  SOP: Auto-descubrimiento _min <= _max", section="consistency")
def t73_dynamic_min_max():
    import yaml
    from pathlib import Path
    
    cfg_path = Path('config/settings.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_dict = yaml.safe_load(f)
        
    errors = []
    
    def scan_dict(d, path=""):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if k.endswith("_min"):
                base = k[:-4]
                max_key = base + "_max"
                if max_key in d:
                    try:
                        v_min = float(v)
                        v_max = float(d[max_key])
                        if v_min > v_max:
                            errors.append(f"{path}.{k}={v_min} > {path}.{max_key}={v_max}")
                    except (ValueError, TypeError):
                        pass
            if isinstance(v, dict):
                scan_dict(v, path=f"{path}.{k}" if path else k)
                
    scan_dict(cfg_dict)
    assert not errors, f"Se detectaron invariantes min/max invertidos: {errors}"
    return "Todos los sufijos _min son <= a sus _max OK"

@test("TEST-74  SOP: Orden cronologico de Temporal Splits y WFB", section="consistency")
def t74_chronology():
    cfg = _cfg()
    try:
        ts = cfg.temporal_splits if not isinstance(cfg.temporal_splits, dict) else cfg.temporal_splits
        # Parses YYYY-MM-DD
        fmt = "%Y-%m-%d"
        
        def get_d(obj, key):
            val = obj[key] if isinstance(obj, dict) else getattr(obj, key)
            return datetime.strptime(str(val)[:10], fmt)
            
        t_s = get_d(ts, 'train_start')
        t_e = get_d(ts, 'train_end')
        v_s = get_d(ts, 'validation_start')
        v_e = get_d(ts, 'validation_end')
        h_s = get_d(ts, 'holdout_start')
        h_e = get_d(ts, 'holdout_end')
        
        assert t_s < t_e, "train_start debe ser < train_end"
        assert t_e <= v_s, "train_end debe ser <= validation_start"
        assert v_s < v_e, "validation_start debe ser < validation_end"
        assert v_e <= h_s, "validation_end debe ser <= holdout_start"
        assert h_s < h_e, "holdout_start debe ser < holdout_end"
        
        # Validar ventanas WFB
        windows = cfg.wfb.windows if not isinstance(cfg.wfb, dict) else cfg.wfb['windows']
        for w in windows:
            w_id = w['id'] if isinstance(w, dict) else getattr(w, 'id', 'UNK')
            w_te = get_d(w, 'train_end')
            w_vs = get_d(w, 'val_start')
            w_ve = get_d(w, 'val_end')
            w_hs = get_d(w, 'holdout_start')
            w_he = get_d(w, 'holdout_end')
            assert w_te <= w_vs, f"WFB {w_id}: train_end > val_start"
            assert w_vs < w_ve, f"WFB {w_id}: val_start >= val_end"
            assert w_ve <= w_hs, f"WFB {w_id}: val_end > holdout_start"
            assert w_hs < w_he, f"WFB {w_id}: holdout_start >= holdout_end"
            
    except Exception as e:
        assert False, f"Error validando cronologia: {e}"
        
    return "Orden cronologico global y de ventanas WFB estricto OK"

@test("TEST-75  SOP: Umbrales internos de Sweep (XGBoost/MetaLabeler)", section="consistency")
def t75_sweeps():
    cfg = _cfg()
    try:
        meta_min = float(cfg.metalabeler.meta_sweep_min)
        meta_max = float(cfg.metalabeler.meta_sweep_max)
        xgb_min = float(cfg.xgboost.threshold_sweep_min)
        xgb_max = float(cfg.xgboost.threshold_sweep_max)
        m_prob = float(cfg.metalabeler.meta_v2_min_prob)
        m_thr = float(cfg.metalabeler.meta_filter_threshold)
        
        assert meta_min < meta_max, f"meta_sweep_min={meta_min} >= meta_sweep_max={meta_max}"
        assert xgb_min < xgb_max, f"xgb_sweep_min={xgb_min} >= xgb_sweep_max={xgb_max}"
        assert m_prob <= m_thr, f"meta_v2_min_prob={m_prob} > meta_filter_threshold={m_thr}"
    except Exception as e:
        assert False, f"Umbrales faltantes o error: {e}"
    return "Rangos de Sweep validos OK"

@test("TEST-76  SOP: Integridad Estructural del SFI (Feature Selection)", section="consistency")
def t76_sfi_integrity():
    cfg = _cfg()
    try:
        c_min = int(cfg.features.sfi_n_clusters_min)
        c_max = int(cfg.features.sfi_n_clusters_max)
        top_n = int(cfg.features.sfi_top_n)
        macro_s = int(cfg.features.sfi_macro_min_slots)
        onch_s = int(cfg.features.sfi_onchain_min_slots)
        cal_s = int(cfg.features.sfi_calendar_min_slots)
        
        assert c_min <= c_max, f"sfi_n_clusters_min={c_min} > sfi_n_clusters_max={c_max}"
        sum_slots = macro_s + onch_s + cal_s
        assert top_n >= sum_slots, f"sfi_top_n={top_n} no puede cubrir suma de slots minimos={sum_slots}"
    except Exception as e:
        assert False, f"SFI params error: {e}"
    return "Integridad SFI OK"

@test("TEST-77  SOP: Arbol de Drawdowns y Position Sizing", section="consistency")
def t77_drawdown_tree():
    cfg = _cfg()
    try:
        kf = float(cfg.kelly_sizer.kelly_fraction)
        d_3q = float(cfg.position_sizer.dd_three_quarter)
        d_half = float(cfg.position_sizer.dd_half_size)
        d_kill = float(cfg.position_sizer.dd_kill_switch)
        d_max = float(cfg.stat.max_drawdown)
        
        assert kf <= 1.0, f"kelly_fraction={kf} > 1.0 (Full Kelly/Excesivo Riesgo)"
        assert d_3q < d_half, f"dd_three_quarter={d_3q} no es menor que dd_half_size={d_half}"
        assert d_half < d_kill, f"dd_half_size={d_half} no es menor que dd_kill_switch={d_kill}"
        assert d_kill <= d_max, f"dd_kill_switch={d_kill} > max_drawdown={d_max}"
    except Exception as e:
        assert False, f"Error en parametros de Drawdown: {e}"
    return "Cascada de Drawdown logica OK"

@test("TEST-78  SOP: Coherencia de Ventanas de Estabilidad (SFI)", section="consistency")
def t78_stability_windows():
    cfg = _cfg()
    try:
        t_win = float(cfg.features.stability_trend_window_years)
        r_win = float(cfg.features.stability_recent_window_years)
        m_min = float(cfg.features.stability_maturity_min_years)
        m_real = float(cfg.features.stability_min_real_years)
        d_thr = float(cfg.features.stability_dead_threshold_years)
        
        assert t_win >= r_win, f"stability_trend_window_years={t_win} < stability_recent_window_years={r_win}"
        assert m_min >= m_real, f"stability_maturity_min_years={m_min} < stability_min_real_years={m_real}"
        assert d_thr <= m_min, f"stability_dead_threshold_years={d_thr} > stability_maturity_min_years={m_min}"
    except Exception as e:
        assert False, f"Error validando SFI Stability: {e}"
    return "Ventanas SFI logicas OK"

@test("TEST-79  SOP: Coherencia HMM y Horizonte Mutuo", section="consistency")
def t79_hmm_coherence():
    cfg = _cfg()
    try:
        n_s = int(cfg.hmm.n_states)
        cands = list(cfg._roadmap.hmm.n_states_candidates)
        oos_w = int(cfg.hmm.oos_window_hours)
        mi_h = int(cfg.hmm.mi_horizon_hours)
        post_w = int(cfg.hmm.post_ath_ath_window_h)
        
        assert n_s in cands, f"n_states={n_s} no esta en candidatos {cands}"
        assert oos_w >= mi_h, f"oos_window_hours={oos_w} < mi_horizon_hours={mi_h}"
        assert mi_h >= post_w, f"mi_horizon_hours={mi_h} < post_ath_ath_window_h={post_w}"
    except Exception as e:
        assert False, f"Error validando HMM: {e}"
    return "HMM Constraints OK"

@test("TEST-80  SOP: Restricciones de Kelly y Position Sizing", section="consistency")
def t80_kelly_sizing():
    cfg = _cfg()
    try:
        k_kf = float(cfg.kelly_sizer.kelly_fraction)
        p_kf = float(cfg.position_sizer.kelly_fraction)
        pt_r = float(cfg.kelly_sizer.pt_ratio)
        sl_r = float(cfg.kelly_sizer.sl_ratio)
        
        assert k_kf == p_kf, f"Inconsistencia critica! kelly_sizer={k_kf} != position_sizer={p_kf}"
        assert k_kf <= 0.25, f"kelly_fraction={k_kf} excede el Quarter-Kelly (0.25) maximo permitido R17"
        assert pt_r > sl_r, f"pt_ratio={pt_r} <= sl_r={sl_r}. Rompe asimetria matematica."
    except Exception as e:
        assert False, f"Error validando Kelly Sizer: {e}"
    return "Kelly y Asimetria OK"

@test("TEST-81  SOP: Coherencia OOD Guard y Momentum Filters", section="consistency")
def t81_ood_momentum():
    cfg = _cfg()
    try:
        d_min = float(cfg.ood_guard.guardian_dvol_min_z)
        d_max = float(cfg.ood_guard.guardian_dvol_max_z)
        m_thr = float(cfg.metalabeler.momentum_filter_threshold)
        m_up = float(cfg.metalabeler.momentum_filter_threshold_upper)
        m_ord = float(cfg.metalabeler.momentum_ordered_correction_threshold)
        m_cra = float(cfg.metalabeler.momentum_crash_speed_threshold)
        
        assert d_min < d_max, f"guardian_dvol_min_z={d_min} >= guardian_dvol_max_z={d_max}"
        assert m_thr <= m_up, f"momentum_filter_threshold={m_thr} > threshold_upper={m_up}"
        assert m_ord <= m_cra, f"ordered_correction={m_ord} > crash_speed={m_cra} (Los valores son negativos, ej -25 <= -5)"
    except Exception as e:
        assert False, f"Error validando OOD/Momentum: {e}"
    return "OOD y Momentum OK"

@test("TEST-82  SOP: Consenso y Activos del Ensemble WFB", section="consistency")
def t82_ensemble_wfb():
    cfg = _cfg()
    try:
        c_thr = int(cfg.wfb.ensemble_consensus_threshold)
        c_adv = int(cfg.wfb.circuit_breaker.min_seeds_adverse)
        m_app = int(cfg.wfb.min_seeds_to_approve)
        m_exp = int(cfg.wfb.max_seeds_to_explore)
        a_len = len(cfg.wfb.active_seeds)
        
        assert c_thr <= c_adv, f"ensemble_consensus_threshold={c_thr} > min_seeds_adverse={c_adv}. (Consenso debe ser <= a semillas adversas CB)"
        assert m_app <= m_exp, f"min_seeds_to_approve={m_app} > max_seeds_to_explore={m_exp}"
        assert a_len <= m_exp, f"Active seeds activadas={a_len} supera el max_seeds_to_explore={m_exp}"
    except Exception as e:
        assert False, f"Error validando WFB Ensemble: {e}"
    return "Ensemble constraints OK"

@test("TEST-83  SOP: Buscador de Duplicidad de Parametros", section="consistency")
def t83_duplicate_params():
    import yaml
    from pathlib import Path
    
    cfg_path = Path('config/settings.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f)
        
    def get_leaves(d, current_path=''):
        leaves = []
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    if isinstance(v, dict):
                        leaves.extend(get_leaves(v, current_path + '.' + k if current_path else k))
                    else:
                        leaves.append((k, current_path + '.' + k if current_path else k))
                else:
                    leaves.append((k, current_path + '.' + k if current_path else k))
        return leaves

    leaves = get_leaves(d)
    counts = {}
    paths = {}
    for k, path in leaves:
        counts[k] = counts.get(k, 0) + 1
        paths.setdefault(k, []).append(path)

    whitelist = {
        'patience', 'enabled', 'use_regime_agents', 'max_pbo', 'min_dsr', 
        'kelly_fraction', 'optuna_metric', 'learning_rate_max', 'learning_rate_min', 
        'n_estimators_max', 'n_estimators_min', 'reg_alpha_max', 'reg_alpha_min', 
        'reg_lambda_max', 'reg_lambda_min', 'optuna_trials', 'weight_decay_alpha', 
        'embargo_hours', 'max_depth_cap', 'gamma_floor', 'pt_mult_min', 'sl_mult_min',
        'n_estimators'
    }
    
    errors = []
    for k, c in counts.items():
        if c > 1 and k not in whitelist:
            errors.append(f"{k} duplicado en: {paths[k]}")
            
    assert not errors, f"Parametros duplicados no autorizados: {errors}"
    return "No hay duplicidades no autorizadas OK"

@test("TEST-84  SOP: Coherencia PSI Prediction Drift (pred_drift_psi_min < pred_drift_psi_max)", section="consistency")
def t84_psi_drift_bounds():
    cfg = _cfg()
    try:
        psi_min = float(cfg.wfb.pred_drift_psi_min)
        psi_max = float(cfg.wfb.pred_drift_psi_max)
    except Exception as e:
        assert False, f"pred_drift_psi_min o pred_drift_psi_max ausentes en cfg.wfb: {e}"
        
    assert psi_min >= 0, f"pred_drift_psi_min={psi_min} debe ser >= 0"
    assert psi_max > psi_min, f"pred_drift_psi_max={psi_max} debe ser > pred_drift_psi_min={psi_min} (Evitar división por cero)"
    # Print de trazabilidad según RULE[fixbugsprints.md]
    print(f"[TEST-84 2026-06-19] Invariante PSI verificado: {psi_min} < {psi_max}")
    return f"psi_min={psi_min}, psi_max={psi_max} OK"


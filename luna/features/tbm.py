"""
Triple Barrier Method (TBM) 2.0 - Luna V1
===================================================
Etiquetado avanzado de eventos de mercado basado en López de Prado.
Cruza señales hipotéticas generadas por un modelo primario (ej. XGBoost) 
contra la realidad futura para determinar si fueron rentables según 3 barreras:
1. PT (Profit Taking) - Limite Superior
2. SL (Stop Loss) - Limite Inferior
3. T1 (Time Stop) - Barrera Vertical

SOP Aplicado:
- R11 (Volatilidad Dinámica): El ancho de las barreras PT/SL es proporcional
  a la volatilidad diaria reciente calculada via EWMA, no porcentajes estáticos.

MEJORA 4 (planes_mejora_v3.md P2):
- Barrera Vertical Dinámica ATR: horizonte calculado por evento en lugar del
  fijo de 96H. Rango: 48H (alta volatilidad) → 168H (baja volatilidad).
  Fórmula: horizon = clip(48H x (ATR_mediana / ATR_actual), 48, 168)
  Activar con dynamic_barrier=True en apply_triple_barrier.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from loguru import logger

def get_daily_volatility(close: pd.Series, span: int = 720) -> pd.Series:
    """
    Calcula la volatilidad diaria usando EWMA (Exponential Weighted Moving Average).
    span=720 asume datos horarios (30 días).
    """
    ret = close.pct_change()
    ewma_vol = ret.ewm(span=span).std()
    
    # Anualizar a volatilidad diaria (asumiendo datos horarios -> * sqrt(24))
    daily_vol = ewma_vol * np.sqrt(24)
    # MEJORA-04 (2026-03-17): limit=24 — no propagar más de 24 barras hacia atrás.
    # Sin limit, el primer valor EWMA válido cubre todo el inicio de la serie → targets TBM
    # artificialmente amplios/estrechos en las primeras horas del dataset.
    daily_vol = daily_vol.bfill(limit=24)
    return daily_vol


def compute_atr(close: pd.Series, span: int = 24) -> pd.Series:
    """
    Calcula el Average True Range (ATR) sobre datos horarios usando True Range real.

    BUG-1 FIX (2026-03-15): corregida implementación anterior que usaba solo (H-L)
    de rolling 24h, ignorando gaps entre cierres consecutivos.
    True Range = max(H-L, |H-prev_close|, |L-prev_close|)

    Con datos de 1H sin high/low explícito, se aproxima con:
      - Intra-barra:  rolling_max(2) - rolling_min(2) (aprox. H-L)
      - Gap vs prev:  |close - prev_close|              (gap entre barras)
    ATR_24h = EWMA del True Range con span=24.
    """
    prev_close = close.shift(1)
    intra_bar  = close.rolling(2).max() - close.rolling(2).min()   # aprox H-L
    gap        = (close - prev_close).abs()                         # gap entre barras
    true_range = pd.concat([intra_bar, gap], axis=1).max(axis=1)
    atr = true_range.ewm(span=span, adjust=False).mean()
    # MEJORA-04 (2026-03-17): mismo límite que get_daily_volatility — max 24 barras.
    return atr.bfill(limit=24)

def compute_asymmetric_ratio(close: pd.Series, span: int = 24) -> pd.Series:
    """
    Calcula ratio de asimetría usando los retornos como proxy para ATR_upside y ATR_downside.
    Ratio = ATR_downside / ATR_upside

    [BUG-2-FIX 2026-06-18] Se añade clip(upper=10.0) antes del bfill porque el EWMA
    de upside inicia en 0 en las primeras barras, produciendo ratios > 1e10 que se
    propagan hacia atrás con bfill. El cap institucional final (tbm_asymmetry_ratio_cap=2.0)
    se aplica en apply_triple_barrier, pero si el bfill ya propagó un valor explosivo
    el damage ya está hecho en esas posiciones. Cap interno=10.0 es conservador: permite
    asimetrías extremas (10:1 bajista vs alcista) sin catastrofizar la inicialización.
    """
    diff = close.diff()
    upside = diff.clip(lower=0)
    downside = diff.clip(upper=0).abs()
    
    atr_up = upside.ewm(span=span, adjust=False).mean()
    # Usar np.where en lugar de replace para manejar también valores muy bajos (no solo 0 exacto)
    atr_up = atr_up.where(atr_up > 1e-8, 1e-8)
    atr_down = downside.ewm(span=span, adjust=False).mean()
    
    ratio = atr_down / atr_up
    # [BUG-2-FIX] clip interno antes de bfill para evitar propagación de valores explosivos
    ratio = ratio.clip(upper=10.0)
    result = ratio.bfill(limit=24)
    
    n_explosive = (ratio > 5.0).sum()
    if n_explosive > 0:
        print(f"[BUG-2-FIX][TBM] compute_asymmetric_ratio: {n_explosive} barras con ratio >5.0 "
              f"(clippeadas a 10.0 antes de bfill). Valores de inicio de serie normalizados.")
    
    return result




def compute_dynamic_horizon(
    atr_series: pd.Series,
    event_ts: pd.Timestamp,
    horizon_min_h: int = 48,
    horizon_max_h: int = 168,
) -> int:
    """
    Calcula el horizonte temporal dinámico para un evento basado en el ATR actual.

    Mejora 4 (planes_mejora_v3.md):
    - Si el mercado está en alta volatilidad (ATR > mediana), el horizonte se reduce.
    - Si el mercado está en baja volatilidad (ATR < mediana), el horizonte aumenta.

    Fórmula: horizon = clip(horizon_min_h x (ATR_mediana / ATR_actual), min, max)

    Args:
        atr_series:    Serie ATR completa (calculada con compute_atr)
        event_ts:      Timestamp del evento actual
        horizon_min_h: Horizonte mínimo en horas (default 48H = alta volatilidad)
        horizon_max_h: Horizonte máximo en horas (default 168H = 1 semana)

    Returns:
        int: horizonte en horas para ese evento
    """
    # BUG-2 FIX (2026-03-16): mediana calculada solo hasta el timestamp del evento.
    # ANTES: atr_series.median() → incluía barras FUTURAS → look-ahead bias.
    # Impacto: ATR_mediana inflada en períodos de alta volatilidad futura →
    # horizontes dinámicos más largos de lo causal → trades con más tiempo para recuperarse.
    atr_before_event = atr_series.loc[:event_ts]
    atr_median = float(atr_before_event.median()) if len(atr_before_event) > 0 else float(atr_series.iloc[0])
    try:
        atr_current = float(atr_series.asof(event_ts))
    except Exception:
        atr_current = atr_median

    if atr_current <= 0 or np.isnan(atr_current):
        return horizon_min_h

    # Ratio inversamente proporcional: más volatilidad -> horizonte más corto
    vol_ratio = atr_median / atr_current
    dynamic_h = int(np.clip(
        horizon_min_h * vol_ratio,
        horizon_min_h,
        horizon_max_h,
    ))
    return dynamic_h


from numba import njit

@njit(cache=True)
def _compute_touches_jit(close_vals, close_times_sec, event_indices, t1_indices,
                         pt_vals, sl_vals, sides, trgts, min_ret,
                         linear_decay_pt, pt_decay_fraction):
    """
    Motor compilado en C para la búsqueda del primer toque de la barrera.
    """
    n_events = len(event_indices)
    pt_idx = np.full(n_events, -1, dtype=np.int64)
    sl_idx = np.full(n_events, -1, dtype=np.int64)
    
    for i in range(n_events):
        loc = event_indices[i]
        t1 = t1_indices[i]
        
        if t1 < 0 or loc < 0 or loc >= len(close_vals) or t1 >= len(close_vals) or loc > t1:
            continue
            
        trgt = trgts[i]
        if trgt < min_ret:
            continue
            
        side = sides[i]
        if side == 0:
            continue
            
        pt = pt_vals[i]
        sl = sl_vals[i]
        
        loc_price = close_vals[loc]
        if loc_price <= 0:
            continue
            
        loc_time = close_times_sec[loc]
        t1_time = close_times_sec[t1]
        vb_h = max(1.0, (t1_time - loc_time) / 3600.0)
        
        for j in range(loc, t1 + 1):
            ret = (close_vals[j] / loc_price) - 1.0
            ret = ret * side
            
            # Check PT
            if pt_idx[i] == -1 and pt > 0:
                if linear_decay_pt:
                    h_elapsed = (close_times_sec[j] - loc_time) / 3600.0
                    pt_dyn = pt * (1.0 - pt_decay_fraction * (h_elapsed / vb_h))
                    if pt_dyn < 0.0:
                        pt_dyn = 0.0
                    if ret >= pt_dyn:
                        pt_idx[i] = j
                else:
                    if ret >= pt:
                        pt_idx[i] = j
                        
            # Check SL
            if sl_idx[i] == -1 and sl > 0:
                if ret <= sl:
                    sl_idx[i] = j
                    
            if pt_idx[i] != -1 and sl_idx[i] != -1:
                break
                
    return pt_idx, sl_idx


def apply_pt_sl_on_t1(
    close: pd.Series, 
    events: pd.DataFrame, 
    pt_sl: list, 
    min_ret: float = 0.005,
    linear_decay_pt: bool = False,
    pt_decay_fraction: float = 0.75
) -> pd.DataFrame:
    """
    Encuentra los timestamps del PT (Profit Taking) o SL (Stop Loss) para cada evento.
    Vectorizado via Numba para máximo rendimiento ($O(N \times H)$ en C compilado).
    """
    out = events[['t1']].copy(deep=True)
    dt_dtype = events['t1'].dtype if 't1' in events.columns else 'datetime64[ns, UTC]'
    out['pt'] = pd.Series(pd.NaT, dtype=dt_dtype, index=events.index)
    out['sl'] = pd.Series(pd.NaT, dtype=dt_dtype, index=events.index)
    
    if len(events) == 0:
        return out
        
    # [FIX-OBS-01] Soporte para multiplicadores PT/SL dinámicos
    if isinstance(pt_sl[0], pd.Series):
        _pt_mult = pt_sl[0].reindex(events.index)
        pt = _pt_mult * events['trgt']
        pt = pt.where(_pt_mult > 0, 0)
    else:
        pt = pt_sl[0] * events['trgt'] if pt_sl[0] > 0 else pd.Series(0, index=events.index)
        
    if isinstance(pt_sl[1], pd.Series):
        _sl_mult = pt_sl[1].reindex(events.index)
        sl = -_sl_mult * events['trgt']
        sl = sl.where(_sl_mult > 0, 0)
    else:
        sl = -pt_sl[1] * events['trgt'] if pt_sl[1] > 0 else pd.Series(0, index=events.index)
        
    # ------------------ COMPILACIÓN VECTORIAL NUMBA ------------------
    _close_sorted = close.sort_index()
    close_idx = _close_sorted.index
    close_vals = _close_sorted.values.astype(np.float64)
    # Convertir timestamps a segundos
    close_times_sec = close_idx.view(np.int64) / 10**9 
    
    # [FIX-TBM] Convertir todos los índices a numpy datetime64[ns] puro para np.searchsorted seguro
    close_idx_ns = close_idx.values.astype('datetime64[ns]')
    events_idx_ns = events.index.values.astype('datetime64[ns]')
    
    event_locs = np.searchsorted(close_idx_ns, events_idx_ns)
    
    events_t1_ns = events['t1'].values.astype('datetime64[ns]')
    mask_nat = pd.isna(events_t1_ns)
    
    # Mapeo a numpy seguro
    t1_clean = np.where(mask_nat, close_idx_ns[0], events_t1_ns)
    event_t1s = np.searchsorted(close_idx_ns, t1_clean)
    event_t1s[mask_nat] = -1
    
    pt_vals = pt.values.astype(np.float64)
    sl_vals = sl.values.astype(np.float64)
    sides = events['side'].values.astype(np.float64) if 'side' in events.columns else np.ones(len(events), dtype=np.float64)
    trgts = events['trgt'].values.astype(np.float64)
    
    pt_res, sl_res = _compute_touches_jit(
        close_vals, close_times_sec, event_locs, event_t1s,
        pt_vals, sl_vals, sides, trgts, float(min_ret),
        bool(linear_decay_pt), float(pt_decay_fraction)
    )
    
    valid_pt = pt_res != -1
    valid_sl = sl_res != -1
    
    out.loc[valid_pt, 'pt'] = close_idx[pt_res[valid_pt]]
    out.loc[valid_sl, 'sl'] = close_idx[sl_res[valid_sl]]
    
    # Forzar dtype datetime64[ns, UTC]
    for col in ['pt', 'sl']:
        out[col] = pd.to_datetime(out[col], utc=True, errors='coerce')
        
    return out

def get_bins(events: pd.DataFrame, close: pd.Series, funding_series: pd.Series = None) -> pd.DataFrame:
    """
    Genera las etiquetas de 1 (rentable) y 0 (no rentable) basadas en qué
    barrera se tocó primero. Aplica Arrastre de Funding Rate si funding_series es provisto.
    """
    events_ = events.dropna(subset=['t1']).copy()
    out = pd.DataFrame(index=events_.index)
    
    # Fix F2: garantizar dtype datetime para min(axis=1) correcto
    for col in ['pt', 'sl']:
        if col in events_.columns:
            events_[col] = pd.to_datetime(events_[col], utc=True, errors='coerce')
        else:
            events_[col] = pd.NaT
    
    # Encontrar la fecha del primer toque entre [t1, pt, sl]
    first_touch = events_[['t1', 'pt', 'sl']].min(axis=1)
    
    out['first_touch'] = first_touch

    # BUG-TBM-01 FIX (2026-04-06): Usar reindex con method='nearest' + tolerance=2H.
    # El reindex exacto (sin tolerancia) produce NaN silencioso cuando first_touch no
    # está exactamente en el índice de close (p.ej. en gaps de datos de Binance).
    # Impacto del bug: ret=NaN → meta_label=0 (trade clasificado como PERDEDOR)
    # aunque el precio de toque real fuera positivo.
    # La tolerancia de 2H es conservadora: si no hay close dentro de 2H del toque,
    # se deja NaN (trade descartado) en lugar de usar un precio lejano incorrecto.
    _close_sorted = close.sort_index()
    _touch_prices = pd.Series(
        _close_sorted.reindex(out['first_touch'], method='nearest', tolerance=pd.Timedelta('2H')).values,
        index=out.index
    )
    _entry_prices = pd.Series(
        _close_sorted.reindex(out.index, method='nearest', tolerance=pd.Timedelta('2H')).values,
        index=out.index
    )
    out['ret'] = _touch_prices / _entry_prices - 1

    # [R14: Funding Rate Drag] Restar acumulado de Funding Rate pagado
    if funding_series is not None:
        try:
            # Calcular suma acumulada del Funding Rate
            _funding_sorted = funding_series.sort_index().fillna(0.0)
            _funding_cumsum = _funding_sorted.cumsum()
            
            # Obtener el Funding Acumulado en el momento de entrada (out.index) y en la salida (out['first_touch'])
            _funding_entry = pd.Series(
                _funding_cumsum.reindex(out.index, method='nearest', tolerance=pd.Timedelta('2H')).values,
                index=out.index
            )
            _funding_exit = pd.Series(
                _funding_cumsum.reindex(out['first_touch'], method='nearest', tolerance=pd.Timedelta('2H')).values,
                index=out.index
            )
            
            _funding_drag = _funding_exit - _funding_entry
            
            # Convencion: Long (1) paga Funding si es positivo, Short (-1) paga Funding negativo
            # Por tanto el costo arrastrado es: FundingRate * side
            _side = events_.get('side', 1.0)
            _cost_paid = _funding_drag * _side
            
            # Descontar el costo pagado del retorno
            out['ret'] -= _cost_paid
            
            _n_impacted = (_cost_paid != 0).sum()
            logger.debug(f"[R14 Funding Drag] Aplicado descuento de Funding Rate a {_n_impacted} eventos en TBM.")
        except Exception as _e_fund:
            logger.warning(f"[R14 Funding Drag] Error calculando drag: {_e_fund}")

    # Advertir si quedan NaN (indica gap de datos > 2H en la ruta del trade)
    _n_nan = out['ret'].isna().sum()
    if _n_nan > 0:
        print(f"[BUG-FIX-LOG 2026-06-05] Corregido formatting logger.warning en tbm.py [BUG-TBM-01]")
        logger.warning(
            "[BUG-TBM-01] {} eventos con ret=NaN tras reindex tolerante 2H "
            "(gaps de datos en el path del trade) — descartados del dataset.",
            _n_nan
        )
    
    if 'side' in events_.columns:
        out['ret'] *= events_['side']
        
    out['bin'] = np.sign(out['ret'])
    
    # Etiqueta para el Meta-Labeler: 1 si acertamos, 0 si fallamos
    out['meta_label'] = (out['ret'] > 0).astype(int)
    
    return out

def get_events(
    close: pd.Series, 
    t_events: pd.DatetimeIndex, 
    pt_sl: list, 
    trgt: pd.Series, 
    min_ret: float, 
    t1: pd.Series = None, 
    side: pd.Series = None,
    linear_decay_pt: bool = False,
    pt_decay_fraction: float = 0.75
) -> pd.DataFrame:
    """
    Arma el DataFrame `events` estructurado con t1 dictaminado y el cálculo 
    de toques PT/SL.
    """
    trgt = trgt.reindex(t_events)
    trgt = trgt[trgt > min_ret]
    
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)
    else:
        t1 = t1.reindex(trgt.index)
    
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
    else:
        side_ = side.reindex(trgt.index)
        
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    # Fix F1: eliminar también eventos donde t1 es NaT para no contaminar el TBM
    events = events.dropna(subset=['t1'])
    
    df0 = apply_pt_sl_on_t1(close, events, pt_sl, min_ret=min_ret, 
                            linear_decay_pt=linear_decay_pt, pt_decay_fraction=pt_decay_fraction)
    
    events['pt'] = df0.get('pt', pd.NaT)
    events['sl'] = df0.get('sl', pd.NaT)
    
    return events

def apply_triple_barrier(
    price_series: pd.Series,
    event_times: pd.DatetimeIndex,
    sides: pd.Series = None,
    pt_sl_multiplier: list = [2.0, 1.0],
    vertical_barrier_hours: int = 96,
    vol_span_hours: int = 720,
    min_return: float = None,
    dynamic_barrier: bool = False,
    dynamic_horizon_min_h: int = 48,
    dynamic_horizon_max_h: int = 168,
    linear_decay_pt: bool = False,
    pt_decay_fraction: float = 0.75,
    funding_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Orquestador maestro para el Triple Barrier Method.
    
    Args:
        price_series:          Serie de precios hora a hora (pd.Series de 'close').
        event_times:           Instantes de tiempo en los que el modelo base generó señal.
        sides:                 pd.Series(1/-1) Dirección de la operación. By default Long=1.
        pt_sl_multiplier:      [TP multiplier, SL multiplier] para aplicar a la volatilidad.
        vertical_barrier_hours: Time Stop fijo. Solo se usa si dynamic_barrier=False.
        vol_span_hours:        Ventana EWMA para target (ej 720h = 30 días).
        min_return:            Retorno mínimo para filtrar señales (SOP R6).
        dynamic_barrier:       Si True, usa barrera vertical dinámica ATR (Mejora 4).
                               El horizonte varía entre dynamic_horizon_min_h y
                               dynamic_horizon_max_h según la volatilidad actual.
        dynamic_horizon_min_h: Horizonte mínimo en horas (default 48H).
        dynamic_horizon_max_h: Horizonte máximo en horas (default 168H = 1 semana).
        funding_series:        Serie del Funding Rate horario para descuento continuo (R14).
        
    Returns:
        DataFrame indexado por event_times con: 
        ['first_touch', 'ret', 'bin', 'meta_label', 'pt', 'sl', 't1']
    """
    # 1. Volatilidad Dinámica
    volatility = get_daily_volatility(price_series, span=vol_span_hours)
    
    # 1b. Enforce No-Fallback for min_return if not provided
    if min_return is None:
        try:
            from config.settings import cfg as _cfg_tbm
            min_return = float(_cfg_tbm.xgboost.tbm_min_return)
        except Exception as e_tbm:
            raise RuntimeError(f"Falta min_return y no se pudo leer cfg.xgboost.tbm_min_return en settings.yaml. Política No-Fallback: {e_tbm}")

    # 1c. Asymmetric TBM (Hipótesis A)
    try:
        from config.settings import cfg as _cfg_tbm
        tbm_asymmetric = bool(getattr(_cfg_tbm.xgboost, "tbm_asymmetric", False))
        tbm_asymmetry_ratio_cap = float(getattr(_cfg_tbm.xgboost, "tbm_asymmetry_ratio_cap", 2.0))
    except Exception as e_tbm:
        # [BUG-4-FIX 2026-06-18] Añadir trazabilidad al fallback (RULE[fixbugsprints.md])
        print(f"[BUG-4-FIX][TBM][WARNING] No se pudo leer tbm_asymmetric de cfg: {e_tbm}. "
              f"Usando fallback tbm_asymmetric=False (TBM simétrico). Revisar settings.yaml.")
        logger.warning("[BUG-4-FIX][TBM] Fallback a TBM simétrico por error leyendo cfg: {}", e_tbm)
        tbm_asymmetric = False
        tbm_asymmetry_ratio_cap = 2.0


    if tbm_asymmetric:
        asym_ratio = compute_asymmetric_ratio(price_series, span=24).clip(lower=1/tbm_asymmetry_ratio_cap, upper=tbm_asymmetry_ratio_cap)
        pt_m, sl_m = pt_sl_multiplier[0], pt_sl_multiplier[1]
        
        pt_mult_s = pd.Series(pt_m, index=price_series.index) if not isinstance(pt_m, pd.Series) else pt_m
        sl_mult_s = pd.Series(sl_m, index=price_series.index) if not isinstance(sl_m, pd.Series) else sl_m
        
        pt_sl_multiplier = [pt_mult_s, sl_mult_s * asym_ratio]
        
        # Logear la activación de la mejora
        logger.info(f"[MEJORA-MATH-A] Asymmetric TBM Activado. Cap: {tbm_asymmetry_ratio_cap}x | Ratio medio asimetría: {float(asym_ratio.median()):.2f}x")

    
    # 2. Time Stop (t1) -> Horizonte fijo o dinámico (Mejora 4)
    if dynamic_barrier:
        # MEJORA 4: calcular ATR y derivar horizonte por evento
        atr = compute_atr(price_series)

        # LOGIC-TBM-01 FIX (2026-04-06): pre-computar mediana expansiva ATR antes del loop.
        # compute_dynamic_horizon() recalculaba atr.loc[:event_ts].median() en CADA iteración
        # → O(N²) en tiempo con N eventos de entrenamiento (50k eventos: varios minutos).
        # Fix: expanding().median() es una operación vectorizada O(N) que se ejecuta UNA VEZ.
        # El valor en cada timestamp es exactamente atr_series.loc[:event_ts].median(),
        # preservando la causalidad estricta del BUG-2 FIX original.
        atr_expanding_median = atr.expanding(min_periods=24).median()

        atr_median = float(atr.median())  # solo para el log final
        n_dynamic = 0
        t1 = []
        for t in event_times:
            # Leer mediana causal pre-computada (sin recalcular en cada paso)
            atr_med_at_t = float(atr_expanding_median.asof(t)) if not atr_expanding_median.empty else atr_median
            if np.isnan(atr_med_at_t) or atr_med_at_t <= 0:
                atr_med_at_t = atr_median if not np.isnan(atr_median) else 1.0
            try:
                atr_current = float(atr.asof(t))
            except Exception:
                atr_current = atr_med_at_t
            if np.isnan(atr_current) or atr_current <= 0:
                atr_current = max(atr_med_at_t, 1e-8)
            vol_ratio = atr_med_at_t / atr_current if atr_current > 0 else 1.0
            horizon_h = int(np.clip(
                dynamic_horizon_min_h * vol_ratio,
                dynamic_horizon_min_h,
                dynamic_horizon_max_h,
            ))
            next_t = t + pd.Timedelta(hours=horizon_h)
            try:
                # [FIX-TBM-O(NlogN)] Usar searchsorted para reducir complejidad de O(N^2) a O(N log N)
                idx_found = price_series.index.searchsorted(next_t)
                closest = price_series.index[idx_found] if idx_found < len(price_series) else price_series.index[-1]
                t1.append(closest)
            except Exception as _e_tbm:
                print(f"[FIX-TBM-O(NlogN)] Error en loop dinámico: {_e_tbm}")
                t1.append(price_series.index[-1])
            if horizon_h != vertical_barrier_hours:
                n_dynamic += 1
        logger.info(
            f"TBM dinámico (Mejora 4): {n_dynamic}/{len(event_times)} eventos con horizonte distinto al fijo {vertical_barrier_hours}H "
            f"(ATR mediana={atr_median:.2f}, rango {dynamic_horizon_min_h}–{dynamic_horizon_max_h}H) [LOGIC-TBM-01: O(N) en lugar de O(N²)]"
        )

    else:
        # [FIX-TBM-O(NlogN)] Trace print and log at start of loop
        print(f"[FIX-TBM-O(NlogN)] [BUG-TBM-FIX] Iniciando búsqueda binaria rápida O(log N) para barrera vertical fija {vertical_barrier_hours}H ({len(event_times)} eventos)...")
        # Comportamiento original (backward-compatible)
        t1 = []
        for t in event_times:
            next_t = t + pd.Timedelta(hours=vertical_barrier_hours)
            if next_t in price_series.index:
                t1.append(next_t)
            else:
                try:
                    # [FIX-TBM-O(NlogN)] Usar searchsorted para reducir complejidad de O(N^2) a O(N log N) en fallbacks
                    idx_found = price_series.index.searchsorted(next_t)
                    closest = price_series.index[idx_found] if idx_found < len(price_series) else price_series.index[-1]
                    t1.append(closest)
                except Exception as _e_tbm2:
                    print(f"[FIX-TBM-O(NlogN)] Error en loop estático: {_e_tbm2}")
                    t1.append(price_series.index[-1])

    t1_series = pd.Series(t1, index=event_times)

    # 3. Calcular toques (eventos)
    barrier_desc = f"dinámica ATR ({dynamic_horizon_min_h}–{dynamic_horizon_max_h}H)" if dynamic_barrier else f"{vertical_barrier_hours}H fija"
    
    if isinstance(pt_sl_multiplier[0], pd.Series):
        pt_desc = f"Dinámico (med={pt_sl_multiplier[0].median():.1f}x)"
    else:
        pt_desc = f"{pt_sl_multiplier[0]:.1f}x"
        
    if isinstance(pt_sl_multiplier[1], pd.Series):
        sl_desc = f"Dinámico (med={pt_sl_multiplier[1].median():.1f}x)"
    else:
        sl_desc = f"{pt_sl_multiplier[1]:.1f}x"

    logger.info(f"Aplicando PT: {pt_desc} / SL: {sl_desc} | Barrera Vertical: {barrier_desc}")

    # Guard: si no hay eventos no hay nada que calcular
    if len(event_times) == 0:
        logger.warning("apply_triple_barrier: event_times está vacío — devolviendo DataFrame vacío.")
        return pd.DataFrame(columns=['t1', 'trgt', 'side', 'pt', 'sl', 'first_touch', 'ret', 'bin', 'meta_label'])

    events = get_events(
        close=price_series,
        t_events=event_times,
        pt_sl=pt_sl_multiplier,
        trgt=volatility,
        min_ret=min_return,
        t1=t1_series,
        side=sides,
        linear_decay_pt=linear_decay_pt,
        pt_decay_fraction=pt_decay_fraction
    )

    # 4. Asignar etiquetas Finales (Meta-Labels)
    labels = get_bins(events, price_series, funding_series)

    # 5. Cruzar información y devolver set consolidado
    result = events.join(labels[['first_touch', 'ret', 'bin', 'meta_label']])
    return result

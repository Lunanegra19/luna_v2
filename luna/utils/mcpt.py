import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from scipy.stats import spearmanr

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    """
    Permutes OHLC data by shuffling intra-bar movements and inter-bar gaps independently.
    This preserves the distribution of returns, kurtosis, and skewness, while destroying
    autocorrelation and causal relationships.
    """
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    if perm_n <= 0:
        return ohlc if n_markets > 1 else ohlc[0]

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        
        # Get prices relative to this bars open
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) separately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]


def evaluate_rule_mcpt(pandas_eval: str, df: pd.DataFrame, n_perms: int = 100, pval_threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Evaluates a pandas_eval rule against MCPT permutations to detect curve-fitting.
    We apply the rule to the REAL features to generate trading signals, but evaluate 
    the profit factor of those signals against PERMUTED future market returns.
    
    Returns:
        (is_genuine, p_value)
    """
    try:
        # 1. Evaluar la regla en los datos reales
        real_mask = df.eval(pandas_eval, engine='python').astype(float).fillna(0)
        
        # Si la regla casi nunca se activa, es muy fragil
        if real_mask.sum() < 5:
            return False, 1.0
            
        r_target = np.log(df['close']).diff().shift(-1).fillna(0)
        
        # Calculo de Profit Factor Real
        # Senal 1 = Long, Senal 0 = Flat (si queremos long/short usar np.where(real_mask == 1, 1, -1))
        # Para Alpha Rules, solemos tratarlas como Long-only masks
        real_rets = real_mask * r_target
        real_rets_nonz = real_rets[real_rets != 0]
        
        if len(real_rets_nonz) == 0:
            return False, 1.0
            
        loss_sum = np.abs(real_rets_nonz[real_rets_nonz < 0]).sum()
        if loss_sum > 0:
            real_pf = real_rets_nonz[real_rets_nonz > 0].sum() / loss_sum
        else:
            real_pf = 3.0 # Arbitrary high cap
            
        if real_pf < 1.0: # Si ya pierde dinero en el mercado real, descartar
            return False, 1.0
            
        # 2. Tribunal MCPT
        better_count = 1 # Empezamos en 1 por la muestra original (conservador)
        ohlc_cols = ['open', 'high', 'low', 'close']
        
        # Aceleracion: Solo necesitamos permutar los OHLC para calcular p_r_target
        df_ohlc = df[ohlc_cols].copy()
        
        for _ in range(n_perms):
            perm_df = get_permutation(df_ohlc, start_index=100)
            p_r_target = np.log(perm_df['close']).diff().shift(-1).fillna(0)
            
            p_rets = real_mask * p_r_target
            p_rets_nonz = p_rets[p_rets != 0]
            
            p_loss = np.abs(p_rets_nonz[p_rets_nonz < 0]).sum()
            perm_pf = p_rets_nonz[p_rets_nonz > 0].sum() / p_loss if p_loss > 0 else 3.0
            
            if perm_pf >= real_pf:
                better_count += 1
                
        pval = better_count / (n_perms + 1)
        
        return pval <= pval_threshold, pval
        
    except Exception as e:
        return False, 1.0

def evaluate_continuous_mcpt_vectorized(feat_vals: np.ndarray, r_target: np.ndarray, synthetic_targets: np.ndarray, pval_threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Tribunal MCPT para features continuas optimizado.
    Recibe los arrays sintéticos precalculados para evitar recalcular permutaciones.
    """
    try:
        real_ic, _ = spearmanr(feat_vals, r_target)
        if np.isnan(real_ic):
            return False, 1.0 # Array constante o Degenerado
            
        better_count = 1
        
        for p_target in synthetic_targets:
            p_ic, _ = spearmanr(feat_vals, p_target)
            if np.isnan(p_ic):
                p_ic = 0.0
            
            # Si el IC sintético es igual o mejor que el IC real en valor absoluto
            if abs(p_ic) >= abs(real_ic):
                better_count += 1
                
        pval = better_count / (len(synthetic_targets) + 1)
        return pval <= pval_threshold, pval
        
    except Exception as e:
        return False, 1.0

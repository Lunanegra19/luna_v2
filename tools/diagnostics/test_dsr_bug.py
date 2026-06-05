import numpy as np
from scipy.stats import norm
import math

def fake_dsr_current(fold_sharpes, n_trials):
    """Implementación actual en Luna V2 (errónea)"""
    sr_mean = np.mean(fold_sharpes)
    sr_std = np.std(fold_sharpes, ddof=1) # ¡VARIANZA TEMPORAL!
    
    em_gamma = 0.5772156649
    p = 1.0 / max(2, n_trials)
    z1 = norm.ppf(1 - p)
    z2 = norm.ppf(1 - p / math.e)
    exp_max = (1 - em_gamma) * z1 + em_gamma * z2
    
    # El benchmark SR* se calcula usando la varianza temporal de ESTA única feature
    sr_star = sr_std * exp_max 
    
    var_sr = (1.0 + 0.5 * (sr_mean ** 2)) / 100 # n_obs = 100 simulado
    dsr = norm.cdf((sr_mean - sr_star) / math.sqrt(var_sr))
    return dsr, sr_star

def true_dsr_lopez_de_prado(fold_sharpes, n_trials, cross_sectional_std):
    """Implementación matemáticamente correcta (López de Prado 2014)"""
    sr_mean = np.mean(fold_sharpes)
    
    em_gamma = 0.5772156649
    p = 1.0 / max(2, n_trials)
    z1 = norm.ppf(1 - p)
    z2 = norm.ppf(1 - p / math.e)
    exp_max = (1 - em_gamma) * z1 + em_gamma * z2
    
    # El benchmark SR* requiere la varianza TRANSVERSAL de TODAS las features testeadas
    sr_star = cross_sectional_std * exp_max 
    
    var_sr = (1.0 + 0.5 * (sr_mean ** 2)) / 100
    dsr = norm.cdf((sr_mean - sr_star) / math.sqrt(var_sr))
    return dsr, sr_star

def test_dsr_bug():
    print("="*60)
    print("TEST: DEFLATED SHARPE RATIO (TEMPORAL VS CROSS-SECTIONAL)")
    print("="*60)
    
    n_trials = 1000 # Testeamos 1000 features en total
    cross_sectional_std = 0.5 # Asumimos que la varianza entre las 1000 features es 0.5
    
    # CASO A: Feature muy estable a lo largo del tiempo (folds muy parecidos)
    # Sharpe mean = 1.0, pero varianza temporal muy baja (0.05)
    folds_stable = [0.95, 1.0, 1.05, 0.98, 1.02]
    
    dsr_curr_A, star_curr_A = fake_dsr_current(folds_stable, n_trials)
    dsr_true_A, star_true_A = true_dsr_lopez_de_prado(folds_stable, n_trials, cross_sectional_std)
    
    print("CASO A: Feature Estable (Mean SR=1.0, Temporal Std=0.04)")
    print(f"  Implementación Actual (Luna) -> SR* = {star_curr_A:.4f} | DSR = {dsr_curr_A:.4f}  <-- MÚLTIPLE TESTING BORRADO!")
    print(f"  Lopez de Prado Correcto    -> SR* = {star_true_A:.4f} | DSR = {dsr_true_A:.4f}  <-- PENALIZADA CORRECTAMENTE")
    print("-" * 60)
    
    # CASO B: Feature inestable a lo largo del tiempo (folds muy volátiles)
    # Sharpe mean = 1.0, pero varianza temporal alta (1.5)
    folds_unstable = [-0.5, 2.5, 0.0, 2.0, 1.0]
    
    dsr_curr_B, star_curr_B = fake_dsr_current(folds_unstable, n_trials)
    dsr_true_B, star_true_B = true_dsr_lopez_de_prado(folds_unstable, n_trials, cross_sectional_std)
    
    print("CASO B: Feature Inestable (Mean SR=1.0, Temporal Std=1.19)")
    print(f"  Implementación Actual (Luna) -> SR* = {star_curr_B:.4f} | DSR = {dsr_curr_B:.4f}  <-- PENALIZADA INJUSTAMENTE!")
    print(f"  Lopez de Prado Correcto    -> SR* = {star_true_B:.4f} | DSR = {dsr_true_B:.4f}  <-- PENALIZACIÓN ESTÁNDAR")

if __name__ == '__main__':
    test_dsr_bug()

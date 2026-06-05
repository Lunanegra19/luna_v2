import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler

def test_focal_loss_gradients():
    print("="*50)
    print("TEST: FOCAL LOSS GRADIENTS (y=0, negative class)")
    print("="*50)
    
    # Simular predicciones desde p=0.1 (correcto) hasta p=0.99 (muy equivocado, falso positivo)
    p_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    gamma = 2.0
    
    print(f"{'Prob(p)':<10} | {'Viejo Grad (+)':<15} | {'Nuevo Grad (-)':<15} | {'Diferencia':<10}")
    print("-" * 55)
    
    for p in p_values:
        # Fórmula con el BUG (vieja)
        grad_old = p**gamma * (p + gamma * (1 - p) * np.log(1 - p))
        
        # Fórmula CORRECTA (nueva)
        grad_new = p**gamma * (p - gamma * (1 - p) * np.log(1 - p))
        
        print(f"{p:<10.2f} | {grad_old:<15.4f} | {grad_new:<15.4f} | {(grad_new - grad_old):<10.4f}")
        
    print("\nCONCLUSION FOCAL LOSS:")
    print("El gradiente antiguo (bug) es MENOR que el nuevo en p=0.99.")
    print("En p=0.99, el modelo está 99% seguro de que es LONG, pero y=0 (era BEAR).")
    print("El gradiente debería ser altísimo para castigar la red. El código viejo lo atenuaba.\n")


def test_granger_ols_collapse():
    print("="*50)
    print("TEST: GRANGER CAUSALITY MATRIX COLLAPSE")
    print("="*50)
    
    np.random.seed(42)
    n = 200
    
    # Serie X con magnitud astronómica (ej. M2 Money Supply en Trillones)
    x = np.random.normal(loc=1e13, scale=1e12, size=n)
    
    # Serie Y con magnitud diminuta (ej. BTC Returns en porcentajes)
    y = np.random.normal(loc=0.0, scale=0.02, size=n)
    
    df_unscaled = pd.DataFrame({'x': x, 'y': y})
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_unscaled), columns=['x', 'y'])
    
    print("Ejecutando Granger en datos NO ESCALADOS (x=10^13, y=10^-2)...")
    try:
        grangercausalitytests(df_unscaled, maxlag=2, verbose=False)
        print("  -> Ejecutado (Revisar si hubo warnings internos de statsmodels)")
    except Exception as e:
        print(f"  -> ERROR OLS FATAL: {e}")
        
    print("\nEjecutando Granger en datos ESCALADOS (x~N(0,1), y~N(0,1))...")
    try:
        grangercausalitytests(df_scaled, maxlag=2, verbose=False)
        print("  -> Ejecutado de forma estable y numéricamente segura.")
    except Exception as e:
        print(f"  -> ERROR: {e}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('always')  # Forzar que se vean los warnings de HessianInversion
    test_focal_loss_gradients()
    test_granger_ols_collapse()

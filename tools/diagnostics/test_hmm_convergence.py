import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

# Capture prints from hmmlearn C backend by redirecting stdout at python level is hard
# but we can capture warnings if any, and check log likelihoods.

# Generate data with collapsing variance
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
flat_data = np.zeros(500) # zero variance
X_raw = np.concatenate([normal_data, flat_data]).reshape(-1, 1)

X = StandardScaler().fit_transform(X_raw)

print("--- TESTING DEFAULT HMM (min_covar default) ---")
m1 = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=50, random_state=42, verbose=True)
try:
    m1.fit(X)
    print(f"Converged: {m1.monitor_.converged}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- TESTING FIXED HMM (min_covar=0.01) ---")
m2 = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=50, random_state=42, verbose=True, min_covar=0.01)
try:
    m2.fit(X)
    print(f"Converged: {m2.monitor_.converged}")
except Exception as e:
    print(f"Error: {e}")

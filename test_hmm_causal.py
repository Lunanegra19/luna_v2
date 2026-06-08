import pandas as pd
import numpy as np
import joblib
from scipy.special import logsumexp

df = pd.read_parquet(r'C:\Users\Usuario\Desktop\ia\luna_v2\data\features\features_holdout_W1.parquet')
m = joblib.load(r'C:\Users\Usuario\Desktop\ia\luna_v2\data\models\hmm_regime.pkl')
X = df[m['features']].fillna(0.0)
Xs = m['scaler'].transform(X)

model = m['model']
n_samples = Xs.shape[0]
n_components = model.n_components
framelogprob = model._compute_log_likelihood(Xs)

log_startprob = np.log(np.maximum(model.startprob_, 1e-10))
log_transmat = np.log(np.maximum(model.transmat_, 1e-10))

log_alpha = np.zeros((n_samples, n_components))
log_alpha[0] = log_startprob + framelogprob[0]

states = np.zeros(n_samples, dtype=int)
states[0] = np.argmax(log_alpha[0])

for t in range(1, n_samples):
    work_buffer = log_alpha[t-1][:, None] + log_transmat
    log_alpha[t] = logsumexp(work_buffer, axis=0) + framelogprob[t]
    states[t] = np.argmax(log_alpha[t])

print("Causal Forward Filter results:")
print(pd.Series(states).map(m['state_map']).value_counts())

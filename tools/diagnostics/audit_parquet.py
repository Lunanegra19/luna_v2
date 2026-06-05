import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('data/features/features_train.parquet')
    cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[cols]).sum().sum()
    nan_counts = df.isna().sum().sum()
    print(f'Total Rows: {len(df)}')
    print(f'Total Columns: {len(df.columns)}')
    print(f'Total NaNs: {nan_counts}')
    print(f'Total Infs: {inf_counts}')
    print('\nTop 5 Max Values (Extreme Value Check):')
    print(df[cols].max().nlargest(5))
except Exception as e:
    print(f'Error: {e}')

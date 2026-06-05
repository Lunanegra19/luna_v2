import pandas as pd
from pathlib import Path

ROOT = Path("g:/Mi unidad/ia/luna_v2")
WFB_DIR = ROOT / "data" / "reports" / "wfb"

for w in [3, 4]:
    trades_path = WFB_DIR / f"oos_trades_W{w}_seed42.parquet"
    if not trades_path.exists():
        print(f"Ventana W{w}: No se encontró el archivo de trades.")
        continue
    
    df = pd.read_parquet(trades_path)
    print(f"\n==================================================")
    print(f"ANALISIS DE TRADES REALES EN VENTANA W{w}")
    print(f"==================================================")
    print(f"Total trades: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print("\nPrimeras filas del dataframe de trades:")
    print(df[['is_win', 'return_pct', 'hmm_regime', 'meta_v2_prob', 'xgb_prob']].head(10))
    
    print("\nDistribución de Regímenes HMM en los trades:")
    print(df['hmm_regime'].value_counts())
    
    print("\nEstadísticas de meta_v2_prob:")
    print(df['meta_v2_prob'].describe())
    
    print("\nEstadísticas de xgb_prob:")
    print(df['xgb_prob'].describe())

    # Agrupar por régimen y ver win rate y meta_v2_prob media
    print("\nMétricas por Régimen HMM:")
    summary = df.groupby('hmm_regime').agg(
        n_trades=('is_win', 'count'),
        win_rate=('is_win', 'mean'),
        avg_ret=('return_pct', 'mean'),
        avg_meta=('meta_v2_prob', 'mean'),
        min_meta=('meta_v2_prob', 'min'),
        max_meta=('meta_v2_prob', 'max'),
    )
    print(summary)

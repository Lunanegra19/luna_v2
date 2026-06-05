import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def generate_timeline_plot():
    data_dir = Path(r"G:\Mi unidad\ia\Luna v1\data")
    
    # Load all 3 dataset splits
    df_train = pd.read_parquet(data_dir / "features/features_train.parquet", columns=['close'])
    df_val = pd.read_parquet(data_dir / "features/features_validation.parquet", columns=['close'])
    df_holdout = pd.read_parquet(data_dir / "features/features_holdout.parquet", columns=['close'])
    
    # Load trades
    trades_path = data_dir / "reports/oos_trades.parquet"
    if trades_path.exists():
        df_trades = pd.read_parquet(trades_path)
    else:
        df_trades = pd.DataFrame()

    # Combine for a continuous line
    df_all = pd.concat([df_train, df_val, df_holdout]).sort_index()
    # Drop duplicates just in case there's overlap we want to see on the X axis, but keep first
    df_all_unique = df_all[~df_all.index.duplicated(keep='first')]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot main continuous line
    ax.plot(df_all_unique.index, df_all_unique['close'], color='gray', linewidth=1, alpha=0.5, label='BTC Close Price')
    
    # Shade regions using actual start/end of indexes
    regions = [
        ("Training (In-Sample)", df_train.index.min(), df_train.index.max(), 'blue', 0.2),
        ("Validation", df_val.index.min(), df_val.index.max(), 'orange', 0.3),
        ("Holdout (OOS)", df_holdout.index.min(), df_holdout.index.max(), 'purple', 0.3)
    ]
    
    # Note overlaps
    overlaps = []
    if df_train.index.max() > df_val.index.min():
        overlaps.append(f"Overlap Train/Val: {df_val.index.min()} to {df_train.index.max()}")
    if df_val.index.max() > df_holdout.index.min():
        overlaps.append(f"Overlap Val/Holdout: {df_holdout.index.min()} to {df_val.index.max()}")
    
    # Shade
    for label, start, end, color, alpha in regions:
        ax.axvspan(start, end, color=color, alpha=alpha, label=f"{label}\n{start.strftime('%Y-%m-%d')} a {end.strftime('%Y-%m-%d')}")
        
    # Plot Trades
    if not df_trades.empty:
        # Plot trade entry points
        ax.scatter(df_trades['entry_time'], df_trades['entry_price'], 
                   color='cyan', marker='^', s=100, zorder=5, label=f'Trades ({len(df_trades)})')
        
    ax.set_title("Auditoría de Fechas del Pipeline (IS vs Val vs OOS)", fontsize=16, pad=20, color='white')
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(color='white', alpha=0.1)
    
    # Add text box with overlaps if any
    info_text = "TRAMOS DE DATOS:\n\n"
    for label, start, end, _, _ in regions:
        info_text += f"[{label}]\n  {start} -> {end}\n  Size: {end - start}\n\n"
        
    info_text += "ANALISIS DE SOLAPAMIENTOS:\n"
    if overlaps:
        for over in overlaps:
            info_text += f"! PELIGRO: {over}\n"
    else:
        info_text += "OK: 0 solapamientos detectados. Particiones limpias.\n"
        
    plt.figtext(0.15, 0.15, info_text, bbox=dict(facecolor='black', alpha=0.8, edgecolor='red' if overlaps else 'green'), 
                color='white', fontsize=10, fontfamily='monospace')
                
    out_path = data_dir / "reports/timeline_audit.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Grafico guardado en: {out_path}")
    print(info_text)

if __name__ == "__main__":
    generate_timeline_plot()

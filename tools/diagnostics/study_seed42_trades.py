import pandas as pd
import numpy as np
from pathlib import Path

# Configurar rutas
wfb_dir = Path("g:/Mi unidad/ia/luna_v2/data/reports/wfb")

def analyze_trades():
    print("================================================================")
    print("  ESTUDIO PROFUNDO DE TRADES OOS - SEMILLA 42 (VENTANAS W1, W2, W3)")
    print("================================================================\n")
    
    # 1. Cargar trades de las 3 ventanas
    dfs = []
    for w in [1, 2, 3]:
        p = wfb_dir / f"oos_trades_W{w}_seed42.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df['window'] = f"W{w}"
            dfs.append(df)
        else:
            print(f"Advertencia: No existe {p}")
            
    if not dfs:
        print("Error: No se encontraron parquets de trades para Seed 42.")
        return
        
    trades = pd.concat(dfs, ignore_index=False)
    # Reset index pero conservar timestamp
    trades = trades.reset_index()
    if 'timestamp' in trades.columns:
        trades['timestamp'] = pd.to_datetime(trades['timestamp'], utc=True)
        trades = trades.sort_values('timestamp')
    
    n_total = len(trades)
    print(f"Total de trades combinados (W1+W2+W3): {n_total}")
    
    # Calcular Sharpe global base
    def get_sharpe(df_sub):
        n = len(df_sub)
        if n <= 1:
            return 0.0
        ret_mean = df_sub['return_pct'].mean()
        ret_std = df_sub['return_pct'].std()
        if ret_std < 1e-10:
            return 0.0
        # Sharpe anualizado simplificado usando la duración temporal de la muestra
        days = (df_sub['timestamp'].max() - df_sub['timestamp'].min()).days
        if days <= 0:
            days = 90 * len(df_sub['window'].unique()) # aproximado (90 dias por ventana)
        n_per_year = n / (days / 365.25)
        return (ret_mean / ret_std) * (n_per_year ** 0.5)

    def get_stats(df_sub):
        n = len(df_sub)
        if n == 0:
            return {"trades": 0, "wr": 0.0, "ev_pct": 0.0, "sharpe": 0.0, "profit_factor": 0.0, "ret_acum": 0.0}
        wr = (df_sub['return_pct'] > 0).mean() * 100
        ev_pct = df_sub['return_pct'].mean() * 100
        sharpe = get_sharpe(df_sub)
        
        gains = df_sub[df_sub['return_pct'] > 0]['return_pct'].sum()
        losses = abs(df_sub[df_sub['return_pct'] < 0]['return_pct'].sum())
        pf = gains / losses if losses > 0 else float('inf') if gains > 0 else 1.0
        
        # Retorno acumulado compuesto simplificado
        ret_acum = (1 + df_sub['return_pct']).prod() - 1
        
        return {
            "trades": n,
            "wr": wr,
            "ev_pct": ev_pct,
            "sharpe": sharpe,
            "profit_factor": pf,
            "ret_acum": ret_acum * 100
        }

    base_stats = get_stats(trades)
    print("\n--- RENDIMIENTO BASE COMBINADO ---")
    print(f"  Trades        : {base_stats['trades']}")
    print(f"  Win Rate      : {base_stats['wr']:.2f}%")
    print(f"  EV Medio (%)  : {base_stats['ev_pct']:.4f}%")
    print(f"  Sharpe Anual  : {base_stats['sharpe']:.4f}")
    print(f"  Profit Factor : {base_stats['profit_factor']:.4f}")
    print(f"  Retorno Acum  : {base_stats['ret_acum']:.4f}%")
    
    # Rendimiento por ventana
    print("\n--- RENDIMIENTO BASE POR VENTANA ---")
    for w in sorted(trades['window'].unique()):
        w_df = trades[trades['window'] == w]
        w_stats = get_stats(w_df)
        print(f"  {w} | Trades: {w_stats['trades']:2d} | WR: {w_stats['wr']:.1f}% | Sharpe: {w_stats['sharpe']:.3f} | Ret Medio: {w_stats['ev_pct']:.4f}% | Ret Acum: {w_stats['ret_acum']:.3f}%")

    # 2. Análisis por Dirección
    print("\n--- ANÁLISIS POR DIRECCIÓN ---")
    for d in ['long', 'short']:
        d_df = trades[trades['direction'] == d]
        if len(d_df) > 0:
            d_stats = get_stats(d_df)
            print(f"  {d.upper():5s} | Trades: {d_stats['trades']:2d} | WR: {d_stats['wr']:.1f}% | Sharpe: {d_stats['sharpe']:.3f} | Ret Medio: {d_stats['ev_pct']:.4f}%")
        else:
            print(f"  {d.upper():5s} | No hay trades")

    # 3. Análisis por Régimen HMM
    print("\n--- ANÁLISIS POR RÉGIMEN HMM ---")
    if 'hmm_regime' in trades.columns:
        for r, r_df in trades.groupby('hmm_regime'):
            r_stats = get_stats(r_df)
            print(f"  {r:25s} | Trades: {r_stats['trades']:2d} | WR: {r_stats['wr']:.1f}% | Sharpe: {r_stats['sharpe']:.3f} | Ret Medio: {r_stats['ev_pct']:.4f}%")
    else:
        print("  Columna 'hmm_regime' no encontrada.")

    # 4. Análisis por Distancia OOD (Out-of-Distribution KL Distance)
    print("\n--- ANÁLISIS POR DISTANCIA OOD (KL DISTANCE) ---")
    if 'ood_kl_distance' in trades.columns:
        # Ver distribución
        q = trades['ood_kl_distance'].quantile([0.25, 0.5, 0.75, 0.9])
        print(f"  Cuantiles ood_kl_distance: Q25={q[0.25]:.4f}, Q50={q[0.5]:.4f}, Q75={q[0.75]:.4f}, Q90={q[0.9]:.4f}")
        # Probar descartar trades con alta distancia OOD
        for max_kl in [0.15, 0.20, 0.25, 0.30]:
            f_df = trades[trades['ood_kl_distance'] <= max_kl]
            f_stats = get_stats(f_df)
            print(f"  Filtro ood_kl_distance <= {max_kl:.2f} | Trades: {f_stats['trades']:2d} ({len(f_df)/n_total:.1%} activos) | WR: {f_stats['wr']:.1f}% | Sharpe: {f_stats['sharpe']:.3f} | Ret Acum: {f_stats['ret_acum']:.3f}%")
    else:
        print("  Columna 'ood_kl_distance' no encontrada.")

    # 5. Análisis del MetaLabeler v2 (Probabilidad de Éxito)
    print("\n--- ANÁLISIS DE UMBRAL DE METAPROBABILIDAD (meta_v2_prob) ---")
    if 'meta_v2_prob' in trades.columns:
        q_meta = trades['meta_v2_prob'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  Cuantiles meta_v2_prob: Q10={q_meta[0.10]:.4f}, Q25={q_meta[0.25]:.4f}, Q50={q_meta[0.5]:.4f}, Q75={q_meta[0.75]:.4f}, Q90={q_meta[0.90]:.4f}")
        
        # Simular subir el umbral mínimo
        for th in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62]:
            f_df = trades[trades['meta_v2_prob'] >= th]
            f_stats = get_stats(f_df)
            # Por ventana
            w_details = []
            for w in ['W1', 'W2', 'W3']:
                w_sub = f_df[f_df['window'] == w]
                w_details.append(f"{w}: {len(w_sub)}t (WR {(w_sub['return_pct']>0).mean()*100:.0f}%)")
            w_str = " | ".join(w_details)
            print(f"  Umbral >= {th:.2f} | Trades: {f_stats['trades']:2d} ({len(f_df)/n_total:.1%}) | WR: {f_stats['wr']:.1f}% | Sharpe: {f_stats['sharpe']:.3f} | Ret Acum: {f_stats['ret_acum']:.3f}% | ({w_str})")
    else:
        print("  Columna 'meta_v2_prob' no encontrada.")

    # 6. Combinación de Filtros Inteligentes (Optimización Teórica)
    print("\n--- SIMULACIÓN DE COMBINACIONES DE MEJORA ---")
    
    # Idea 1: Meta_v2_prob >= 0.55
    if 'meta_v2_prob' in trades.columns:
        print("\n[Propuesta A] Subir Umbral de MetaLabeler a 0.55")
        df_prop = trades[trades['meta_v2_prob'] >= 0.55]
        show_proposal_results(df_prop, get_stats)

    # Idea 2: Meta_v2_prob >= 0.55 y Evitar Regímenes Conflictivos
    # Veamos qué regímenes son malos:
    # Del análisis HMM, si hay alguno con Sharpe muy negativo, lo podemos excluir.
    if 'meta_v2_prob' in trades.columns and 'hmm_regime' in trades.columns:
        # Excluir regímenes malos si los hay
        bad_regimes = []
        for r, r_df in trades.groupby('hmm_regime'):
            if r_df['return_pct'].mean() < -0.01 / 100: # Retorno negativo significativo
                bad_regimes.append(r)
        
        print(f"\n[Propuesta B] Umbral Meta >= 0.55 AND Excluir Regímenes con EV Negativo {bad_regimes}")
        df_prop2 = trades[(trades['meta_v2_prob'] >= 0.55) & (~trades['hmm_regime'].isin(bad_regimes))]
        show_proposal_results(df_prop2, get_stats)

        # Idea 3: Umbral Meta >= 0.54 AND ood_kl_distance <= 0.25 (Evitar anomalías extremas)
        if 'ood_kl_distance' in trades.columns:
            print("\n[Propuesta C] Umbral Meta >= 0.54 AND Evitar Anomalías OOD (kl_dist <= 0.25)")
            df_prop3 = trades[(trades['meta_v2_prob'] >= 0.54) & (trades['ood_kl_distance'] <= 0.25)]
            show_proposal_results(df_prop3, get_stats)

def show_proposal_results(df_sub, get_stats_fn):
    stats = get_stats_fn(df_sub)
    print(f"  Trades totales: {stats['trades']} | Win Rate: {stats['wr']:.2f}% | Sharpe: {stats['sharpe']:.4f} | RetAcum: {stats['ret_acum']:.4f}% | ProfitFactor: {stats['profit_factor']:.4f}")
    # Por ventana
    for w in sorted(df_sub['window'].unique()):
        w_df = df_sub[df_sub['window'] == w]
        w_stats = get_stats_fn(w_df)
        print(f"    {w} | Trades: {w_stats['trades']:2d} | WR: {w_stats['wr']:.1f}% | Sharpe: {w_stats['sharpe']:.3f} | Ret Medio: {w_stats['ev_pct']:.4f}%")

if __name__ == "__main__":
    analyze_trades()

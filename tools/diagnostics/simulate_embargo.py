"""
tools/diagnostics/simulate_embargo.py
======================================
Simulación de alternativas al embargo dinámico sobre trades OOS ya generados.

NO requiere reentrenamiento. Lee los oos_trades.parquet de cada ventana WFB,
concatena en orden cronológico y re-aplica distintos escenarios de embargo
para proyectar cuántos trades sobrevivirían y con qué métricas estadísticas.

Escenarios testeados:
  A) Embargo actual     → BULL_TREND_WEAK=72H, otros regímenes según mapa
  B) Embargo reducido   → BULL_TREND_WEAK=24H (resto igual)
  C) Embargo mínimo     → todos los regímenes a 24H
  D) Sin embargo        → 0H (todas las señales por ventana se consolidan)
  E) Solo ventanas      → usa los trades raw de cada ventana sin re-embargar el OOS final

Uso:
    python tools/diagnostics/simulate_embargo.py [--seed SEED] [--all]

Ejemplo:
    python tools/diagnostics/simulate_embargo.py --seed 38990
    python tools/diagnostics/simulate_embargo.py --all
"""

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Importar configuración institucional para garantizar No-Fallback
try:
    from luna.core.config import settings as _cfg
except ImportError:
    import yaml
    _cfg_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    class _MockCfg:
        pass
    _cfg = _MockCfg()
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _yaml_data = yaml.safe_load(_f)
        _cfg.stat = _yaml_data.get("stat", {})
        _cfg.xgboost = _yaml_data.get("xgboost", {})

# Forzar UTF-8 en stdout para evitar UnicodeEncodeError en Windows cp1252
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuración institucional (SOP thresholds)
# ---------------------------------------------------------------------------
# Leemos estricto desde settings.yaml (No-Fallback)
_stat = getattr(_cfg, "stat", {}) if hasattr(_cfg, "stat") else getattr(_cfg, "gauntlet", {})
if isinstance(_stat, dict):
    MIN_TRADES      = int(_stat.get("min_trades", 30))
    MIN_DSR         = float(_stat.get("min_dsr", 0.75))
    MAX_PBO_PCT     = float(_stat.get("max_pbo", 0.22)) * 100.0
    MAX_DD_PCT      = float(_stat.get("max_drawdown", 0.60)) * 100.0
else:
    MIN_TRADES      = int(getattr(_stat, "min_trades", 30))
    MIN_DSR         = float(getattr(_stat, "min_dsr", 0.75))
    MAX_PBO_PCT     = float(getattr(_stat, "max_pbo", 0.22)) * 100.0
    MAX_DD_PCT      = float(getattr(_stat, "max_drawdown", 0.60)) * 100.0

# ---------------------------------------------------------------------------
# Mapa de embargo por régimen (estado actual de producción)
# ---------------------------------------------------------------------------
_xgb = getattr(_cfg, "xgboost", {})
if isinstance(_xgb, dict):
    DEFAULT_WAIT_HOURS = float(_xgb.get("embargo_hours", 72.0))
else:
    DEFAULT_WAIT_HOURS = float(getattr(_xgb, "embargo_hours", 72.0))

HMM_EMBARGO_PRODUCTION = {
    "1_BULL_TREND":        72.0,
    "1_VOLATILE_BULL":     96.0,
    "1_BULL_GRIND":        72.0,
    "2_CALM_RANGE":       144.0,
    "2_VOLATILE_RANGE":   168.0,
    "3_CALM_BEAR":        168.0,
    "3_BEAR_CRASH":       168.0,
    "4_BEAR_FORCED":      168.0,
    "1_BULL_TREND_B":      72.0,
    "1_BULL_TREND_C":      72.0,
    "1_BULL_TREND_D":      72.0,
    "1_BULL_TREND_WEAK":   72.0,   # RÉGIMEN DOMINANTE
    "1_VOLATILE_BULL_B":   96.0,
    "1_VOLATILE_BULL_C":   96.0,
    "1_VOLATILE_BULL_D":   96.0,
    "2_CALM_RANGE_B":     144.0,
    "2_CALM_RANGE_C":     144.0,
    "2_VOLATILE_RANGE_B": 168.0,
    "3_CALM_BEAR_B":      168.0,
    "3_BEAR_CRASH_B":     168.0,
}


def _build_embargo_map(scenario: str) -> dict:
    """Construye el mapa de embargo según el escenario."""
    base = HMM_EMBARGO_PRODUCTION.copy()
    if scenario == "production":
        return base
    elif scenario == "reduced_weak":
        # Solo cambia BULL_TREND_WEAK de 72H a 24H
        new = base.copy()
        new["1_BULL_TREND_WEAK"] = 24.0
        new["1_BULL_TREND"]      = 24.0   # también reducimos el base
        new["1_BULL_TREND_B"]    = 24.0
        new["1_BULL_TREND_C"]    = 24.0
        new["1_BULL_TREND_D"]    = 24.0
        print("[SCENARIO] reduced_weak: BULL_TREND_WEAK/B/C/D/base 72H -> 24H")
        return new
    elif scenario == "all_24h":
        new = {k: 24.0 for k in base}
        print("[SCENARIO] all_24h: todos los regímenes a 24H")
        return new
    elif scenario == "no_embargo":
        new = {k: 0.0 for k in base}
        print("[SCENARIO] no_embargo: embargo = 0H (todas las señales)")
        return new
    else:
        raise ValueError(f"Escenario desconocido: {scenario}")


def apply_embargo_simulation(
    df: pd.DataFrame,
    embargo_map: dict,
    fallback: float = 168.0,
) -> pd.DataFrame:
    """
    Re-aplica embargo secuencial al DataFrame de trades concatenado.
    El índice debe ser DatetimeTZAware y ordenado cronológicamente.
    Devuelve solo los trades que superan el embargo.
    """
    if df.empty:
        return df

    regime_col = next(
        (c for c in ["hmm_regime", "HMM_Semantic", "regime"] if c in df.columns),
        None,
    )

    selected_indices = []
    last_time = None

    for ts, row in df.iterrows():
        regime = str(row[regime_col]) if regime_col else "UNKNOWN"
        emb_h = embargo_map.get(regime, fallback)

        if emb_h == 0.0:
            # Sin embargo: todos pasan
            selected_indices.append(ts)
            last_time = ts
        elif last_time is None:
            selected_indices.append(ts)
            last_time = ts
        else:
            delta_h = (ts - last_time).total_seconds() / 3600.0
            if delta_h >= emb_h:
                selected_indices.append(ts)
                last_time = ts

    return df.loc[selected_indices]


# ---------------------------------------------------------------------------
# Métricas estadísticas
# ---------------------------------------------------------------------------

def compute_sharpe(returns: pd.Series) -> float:
    """Sharpe anualizado (asume retornos en fracción de capital, 4H barras)."""
    if len(returns) < 2:
        return float("nan")
    bars_per_year = 365 * 6  # 4H bars
    mean_r = returns.mean()
    std_r  = returns.std(ddof=1)
    if std_r == 0:
        return float("nan")
    return mean_r / std_r * math.sqrt(bars_per_year)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Máximo drawdown como fracción del capital (0-1)."""
    if returns.empty:
        return float("nan")
    equity = (1 + returns).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    return float(abs(dd.min()))


def compute_dsr(returns: pd.Series, n_trials: int = 100) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Aproximación con corrección por selección de hiperparámetros (n_trials).
    """
    n = len(returns)
    if n < 3:
        return 0.0
    sr = compute_sharpe(returns)
    if math.isnan(sr) or sr <= 0:
        return 0.0
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())
    # SR máximo esperado por azar (corrección de Bailey)
    gamma = 0.5772156649  # Euler-Mascheroni
    e_max_sr = (1 - gamma) * math.sqrt(2 * math.log(n_trials)) + \
               gamma / math.sqrt(2 * math.log(n_trials))
    # DSR: probabilidad de que SR > E[SR_max]
    _variance_sr = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1)
    if _variance_sr <= 0:
        # Combinación extrema de skew/kurtosis — DSR indeterminado
        return 0.0
    sigma_sr = math.sqrt(_variance_sr)
    if sigma_sr <= 0:
        return 0.0
    z = (sr - e_max_sr) / sigma_sr
    # CDF normal estándar
    try:
        from scipy.stats import norm
        dsr = float(norm.cdf(z))
    except ImportError:
        # Aproximación sin scipy
        dsr = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(0.0, min(1.0, dsr))


def compute_pbo(returns: pd.Series, n_blocks: int = 8) -> float:
    """
    Probability of Backtest Overfitting (CSCV, Bailey et al.).
    Estimación simplificada: si n < n_blocks*2, devuelve 0.5 (indeterminado).
    """
    n = len(returns)
    if n < n_blocks * 2:
        return 0.5  # indeterminado
    # Dividir en N bloques, combinaciones IS/OOS
    block_size = n // n_blocks
    blocks = [returns.iloc[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    lambdas = []
    for i in range(n_blocks):
        is_blocks = [blocks[j] for j in range(n_blocks) if j != i]
        oos_block = blocks[i]
        is_sr  = compute_sharpe(pd.concat(is_blocks))
        oos_sr = compute_sharpe(oos_block)
        if not math.isnan(is_sr) and not math.isnan(oos_sr):
            lambdas.append(1 if oos_sr < is_sr else 0)
    if not lambdas:
        return 0.5
    return sum(lambdas) / len(lambdas)


def compute_win_rate(df: pd.DataFrame) -> float:
    win_col = next((c for c in ["is_win", "label", "outcome"] if c in df.columns), None)
    if win_col:
        return float(df[win_col].mean())
    ret_col = next((c for c in ["return_pct", "ret_pct"] if c in df.columns), None)
    if ret_col:
        return float((df[ret_col] > 0).mean())
    return float("nan")


def evaluate_scenario(df: pd.DataFrame, scenario_name: str) -> dict:
    """Calcula todas las métricas para un DataFrame de trades dado."""
    n = len(df)
    if n == 0:
        return {
            "scenario": scenario_name, "n_trades": 0,
            "win_rate": 0.0, "total_ret_pct": 0.0,
            "sharpe": float("nan"), "max_dd_pct": float("nan"),
            "dsr": 0.0, "pbo": 0.5,
            "pass_trades": False, "pass_dsr": False,
            "pass_pbo": False, "pass_dd": False,
            "APPROVED": False,
        }

    ret_col = next((c for c in ["return_pct", "ret_pct"] if c in df.columns), None)
    returns = df[ret_col] if ret_col else pd.Series([], dtype=float)

    wr      = compute_win_rate(df)
    sharpe  = compute_sharpe(returns)
    max_dd  = compute_max_drawdown(returns)
    dsr     = compute_dsr(returns)
    pbo     = compute_pbo(returns)
    tot_ret = float(returns.sum()) * 100 if not returns.empty else float("nan")

    pass_trades = n >= MIN_TRADES
    pass_dsr    = (not math.isnan(dsr)) and dsr >= MIN_DSR
    pass_pbo    = (not math.isnan(pbo)) and pbo * 100 <= MAX_PBO_PCT
    pass_dd     = (not math.isnan(max_dd)) and max_dd * 100 <= MAX_DD_PCT
    approved    = pass_trades and pass_dsr and pass_pbo and pass_dd

    return {
        "scenario":     scenario_name,
        "n_trades":     n,
        "win_rate":     round(wr * 100, 1),
        "total_ret_pct": round(tot_ret, 3),
        "sharpe":       round(sharpe, 3) if not math.isnan(sharpe) else None,
        "max_dd_pct":   round(max_dd * 100, 1) if not math.isnan(max_dd) else None,
        "dsr":          round(dsr, 4),
        "pbo":          round(pbo * 100, 1),
        "pass_trades":  pass_trades,
        "pass_dsr":     pass_dsr,
        "pass_pbo":     pass_pbo,
        "pass_dd":      pass_dd,
        "APPROVED":     approved,
    }


def print_result(r: dict):
    ok = lambda b: "[OK]" if b else "[--]"
    approved_str = "[APROBADO]" if r["APPROVED"] else "[RECHAZADO]"
    print(f"  Trades : {r['n_trades']:>4}  {ok(r['pass_trades'])}  (min={MIN_TRADES})")
    print(f"  WinRate: {r['win_rate']:>5.1f}%")
    print(f"  Sharpe : {r['sharpe']}")
    print(f"  MaxDD  : {r['max_dd_pct']:>5.1f}%  {ok(r['pass_dd'])}  (max={MAX_DD_PCT}%)")
    print(f"  DSR    : {r['dsr']:>6.4f}  {ok(r['pass_dsr'])}  (min={MIN_DSR})")
    print(f"  PBO    : {r['pbo']:>5.1f}%   {ok(r['pass_pbo'])}  (max={MAX_PBO_PCT}%)")
    print(f"  Ret Tot: {r['total_ret_pct']:>+7.3f}%")
    print(f"  VEREDICTO: {approved_str}")


def run_seed_simulation(seed: str, runs_base: Path):
    """
    Ejecuta todos los escenarios de embargo para una seed dada.
    Busca el run más reciente que contenga esa seed.
    """
    # Buscar el run más reciente para la seed con todos los parquets
    candidates = sorted(runs_base.glob(f"WFB_*_seed{seed}"), reverse=True)
    run_dir = None
    for c in candidates:
        seed_subdir = c / f"seed{seed}"
        if not seed_subdir.exists():
            seed_subdir = c
        parquets = list(seed_subdir.rglob("oos_trades.parquet"))
        if parquets:
            run_dir = c
            break

    if run_dir is None:
        print(f"[ERROR] No se encontró run con parquets para seed={seed}")
        return

    seed_subdir = run_dir / f"seed{seed}"
    if not seed_subdir.exists():
        seed_subdir = run_dir

    print(f"\n{'='*70}")
    print(f"SEED {seed}  |  Run: {run_dir.name}")
    print(f"{'='*70}")

    # Cargar y concatenar todos los parquets en orden cronológico
    all_frames = []
    for w_dir in sorted(seed_subdir.glob("W*")):
        p = w_dir / "oos_trades.parquet"
        if not p.exists():
            print(f"  {w_dir.name}: sin parquet — ventana vacía (posiblemente BEAR)")
            continue
        df_w = pd.read_parquet(p)
        if df_w.empty:
            print(f"  {w_dir.name}: DataFrame vacío")
            continue
        n = len(df_w)
        regime_col = next((c for c in ["hmm_regime","HMM_Semantic"] if c in df_w.columns), None)
        regime_dist = dict(df_w[regime_col].value_counts()) if regime_col else {}
        ret_col = next((c for c in ["return_pct","ret_pct"] if c in df_w.columns), None)
        mean_ret = df_w[ret_col].mean()*100 if ret_col else float("nan")
        print(f"  {w_dir.name}: {n:>3} trades | ret_medio={mean_ret:+.4f}% | regímenes={regime_dist}")
        all_frames.append(df_w)

    if not all_frames:
        print(f"  [ERROR] No hay trades para esta seed")
        return

    df_combined = pd.concat(all_frames).sort_index()
    n_raw = len(df_combined)
    date_range = f"{df_combined.index.min().date()} -> {df_combined.index.max().date()}"
    print(f"\n  Total raw (sin embargo OOS final): {n_raw} trades | {date_range}")

    # -------------------------------------------------------------------
    # ESCENARIO E: Sin re-embargar (usar raw de ventanas tal cual)
    # -------------------------------------------------------------------
    print(f"\n  {'-'*60}")
    print(f"  [E] SIN RE-EMBARGO (trades raw por ventana, igual que WFB)")
    print(f"  {'-'*60}")
    r_e = evaluate_scenario(df_combined, "sin_re_embargo")
    print_result(r_e)

    # -------------------------------------------------------------------
    # Escenarios con re-embargo sobre el OOS consolidado
    # -------------------------------------------------------------------
    scenarios = [
        ("production",    "A) Producción actual   (72H BULL_WEAK, 168H fallback)"),
        ("reduced_weak",  "B) Reducido bull-trend  (24H BULL_WEAK/B/C/D/base)"),
        ("all_24h",       "C) Embargo mínimo       (todos regímenes 24H)"),
        ("no_embargo",    "D) Sin embargo          (0H — todas las señales)"),
    ]

    for sc_key, sc_label in scenarios:
        emb_map = _build_embargo_map(sc_key)
        df_sc   = apply_embargo_simulation(df_combined, emb_map, DEFAULT_WAIT_HOURS)
        print(f"\n  {'-'*60}")
        print(f"  [{sc_key.upper()}] {sc_label}")
        print(f"  {'-'*60}")
        print(f"  Candidatos: {n_raw} -> retenidos: {len(df_sc)}")
        r = evaluate_scenario(df_sc, sc_key)
        print_result(r)

    return


def run_all_seeds(runs_base: Path):
    """Ejecuta la simulación para todas las seeds con parquets disponibles de hoy."""
    # Buscar runs del 20260518 con parquets
    today_runs = sorted(runs_base.glob("WFB_20260518_*"))
    seeds_done = set()
    
    for run_dir in today_runs:
        # Extraer seed del nombre de directorio
        name = run_dir.name
        if "seed" not in name:
            continue
        seed = name.split("seed")[-1]
        if seed in seeds_done:
            continue
        # Verificar que tenga parquets
        parquets = list(run_dir.rglob("oos_trades.parquet"))
        if not parquets:
            continue
        seeds_done.add(seed)
        run_seed_simulation(seed, runs_base)

    print(f"\n{'='*70}")
    print(f"Simulación completada para {len(seeds_done)} seeds: {seeds_done}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Simula escenarios de embargo sobre trades OOS ya generados."
    )
    parser.add_argument(
        "--seed", type=str, default=None,
        help="Seed a simular (ej: 38990). Si no se especifica, usa --all."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Simular todas las seeds con datos disponibles."
    )
    parser.add_argument(
        "--runs-dir", type=str,
        default=r"G:\Mi unidad\ia\luna_v2\data\runs",
        help="Directorio base de runs WFB."
    )
    args = parser.parse_args()

    runs_base = Path(args.runs_dir)
    if not runs_base.exists():
        print(f"[ERROR] runs_dir no existe: {runs_base}")
        sys.exit(1)

    print("=" * 70)
    print("SIMULADOR DE EMBARGO - Luna V2 (sin reentrenamiento)")
    print("=" * 70)
    print(f"Umbrales institucionales: min_trades={MIN_TRADES} | "
          f"min_DSR={MIN_DSR} | max_PBO={MAX_PBO_PCT}% | max_DD={MAX_DD_PCT}%")

    if args.seed:
        run_seed_simulation(args.seed, runs_base)
    elif args.all:
        run_all_seeds(runs_base)
    else:
        # Por defecto: todas las seeds de hoy
        run_all_seeds(runs_base)


if __name__ == "__main__":
    main()

"""arch01_exit_mechanism_audit.py
Verifica que mecanismo de salida esta activo realmente.
El 100% de los trades tiene |ret| < 0.3%, pero vbh=72H y tbm_min_return=0.003 (0.3%).
Esto es imposible si el TBM funciona correctamente — ningun trade deberia cerrarse
por TP sin haber alcanzado al menos el tbm_min_return=0.3%.
HIPOTESIS: hay otra logica de salida (stop-loss fijo, trailing stop, o los
           multiplicadores ATR estan generando TP/SL en rangos muy pequenos).
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("[ARCH-01] MECANISMO DE SALIDA — ¿POR QUE ret < 0.3% EN TODOS?")
print("="*70)

# ── Cargar trades ─────────────────────────────────────────────────────────────
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
runs_dir = ROOT / "data" / "runs"
all_dfs = []
for f in runs_dir.rglob("oos_trades.parquet"):
    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
    if mtime >= cutoff:
        try:
            df = pd.read_parquet(f)
            df["_run_id"] = f.parts[-4]
            df["_window"] = f.parts[-2]
            all_dfs.append(df)
        except Exception:
            pass

df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
print(f"  Trades: {len(df_all)}")
print(f"  Todas las columnas disponibles:")
for col in df_all.columns:
    sample = df_all[col].dropna().iloc[0] if df_all[col].dropna().shape[0] > 0 else "N/A"
    print(f"    {col:35} | ejemplo: {str(sample)[:60]}")

# ── Analizar tribe_mult ───────────────────────────────────────────────────────
print("\n[1] tribe_mult — ¿QUE ES ESTE MULTIPLICADOR?")
print("-"*60)
if "tribe_mult" in df_all.columns:
    tm = df_all["tribe_mult"].dropna()
    print(f"  Valores unicos: {sorted(tm.unique())}")
    print(f"  Media: {tm.mean():.4f}")
    print(f"  Stats: min={tm.min():.4f} max={tm.max():.4f}")
    
    # Correlacion tribe_mult vs retorno
    tm_ret = df_all[["tribe_mult","return_pct"]].dropna()
    if len(tm_ret) > 5:
        corr = tm_ret.corr().iloc[0,1]
        print(f"  Correlacion tribe_mult vs return_pct: {corr:.4f}")

# ── Analizar filter_fallback_level ────────────────────────────────────────────
print("\n[2] filter_fallback_level — ¿QUE FILTRO SE APLICO?")
print("-"*60)
if "filter_fallback_level" in df_all.columns:
    ffl = df_all["filter_fallback_level"].dropna()
    print(f"  Distribucion:")
    print(ffl.value_counts().to_string())

# ── Leer el codigo que genera return_pct ─────────────────────────────────────
print("\n[3] BUSCANDO LA LOGICA DE CALCULO DE return_pct EN EL CODIGO")
print("-"*60)
# Buscar en generate_oos_predictions.py
for script_name in ["generate_oos_predictions.py","generate_oos.py","wfb_worker.py","run_wfb_orchestrator.py"]:
    script_path = ROOT / "scripts" / script_name
    if not script_path.exists():
        script_path = ROOT / "luna" / "models" / script_name
    if not script_path.exists():
        continue
    
    content = script_path.read_text("utf-8", errors="replace")
    lines = content.splitlines()
    
    # Buscar "return_pct" en el codigo
    ret_pct_lines = [(i+1, l) for i, l in enumerate(lines) if "return_pct" in l]
    print(f"\n  {script_name} — {len(ret_pct_lines)} referencias a 'return_pct':")
    for lno, line in ret_pct_lines[:15]:
        print(f"    L{lno:4}: {line.strip()[:110]}")

# Buscar en luna/models/
for py_file in (ROOT / "luna" / "models").glob("*.py"):
    content = py_file.read_text("utf-8", errors="replace")
    if "return_pct" in content and "oos_trades" in content:
        lines = content.splitlines()
        ret_pct_lines = [(i+1, l) for i, l in enumerate(lines) if "return_pct" in l or "oos_trades" in l]
        print(f"\n  {py_file.name} — referencias return_pct/oos_trades:")
        for lno, line in ret_pct_lines[:10]:
            print(f"    L{lno:4}: {line.strip()[:110]}")

print("\n[ARCH-01] Exit mechanism audit completado.")

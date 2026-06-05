"""Investiga en profundidad los bugs críticos del audit."""
import sys, joblib, traceback
from pathlib import Path

sys.path.insert(0, "/root/luna_v2")
ROOT = Path("/root/luna_v2")

print("=" * 70)
print("BUG F — MODELOS .pkl corruptos")
print("=" * 70)
for seed in ["seed99", "seed1337", "seed2025"]:
    for model_name in ["ood_guard.pkl", "hmm_regime.pkl"]:
        pkl_path = ROOT / "data" / "models" / "prod" / seed / model_name
        if pkl_path.exists():
            try:
                obj = joblib.load(pkl_path)
                print(f"  ✅ {seed}/{model_name}: tipo={type(obj).__name__}, keys={list(obj.keys()) if hasattr(obj,'keys') else 'N/A'}")
            except Exception as e:
                print(f"  ❌ {seed}/{model_name}: ERROR = {repr(e)}")
                tb = traceback.format_exc()
                print(f"     Traceback:\n{tb[:500]}")
        else:
            print(f"  ⚠️  {seed}/{model_name}: ARCHIVO NO EXISTE")

print("\n" + "=" * 70)
print("BUG D — Parámetros min_trades y embargo_hours en settings.yaml")
print("=" * 70)
import yaml
settings_path = ROOT / "config" / "settings.yaml"
with open(settings_path) as f:
    raw = yaml.safe_load(f)

print("  Secciones de settings.yaml:", list(raw.keys()))
for section_name, section_data in raw.items():
    if isinstance(section_data, dict):
        for key, val in section_data.items():
            if any(k in key.lower() for k in ["trade", "embargo", "gauntlet", "purge", "wfb"]):
                print(f"  [{section_name}] {key} = {val}")

print("\n" + "=" * 70)
print("BUG E — Tablas heartbeats y trade_history en PostgreSQL")
print("=" * 70)
from luna.database.db_manager import DatabaseManager
from psycopg2.extras import DictCursor
db = DatabaseManager()
with db.get_connection() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]
        print(f"  Tablas existentes en PostgreSQL: {tables}")

print("\n" + "=" * 70)
print("BUG A — 7 features skew: ¿están en el pipeline feature?")
print("=" * 70)
MISSING = [
    "ETF_Flow_Proxy", "FundingRate_EMA3", "FundingRate_Pct90d",
    "OI_High_USD", "OI_Low_USD", "OI_Open_USD", "dv_etf_flow_proxy"
]
fp_src = (ROOT / "luna" / "features" / "feature_pipeline.py").read_text(errors="replace")
for feat in MISSING:
    in_fp = feat in fp_src
    # Buscar también en los fetchers
    fetchers_src = ""
    for fp in (ROOT / "luna" / "data").rglob("fetch_*.py"):
        fetchers_src += fp.read_text(errors="replace")
    in_fetcher = feat in fetchers_src
    base_map = {
        "ETF_Flow_Proxy":    "ETF_Flow_Proxy — raw de ETF fetcher",
        "FundingRate_EMA3":  "FundingRate.ewm(span=3)",
        "FundingRate_Pct90d":"FundingRate.rolling(90*24).rank(pct=True)",
        "OI_High_USD":       "raw de derivatives fetcher",
        "OI_Low_USD":        "raw de derivatives fetcher",
        "OI_Open_USD":       "raw de derivatives fetcher",
        "dv_etf_flow_proxy": "alias de ETF_Flow_Proxy",
    }
    print(f"  {feat}:")
    print(f"    En feature_pipeline.py: {'✅' if in_fp else '❌ NO'}")
    print(f"    En fetchers data/:       {'✅' if in_fetcher else '❌ NO'}")
    print(f"    Fuente esperada:         {base_map.get(feat, '?')}")

"""check_model_headers.py — Verifica si modelos XGBoost son binarios reales o mock JSON"""
from pathlib import Path

PROD = Path("/root/luna_v2/data/models/prod")

for sd in sorted(PROD.iterdir()):
    if not sd.is_dir() or not sd.name.startswith("seed"):
        continue
    bull_path = sd / "xgboost_meta_bull_long.model"
    if bull_path.exists():
        with open(bull_path, "rb") as f:
            header = f.read(10)
        # XGBoost binario real: empieza con bytes no-ASCII o con {L (0x7b 0x4c)
        # Mock JSON: empieza con { seguido de comillas (0x7b 0x22) o { seguido de espacio (0x7b 0x20)
        first_2 = header[:2]
        is_json_mock = (first_2 == b'{"') or (first_2 == b'{ ') or (header[1:2] in [b'"', b' ', b'\n'])
        is_xgb_binary = (first_2 == b'{L') or (header[0:1] == b'{' and header[1:2] == b'\x4c')
        print(f"{sd.name}/xgb_bull: hex={header[:4].hex()} | is_json_mock={is_json_mock} | is_xgb_binary={is_xgb_binary} | size={bull_path.stat().st_size}B")
    else:
        print(f"{sd.name}/xgb_bull: FALTA")

# Tambien verificar seed99 range
p99 = PROD / "seed99" / "xgboost_meta_range_long.model"
print("seed99 range exists:", p99.exists())

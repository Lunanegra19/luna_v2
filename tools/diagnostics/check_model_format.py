"""check_model_format.py — verifica si los modelos prod son mocks o XGBoost real"""
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent.parent

test_files = [
    ROOT / "data" / "models" / "prod" / "seed42" / "xgboost_meta_bull_long.model",
    ROOT / "data" / "models" / "xgboost_meta_bear_long.model",
    ROOT / "data" / "models" / "xgboost_meta_range_long.model",
]

for test_file in test_files:
    if not test_file.exists():
        print(f"NO EXISTE: {test_file.name}")
        continue
    
    with open(test_file, 'rb') as f:
        header = f.read(20)
    
    print(f"\n{test_file.relative_to(ROOT)}")
    print(f"  Header hex: {header.hex()}")
    print(f"  Header bytes[0]: {header[0]:02x} = '{chr(header[0]) if 32 <= header[0] < 127 else '?'}'")
    
    # Intentar leer como texto
    try:
        content = test_file.read_text(encoding='utf-8', errors='replace')[:200].strip()
        if content.startswith('{'):
            try:
                # Verificar si es un mock con mocked:true
                data = json.loads(content[:500] if len(content) > 500 else content)
                mocked = data.get("mocked", False)
                print(f"  Es JSON valido. mocked={mocked}")
                if mocked:
                    print(f"  --> MOCK PLACEHOLDER")
                else:
                    print(f"  --> XGBoost JSON format (real, no mock)")
            except json.JSONDecodeError:
                print(f"  Es JSON parcial (XGBoost JSON format nativo)")
        else:
            print(f"  No empieza con '{{' -> formato binario XGBoost legacy")
    except Exception as e:
        print(f"  Error leyendo como texto: {e}")

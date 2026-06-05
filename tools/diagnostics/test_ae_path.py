import os
import re
from pathlib import Path

window_id = os.environ.get("LUNA_WINDOW_ID", "W2")
root_dir = Path(r"g:\Mi unidad\ia\luna_v2")

m = re.match(r"W(\d+)", window_id)
if m:
    w_idx = int(m.group(1))
    if w_idx > 1:
        prev_window = f"W{w_idx - 1}"
        prev_model_path = root_dir / "data" / "wfb_cache" / prev_window / "models" / "autoencoder_state.pt"
        print(f"Path to check: {prev_model_path}")
        print(f"Exists: {prev_model_path.exists()}")
        if prev_model_path.exists():
            print("Would load state dict!")
        else:
            print("File not found.")

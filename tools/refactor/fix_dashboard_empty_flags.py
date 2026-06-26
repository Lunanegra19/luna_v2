import sys
import re

file_path = "dashboard/server.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                  if s_metrics:
                      run_champions.append(s_metrics)
                  else:
                      if True: run_champions.append({'''

replacement = '''                  if s_metrics:
                      run_champions.append(s_metrics)
                  else:
                      wfb_reports_dir = PROJECT_ROOT / "data" / "reports" / "wfb"
                      empty_windows = {}
                      if wfb_reports_dir.exists():
                          s_raw = s.replace("_LONG", "").replace("_SHORT", "")
                          s_dir = s.split("_")[1].lower() if "_" in s else ""
                          pattern = f"oos_trades_W*_seed{s_raw}_{s_dir}_EMPTY.flag" if s_dir else f"oos_trades_W*_seed{s_raw}_EMPTY.flag"
                          for flag_file in wfb_reports_dir.glob(pattern):
                              w_match = re.search(r"oos_trades_(W\d+)_", flag_file.name)
                              if w_match:
                                  w_name = w_match.group(1)
                                  empty_windows[w_name] = {"trades": 0, "win_rate": 0.0}
                                  
                      run_champions.append({
                          "windows": empty_windows,'''

content = content.replace(target, replacement)
# remove the empty dict windows line
content = content.replace('"windows": {},', '')

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Dashboard server.py updated.")

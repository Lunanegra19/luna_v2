import glob
import re

reports = glob.glob(r"C:\Users\Usuario\Desktop\ia\luna_v2\data\reports\*FINAL_Statistical_Validation_Report.md")

results = {}

for report in reports:
    seed_match = re.search(r"seed(\d+)", report)
    if not seed_match:
        continue
    seed = seed_match.group(1)
    results[seed] = {}
    
    with open(report, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Buscar la tabla WFV:
    # | W1 | 55 | 44.3% | ...
    
    lines = content.split('\n')
    in_wfv_table = False
    for line in lines:
        if "| Ventana | Trades | Win Rate | Rango |" in line:
            in_wfv_table = True
            continue
        
        if in_wfv_table:
            if line.startswith("| W"):
                parts = line.split("|")
                if len(parts) >= 4:
                    w = parts[1].strip()
                    n = parts[2].strip()
                    wr = parts[3].strip()
                    results[seed][w] = {"n": n, "wr": wr}
            elif line.strip() == "" or line.startswith("---"):
                if len(results[seed]) > 0:
                    in_wfv_table = False

# Ahora formatear en markdown
print("## Detalle de Resultados por Semilla y Ventana (WFV)")
print("| Semilla | W1 (Trades - WR) | W2 (Trades - WR) | W3 (Trades - WR) | W4 (Trades - WR) | W5 (Trades - WR) |")
print("|---------|-------------------|-------------------|-------------------|-------------------|-------------------|")

for seed, data in results.items():
    row = [seed]
    for w in ["W1", "W2", "W3", "W4", "W5"]:
        if w in data:
            row.append(f"{data[w]['n']} ({data[w]['wr']})")
        else:
            row.append("-")
    print("| " + " | ".join(row) + " |")


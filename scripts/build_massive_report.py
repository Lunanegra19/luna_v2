import os
import json
import glob
from pathlib import Path
from datetime import datetime, timezone

def build_massive_report():
    base_dir = Path("C:/Users/Usuario/Desktop/ia/luna_v2")
    reports_dir = base_dir / "data" / "reports"
    wfb_reports_dir = reports_dir / "wfb"
    output_file = Path("C:/Users/Usuario/.gemini/antigravity-ide/brain/4611b3b5-daac-4472-ac62-b081a3a08da6/informe_wfb_exhaustivo.md")
    
    # Run start time (approx 2026-06-08 05:30:00 UTC)
    cutoff_time = datetime(2026, 6, 8, 5, 30, 0, tzinfo=timezone.utc).timestamp()
    
    lines = []
    lines.append("# INFORME WFB MASIVO Y EXHAUSTIVO (NIVEL OMEGA)")
    lines.append(f"> Generado el: {datetime.now(timezone.utc).isoformat()}")
    lines.append("\n## INTRODUCCIÓN")
    lines.append("Este documento contiene **ABSOLUTAMENTE TODOS LOS DATOS** de todas las semillas procesadas en la ejecución actual. Incluye los reportes de validación completos, veredictos estadísticos crudos, cortes de embudo por compuerta (Gates), early stops, y telemetría de entrenamiento. Diseñado para inspección en profundidad sin resúmenes superficiales.\n")
    
    # 1. Gather all statistical verdicts and tearsheets
    md_files = list(reports_dir.glob("*_FINAL_Statistical_Validation_Report.md"))
    json_files = list(reports_dir.glob("*_FINAL_statistical_verdict.json"))
    
    # Filter by recency
    md_files = [f for f in md_files if f.stat().st_mtime > cutoff_time]
    json_files = [f for f in json_files if f.stat().st_mtime > cutoff_time]
    
    md_files.sort(key=lambda x: x.stat().st_mtime)
    json_files.sort(key=lambda x: x.stat().st_mtime)
    
    lines.append(f"### SEMILLAS PROCESADAS HASTA EL MOMENTO: {len(md_files)}\n")
    
    # Extract seeds from json filenames (e.g. ..._seed42_...)
    seeds = []
    for jf in json_files:
        name = jf.name
        if "_seed" in name:
            s = name.split("_seed")[1].split("_")[0]
            if s not in seeds:
                seeds.append(s)
                
    # Section: Raw Veridict JSONs
    lines.append("## PARTE 1: VEREDICTOS ESTADÍSTICOS JSON CRUDOS (POR SEMILLA)\n")
    for jf in json_files:
        lines.append(f"### Archivo: `{jf.name}`")
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                lines.append("```json")
                lines.append(json.dumps(data, indent=4))
                lines.append("```\n")
        except Exception as e:
            lines.append(f"> Error leyendo {jf.name}: {e}\n")

    # Section: Gate JSONs from wfb/
    lines.append("## PARTE 2: EMBONADO DE SEÑALES - GATES Y EARLY STOPS\n")
    for s in seeds:
        lines.append(f"### Embudo de Señales: Semilla {s}")
        gate_files = list(wfb_reports_dir.glob(f"*_seed{s}.json"))
        for gf in gate_files:
            if gf.stat().st_mtime > cutoff_time:
                lines.append(f"#### {gf.name}")
                try:
                    with open(gf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        lines.append("```json")
                        lines.append(json.dumps(data, indent=4))
                        lines.append("```\n")
                except:
                    pass

    # Section: Full MD Reports
    lines.append("## PARTE 3: REPORTES DE VALIDACIÓN ESTADÍSTICA COMPLETOS POR SEMILLA\n")
    for mdf in md_files:
        lines.append(f"### ================= REPORTE: {mdf.name} =================")
        try:
            with open(mdf, "r", encoding="utf-8") as f:
                content = f.read()
                lines.append(content)
                lines.append("\n---\n")
        except:
            pass

    # Section: Log extracts (Autoencoder loss, OOD guard)
    lines.append("## PARTE 4: EXTRACTOS CLAVE DEL WORKER LOG (task-9003.log)\n")
    log_path = Path("C:/Users/Usuario/.gemini/antigravity-ide/brain/4611b3b5-daac-4472-ac62-b081a3a08da6/.system_generated/tasks/task-9003.log")
    if log_path.exists():
        lines.append("```log")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Epoch" in line and "MSE Loss" in line:
                    lines.append(line.strip())
                elif "OOD" in line or "HMM" in line:
                    lines.append(line.strip())
        lines.append("```\n")

    final_content = "\n".join(lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_content)
        
    print(f"Report generated at: {output_file}")
    print(f"Total lines: {len(lines)}")

if __name__ == "__main__":
    build_massive_report()

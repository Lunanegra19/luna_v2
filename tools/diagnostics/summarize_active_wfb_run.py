# -*- coding: utf-8 -*-
"""
summarize_active_wfb_run.py
--------------------------
Mantenimiento y Diagnóstico: Lee todos los veredictos de la run WFB actual (2026-06-20)
y genera una tabla resumen con estadísticas clave del ensemble de 29 semillas.
"""
import os
import glob
import json
import re
import pandas as pd
from datetime import datetime

def main():
    reports_dir = r"c:\Users\Usuario\Desktop\ia\luna_v2\data\reports"
    verdict_files = glob.glob(os.path.join(reports_dir, "*_statistical_verdict.json"))
    
    if not verdict_files:
        print("No statistical verdict files found in data/reports.")
        return

    # Filtrar solo archivos generados hoy (2026-06-20)
    today_prefix = "2026-06-20"
    active_verdicts = []
    
    for f in verdict_files:
        basename = os.path.basename(f)
        if basename.startswith(today_prefix):
            # Filtrar runs de la mañana (antes de las 09:50)
            time_match = re.search(r"_T(\d{2})(\d{2})_", basename)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                if hour < 9 or (hour == 9 and minute < 50):
                    continue
            active_verdicts.append(f)
            
    if not active_verdicts:
        print("No verdict files found for today's run.")
        return
        
    # Ordenar por fecha y hora de creación en el nombre de archivo
    # Ej: 2026-06-20_T0951_...
    def extract_time_key(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r"(\d{4}-\d{2}-\d{2})_T(\d{4})", basename)
        if match:
            return match.group(1) + "_" + match.group(2)
        return ""
        
    active_verdicts = sorted(active_verdicts, key=extract_time_key)
    
    records = []
    for f in active_verdicts:
        basename = os.path.basename(f)
        
        # Extraer seed y timestamp del nombre de archivo
        # Ej: 2026-06-20_T1743_WFB_20260620_172427_36576_seed43609_FINAL_statistical_verdict.json
        seed_match = re.search(r"seed(\d+)", basename)
        seed = seed_match.group(1) if seed_match else "Unknown"
        
        time_match = re.search(r"_T(\d{2})(\d{2})_", basename)
        time_str = f"{time_match.group(1)}:{time_match.group(2)}" if time_match else "N/A"
        
        try:
            with open(f, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
            
            metrics = data.get("metrics", {})
            audit = data.get("statistical_audit", {})
            summary = data.get("summary", {})
            flags = data.get("flags", {})
            
            # Obtener métricas
            total_trades = metrics.get("total_trades", summary.get("total_trades", 0))
            win_rate = metrics.get("win_rate", summary.get("win_rate_pct", 0.0) / 100.0)
            win_rate_pct = round(win_rate * 100, 2)
            total_return = round(metrics.get("total_return_pct", summary.get("total_return_pct", 0.0)), 2)
            max_dd = round(metrics.get("max_drawdown_pct", summary.get("max_drawdown_pct", 0.0)), 2)
            sharpe = round(metrics.get("sharpe_crudo", summary.get("sharpe_crudo", 0.0)), 2)
            calmar = round(metrics.get("calmar_ratio", summary.get("calmar_ratio", 0.0)), 2)
            
            # DSR base y DSR ajustado
            dsr_base = round(audit.get("dsr", summary.get("dsr", 0.0)), 4)
            dsr_adj = round(data.get("dsr_adjusted", dsr_base), 4)
            
            approved = data.get("deploy_approved", summary.get("deploy_approved", False))
            rejection = data.get("rejection_reason", "N/A")
            if rejection == "N/A" and not approved:
                # Buscar en flags por qué rechazó
                reasons = []
                if not flags.get("pass_sharpe", True): reasons.append("Sharpe")
                if not flags.get("pass_dsr", True): reasons.append("DSR")
                if not flags.get("pass_pbo", True): reasons.append("PBO")
                if not flags.get("pass_trades", True): reasons.append("Trades")
                if not flags.get("pass_drawdown", True): reasons.append("MaxDD")
                rejection = f"Failed gates: {', '.join(reasons)}" if reasons else "Rejected"
                
            records.append({
                "Time": time_str,
                "Seed": seed,
                "Trades": total_trades,
                "WR %": win_rate_pct,
                "Return %": total_return,
                "MaxDD %": max_dd,
                "Sharpe": sharpe,
                "Calmar": calmar,
                "DSR Base": dsr_base,
                "DSR Adj": dsr_adj,
                "Deploy": "Approved" if approved else "Rejected",
                "Reason": rejection
            })
        except Exception as e:
            print(f"Error parsing {basename}: {e}")
            
    if not records:
        print("No valid records parsed.")
        return
        
    df = pd.DataFrame(records)
    
    # Generar salida formateada
    print("\n# === INFORME DE SEEDS TERMINADAS (RUN ACTIVA 2026-06-20) ===\n")
    print(f"Total seeds completadas hasta ahora: {len(df)} / 29\n")
    
    # Imprimir tabla markdown
    print(df.to_markdown(index=False))
    
    # Resumen cuantitativo del ensemble
    approved_df = df[df["Deploy"] == "Approved"]
    print("\n# === ANÁLISIS DEL ENSEMBLE ===")
    print(f"Semillas Aprobadas (Pasaron el Gauntlet): {len(approved_df)} de {len(df)} ({len(approved_df)/len(df)*100:.1f}%)")
    
    if len(approved_df) > 0:
        print(f"  - Retorno promedio de aprobadas: {approved_df['Return %'].mean():.2f}%")
        print(f"  - Max Drawdown promedio de aprobadas: {approved_df['MaxDD %'].mean():.2f}%")
        print(f"  - Win Rate promedio de aprobadas: {approved_df['WR %'].mean():.2f}%")
        print(f"  - Sharpe promedio de aprobadas: {approved_df['Sharpe'].mean():.2f}%")
        print(f"  - Calmar promedio de aprobadas: {approved_df['Calmar'].mean():.2f}%")
    else:
        print("  - Ninguna semilla ha sido aprobada por el Gauntlet estadístico ajustado (DSR corrigiendo por comparaciones múltiples).")
        
    print(f"\n- Retorno promedio general: {df['Return %'].mean():.2f}%")
    print(f"- Max Drawdown promedio general: {df['MaxDD %'].mean():.2f}%")
    print(f"- Sharpe promedio general: {df['Sharpe'].mean():.2f}%")
    print(f"- Win Rate promedio general: {df['WR %'].mean():.2f}%")

if __name__ == "__main__":
    main()

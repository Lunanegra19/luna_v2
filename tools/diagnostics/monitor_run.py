#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
monitor_run.py — Seguimiento read-only de un run WFB en curso (smoke o full).
Reporta: proceso vivo, fase actual, ventana/seed, banda DVOL aplicada, errores, veredicto.

Uso:
    python tools/diagnostics/monitor_run.py <ruta_log>
    python tools/diagnostics/monitor_run.py  (usa el smoke log por defecto)
"""
import sys, os, re, io, glob
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

DEFAULT_LOG = r"C:/Users/Usuario/AppData/Local/Temp/claude/c--Users-Usuario-Desktop-ia-luna-v2/185cf39a-74f4-4d95-80ce-5e9ffa4955f0/scratchpad/f1_longonly_smoke.log"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHASES = [
    ("Feature Pipeline (Base", "Feature Pipeline Base"),
    ("Build Dataset", "AI Mining"),
    ("Feature Pipeline (Pre-SFI", "Feature Pre-SFI"),
    ("SFI Feature Selection", "SFI (fase larga)"),
    ("Feature Pipeline (Post-SFI", "Feature Post-SFI"),
    ("HMM Regime Model", "HMM"),
    ("XGBoost Core Model", "XGBoost"),
    ("MetaLabeler V2", "MetaLabeler"),
    ("Calibrador de Probabilidades", "Calibrador"),
    ("Generador de Predicciones OOS", "OOS (aplica banda DVOL)"),
    ("Validación Estadística", "Gauntlet"),
]


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    if not os.path.exists(log):
        print(f"[!] log no existe: {log}"); return
    txt = open(log, encoding="utf-8", errors="ignore").read()
    lines = txt.splitlines()
    size = os.path.getsize(log)
    print(f"=== MONITOR | {os.path.basename(log)} ({size//1024} KB, {len(lines)} lineas) ===")

    # Fase actual (ultima fase iniciada)
    last_phase = None; last_phase_line = ""
    for ln in lines:
        for key, name in PHASES:
            if key in ln and ("Iniciando" in ln or "Fase" in ln or key in ln):
                last_phase = name; last_phase_line = ln.strip()[-120:]
    # ventana/seed
    win = re.findall(r"seed(\d+)/(W\d+)", txt)
    cur_win = win[-1] if win else None
    # banda DVOL aplicada?
    dvol = re.findall(r"DVOL_kz fuera de \[([-0-9.]+), ([0-9.]+)\]", txt)
    # errores / fatal
    errs = [l for l in lines if re.search(r"FATAL|Traceback|abortó con código|ERROR.*abort|CRITICO.*FAIL", l)][-5:]
    # veredicto
    verdict = [l for l in lines if "VEREDICTO" in l or "DEPLOY APROBADO" in l or "RECHAZADO" in l][-3:]
    # SFI reuse / compute
    sfi_reuse = txt.count("SFI reutilizado")
    sfi_run = txt.count("Iniciando Fase Compartida: SFI") + txt.count("[D] ")

    print(f"FASE ACTUAL : {last_phase or '(arrancando)'}")
    if cur_win: print(f"VENTANA     : seed{cur_win[0]} / {cur_win[1]}")
    print(f"BANDA DVOL  : {'APLICADA -> ' + str(dvol[-1]) if dvol else 'aun no (no llego a OOS)'}", )
    print(f"SFI         : reutilizadas={sfi_reuse}")
    if verdict:
        print("VEREDICTO   :")
        for v in verdict: print("   ", v.strip()[-140:])
    if errs:
        print("ERRORES/FATAL:")
        for e in errs: print("   ", e.strip()[-160:])
    else:
        print("ERRORES     : ninguno detectado")
    print("ULTIMA LINEA:", lines[-1][-160:] if lines else "(vacio)")


if __name__ == "__main__":
    main()

"""Auditoria de origen de los parquets - verifica si son de una sola run"""
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

data_dir = Path("data/predictions")
files = sorted(data_dir.glob("oos_trades_seed*.parquet"))
print(f"Total archivos parquet: {len(files)}")
print()
print(f"{'Archivo':<45} {'Size_KB':>8} {'Modificado':>22} {'N_trades':>9}  Ventanas")
print("-" * 110)

total_trades = 0
fechas = []
for f in files:
    stat = os.stat(f)
    mod = datetime.fromtimestamp(stat.st_mtime)
    fechas.append(mod)
    d = pd.read_parquet(f)
    w_col = "wfb_window" if "wfb_window" in d.columns else "window"
    wins = sorted(d[w_col].unique()) if w_col in d.columns else ["?"]
    total_trades += len(d)
    print(f"{f.name:<45} {stat.st_size/1024:>8.1f}  {mod.strftime('%Y-%m-%d %H:%M:%S'):>22}  {len(d):>9}  {wins}")

print()
print(f"TOTAL TRADES: {total_trades}")
print()
print(f"Fecha mas antigua : {min(fechas).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Fecha mas reciente: {max(fechas).strftime('%Y-%m-%d %H:%M:%S')}")
delta = max(fechas) - min(fechas)
print(f"Diferencia entre primer y ultimo archivo: {delta}")
print()

secs = delta.total_seconds()
if secs < 3600:
    print("CONCLUSION: Todos los archivos son de la MISMA RUN (diferencia < 1h)")
    print("Los 7.718 trades provienen de UNA SOLA run con config uniforme.")
elif secs < 86400:
    print(f"AVISO: Diferencia de {secs/3600:.1f}h entre archivos -> POSIBLE MEZCLA DE RUNS")
    print("Revisar manualmente cuales seeds son de runs diferentes.")
else:
    print(f"ALERTA: Diferencia de {secs/86400:.1f} dias entre archivos -> HAY MEZCLA DE RUNS")
    print("El analisis sobre 7.718 trades esta CONTAMINADO con runs de otras arquitecturas.")

print()
# Analisis adicional: ver si los timestamps tienen agrupaciones (clusters)
print("Distribucion temporal de los archivos (minutos desde el primero):")
t0 = min(fechas)
for f, mod in sorted(zip(files, fechas), key=lambda x: x[1]):
    diff_min = (mod - t0).total_seconds() / 60
    bar = "#" * int(diff_min / 5)
    print(f"  {f.name:<42}  +{diff_min:>6.1f}min  {bar}")

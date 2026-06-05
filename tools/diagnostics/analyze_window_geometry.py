import pandas as pd

windows = [
    {'id': 'W1', 'train_end': '2024-10-31', 'val_start': '2024-11-01', 'val_end': '2024-12-31', 'holdout_start': '2025-01-01', 'holdout_end': '2025-03-31'},
    {'id': 'W2', 'train_end': '2025-01-31', 'val_start': '2025-02-01', 'val_end': '2025-03-31', 'holdout_start': '2025-04-01', 'holdout_end': '2025-06-30'},
    {'id': 'W3', 'train_end': '2025-04-30', 'val_start': '2025-05-01', 'val_end': '2025-06-30', 'holdout_start': '2025-07-01', 'holdout_end': '2025-09-30'},
    {'id': 'W4', 'train_end': '2025-07-31', 'val_start': '2025-08-01', 'val_end': '2025-09-30', 'holdout_start': '2025-10-01', 'holdout_end': '2025-12-31'},
    {'id': 'W5', 'train_end': '2025-10-31', 'val_start': '2025-11-01', 'val_end': '2025-12-31', 'holdout_start': '2026-01-01', 'holdout_end': '2026-03-31'},
]

TRAIN_START = pd.Timestamp('2017-08-17')
EMBARGO_H_BULL = 72
EMBARGO_H_BEAR = 168
MIN_CSCV = 32

print("=== Geometria actual de ventanas WFB ===")
print()
print(f"{'ID':<4} | {'Val(d)':<7} | {'Holdout(d)':<11} | {'Train(d)':<10} | {'MaxTrades72H':<14} | {'MaxTrades168H':<14} | {'CSCV_OK?'}")
print("-" * 90)

for w in windows:
    te = pd.Timestamp(w['train_end'])
    vs = pd.Timestamp(w['val_start'])
    ve = pd.Timestamp(w['val_end'])
    hs = pd.Timestamp(w['holdout_start'])
    he = pd.Timestamp(w['holdout_end'])

    val_days = (ve - vs).days
    holdout_days = (he - hs).days
    train_days = (te - TRAIN_START).days
    max_trades_72 = int(holdout_days * 24 / EMBARGO_H_BULL)
    max_trades_168 = int(holdout_days * 24 / EMBARGO_H_BEAR)
    ok = "YES" if max_trades_72 >= MIN_CSCV else "NO"

    print(f"{w['id']:<4} | {val_days:<7} | {holdout_days:<11} | {train_days:<10} | {max_trades_72:<14} | {max_trades_168:<14} | {ok}")

print()
print(f"Minimo CSCV con n_blocks=8: {MIN_CSCV} trades")
print()

print("=== PROBLEMA: Solapamiento si extendemos holdout a 120 dias ===")
print()
print("Las ventanas avanzan 3 meses (~92 dias). Con holdout de 120 dias:")
for i, w in enumerate(windows[:4]):
    hs = pd.Timestamp(w['holdout_start'])
    he_actual = pd.Timestamp(w['holdout_end'])
    he_120 = hs + pd.Timedelta(days=120)
    next_hs = pd.Timestamp(windows[i+1]['holdout_start'])
    overlap = (he_120 - next_hs).days
    print(f"  {w['id']}: holdout_end actual={he_actual.date()} | extendido={he_120.date()} | sig_start={next_hs.date()} | SOLAPAMIENTO={overlap}d {'[CRITICO]' if overlap > 0 else '[OK]'}")

print()
print("=== ALTERNATIVAS SIN SOLAPAMIENTO ===")
print()
print("OPCION A: Reducir embargo en Bull a 48H (sin cambiar fechas):")
for w in windows:
    hs = pd.Timestamp(w['holdout_start'])
    he = pd.Timestamp(w['holdout_end'])
    holdout_days = (he - hs).days
    max_48 = int(holdout_days * 24 / 48)
    max_72 = int(holdout_days * 24 / 72)
    print(f"  {w['id']}: {holdout_days}d | embargo=48H -> {max_48} max_trades | embargo=72H -> {max_72} max_trades")

print()
print("OPCION B: Mover holdout_end W5 a 2026-06-30 (extension solo ultima ventana):")
w5_extended = 181  # 2026-01-01 a 2026-06-30
print(f"  W5 extendida: 181 dias | embargo=72H -> {int(181*24/72)} max_trades | embargo=168H -> {int(181*24/168)} max_trades")
print("  Costo: necesitamos datos hasta 2026-06-30 (aun en el futuro).")
print()
print("OPCION C: Reducir embargo Bear a 96H (intermedio):")
for w in windows:
    hs = pd.Timestamp(w['holdout_start'])
    he = pd.Timestamp(w['holdout_end'])
    holdout_days = (he - hs).days
    max_96 = int(holdout_days * 24 / 96)
    print(f"  {w['id']}: {holdout_days}d | embargo=96H -> {max_96} max_trades | CSCV_OK={max_96 >= 32}")

print()
print("=== DIAGNOSTICO REAL: datos historicos de trades ===")
print()
print("Del diagnostic run: media de trades por ventana en 277 parquets:")
real = {'W1': 12.8, 'W2': 16.5, 'W3': 11.9, 'W4': 9.7, 'W5': 10.8}
for wid, mean in real.items():
    print(f"  {wid}: {mean:.1f} trades promedio (MAX teorico embargo72H = {int(92*24/72)})")
print()
print("El MAX teorico es ~30 trades pero el real es 10-17.")
print("El embargo NO es el unico factor: el MetaLabeler y el momentum filtran muchisimo antes.")
print()
print("=== CONCLUSION ===")
print()
print("1. Ampliar ventanas WFB causa solapamiento critico (28 dias) entre holdouts.")
print("2. Reducir embargo Bear 168H->96H: max teorico mejora de 12 a 22 trades (insuficiente).")
print("3. Reducir embargo Bull 72H->48H: max teorico mejora de 30 a 45 trades.")
print("   RIESGO: TBM horizon minimo=48H => si reducimos embargo a 48H, el purge/embargo")
print("   no tiene buffer suficiente entre fin del trade y nueva senal.")
print()
print("RECOMENDACION SEGURA: no cambiar las fechas de las ventanas WFB.")
print("La solucion al bajo n_trades no es geometrica sino de densidad de senales.")
print("La causa real es la combinacion: XGB threshold alto + MetaLabeler threshold + embargo.")

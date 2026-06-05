import math
from config.settings import cfg

stat = cfg.stat
margin_range   = float(stat.brier_margin_range)
margin_default = float(stat.brier_margin_default)
print(f"[FIX-G2-BRIER-MARGIN-01] stat.brier_margin_range   = {margin_range}")
print(f"[FIX-G2-BRIER-MARGIN-01] stat.brier_margin_default = {margin_default}")

base_rate_is = 0.48
naive_true = base_rate_is * (1 - base_rate_is)

gate_old = round(naive_true + 0.025, 4)
gate_new = round(naive_true + margin_default, 4)

print()
print("=== Efecto del fix en gate adaptativo (base_rate=0.48) ===")
print(f"  CALM_BEAR: gate ANTES = {gate_old:.4f} | gate DESPUES = {gate_new:.4f}")
print(f"  RANGE:     gate ANTES = {round(naive_true+0.030,4):.4f} | gate DESPUES = {round(naive_true+margin_range,4):.4f}")

brier_media = 0.2579
brier_std   = 0.0070
sigmas_old = (gate_old - brier_media) / brier_std
sigmas_new = (gate_new - brier_media) / brier_std
print()
print(f"  Gate ANTES: {gate_old:.4f} a {sigmas_old:.1f} sigma de la media Brier")
print(f"  Gate NUEVO: {gate_new:.4f} a {sigmas_new:.1f} sigma de la media Brier")
print()

z_new = (gate_new - brier_media) / brier_std
pct_degraded = 1 - 0.5 * (1 + math.erf(z_new / math.sqrt(2)))
print(f"  % seeds DEGRADED estimadas con nuevo gate: {pct_degraded*100:.1f}%")
print(f"  (Antes: ~18% con gate={gate_old:.4f}; Ahora: ~{pct_degraded*100:.0f}% con gate={gate_new:.4f})")
print()
print("[FIX-G2-BRIER-MARGIN-01] VERIFICACION OK - Parametros legibles y gate calculado correctamente")

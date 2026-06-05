"""[TEST-SIZER-KELLY] Verifica el nuevo print de Kelly telemetry en el sizer."""
import sys
sys.path.insert(0, "/root/luna_v2")
from luna.live.position_sizer import PositionSizer

sizer = PositionSizer(base_capital=100000.0, base_risk_fraction=0.20)

print("=" * 70)
print("[TEST-SIZER-KELLY] Escenario HOLD (confianza=0, régimen BEAR=0)")
print("=" * 70)
r = sizer.calculate_position_size(
    action="HOLD", confidence=0.0, hmm_regime=0,
    current_drawdown=0.0, current_volatility=0.02, historical_volatility=0.02
)
print("Resultado:", r)

print("\n" + "=" * 70)
print("[TEST-SIZER-KELLY] Escenario LONG (confianza=0.72, régimen BULL=2)")
print("=" * 70)
r2 = sizer.calculate_position_size(
    action="LONG", confidence=0.72, hmm_regime=2,
    current_drawdown=0.0, current_volatility=0.015, historical_volatility=0.02,
    asset_price=96000.0, tribe_id=-1
)
print("size_usd:", r2["size_usd"])
print("contracts:", r2["contracts"])
print("regime_kelly_cap:", r2["regime_kelly_cap"])

print("\n" + "=" * 70)
print("[TEST-SIZER-KELLY] Escenario LONG con DD severo (confianza=0.65, DD=12%)")
print("=" * 70)
r3 = sizer.calculate_position_size(
    action="LONG", confidence=0.65, hmm_regime=2,
    current_drawdown=0.12, current_volatility=0.03, historical_volatility=0.02,
    asset_price=96000.0, tribe_id=1
)
print("size_usd:", r3["size_usd"])
print("✅ TEST COMPLETADO")

import sys
from pathlib import Path
import json

base_dir = Path("c:/Users/Usuario/Desktop/ia/luna_v2")
sys.path.append(str(base_dir))

class MockConnector:
    def __init__(self):
        self.pos_side = "HOLD"
    def get_position(self, symbol):
        return {"side": self.pos_side}
    def close_position(self, symbol):
        print(f"   [MockConnector] cerrando posicion {self.pos_side} en {symbol}")
        self.pos_side = "HOLD"
        return {"status": "closed"}
    def execute_order(self, side, contracts, params):
        print(f"   [MockConnector] executing {side} order for {contracts} contracts")
        self.pos_side = "LONG" if side == "buy" else "SHORT"
        return {"status": "executed", "side": side}

class MockLiveEngine:
    def __init__(self):
        self.connector = MockConnector()
        
    def run_cycle(self, action, decision):
        # Replica la logica de execute_motor
        symbol = "BTC/USDT"
        size_usd = 1000
        price = 100000
        
        print(f"\n--- Probando accion: {action} con decision: {decision} ---")
        
        if action == "LONG":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] == "SHORT":
                print(f"[P3+P2-EXEC] Posicion SHORT detectada. Cerrando antes de abrir LONG...")
                self.connector.close_position(symbol)
            elif current_pos["side"] == "LONG":
                print(f"[P1-DYNAMIC-HOLD] Posicion LONG ya existe. Ignorando senal de re-compra para evitar compounding infinito y proteger margen.")
                return {}

            contracts = round(size_usd / price, 4)
            if contracts < 0.01: return {}
            close_params = {'reduceOnly': False}
            print(f"[P3+P2-EXEC/LONG] Abriendo LONG: {contracts:.4f} contratos @ ~${price:,.2f} (${size_usd:,.2f} USD)")
            order = self.connector.execute_order(side="buy", contracts=contracts, params=close_params)

        elif action == "SHORT":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] == "LONG":
                print(f"[P3+P2-EXEC] Posicion LONG detectada. Cerrando antes de abrir SHORT...")
                self.connector.close_position(symbol)
            elif current_pos["side"] == "SHORT":
                print(f"[P1-DYNAMIC-HOLD] Posicion SHORT ya existe. Ignorando re-venta.")
                return {}
            print(f"[P3+P2-EXEC/WARN] Accion SHORT ignorada temporalmente en entorno Only-Long.")
            return {}

        elif action == "CLOSE":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] != "HOLD":
                print(f"[P3+P2-EXEC/CLOSE] Forzando cierre de posicion {current_pos['side']} en {symbol}...")
                order = self.connector.close_position(symbol)
            else:
                print(f"[P3+P2-EXEC/CLOSE] Sin posicion abierta. Nada que cerrar.")
                return {}
                
        elif action == "HOLD":
            current_pos = self.connector.get_position(symbol)
            if current_pos["side"] == "LONG":
                exit_prob = 0.48
                xgb_prob = float(decision.get("xgb_prob", 0.5))
                if xgb_prob >= exit_prob:
                    print(f"[P1-DYNAMIC-HOLD/HOLD] Manteniendo ganancia de LONG activa. (xgb_prob {xgb_prob:.4f} >= {exit_prob}).")
                    return {}
                else:
                    print(f"[P1-DYNAMIC-HOLD/EXIT] Probabilidad alcista deteriorada (xgb_prob {xgb_prob:.4f} < {exit_prob}). Liquidando LONG para asegurar.")
                    order = self.connector.close_position(symbol)
            elif current_pos["side"] == "SHORT":
                exit_prob = 0.48
                xgb_prob = float(decision.get("xgb_prob", 0.5))
                if xgb_prob <= (1.0 - exit_prob):
                    print(f"[P1-DYNAMIC-HOLD/HOLD] Manteniendo ganancia de SHORT activa. (xgb_prob {xgb_prob:.4f} <= {(1.0 - exit_prob):.2f}).")
                    return {}
                else:
                    print(f"[P1-DYNAMIC-HOLD/EXIT] Probabilidad bajista deteriorada. Liquidando SHORT.")
                    order = self.connector.close_position(symbol)
            else:
                print(f"[P3+P2-EXEC/HOLD] Sin posicion abierta. Quedando plano.")
                return {}
        else:
            print(f"[P3+P2-EXEC/WARN] Accion desconocida: {action}. Ignorando.")
            return {}

engine = MockLiveEngine()
engine.run_cycle("HOLD", {"xgb_prob": 0.3})
engine.run_cycle("LONG", {"xgb_prob": 0.6})
engine.run_cycle("LONG", {"xgb_prob": 0.6}) # Debe bloquear pyramiding
engine.run_cycle("HOLD", {"xgb_prob": 0.5}) # Debe hacer dynamic hold
engine.run_cycle("HOLD", {"xgb_prob": 0.45}) # Debe liquidar por bajada de prob
engine.run_cycle("HOLD", {"xgb_prob": 0.45}) # Debe quedarse plano

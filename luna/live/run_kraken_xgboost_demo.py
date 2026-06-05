import sys
import os
import time
import json
import ccxt
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Carga The Entorno
env_path = PROJECT_ROOT / ".env.sandbox"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv(PROJECT_ROOT / ".env")

from luna.data.data_collector import DataCollector
from luna.features.feature_pipeline import FeaturePipeline

class PureXGBoostInference:
    """Clase simplificada que aísla la carga y ejecución de XGBoost puro."""
    def __init__(self):
        self.models_dir = PROJECT_ROOT / "data" / "models"
        self.data_dir = PROJECT_ROOT / "data"
        self.xgb_model = self._load_xgb("xgboost_meta.model")
        self.selected_features = self._load_selected_features()
        if not self.xgb_model:
            raise RuntimeError("Falta el modelo xgboost_meta.model")

    def _load_xgb(self, filename: str):
        import xgboost as xgb
        path = self.models_dir / filename
        if not path.exists():
            print(f"[!] No se encontró el modelo: {path}")
            return None
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model

    def _load_selected_features(self) -> list:
        feat_path = self.data_dir / "features" / "selected_features.json"
        if feat_path.exists():
            with open(feat_path, 'r') as f:
                return json.load(f).get("selected_features", [])
        return []

    def predict(self):
        print("  -> Recolectando Datos (Mínimos)...")
        DataCollector().build(mode='incremental')
        
        print("  -> Pipeline de Extracción de Features...")
        result = FeaturePipeline().run(skip_fracdiff=False, skip_sfi=True, live_mode=True)
        df = result.get('live', None)
        if df is None or df.empty:
            return None, "Data vacía"
        df = df.sort_index()

        xgb_cols = self.xgb_model.feature_names_in_
        x_input = df[xgb_cols].iloc[-1:]
        xgb_prob = float(self.xgb_model.predict_proba(x_input)[0][1])
        
        # Lógica PURA: >= 0.5 = LONG, de lo contrario SHORT
        direction = "LONG" if xgb_prob >= 0.5 else "SHORT"
        
        return {
            "action": direction,
            "xgb_prob": xgb_prob,
            "price": float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0,
            "timestamp": df.index[-1]
        }, "OK"

class KrakenDemoBot:
    """Orquestador Demo que lee a XGBoost Puro y coloca órdenes reales en Sandbox."""
    def __init__(self):
        self.exchange = ccxt.krakenfutures({
            'apiKey': os.getenv('KRAKEN_FUTURES_SANDBOX_API_KEY', 'dummy'),
            'secret': os.getenv('KRAKEN_FUTURES_SANDBOX_API_SECRET', 'dummy'),
            'enableRateLimit': True,
        })
        self.exchange.set_sandbox_mode(True)
        self.brain = PureXGBoostInference()
        self.symbol = 'BTC/USD:BTC' # Symbol CCXT unficado para futuros inversos en Kraken
        self.fixed_size_usd = 50.0  # Fricción mínima
        
    def _execute_order(self, direction, price):
        print(f"  -> Resolviendo órden en Exchange para {direction}...")
        try:
            # Obtener posición actual para no generar sobre-exposición
            positions = self.exchange.fetch_positions([self.symbol])
            current_position = 0.0
            if positions:
                # Normalmente retorna una lista, el monto expuesto total en el symbol
                pos = positions[0]
                current_position = float(pos.get('contracts', 0.0))
                side = pos.get('side', '')
                if side == 'short':
                    current_position = -current_position

            is_long = current_position > 0
            is_short = current_position < 0
            
            amount = self.fixed_size_usd

            if direction == "LONG" and not is_long:
                if is_short:
                    print("  -> Cerrando posicion SHORT anterior (Inversión de régimen)...")
                    self.exchange.create_order(self.symbol, 'market', 'buy', abs(current_position), params={"reduceOnly": True})
                print(f"  -> Abriendo posicion LONG por {amount}")
                self.exchange.create_order(self.symbol, 'market', 'buy', amount)
                print("  [✓] Ejecutado Exitosamente")
                
            elif direction == "SHORT" and not is_short:
                if is_long:
                    print("  -> Cerrando posicion LONG anterior (Inversión de régimen)...")
                    self.exchange.create_order(self.symbol, 'market', 'sell', abs(current_position), params={"reduceOnly": True})
                print(f"  -> Abriendo posicion SHORT por {amount}")
                self.exchange.create_order(self.symbol, 'market', 'sell', amount)
                print("  [✓] Ejecutado Exitosamente")
                
            else:
                print(f"  -> Ya estamos en posición {direction}. No se requiere emitir orden repetida.")
        except Exception as e:
            print(f"  [X] Fallo al ejecutar la orden en API Kraken: {e}")

    def run(self):
        print("==========================================")
        print("🚀 KRAKEN DEMO PURE XGBOOST EXECUTOR")
        print("==========================================")
        while True:
            try:
                print(f"\n--- [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] Iniciando Inferencia ---")
                decision, status = self.brain.predict()
                if not decision:
                     print(f"  [!] Fallo de inferencia: {status}")
                else:
                     act = decision['action']
                     prob = decision['xgb_prob']
                     print(f"  [+] Inferencia Completada: XGBoost dice {act} (Probabilidad LONG: {prob:.4f}) a Precio Referencia ${decision['price']:.2f}")
                     
                     self._execute_order(act, decision['price'])
            except Exception as e:
                print(f"  [!] Error crítico en el ciclo de orquestación: {e}")
                traceback.print_exc()
            
            print("  -> Esperando 60 segundos antes de recálculo (Fast Demo Polling)...")
            time.sleep(60)

if __name__ == "__main__":
    KrakenDemoBot().run()

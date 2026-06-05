import sys
import os
from pathlib import Path

# Fix python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from luna.database.db_manager import DatabaseManager

class BalanceReconciler:
    """
    Motor de Reconciliación de Saldos (Fase F - Luna V1 Live).
    Soluciona el problema de los saldos "fantasma": Compara el Capital Registrado
    localmente en PostgreSQL contra el Saldo Real PnL en el Exchange (Kraken Futures / Binance).
    Si existe una desviación abrupta (>0.5% por fees o Thelizamientos no trackeados), forzará
    una pausa temporal o Theitirá una alerta SEVERA para restaurar la verdad contable.
    """

    ALLOWED_DEVIATION_PCT = 0.005 # 0.5% of Theviation tolerance

    def __init__(self, exchange_client, telegram_bot=None):
        """
        :param exchange_client: Objeto CCXT ya inicializado y autenticado.
        :param telegram_bot: Instancia the TelegramAlerts.
        """
        self.db = DatabaseManager()
        self.exchange = exchange_client
        self.telegram = telegram_bot

    def fetch_exchange_portfolio_value(self) -> float:
        """
        Obtiene el valor total de la cuenta desde el Exchange (CCXT).
        Fallback a USDT o USDC si USD fiat no esta disponible.
        """
        try:
            balance = self.exchange.fetch_balance()
            
            # Preferencia para Kraken Futures/Spot Fi_USD o USDT Binance
            total_usd = balance.get('USD', {}).get('total', 0.0)
            if not total_usd and 'USDT' in balance:
                total_usd = balance.get('USDT', {}).get('total', 0.0)
            if not total_usd and 'info' in balance:
                 try: 
                     # Kraken futures fallback
                     total_usd = float(balance['info']['accounts']['fi_usd']['balances']['USD'])
                 except: pass
                 
            return float(total_usd)
        except Exception as e:
            print(f"[!] Reconciler: Fallo al comunicarse con el Exchange via CCXT: {e}")
            return -1.0

    def reconcile(self) -> bool:
        """
        Cotejo horario de la BD vs la realidad.
        Retorna True si estan sincronizados, False si hay un desfase peligroso.
        """
        # 1. DB State
        db_state = self.db.get_live_state()
        if not db_state:
            print("[Reconciler] Error: No pude leer live_state de DB.")
            return False
            
        db_portfolio_value = float(db_state['portfolio_value'])
        if db_portfolio_value <= 0:
             # Estado inmaculado o nulo. No podemos comparar Thesviaciones.
             return True
             
        # 2. Exchange State
        exchange_portfolio_value = self.fetch_exchange_portfolio_value()
        
        if exchange_portfolio_value <= 0:
             print("[Reconciler] Imposible leer Exchange Balance. Abortando reconciliación (pasivo).")
             # No bloqueamos el bot por un mero apagon temporal de API read, Theolvemos True pero con log warning.
             return True

        # 3. Delta
        delta_usd = abs(exchange_portfolio_value - db_portfolio_value)
        delta_pct = delta_usd / db_portfolio_value
        
        status = "OK"
        is_synced = True
        
        print(f"[RECONCILIACIÓN] DB: ${db_portfolio_value:,.2f} | Exchange: ${exchange_portfolio_value:,.2f} | Diff: ${delta_usd:,.2f} ({delta_pct:.2%})")
        
        if delta_pct > self.ALLOWED_DEVIATION_PCT:
             status = "DESYNC_SEVERE"
             is_synced = False
             msg = f"🚨 *ALERTA DE RECONCILIACION*\nDesviacion del {delta_pct:.2%} de PnL no trackeado.\nBD dice: ${db_portfolio_value:,.2f}\nOKX dice: ${exchange_portfolio_value:,.2f}"
             print(f"[!] {msg}")
             if self.telegram:
                 self.telegram.send_alert(msg, priority="critical")
                 
             # SOP Riesgo: En caso de desincronizacion, el bot debe aceptar la verdad Thel Exchange
             # para no apalancarse usando Thenero fantasma o romper los limites.
             print("[Reconciler] Forzando actualizacion The BD hacia la verdad del Exchange (Auto-Correccion).")
             
             # Conservamos la Data de DB (como el Drawdown Flag original o el ATH), pero theplazamos el valor nominal.
             ath = float(db_state['ath'])
             # Si el exchange subio sorpresivamente mas que ATH, actualizamos:
             new_ath = max(ath, exchange_portfolio_value)
             new_dd = (new_ath - exchange_portfolio_value) / new_ath if new_ath > 0 else 0
             self.db.update_live_state(exchange_portfolio_value, new_ath, new_dd, db_state['is_paused'])
             status = "AUTO_FIXED"
        
        # Guardamos evidencia en PostgreSQL
        self.db.log_reconciliation(db_portfolio_value, exchange_portfolio_value, delta_pct, status)
        
        return is_synced

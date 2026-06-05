import os
import sys
import time

# Reconfigure stdout for UTF-8 encoding on Windows to prevent charmap crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import ccxt
from pathlib import Path
from loguru import logger
import numpy as np


# Fix python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from luna.database.db_manager import DatabaseManager
from luna.live.position_sizer import PositionSizer

class OKXBrokerConnector:
    """
    [LUNA-V2-LIVE] Broker Connector for OKX using CCXT.
    Supports secure API signing, sandbox/demo trading mode, live balance retrieval,
    position auditing, and execution of market orders for futures and X-Perps (MiCA compliance).
    """
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.db = DatabaseManager()
        
        # Load API keys and regional/network configurations from environment variables
        self.api_key = os.getenv("OKX_API_KEY")
        self.secret_key = os.getenv("OKX_SECRET_KEY")
        self.passphrase = os.getenv("OKX_PASSPHRASE")
        self.hostname = os.getenv("OKX_HOSTNAME", "eea.okx.com")
        self.force_ipv4 = os.getenv("OKX_FORCE_IPV4", "True").lower() in ("true", "1", "yes")
        
        # Apply IPv4 forcing if enabled to prevent IP whitelisting rejection over IPv6
        if self.force_ipv4:
            try:
                import socket
                orig_getaddrinfo = socket.getaddrinfo
                def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
                    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
                socket.getaddrinfo = ipv4_only_getaddrinfo
                print("[OKX_BOOT] Red: Forzada resolucion DNS a IPv4 (AF_INET) para emparejamiento con Whitelist.")
            except Exception as e:
                print(f"[OKX_BOOT/WARN] No se pudo establecer la monkey-patch de socket IPv4: {e}")
        
        # Log loading status safely (without printing sensitive credentials)
        print(f"[OKX_BOOT] Inicializando conector OKX. API Key detectada: {'SÍ' if self.api_key else 'NO'} | "
              f"Secret detectado: {'SÍ' if self.secret_key else 'NO'} | Passphrase detectada: {'SÍ' if self.passphrase else 'NO'} | "
              f"Host de region: {self.hostname} | Forzar IPv4: {self.force_ipv4}")
        
        # CCXT config
        config = {
            'apiKey': self.api_key if self.api_key else "dummy_key",
            'secret': self.secret_key if self.secret_key else "dummy_secret",
            'password': self.passphrase if self.passphrase else "dummy_passphrase",
            'enableRateLimit': True,
            'hostname': self.hostname,
        }
        
        # Create exchange instance
        self.exchange = ccxt.okx(config)
        
        # Set sandbox / demo mode
        if self.demo_mode:
            self.exchange.set_sandbox_mode(True)
            logger.info("[OKX_BOOT] Modo DEMO TRADING (Sandbox) ACTIVADO. No se enviaran ordenes reales a produccion.")
            print("[OKX_BOOT] Modo DEMO TRADING (Sandbox) ACTIVADO. Las ordenes iran a la cuenta Demo de OKX.")
        else:
            logger.warning("[OKX_BOOT] ¡¡MODO PRODUCCIÓN REAL ACTIVADO!! CUIDADO: Dinero real en riesgo.")
            print("[OKX_BOOT] ¡¡MODO PRODUCCIÓN REAL ACTIVADO!! Operando en mercado real.")
            
        try:
            self.exchange.load_markets()
            print("[OKX_BOOT] Mercados cargados con exito de OKX.")
        except Exception as e:
            logger.error(f"[OKX_BOOT] Error al precargar mercados de OKX: {e}")
            print(f"[OKX_BOOT/ERROR] No se pudieron cargar los mercados de OKX: {e}")

    def fetch_equity(self) -> float:
        """
        Obtiene la equidad (Equity) neta de la cuenta de OKX.
        Utiliza el endpoint de balance unificado de OKX. Mapea fallbacks si no hay saldo en USDT.
        """
        try:
            balance = self.exchange.fetch_balance()
            
            # 1. En la cuenta unificada de OKX, la equidad total esta en info -> data -> eqUsd
            total_equity = 0.0
            if 'info' in balance and 'data' in balance['info']:
                try:
                    total_equity = float(balance['info']['data'][0]['eqUsd'])
                except Exception:
                    pass
            
            # 2. Fallback clasico por balance directo
            if total_equity <= 0:
                total_equity = float(balance.get('USDT', {}).get('total', 0.0))
                
            if total_equity <= 0:
                total_equity = float(balance.get('USDC', {}).get('total', 0.0))
                
            if total_equity <= 0:
                total_equity = float(balance.get('EUR', {}).get('total', 0.0))
                
            if total_equity <= 0:
                # Si sigue siendo 0, intentamos recuperar el total de USD o cualquier balance con valor
                total_equity = float(balance.get('total', {}).get('USDT', 0.0))

            print(f"[OKX_BALANCE] Balance cargado de OKX: ${total_equity:,.2f} USD/EUR")
            logger.info(f"OKX Account Equity: {total_equity:.2f} USD/EUR")
            return total_equity
        except Exception as e:
            logger.error(f"[OKX_BALANCE] Error al obtener el balance de OKX: {e}")
            print(f"[OKX_BALANCE/ERROR] Fallo al recuperar saldo de OKX: {e}")
            return -1.0

    def get_position(self, symbol: str) -> dict:
        """
        Audita la posicion abierta en el exchange para un simbolo.
        Retorna dict con side ('LONG', 'SHORT', 'HOLD'), contratos, precio de entrada y PnL no realizado.
        Soporta inteligentemente mercados Spot (mapeando balances de la moneda base a LONG/HOLD).
        """
        try:
            # 1. Detectar si el par es Spot o derivado
            is_spot = False
            try:
                market = self.exchange.market(symbol)
                is_spot = market.get('spot', False)
            except Exception:
                # Fallback si no esta precargado
                if '/' in symbol and ':' not in symbol:
                    is_spot = True
            
            if is_spot:
                # 2. Auditar balance para Spot
                market = self.exchange.market(symbol)
                base_asset = market['base']
                quote_asset = market['quote']
                
                balance = self.exchange.fetch_balance()
                base_balance = float(balance.get(base_asset, {}).get('total', 0.0) or 0.0)
                
                # Obtener ultimo precio para calcular valor
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get('last', ticker.get('close', 0.0)) or 0.0)
                value_quote = base_balance * price
                
                # [LUNA-V2-LIVE-FIX] Cargar dinámicamente el umbral de polvo de settings.yaml (No-Fallback)
                from config.settings import cfg
                try:
                    dust_threshold = float(cfg.position_sizer.spot_dust_threshold)
                except AttributeError as e:
                    print("[OKX_SPOT_POSITION/CRITICAL] Parámetro 'spot_dust_threshold' ausente en settings.yaml!")
                    raise KeyError("Falta 'spot_dust_threshold' en config/settings.yaml bajo la sección 'position_sizer'.") from e
                
                # Umbral de polvo dinámico leído de settings.yaml para evitar falsos positivos
                if value_quote > dust_threshold:
                    mapped_side = "LONG"
                    size = base_balance
                    entry_price = price
                    unrealized_pnl = 0.0
                    
                    # [LUNA-V2-LIVE-DEMO-SPOT-BOOT-FIX] Check if the database actually has a record of a LONG position.
                    # If the database's last recorded action is not LONG (e.g. it is HOLD or empty),
                    # then we should treat this exchange balance as default/external assets and set mapped_side to HOLD.
                    try:
                        with self.db.get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute("SELECT action FROM audit_logs ORDER BY id DESC LIMIT 1")
                                row = cur.fetchone()
                                last_action = row[0] if row else None
                        
                        if last_action != "LONG":
                            print(f"[BUGFIX-DEMO-BOOT] Balance de {base_balance} {base_asset} detectado en OKX, "
                                  f"pero la base de datos registra '{last_action}' como última acción. "
                                  f"Mapeando posición a HOLD para evitar liquidación fantasma de activos por defecto del Demo account.")
                            mapped_side = "HOLD"
                            size = 0.0
                            entry_price = 0.0
                    except Exception as db_err:
                        print(f"[BUGFIX-DEMO-BOOT/WARN] Error al verificar último estado en la base de datos: {db_err}. "
                              f"Continuando con el mapeo por defecto '{mapped_side}' por seguridad.")
                else:
                    mapped_side = "HOLD"
                    size = 0.0
                    entry_price = 0.0
                    unrealized_pnl = 0.0
                    
                print(f"[BUGFIX-3] Balance auditado para Spot {symbol}: Asset={base_asset} | "
                      f"Cantidad={size} | Valor={value_quote:.2f} {quote_asset} | "
                      f"Umbral={dust_threshold:.1f} | Side={mapped_side}")
                logger.info(f"Audited spot position for {symbol}: {mapped_side} ({size} units)")
                
                return {
                    "side": mapped_side,
                    "contracts": size,
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized_pnl
                }
            
            # 3. Flujo original para futuros/derivados
            positions = self.exchange.fetch_positions([symbol])
            
            if not positions:
                print(f"[OKX_POSITION] Sin posicion abierta detectada para {symbol}.")
                return {"side": "HOLD", "contracts": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0}
            
            # Buscar la posicion correspondiente al simbolo
            pos = positions[0]
            size = float(pos.get('contracts', 0.0) or pos.get('contractSize', 0.0))
            side = pos.get('side', '').upper() # 'long', 'short'
            entry_price = float(pos.get('entryPrice', 0.0) or 0.0)
            unrealized_pnl = float(pos.get('unrealizedPnl', 0.0) or 0.0)
            
            # CCXT side mapping
            mapped_side = "HOLD"
            if size > 0:
                if side == "LONG":
                    mapped_side = "LONG"
                elif side == "SHORT":
                    mapped_side = "SHORT"
            
            print(f"[OKX_POSITION] Posicion auditada: Simbolo={symbol} | Side={mapped_side} | Contratos={size} | "
                  f"EntryPrice=${entry_price:,.2f} | UnrPnL=${unrealized_pnl:,.2f}")
            logger.info(f"Audited position for {symbol}: {mapped_side} ({size} contracts)")
            
            return {
                "side": mapped_side,
                "contracts": size,
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl
            }
        except Exception as e:
            logger.error(f"[OKX_POSITION] Error al obtener posicion en {symbol}: {e}")
            print(f"[OKX_POSITION/ERROR] Error auditando posicion para {symbol}: {e}")
            return {"side": "HOLD", "contracts": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0}

    def execute_market_order(self, symbol: str, side: str, contracts: float, params: dict = None) -> dict:
        """
        Ejecuta una orden a mercado (Taker) de compra o venta de contratos de futuros o Spot.
        side: 'buy' o 'sell'
        """
        if contracts <= 0:
            print(f"[OKX_ORDER] Cantidad de contratos invalida: {contracts}. Ignorando orden.")
            return {}
            
        # Parse params dynamically to respect Spot vs. Derivatives MICA/ESMA limits
        order_params = params.copy() if params else {}
        is_spot = False
        try:
            market = self.exchange.market(symbol)
            is_spot = market.get('spot', False)
        except Exception:
            if '/' in symbol and ':' not in symbol:
                is_spot = True
        if is_spot and 'reduceOnly' in order_params:
            del order_params['reduceOnly']

        # Capturar precio ideal de referencia (mid-price actual)
        ideal_price = 0.0
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            bid = float(ticker.get('bid', 0.0))
            ask = float(ticker.get('ask', 0.0))
            ideal_price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(ticker.get('last', 0.0))
        except Exception as pe:
            print(f"    [!] Error al obtener mid-price ideal antes de la orden: {pe}")

        print(f"[OKX_ORDER/INIT] Enviando orden a mercado: Symbol={symbol} | Side={side.upper()} | Contratos={contracts:.6f} | Precio Ideal=${ideal_price:,.2f} | Params={order_params}...")
        logger.info(f"Sending market order to OKX: {side.upper()} {contracts} contracts on {symbol}")
        
        try:
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side.lower(),
                amount=contracts,
                params=order_params
            )
            
            order_id = order.get('id', 'N/D')
            filled_price = float(order.get('average', 0.0) or order.get('price', 0.0) or 0.0)
            status = order.get('status', 'open')
            filled = float(order.get('filled', 0.0) or 0.0)
            
            if status == 'canceled' and filled == 0.0:
                cancel_reason = order.get('info', {}).get('cancelSourceReason', 'Desconocido (slippage o falta de liquidez en Sandbox)')
                print(f"[OKX_ORDER/CANCEL] La orden ID={order_id} fue CANCELADA inmediatamente por el exchange. Motivo: {cancel_reason}")
                logger.warning(f"Market order was immediately canceled: {cancel_reason}")
                return {}
            
            # Extraer comisiones (Fees)
            fee_cost = 0.0
            fee_currency = "USDT"
            if order.get('fee'):
                fee_cost = float(order['fee'].get('cost', 0.0) or 0.0)
                fee_currency = order['fee'].get('currency', 'USDT')
            elif order.get('info') and 'fee' in order['info']:
                try:
                    fee_cost = abs(float(order['info']['fee']))
                except Exception:
                    pass

            # Calcular deslizamiento (Slippage)
            slippage_pct = 0.0
            if ideal_price > 0 and filled_price > 0:
                if side.lower() == "buy":
                    slippage_pct = (filled_price - ideal_price) / ideal_price
                else:
                    slippage_pct = (ideal_price - filled_price) / ideal_price

            # Enriquecer el diccionario retornado
            order['fee_cost'] = fee_cost
            order['fee_currency'] = fee_currency
            order['slippage_pct'] = slippage_pct
            order['ideal_price'] = ideal_price
            order['avg_price'] = filled_price

            print(f"[OKX_ORDER/SUCCESS] Orden ejecutada en OKX. ID={order_id} | Status={status} | Precio Promedio=${filled_price:,.2f} | Fee={fee_cost} {fee_currency} | Slippage={slippage_pct:.4%}")
            logger.success(f"Market order executed. ID: {order_id} | Avg Price: {filled_price} | Fee: {fee_cost} {fee_currency} | Slippage: {slippage_pct:.4%}")
            return order
        except Exception as e:
            logger.error(f"[OKX_ORDER/FATAL] Fallo al ejecutar orden en OKX: {e}")
            print(f"[OKX_ORDER/ERROR] Fallo critico en la transmision de la orden: {e}")
            return {}

    def execute_hybrid_order(self, symbol: str, side: str, contracts: float, timeout_seconds: int = 15, params: dict = None) -> dict:
        """
        [LUNA-V2-LIVE-EXEC] Ejecuta una orden híbrida (Limit con Persecución Activa y Fallback a Market).
        1. Coloca orden LIMIT Maker-Only (postOnly=True) al Mid-price actual del libro de órdenes de OKX.
        2. Monitorea durante 'timeout_seconds' (def: 15s) en intervalos cortos (cada 3s).
        3. Si el precio de mercado se desvía más de 0.02% del precio límite, cancela la orden y coloca una
           nueva orden límite al nuevo Mid-price (Order Chasing/Limit Pegging) por el remanente.
        4. Si expira el tiempo y no se ha llenado del todo, cancela y ejecuta a MARKET el remanente (si supera el min).
        """
        if contracts <= 0:
            print(f"[LUNA-V2-LIMIT-EXEC] Cantidad de contratos inválida: {contracts}. Ignorando.")
            return {}

        # 1. Resolver el mercado y configurar parámetros Maker-Only
        is_spot = False
        try:
            market = self.exchange.market(symbol)
            is_spot = market.get('spot', False)
        except Exception:
            if '/' in symbol and ':' not in symbol:
                is_spot = True
        
        min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.0001) if 'market' in locals() or 'market' in globals() else 0.0001
        price_precision = market.get('precision', {}).get('price', 2) if 'market' in locals() or 'market' in globals() else 2
        
        if isinstance(price_precision, float):
            import math
            if price_precision > 0:
                price_precision = int(round(abs(math.log10(price_precision))))
            else:
                price_precision = 2

        order_params = params.copy() if params else {}
        order_params['postOnly'] = True
        if is_spot and 'reduceOnly' in order_params:
            del order_params['reduceOnly']

        # Función auxiliar para obtener el Mid-price actual redondeado
        def get_current_limit_price() -> float:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                bid = float(ticker.get('bid', 0.0))
                ask = float(ticker.get('ask', 0.0))
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(ticker.get('last', 0.0))
                if mid > 0:
                    return round(mid, int(price_precision))
            except Exception as pe:
                print(f"    [!] Error al calcular Mid-price para re-cotización: {pe}")
            return 0.0

        # Obtener el precio inicial
        limit_price = get_current_limit_price()
        if limit_price <= 0:
            print("[LUNA-V2-LIMIT-EXEC/WARN] No se pudo obtener el precio de referencia. Rebotando a Mercado...")
            return self.execute_market_order(symbol, side, contracts, params=params)

        ideal_price = limit_price
        placed_orders = []

        print(f"[LUNA-V2-LIMIT-EXEC/INIT] Lanzando orden límite Maker: Symbol={symbol} | Side={side.upper()} | "
              f"Target={contracts:.6f} conts | Límite=${limit_price:,.2f} | Params={order_params}")

        # Colocar orden límite inicial
        try:
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side.lower(),
                amount=contracts,
                price=limit_price,
                params=order_params
            )
            order_id = order.get('id')
            if not order_id:
                raise RuntimeError("El exchange no devolvió un ID de orden válido.")
            placed_orders.append(order_id)
        except Exception as e:
            logger.error(f"[LUNA-V2-LIMIT-EXEC/ERR] Error al colocar orden límite inicial: {e}. Rebotando a Mercado...")
            print(f"  [LUNA-V2-LIMIT-EXEC/ERR] Fallo inicial: {e}. Rebotando directamente a mercado...")
            return self.execute_market_order(symbol, side, contracts, params=params)

        # 2. Bucle de Monitoreo y Persecución Activa (Order Chasing)
        start_time = time.time()
        filled = 0.0
        remaining = contracts
        status = 'open'
        
        print(f"  [LUNA-V2-LIMIT-EXEC/CHASING] Iniciando persecución activa (timeout: {timeout_seconds}s)...")
        while time.time() - start_time < timeout_seconds:
            try:
                time.sleep(3)
                check_order = self.exchange.fetch_order(order_id, symbol)
                status = check_order.get('status', 'open')
                filled = float(check_order.get('filled', 0.0))
                remaining = contracts - filled
                
                print(f"    - ID={order_id} | Estado: {status.upper()} | Llenado: {filled:.6f}/{contracts:.6f} conts (Faltan: {remaining:.6f})")
                
                if status == 'closed':
                    print(f"[LUNA-V2-LIMIT-EXEC/SUCCESS] ¡Orden completamente ejecutada en el libro como Maker! ID={order_id}")
                    logger.success(f"Limit order fully filled. ID: {order_id}")
                    break

                # Comprobar si el precio se ha desviado más de 0.02%
                new_mid = get_current_limit_price()
                if new_mid > 0 and abs(new_mid - limit_price) / limit_price > 0.0002:
                    drift_pct = (abs(new_mid - limit_price) / limit_price) * 100
                    print(f"    ⚠️ [CHASING/DRIFT] Desviación de precio detectada: {drift_pct:.4f}% (>0.02%). Re-cotizando...")
                    
                    # Cancelar la orden actual
                    try:
                        self.exchange.cancel_order(order_id, symbol)
                        # Re-auditar el llenado definitivo de la cancelada
                        final_state = self.exchange.fetch_order(order_id, symbol)
                        filled = float(final_state.get('filled', 0.0))
                        remaining = contracts - filled
                    except Exception as ce:
                        print(f"      [!] Error al cancelar en re-cotización: {ce}")
                    
                    if remaining <= min_amount:
                        print(f"      [CHASING] Cantidad remanente {remaining:.6f} por debajo del lote mínimo. Finalizando.")
                        break
                        
                    # Colocar nueva orden límite al nuevo precio
                    limit_price = new_mid
                    print(f"      [CHASING/REPLACE] Colocando nueva orden límite Maker: Precio=${limit_price:,.2f} | Remanente={remaining:.6f}...")
                    try:
                        order = self.exchange.create_limit_order(
                            symbol=symbol,
                            side=side.lower(),
                            amount=remaining,
                            price=limit_price,
                            params=order_params
                        )
                        order_id = order.get('id')
                        if not order_id:
                            raise RuntimeError("El exchange no devolvió un ID de orden válido al re-cotizar.")
                        placed_orders.append(order_id)
                    except Exception as re_err:
                        print(f"      [!] Error al re-cotizar: {re_err}. Rebotando remanente a Mercado...")
                        market_order = self.execute_market_order(symbol, side, remaining, params=params)
                        if market_order and market_order.get('id'):
                            placed_orders.append(market_order.get('id'))
                        break

            except Exception as poll_err:
                print(f"    [!] Error en el bucle de persecución: {poll_err}")

        # 3. Expiración y Fallback a Mercado
        if remaining > min_amount and (time.time() - start_time >= timeout_seconds):
            print(f"⚠️ [LUNA-V2-LIMIT-EXEC/TIMEOUT] Límite superado sin ejecución completa. Llenado final: {filled:.6f}/{contracts:.6f} conts.")
            try:
                # Cancelar orden activa
                print(f"  - Cancelando orden límite activa {order_id}...")
                self.exchange.cancel_order(order_id, symbol)
                final_order = self.exchange.fetch_order(order_id, symbol)
                filled = float(final_order.get('filled', 0.0))
                remaining = contracts - filled
                print(f"  - Llenado total Maker: {filled:.6f} conts. Remanente a mercado: {remaining:.6f} conts.")
            except Exception as cancel_err:
                print(f"  [!] Error al cancelar orden final: {cancel_err}. Rellenando remanente estimado a mercado...")
                
            if remaining > min_amount:
                print(f"🚀 [LUNA-V2-LIMIT-EXEC/FALLBACK] Ejecutando orden MARKET para llenar remanente de {remaining:.6f} contratos...")
                market_order = self.execute_market_order(symbol, side, remaining, params=params)
                if market_order and market_order.get('id'):
                    placed_orders.append(market_order.get('id'))
            else:
                print(f"  [LUNA-V2-LIMIT-EXEC] Remanente {remaining:.6f} es inferior al lote mínimo ({min_amount}). Omitiendo orden de mercado.")

        # --- TELEMETRY AGGREGATION BLOCK ---
        total_filled = 0.0
        total_value = 0.0
        total_fee_cost = 0.0
        fee_currency = "USDT"
        
        last_fetched_order = None
        for oid in placed_orders:
            try:
                o_info = self.exchange.fetch_order(oid, symbol)
                last_fetched_order = o_info
                
                f = float(o_info.get('filled', 0.0) or 0.0)
                ap = float(o_info.get('average', o_info.get('price', 0.0)) or 0.0)
                
                total_filled += f
                total_value += (f * ap)
                
                if o_info.get('fee'):
                    total_fee_cost += float(o_info['fee'].get('cost', 0.0) or 0.0)
                    fee_currency = o_info['fee'].get('currency', 'USDT')
                elif o_info.get('info') and 'fee' in o_info['info']:
                    try:
                        total_fee_cost += abs(float(o_info['info']['fee']))
                    except Exception:
                        pass
            except Exception as fe:
                print(f"    [!] Error al recuperar telemetría de orden {oid}: {fe}")
        
        avg_price = total_value / total_filled if total_filled > 0 else limit_price
        
        slippage_pct = 0.0
        if ideal_price > 0 and avg_price > 0:
            if side.lower() == "buy":
                slippage_pct = (avg_price - ideal_price) / ideal_price
            else:
                slippage_pct = (ideal_price - avg_price) / ideal_price
                
        result_order = last_fetched_order if last_fetched_order else (order if 'order' in locals() else {})
        result_order['fee_cost'] = total_fee_cost
        result_order['fee_currency'] = fee_currency
        result_order['slippage_pct'] = slippage_pct
        result_order['ideal_price'] = ideal_price
        result_order['avg_price'] = avg_price
        result_order['filled'] = total_filled
        result_order['status'] = 'closed' if total_filled >= contracts else 'open'
        
        print(f"[LUNA-V2-LIMIT-EXEC/SUMMARY] Ejecución híbrida finalizada. Total Llenado={total_filled:.6f}/{contracts:.6f} conts | Precio Promedio=${avg_price:,.2f} | Fee Total={total_fee_cost:.4f} {fee_currency} | Slippage Total={slippage_pct:.4%}")
        logger.success(f"Hybrid execution completed. Filled: {total_filled}/{contracts} | Avg Price: {avg_price} | Fee: {total_fee_cost} {fee_currency} | Slippage: {slippage_pct:.4%}")
        
        return result_order

    def close_position(self, symbol: str) -> dict:
        """
        Cierra completamente cualquier posicion abierta para un simbolo especifico.
        Realiza una orden inversa al tamaño de contratos actual.
        Retorna el diccionario de la orden de cierre enriquecido con telemetría.
        """
        print(f"[OKX_CLOSE] Solicitud de cierre completo para {symbol}...")
        pos = self.get_position(symbol)
        
        if pos["side"] == "HOLD" or pos["contracts"] <= 0:
            print(f"[OKX_CLOSE] Sin posicion abierta que requiera cierre para {symbol}.")
            return {}
            
        opposite_side = "sell" if pos["side"] == "LONG" else "buy"
        contracts = pos["contracts"]
        
        # Check Spot vs. Derivatives to pass reduceOnly dynamically (R6 compliance)
        is_spot = False
        try:
            market = self.exchange.market(symbol)
            is_spot = market.get('spot', False)
        except Exception:
            if '/' in symbol and ':' not in symbol:
                is_spot = True
                
        close_params = {}
        if not is_spot:
            close_params['reduceOnly'] = True
            
        print(f"[OKX_CLOSE/EXEC] Posicion actual {pos['side']} de {contracts} contratos. Ejecutando orden inversa {opposite_side.upper()} | Params={close_params}...")
        order = self.execute_market_order(symbol, opposite_side, contracts, params=close_params)
        
        if order:
            print(f"[OKX_CLOSE/SUCCESS] Posicion cerrada exitosamente para {symbol}.")
            return order
        else:
            print(f"[OKX_CLOSE/ERROR] No se pudo cerrar la posicion para {symbol}.")
            return {}

    def calculate_live_size(
        self,
        action: str,
        confidence: float,
        hmm_regime,
        current_vol: float,
        historical_vol: float,
        sizer: PositionSizer,
        asset_price: float,
        tribe_id: int = -1,
        hmm_transition_matrix = None
    ) -> dict:
        """
        Calcula el tamaño de posicion exacto utilizando la equidad actual en OKX y el PositionSizer.
        Actualiza live_state en DB e inyecta la equidad real.
        """
        print("\n[OKX_SIZER] Iniciando calculo de tamaño de posicion en vivo...")
        
        # 1. Recuperar equidad en vivo de OKX
        live_equity = self.fetch_equity()
        if live_equity <= 0:
            print("[OKX_SIZER/WARN] Equidad invalida recuperada. Cayendo al capital base del sizer.")
            live_equity = sizer.base_capital
        else:
            # Sincronizar el capital base del sizer con la realidad del exchange
            sizer.base_capital = live_equity
            
        # 2. Obtener estado de riesgo de DB
        db_state = self.db.get_live_state()
        if db_state:
            current_drawdown = float(db_state.get('drawdown', 0.0))
            ath = float(db_state.get('ath', live_equity))
        else:
            current_drawdown = 0.0
            ath = live_equity
            
        # Actualizar base de datos con la equidad real recuperada
        new_ath = max(ath, live_equity)
        new_dd = (new_ath - live_equity) / new_ath if new_ath > 0 else 0.0
        self.db.update_live_state(
            portfolio_value=live_equity,
            ath=new_ath,
            drawdown=new_dd,
            is_paused=bool(db_state.get('is_paused', False)) if db_state else False
        )
        
        # 3. Calcular con el PositionSizer
        sizing = sizer.calculate_position_size(
            action=action,
            confidence=confidence,
            hmm_regime=hmm_regime,
            current_drawdown=new_dd,
            current_volatility=current_vol,
            historical_volatility=historical_vol,
            asset_price=asset_price,
            tribe_id=tribe_id,
            hmm_transition_matrix=hmm_transition_matrix
        )
        
        # Imprimir todas las estadisticas solicitadas en RULE[windowstats.md]
        print("+" + "-"*78 + "+")
        print(f"| {'ESTADÍSTICAS DE APALANCAMIENTO Y RIESGO EN VIVO (LUNA V2)':^76} |")
        print("+" + "-"*78 + "+")
        print(f"| Equidad Real OKX  : ${live_equity:,.2f} USD" + " "*(46 - len(f"{live_equity:,.2f}")) + "|")
        print(f"| Drawdown Actual   : {new_dd:.2%}" + " "*(52 - len(f"{new_dd:.2%}")) + "|")
        print(f"| Régimen HMM       : {sizing.get('hmm_regime_name', 'UNKNOWN')}" + " "*(52 - len(sizing.get('hmm_regime_name', 'UNKNOWN'))) + "|")
        print(f"| Acción Consenso   : {action}" + " "*(52 - len(action)) + "|")
        print(f"| Confianza Calib.  : {confidence:.2%}" + " "*(52 - len(f"{confidence:.2%}")) + "|")
        print(f"| Tamaño Asignado   : ${sizing.get('size_usd', 0.0):,.2f} USD" + " "*(46 - len(f"{sizing.get('size_usd', 0.0):,.2f}")) + "|")
        
        # Calcular apalancamiento teorico
        target_size = sizing.get('size_usd', 0.0)
        leverage = target_size / live_equity if live_equity > 0 else 0.0
        print(f"| Apalancamiento    : {leverage:.2f}x (Límite Kelly Máximo: 5x-10x)" + " "*(37 - len(f"{leverage:.2f}")) + "|")
        
        # Kelly óptimo y caps
        kelly_cap = sizing.get('regime_kelly_cap', 0.02)
        print(f"| Kelly Cap Régimen : {kelly_cap:.1%}" + " "*(51 - len(f"{kelly_cap:.1%}")) + "|")
        print(f"| Breakdown         : {sizing.get('multiplier_breakdown', '')[:60]}..." + " "*(8 - min(8, len(sizing.get('multiplier_breakdown', '')))) + "|")
        print("+" + "-"*78 + "+")
        
        return sizing

if __name__ == "__main__":
    print("🌙 [TEST] Inicializando OKXBrokerConnector en modo Sandbox/Demo...")
    try:
        connector = OKXBrokerConnector(demo_mode=True)
        equity = connector.fetch_equity()
        print(f"✅ [TEST/SUCCESS] Conexión establecida. Saldo Demo: ${equity:,.2f}")
        
        print("\n[TEST] Verificando posicion para 'BTC/USDT:USDT'...")
        pos = connector.get_position("BTC/USDT:USDT")
        print(f"✅ [TEST/SUCCESS] Posición auditada: {pos}")
        
        print("\n[TEST] Probando integracion de Position Sizer con balance real...")
        sizer = PositionSizer()
        sizing = connector.calculate_live_size(
            action="LONG",
            confidence=0.72,
            hmm_regime="BULL_TREND",
            current_vol=0.002,
            historical_vol=0.0015,
            sizer=sizer,
            asset_price=67250.0
        )
        print(f"✅ [TEST/SUCCESS] Dimensionamiento en vivo calculado: {sizing}")
    except Exception as e:
        print(f"❌ [TEST/ERROR] Fallo al probar el OKXBrokerConnector: {e}")

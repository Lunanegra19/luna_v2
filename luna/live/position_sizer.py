import numpy as np
import sys
from pathlib import Path

# Fix python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from loguru import logger
from luna.utils.debug_guards import check_kelly, vlog, check_invariant

# ---- Mapeo de régimen HMM entero → nombre -----------------------------------
# El HMM puede entregar el régimen como entero (legacy) o como string (nuevo)
_REGIME_INT_TO_NAME = {
    0: 'BEAR_TREND',
    1: 'CALM_RANGE',
    2: 'BULL_TREND',
    3: 'VOLATILE_RANGE',
}

class PositionSizer:
    """
    Dimensionador de Posiciones Dinámico.
    Orquesta el volumen (%) de capital a arriesgar (Kelly Fraccionario) en base a:
      - Confianza calibrada de la predicción (MetaLabeler)
      - Régimen dictado por HMM — ahora como DICTADOR de Kelly (MEJORA 5 / P1-3)
      - Volatilidad de Corto vs Largo plazo (EWMA Target)
      - Drawdown acumulado.
      - Profundidad de Liquidez Live (Orderbook Check)
    """

    # --- MEJORA 5 (P1-3) --- HMM como dictador de Position Sizing ------------------
    # Fraccion de Kelly MAXIMA permitida por régimen HMM.
    # Estos caps son límites DUROS: ningún otro multiplicador puede superarlos.
    # Con WR=41.7% histórico, la causa del MaxDD=99% era compounding sin restriccion.
    #
    # [FIX-KELLY-WEAK-01] (2026-05-18): Añadidos regímenes _WEAK y variantes semánticas.
    # PROBLEMA CONFIRMADO: '1_BULL_TREND_WEAK' (71% de trades, WR=48%) caía en 'UNKNOWN'
    # (cap=2%) pero el Kelly real era ~4.88% porque _resolve_regime_name no mapeaba
    # el formato 'N_SEMANTIC_NAME' a su canonical. El cap más alto se aplicaba al régimen
    # de menor WR — exactamente al revés de la intención del riesgo institucional.
    REGIME_KELLY_CAP = {
        # ── Regímenes Bull ────────────────────────────────────────────────────
        'BULL_TREND':        0.25,  # 25%: tendencia alcista confirmada
        'VOLATILE_BULL':     0.18,  # 18%: alcista pero con alta volatilidad
        'BULL_GRIND':        0.15,  # 15%: alcista lento, bajo momentum
        'BULL_TREND_WEAK':   0.12,  # 12%: [FIX-KELLY-WEAK-01] alcista débil — WR historico 48%
        'BULL_TREND_B':      0.20,  # 20%: variante B de bull trend
        'VOLATILE_BULL_B':   0.15,  # 15%: variante B de volatile bull
        # ── Regímenes Range ───────────────────────────────────────────────────
        'CALM_RANGE':        0.20,  # 20%: rango calmado (buena visibilidad)
        'CALM_RANGE_B':      0.18,  # 18%: variante B de calm range
        'VOLATILE_RANGE':    0.05,  # 5%: rango volátil — exposición mínima
        'VOLATILE_RANGE_B':  0.05,  # 5%: variante B de volatile range
        'VOLATILE_RANGE_C':  0.04,  # 4%: variante C de volatile range (alta incertidumbre)
        # ── Regímenes Bear ────────────────────────────────────────────────────
        'BEAR_TREND':        0.10,  # 10%: tendencia bajista (modo defensivo)
        'CALM_BEAR':         0.08,  # 8%: bajista calmado
        'BEAR_CRASH':        0.03,  # 3%: crash activo — exposición casi nula
        'BEAR_FORCED':       0.00,  # 0%: bloqueo forzado por risk-off
        # ── Fallback ──────────────────────────────────────────────────────────
        'UNKNOWN':           0.02,  # 2%: régimen no determinado
    }

    # Multiplicador adicional por probabilidad de transición de régimen
    TRANSITION_RISK_MULTIPLIER = {
        'LOW':    1.0,   # Sin reducción si régimen es estable (prob_transicion < 10%)
        'MEDIUM': 0.5,   # Reducción 50% si hay riesgo de cambio (10-25%)
        'HIGH':   0.1,   # Reducción 90% si cambio inminente (> 25%)
    }

    # Cap absoluto: tope global del 25% para permitir exposición asimétrica (BULL_TREND)
    ABSOLUTE_POSITION_CAP = 0.25  # 25% del capital total

    def __init__(self, base_capital: float = None, base_risk_fraction: float = None):
        # [FIX-KELLY-001/003] Leer capital y fracción de riesgo de settings.yaml.
        # Política No-Fallback: si settings.yaml existe y le faltan los parámetros → KeyError visible.
        # Solo si el archivo no existe (test aislado) → usa legacy defaults con WARNING explícito.
        _cfg_loaded = False
        try:
            from pathlib import Path as _PPath
            import yaml as _yaml_ps
            _settings_path = _PPath(__file__).resolve().parents[2] / "config" / "settings.yaml"
            if _settings_path.exists():
                with open(_settings_path, "r", encoding="utf-8") as _f_ps:
                    _raw_ps = _yaml_ps.safe_load(_f_ps)
                _ps_cfg = _raw_ps.get("position_sizer", {})

                # [No-Fallback] base_capital es un parámetro CRÍTICO de riesgo.
                # Si no está en settings.yaml (y el archivo existe), es un error de configuración.
                if "base_capital" not in _ps_cfg:
                    raise KeyError(
                        "[FIX-KELLY-001/003][CRITICAL] 'base_capital' no encontrado en "
                        "settings.yaml → position_sizer. Añadir 'base_capital: 100000.0' "
                        "a la sección position_sizer."
                    )
                if "base_risk_fraction" not in _ps_cfg:
                    raise KeyError(
                        "[FIX-KELLY-001/003][CRITICAL] 'base_risk_fraction' no encontrado en "
                        "settings.yaml → position_sizer. Añadir 'base_risk_fraction: 0.20'."
                    )

                _base_capital_cfg = float(_ps_cfg["base_capital"])
                _base_risk_cfg = float(_ps_cfg["base_risk_fraction"])

                # Si se pasó explícitamente en el constructor, el argumento tiene precedencia
                self.base_capital = base_capital if base_capital is not None else _base_capital_cfg
                self.base_risk_fraction = base_risk_fraction if base_risk_fraction is not None else _base_risk_cfg
                _cfg_loaded = True
                logger.info(
                    f"[FIX-KELLY-001/003] PositionSizer inicializado desde settings.yaml: "
                    f"base_capital=${self.base_capital:,.0f} | base_risk_fraction={self.base_risk_fraction:.2%}"
                )
                print(
                    f"[FIX-KELLY-001/003] PositionSizer: capital=${self.base_capital:,.0f} | "
                    f"risk_fraction={self.base_risk_fraction:.2%} | "
                    f"exposure_por_trade=${self.base_capital * self.base_risk_fraction:,.0f}"
                )
        except KeyError:
            raise  # Re-lanzar KeyError de parámetros faltantes — No-Fallback estricto
        except Exception as _e_cfg:
            # settings.yaml no accesible (entorno de test aislado, etc.)
            logger.warning(
                f"[FIX-KELLY-001/003][WARN] No se pudo leer settings.yaml: {_e_cfg}. "
                f"Usando legacy defaults (base_capital=5000, base_risk_fraction=0.20). "
                f"ESTO ES UN ERROR EN PRODUCCION."
            )
            print(
                f"[FIX-KELLY-001/003][WARN] settings.yaml no accesible: {_e_cfg}. "
                f"Usando legacy defaults. VERIFICAR EN PRODUCCION."
            )

        if not _cfg_loaded:
            # Legacy defaults — solo en entorno de test sin settings.yaml
            self.base_capital = base_capital if base_capital is not None else 5000.0
            self.base_risk_fraction = base_risk_fraction if base_risk_fraction is not None else 0.20

        self.base_capital = float(self.base_capital)
        self.base_risk_fraction = float(self.base_risk_fraction)

        
        # Umbrales escalonados de Confianza
        self.conf_tier_1 = 0.70  # Full size
        self.conf_tier_2 = 0.60  # Half size
        self.conf_tier_3 = 0.50  # Quarter size

        # Multiplicadores HMM legacy (entero): mantenidos para retrocompatibilidad
        # Cuando se usa REGIME_KELLY_CAP, estos multipliers ya no son el factor principal
        self.regime_multipliers = {
            0: 0.50,
            1: 0.80,
            2: 1.00,
            3: 1.00
        }

        # Fix A-05: tribe_kelly_mult se cargaba con WRs hardcodeados de una run especifica.
        # Despues de cada run_ai_mining.py los IDs de tribu KMeans pueden reasignarse.
        # Ahora se carga dinamicamente desde alpha_rules.TRIBE_WR_MAP.
        self.tribe_kelly_mult = self._load_tribe_kelly_mult()

    def _load_tribe_kelly_mult(self) -> dict:
        """Carga multiplicadores Kelly por tribu con sistema de fallback multi-nivel:
        1. Ledger JSON: Carga desde data/metadata/tribe_analytics.json (desacoplamiento total).
        2. Fallback clásico: Intenta importación de alpha_rules.TRIBE_WR_MAP.
        3. Fallback neutro: Dict vacío (todos los multiplicadores por tribu serán 1.0 / neutrales).
        
        Convierte WR (fracción) a multiplicador Kelly:
          WR >= 0.58 → 1.00 (full Kelly)
          WR >= 0.55 → 0.85
          WR >= 0.52 → 0.65
          WR >= 0.50 → 0.45
          WR < 0.50  → 0.30 (protección de capital)
        """
        # --- Multiplier Helper Function ---
        def _get_mult_map(wr_map: dict) -> dict:
            mult = {}
            for tribe_id_str, wr in wr_map.items():
                try:
                    tribe_id = int(tribe_id_str)
                    if wr >= 0.58:
                        mult[tribe_id] = 1.00
                    elif wr >= 0.55:
                        mult[tribe_id] = 0.85
                    elif wr >= 0.52:
                        mult[tribe_id] = 0.65
                    elif wr >= 0.50:
                        mult[tribe_id] = 0.45
                    else:
                        mult[tribe_id] = 0.30
                except (ValueError, TypeError):
                    continue
            return mult

        # ── NIVEL 1: Carga desde el Ledger JSON (Desacoplado) ──
        try:
            import json as _json
            ledger_path = PROJECT_ROOT / "data" / "metadata" / "tribe_analytics.json"
            if ledger_path.exists():
                with open(ledger_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                wr_map = data.get("tribe_wr_map", {})
                mult = _get_mult_map(wr_map)
                if mult:
                    logger.success(f"[FIX-COUPLING-03] [SUCCESS] Multiplicadores de tribu cargados desde Ledger JSON: {list(mult.keys())}")
                    print(f"[FIX-COUPLING-03] [SUCCESS] Metadatos de tribu cargados exitosamente desde JSON: {ledger_path}")
                    return mult
        except Exception as e:
            logger.warning(f"[FIX-COUPLING-03] Fallo al leer ledger JSON de tribus: {e}. Intentando fallback...")
            print(f"[FIX-COUPLING-03/WARN] Error leyendo Ledger JSON de tribus: {e}")

        # ── NIVEL 2: Fallback clásico (Importación dinámica de alpha_rules) ──
        try:
            logger.info("[FIX-COUPLING-03] Intentando fallback clásico a importación estática de alpha_rules...")
            from luna.features.alpha_rules import TRIBE_WR_MAP
            mult = _get_mult_map(TRIBE_WR_MAP)
            if mult:
                logger.success(f"[FIX-COUPLING-03] [SUCCESS] Fallback exitoso: Multiplicadores cargados desde alpha_rules.py")
                print(f"[FIX-COUPLING-03] [SUCCESS] Metadatos de tribu cargados desde import fallback (alpha_rules.py)")
                return mult
        except ImportError as e:
            logger.warning(f"[FIX-COUPLING-03] alpha_rules.py no generado aún: {e}")
            print(f"[FIX-COUPLING-03/WARN] alpha_rules.py no disponible para importación.")
        except Exception as e:
            logger.warning(f"[FIX-COUPLING-03] Error inesperado en importación clásica: {e}")
            print(f"[FIX-COUPLING-03/ERROR] Fallo en fallback clásico: {e}")

        # ── NIVEL 3: Fallback neutro definitivo ──
        logger.error("[FIX-COUPLING-03] CRITICAL: Fallo total de carga de multiplicadores de tribu. Usando estado neutral.")
        print("[FIX-COUPLING-03/CRITICO] Fallo total de metadatos de tribu. Retornando mapa vacío (fallbacks inactivos).")
        return {}
        
    def _kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Fix F19: Kelly Fraccionario real (half-Kelly por seguridad, cap 25%).
        Kelly: f* = (p*b - q) / b  donde b = avg_win/avg_loss, q = 1 - win_rate.
        Antes el docstring mencionaba Kelly pero se implementaban multiplicadores fijos.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return self.base_risk_fraction
        b = avg_win / avg_loss          # Payout ratio
        q = 1 - win_rate
        kelly_f = (win_rate * b - q) / b
        # Half-Kelly para reducción de volatilidad del equity, cap 25%
        return float(np.clip(kelly_f / 2, 0.01, 0.25))

    def check_scaling_out_suggested(self, current_pnl_pct: float, mvrv_zscore: float) -> bool:
        """
        Salida Parcial (Take Profit Parcial).
        Si estamos flotando > +10% de ganancia, y MVRV Z-Score > 2.5 (sobrecalentamiento de blockchain),
        sugerimos liquidar el 50% de nuestra apuesta de inmediato.
        """
        if current_pnl_pct >= 0.10 and mvrv_zscore > 2.5:
            return True
        return False

    def _get_confidence_multiplier(self, confidence: float) -> float:
        if confidence >= self.conf_tier_1:
            return 1.0
        elif confidence >= self.conf_tier_2:
            return 0.5
        elif confidence > self.conf_tier_3:
            return 0.25
        return 0.0

    def _get_volatility_multiplier(self, current_vol: float, historical_vol: float) -> float:
        """EWMA Vol Targeting = Volatilidad Historica Normal / Volatilidad Actual Asesinadora."""
        if current_vol <= 0 or historical_vol <= 0:
            return 1.0
        
        ratio = historical_vol / current_vol
        # Clampear entre limites seguros Thealancamiento/desapalancamiento (0.5x hasta 1.5x)
        return float(np.clip(ratio, 0.5, 1.5))

    # --- MEJORA 5 (P1-3) --- Métodos de cap por régimen --------------------------

    def _resolve_regime_name(self, hmm_regime) -> str:
        """
        Normaliza el régimen HMM a un string canónico para lookup en REGIME_KELLY_CAP.

        [FIX-KELLY-WEAK-01] Acepta:
          - Enteros legacy (0/1/2/3) → mapeados via _REGIME_INT_TO_NAME
          - Strings canónicos ('BULL_TREND', 'VOLATILE_RANGE', etc.)
          - Strings semánticos con prefijo numérico ('1_BULL_TREND_WEAK', '2_CALM_RANGE')
            El prefijo 'N_' se elimina antes del lookup para unificar el diccionario.
        """
        if isinstance(hmm_regime, str):
            regime_str = hmm_regime.strip()
            # Lookup directo si ya es un canónico conocido
            if regime_str in self.REGIME_KELLY_CAP:
                return regime_str
            # [FIX-KELLY-WEAK-01] Strip prefijo numérico: '1_BULL_TREND_WEAK' -> 'BULL_TREND_WEAK'
            # El formato del HMM es 'N_SEMANTIC_NAME' donde N es el índice de estado (1-4)
            import re as _re_regime
            _stripped = _re_regime.sub(r'^\d+_', '', regime_str)
            if _stripped in self.REGIME_KELLY_CAP:
                print(
                    f"[FIX-KELLY-WEAK-01] Regime '{regime_str}' -> canonical '{_stripped}' "
                    f"cap={self.REGIME_KELLY_CAP[_stripped]:.0%}"
                )
                return _stripped
            # También strip sufijo '_B', '_C' para mapear variantes a su base si no están en el dict
            _base = _re_regime.sub(r'_[BC]$', '', _stripped)
            if _base in self.REGIME_KELLY_CAP:
                print(
                    f"[FIX-KELLY-WEAK-01] Regime '{regime_str}' -> base '{_base}' "
                    f"cap={self.REGIME_KELLY_CAP[_base]:.0%}"
                )
                return _base
            print(
                f"[FIX-KELLY-WEAK-01] Regime '{regime_str}' no encontrado en REGIME_KELLY_CAP -> UNKNOWN (cap=2%)"
            )
            return 'UNKNOWN'
        # Entero legacy (retrocompatibilidad)
        return _REGIME_INT_TO_NAME.get(int(hmm_regime), 'UNKNOWN')

    def _apply_regime_hard_cap(self, size_usd: float, regime_name: str,
                                hmm_transition_matrix=None) -> tuple:
        """
        Aplica el cap duro de Kelly por régimen HMM (MEJORA 5 / P1-3).
        Retorna (size_usd_capped, breakdown_str).

        - regime_name: 'BULL_TREND' | 'CALM_RANGE' | 'BEAR_TREND' | 'VOLATILE_RANGE' | 'UNKNOWN'
        - hmm_transition_matrix: matriz de transición N×N del HMM (opcional).
          Si se pasa, ajusta el tamaño por riesgo de cambio de régimen inminente.
        """
        # 1. Cap máximo de capital por régimen
        regime_cap_fraction = self.REGIME_KELLY_CAP.get(regime_name, 0.02)
        max_by_regime = self.base_capital * regime_cap_fraction

        # 2. Ajuste por probabilidad de transición
        transition_mult = 1.0
        if hmm_transition_matrix is not None:
            trans_prob = self._get_max_transition_prob(hmm_transition_matrix, regime_name)
            if trans_prob >= 0.25:
                transition_mult = self.TRANSITION_RISK_MULTIPLIER['HIGH']
            elif trans_prob >= 0.10:
                transition_mult = self.TRANSITION_RISK_MULTIPLIER['MEDIUM']
            else:
                transition_mult = self.TRANSITION_RISK_MULTIPLIER['LOW']
            max_by_regime *= transition_mult

        # 3. Cap absoluto: límite global (ej. 25%)
        absolute_cap = self.base_capital * self.ABSOLUTE_POSITION_CAP

        capped = min(size_usd, max_by_regime, absolute_cap)
        breakdown = (f"HMM-Cap({regime_name}:{regime_cap_fraction*100:.0f}%) "
                     f"Trans({transition_mult:.1f}x) AbsCap({self.ABSOLUTE_POSITION_CAP*100:.0f}%)")
        return capped, breakdown

    def _get_max_transition_prob(self, transition_matrix, current_regime: str) -> float:
        """
        Probabilidad de transición a CUALQUIER otro régimen desde el actual.
        transition_matrix puede ser:
          - dict de nombre→dict de nombre→prob (p.ej. {'BULL': {'BEAR': 0.15, ...}})
          - matriz numpy N×N (orden: BEAR=0, CALM=1, BULL=2, VOLATILE=3)
        """
        try:
            if isinstance(transition_matrix, dict):
                row = transition_matrix.get(current_regime, {})
                other_probs = [v for k, v in row.items() if k != current_regime]
                return max(other_probs) if other_probs else 0.0
            elif hasattr(transition_matrix, '__len__'):
                # Matriz numpy o lista de listas
                regime_to_idx = {'BEAR_TREND': 0, 'CALM_RANGE': 1,
                                 'BULL_TREND': 2, 'VOLATILE_RANGE': 3}
                idx = regime_to_idx.get(current_regime, -1)
                if idx < 0 or idx >= len(transition_matrix):
                    return 0.0
                row = list(transition_matrix[idx])
                other_probs = [p for i, p in enumerate(row) if i != idx]
                return max(other_probs) if other_probs else 0.0
        except Exception:
            pass
        return 0.0

    def _check_orderbook_liquidity(self, size_usd: float, exchange_id: str = None, symbol: str = None) -> float:
        """
        Escanea el REST Orderbook Top 20 del exchange operativo para saber si nuestro 'size_usd'
        generará excesivo Slippage.
        [FIX-KELLY-002] exchange_id y symbol se leen de settings.yaml (broker section).
        Default: okx (exchange real de operación, no Binance).
        """
        if size_usd <= 0: return 1.0

        # [FIX-KELLY-002] Leer exchange_id y symbol de settings.yaml si no se pasan explícitamente
        if exchange_id is None or symbol is None:
            try:
                from pathlib import Path as _PP_ob
                import yaml as _yaml_ob
                _sp = _PP_ob(__file__).resolve().parents[2] / "config" / "settings.yaml"
                if _sp.exists():
                    with open(_sp, "r", encoding="utf-8") as _f_ob:
                        _raw_ob = _yaml_ob.safe_load(_f_ob)
                    _broker_cfg = _raw_ob.get("broker", {})
                    exchange_id = exchange_id or _broker_cfg.get("exchange_id", "okx")
                    symbol = symbol or _broker_cfg.get("symbol", "BTC/USDT")
                else:
                    exchange_id = exchange_id or "okx"
                    symbol = symbol or "BTC/USDT"
            except Exception as _e_ob:
                exchange_id = exchange_id or "okx"
                symbol = symbol or "BTC/USDT"

        print(f"[FIX-KELLY-002] OrderBook check: exchange={exchange_id} | symbol={symbol} | size_usd={size_usd:.0f}")

        try:
            import ccxt
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': False})
            orderbook = exchange.fetch_order_book(symbol, limit=20)

            bids = orderbook['bids']
            asks = orderbook['asks']

            # Suma Total Top 10 Orderbook Liquidity (Bids + Asks sides)
            top_10_bids_usd = sum(price * amount for price, amount in bids[:10])
            top_10_asks_usd = sum(price * amount for price, amount in asks[:10])

            min_liquidity = min(top_10_bids_usd, top_10_asks_usd)

            safe_capacity = min_liquidity * 0.05  # Usar menos del 5% del orderbook público
            if size_usd > safe_capacity:
                penalty = safe_capacity / size_usd
                print(f"  [FIX-KELLY-002][WARN] Liquidity Alert: Profundidad insuficiente en {exchange_id}. Reduciendo X{penalty:.2f}")
                return float(np.clip(penalty, 0.1, 1.0))
            return 1.0
        except Exception as e:
            print(f"  [FIX-KELLY-002][WARN] Timeout/error al comprobar Orderbook en {exchange_id}: {e}")
            return 1.0


    def calculate_position_size(
        self,
        action: str,
        confidence: float,
        hmm_regime,              # int legacy O str ('BULL_TREND', etc.)
        current_drawdown: float,
        current_volatility: float,
        historical_volatility: float,
        asset_price: float = 1.0,       # Fix F18: precio del activo para contratos reales
        tribe_id: int = -1,             # M2: tribu KMeans activa (-1 = desconocida)
        hmm_transition_matrix=None,     # MEJORA 5 (P1-3): matriz transición HMM opcional
    ) -> dict:
        """
        Devuelve el dimensionamiento real ($ USD y Breakdowns) que se le pasará al Exchange.

        Mejora P1-3: se añade hmm_transition_matrix para ajuste por riesgo de transición.
        El HMM ahora actúa como DICTADOR: sus caps duros prevalecen sobre todos los demás
        multiplicadores. Esto es el mecanismo por el que se espera reducir MaxDD 99% → <35%.
        """
        if action == "HOLD":
            # [SIZER-KELLY] En HOLD el sizer no ejecuta, pero mostramos el régimen y cap activo
            regime_name_hold = self._resolve_regime_name(hmm_regime)
            kelly_cap_hold = self.REGIME_KELLY_CAP.get(regime_name_hold, 0.02)
            print(
                f"[SIZER] ⏸ HOLD — Sizing no ejecutado. "
                f"Régimen: {regime_name_hold} | Kelly-Cap: {kelly_cap_hold:.0%} | "
                f"Confianza: {confidence:.4f} | Capital: ${self.base_capital:,.0f}"
            )
            return {"size_usd": 0.0, "reason": "Accion es HOLD."}

        # 1. Resolver nombre de régimen (int legacy o string)
        regime_name = self._resolve_regime_name(hmm_regime)

        # 2. Base Constante USD Nominal de exposicion.
        base_exposure = self.base_capital * self.base_risk_fraction

        # 3. Multiplicadores clásicos
        conf_mult   = self._get_confidence_multiplier(confidence)
        # Multiplicador regime legacy (retrocompatibilidad, ahora el cap duro es el verdadero ctrl)
        regime_mult = self.regime_multipliers.get(int(hmm_regime) if isinstance(hmm_regime, int) else 2, 1.0)

        # Si Drawdown es Severo (> 10%), achicamos a la mitad el volumen asimetrico
        dd_mult = 0.5 if current_drawdown >= 0.10 else 1.0

        vol_mult = self._get_volatility_multiplier(current_volatility, historical_volatility)

        # M2: multiplicador tribal Kelly fraccional
        tribe_mult = self.tribe_kelly_mult.get(int(tribe_id), 0.5 if tribe_id >= 0 else 1.0)

        # 4. Size intermedio matematico (logica clasica)
        initial_size_usd = base_exposure * conf_mult * regime_mult * dd_mult * vol_mult * tribe_mult

        # 5. Impacto Realista de Slippage en Execution Flow
        ob_mult = self._check_orderbook_liquidity(initial_size_usd)
        raw_size_usd = float(initial_size_usd * ob_mult)

        # 6. MEJORA 5 (P1-3): aplicar cap duro por régimen HMM (DICTADOR)
        #    Este paso es posterior al cálculo clásico para que el cap sea definitivo.
        final_size_usd, cap_breakdown = self._apply_regime_hard_cap(
            raw_size_usd, regime_name, hmm_transition_matrix
        )
        final_size_usd = float(np.clip(final_size_usd, 0, self.base_capital))

        # Fix F18: calcular contratos como size_usd / precio_activo.
        contracts = final_size_usd / asset_price if asset_price > 0 else 0.0

        breakdown = (
            f"Base(${base_exposure:,.0f}) x "
            f"Conf({conf_mult}x) x "
            f"Regime-Legacy({regime_mult}x) x "
            f"DD({dd_mult}x) x "
            f"Vol({vol_mult:.2f}x) x "
            f"Tribe({tribe_mult}x|T{tribe_id}) x "
            f"OB({ob_mult:.2f}x) "
            f"→ raw=${raw_size_usd:.0f} | {cap_breakdown} → final=${final_size_usd:.0f}"
        )

        # [SIZER-KELLY] Telemetría detallada del cálculo de Kelly para trazabilidad en logs
        kelly_cap_pct = self.REGIME_KELLY_CAP.get(regime_name, 0.02)
        exposure_pct  = final_size_usd / self.base_capital if self.base_capital > 0 else 0.0
        print(
            f"[SIZER] 📐 Kelly Sizing — Acción: {action} | Capital: ${self.base_capital:,.0f}\n"
            f"  Régimen HMM:   {regime_name} → Kelly-Cap duro: {kelly_cap_pct:.0%} (${self.base_capital * kelly_cap_pct:,.0f} máx)\n"
            f"  Multiplicadores:\n"
            f"    Base Risk:   {self.base_risk_fraction:.0%} → ${base_exposure:,.0f}\n"
            f"    Confianza:   {confidence:.4f} → mult={conf_mult}x  (tiers: 0.70→1x | 0.60→0.5x | 0.50→0.25x)\n"
            f"    HMM legacy:  régimen={int(hmm_regime) if isinstance(hmm_regime, int) else hmm_regime} → mult={regime_mult}x\n"
            f"    Drawdown:    {current_drawdown:.2%} → mult={dd_mult}x  ({'⚠ DD>10% activo' if dd_mult < 1.0 else 'OK'})\n"
            f"    Volatilidad: curr={current_volatility:.4f} hist={historical_volatility:.4f} → mult={vol_mult:.2f}x\n"
            f"    Tribu KMeans: T{tribe_id} → mult={tribe_mult}x\n"
            f"    Orderbook:   liquidez → mult={ob_mult:.2f}x\n"
            f"  Size bruto:    ${initial_size_usd:,.2f} → post-OB: ${raw_size_usd:,.2f}\n"
            f"  Cap HMM:       {cap_breakdown}\n"
            f"  → FINAL:       ${final_size_usd:,.2f} ({exposure_pct:.1%} del capital) | {contracts:.6f} contratos @ ${asset_price:,.2f}"
        )

        return {
            "size_usd": round(final_size_usd, 2),
            "contracts": round(contracts, 6),
            "multiplier_breakdown": breakdown,
            "hmm_regime_name": regime_name,
            "regime_kelly_cap": self.REGIME_KELLY_CAP.get(regime_name, 0.02),
            "components": {
                "base": base_exposure,
                "conf_mult": conf_mult,
                "regime_mult": regime_mult,
                "dd_mult": dd_mult,
                "vol_mult": round(vol_mult, 2),
                "tribe_mult": tribe_mult,
                "tribe_id": tribe_id,
                "ob_mult": ob_mult,
                "regime_cap_applied": self.REGIME_KELLY_CAP.get(regime_name, 0.02),
            }
        }

    def compute_position_size_atr(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        hmm_regime,
        atr_24h: float,
        current_price: float,
        hmm_transition_matrix=None,
    ) -> dict:
        """
        Mejora 5 completo (planes_mejora_v3.md §5.2): ATR-based position sizing.

        Dimensionamiento basado en ATR para expresar el riesgo en USD reales:
            position_units = (capital × kelly_final) / ATR_24h
            position_usd   = position_units × current_price

        El ATR_24h representa cuánto puede moverse el activo en un día —
        dividir por él normaliza el riesgo absoluto en USD independientemente
        del precio actual del activo.

        Args:
            capital:               Capital total disponible en USD.
            win_rate:              Win rate histórico (fracción, ej: 0.55).
            avg_win:               Ganancia media por trade ganador (fracción).
            avg_loss:              Pérdida media por trade perdedor (fracción positiva).
            hmm_regime:            Nombre de régimen ('BULL_TREND', etc.) o entero legacy.
            atr_24h:               ATR de 24H calculado sobre datos de precios.
            current_price:         Precio actual del activo.
            hmm_transition_matrix: Matriz de transición HMM (opcional) para ajuste de riesgo.

        Returns:
            dict con size_usd, position_units, kelly_final, regime_cap, breakdown.
        """
        if atr_24h <= 0 or current_price <= 0:
            return {"size_usd": 0.0, "reason": "ATR o precio inválido"}

        # 1. Kelly fraccionado base (half-Kelly capped)
        kelly_base = self._kelly_fraction(win_rate, avg_win, avg_loss)

        # 2. Cap duro por régimen HMM
        regime_name = self._resolve_regime_name(hmm_regime)
        regime_cap = self.REGIME_KELLY_CAP.get(regime_name, 0.02)
        kelly_regime = min(kelly_base, regime_cap)

        # 3. Ajuste por riesgo de transición HMM
        transition_mult = 1.0
        if hmm_transition_matrix is not None:
            trans_prob = self._get_max_transition_prob(hmm_transition_matrix, regime_name)
            if trans_prob >= 0.25:
                transition_mult = self.TRANSITION_RISK_MULTIPLIER['HIGH']
            elif trans_prob >= 0.10:
                transition_mult = self.TRANSITION_RISK_MULTIPLIER['MEDIUM']
            else:
                transition_mult = self.TRANSITION_RISK_MULTIPLIER['LOW']
        kelly_final = kelly_regime * transition_mult

        # ── Guards y logging detallado ──
        check_kelly(kelly_base, label="PositionSizer.kelly_base")
        check_kelly(kelly_final, label="PositionSizer.kelly_final")
        check_invariant(atr_24h > 0, f"ATR inválido: {atr_24h}")
        check_invariant(current_price > 0, f"Precio inválido: {current_price}")
        vlog(
            f"PositionSizer.ATR | regime={regime_name} | "
            f"kelly_base={kelly_base:.4f} → cap={kelly_regime:.4f} → "
            f"trans={transition_mult:.2f}x → kelly_final={kelly_final:.4f} | "
            f"capital={capital:.0f} | ATR={atr_24h:.2f} | price={current_price:.2f}"
        )

        # 4. ATR-based sizing: normalizar riesgo por volatilidad absoluta del activo
        risk_usd = capital * kelly_final
        position_units = risk_usd / atr_24h
        position_usd = position_units * current_price

        # 5. Hard cap absoluto
        absolute_cap_usd = capital * self.ABSOLUTE_POSITION_CAP
        final_usd = min(position_usd, absolute_cap_usd)

        breakdown = (
            f"Kelly({kelly_base:.4f}) → Regime-cap({regime_cap:.2f}) → "
            f"Transition({transition_mult:.1f}x) → kelly_final={kelly_final:.4f} | "
            f"risk_usd=${risk_usd:.0f} / ATR({atr_24h:.0f}) × price({current_price:.0f}) "
            f"= ${position_usd:.0f} | AbsCap({self.ABSOLUTE_POSITION_CAP*100:.0f}%)=${absolute_cap_usd:.0f} → final=${final_usd:.0f}"
        )

        return {
            "size_usd": round(final_usd, 2),
            "position_units": round(position_units, 6),
            "kelly_final": round(kelly_final, 4),
            "kelly_base": round(kelly_base, 4),
            "regime_name": regime_name,
            "regime_cap": regime_cap,
            "transition_mult": transition_mult,
            "atr_24h": atr_24h,
            "breakdown": breakdown,
        }

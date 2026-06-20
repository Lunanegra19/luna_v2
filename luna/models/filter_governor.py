"""
Luna V2 — Filter Governor
=========================
Determina en base al historial acumulado de las ventanas completadas previas (causales)
si los filtros secundarios (DVOL Guardian y MetaLabeler V2) están censurando en exceso
las señales baselines (rentables) y calcula un factor de relajación dinámico [0.0, 1.0].
"""
import os
import re
import pandas as pd
from pathlib import Path
from loguru import logger
from config.settings import cfg

class FilterGovernor:
    """
    Governor dinámico y auto-regulado para el pipeline de filtrado de señales.
    Calcula el factor de relajación basándose únicamente en el histórico causal de ventanas completadas.
    """
    def __init__(self, models_dir: Path, seed: int):
        self.models_dir = Path(models_dir)
        self.seed = int(seed)
        
        # 1. Validar e importar la configuración (No-Fallback Silencioso)
        try:
            gov_cfg = cfg.filter_governor
            self.enabled = bool(gov_cfg.enabled)
            self.min_completed = int(gov_cfg.min_completed_windows)
            self.perf_threshold = float(gov_cfg.performance_threshold_ratio)
            self.cost_rt = float(cfg.sop.cost_pct)
        except Exception as e:
            raise RuntimeError(
                f"[FILTER-GOVERNOR][CRITICAL] Fallo de configuración en settings.yaml. "
                f"La política de No-Fallback está activa: {e}"
            )

    def extract_current_window_id(self) -> str:
        """Extrae el ID de la ventana (ej. 'W3') desde el entorno o el path actual."""
        w_env = os.environ.get("LUNA_WINDOW_ID", "")
        if re.match(r"^W\d+$", w_env):
            return w_env
            
        for part in self.models_dir.parts:
            if re.match(r"^W\d+$", part):
                return part
                
        if re.match(r"^W\d+$", self.models_dir.name):
            return self.models_dir.name
            
        return ""

    def get_relaxation_factor(self) -> float:
        """
        Calcula el factor de relajación dinámico:
        0.0 = Máxima restricción (filtros por defecto)
        1.0 = Máxima relajación (filtros al límite permisivo)
        """
        if not self.enabled:
            logger.info("[FILTER-GOVERNOR] Desactivado en la configuración. Factor de relajación = 0.0")
            return 0.0

        current_w = self.extract_current_window_id()
        if not current_w:
            logger.warning(
                f"[FILTER-GOVERNOR] No se pudo identificar la ventana actual en {self.models_dir}. "
                f"Defaulting to relaxation = 0.0"
            )
            return 0.0

        # Encontrar el índice numérico de la ventana actual
        try:
            curr_idx = int(current_w[1:])
        except ValueError:
            logger.warning(f"[FILTER-GOVERNOR] Ventana {current_w} no tiene formato W<numero>. Relaxation = 0.0")
            return 0.0

        # Ventanas previas completadas (por ejemplo, si actual = W3, previas = W1, W2)
        completed_nums = list(range(1, curr_idx))
        if len(completed_nums) < self.min_completed:
            logger.info(
                f"[FILTER-GOVERNOR] Ventana actual {current_w} tiene {len(completed_nums)} ventanas previas completadas, "
                f"menor que el mínimo {self.min_completed}. Relaxation = 0.0"
            )
            return 0.0

        comp_base_rets = []
        comp_filt_rets = []
        
        project_root = Path(__file__).resolve().parents[2]

        # Recopilar trades de ventanas completadas
        for w_num in completed_nums:
            # Buscar en 3 ubicaciones posibles (activa, cache, reportes)
            p_filt = None
            p_base = None
            
            # Ubicación 1: Carpeta de run activa (estructura de runs)
            candidate_parent = self.models_dir.parent
            filt_run = candidate_parent / f"W{w_num}" / "oos_trades.parquet"
            base_run = candidate_parent / f"W{w_num}" / "oos_trades_xgb_baseline.parquet"
            if filt_run.exists() and base_run.exists():
                p_filt, p_base = filt_run, base_run
            else:
                # Ubicación 2: wfb cache
                filt_cache = project_root / "data" / "wfb_cache" / f"W{w_num}" / "features" / f"oos_trades_W{w_num}_seed{self.seed}.parquet" # or similar
                # Ubicación 3: wfb reports (usado en simulaciones y test)
                filt_rep = project_root / "data" / "reports" / "wfb" / f"oos_trades_W{w_num}_seed{self.seed}.parquet"
                base_rep = project_root / "data" / "reports" / "wfb" / f"oos_trades_xgb_baseline_W{w_num}_seed{self.seed}.parquet"
                if filt_rep.exists() and base_rep.exists():
                    p_filt, p_base = filt_rep, base_rep
                elif filt_cache.exists():
                    # si solo cache existe...
                    pass
            
            if p_filt and p_base:
                try:
                    df_base = pd.read_parquet(p_base)
                    df_filt = pd.read_parquet(p_filt)
                    
                    base_net = (df_base["return_raw"] - self.cost_rt).sum() if not df_base.empty else 0.0
                    filt_net = (df_filt["return_raw"] - self.cost_rt).sum() if not df_filt.empty else 0.0
                    
                    comp_base_rets.append(base_net)
                    comp_filt_rets.append(filt_net)
                    logger.debug(
                        f"[FILTER-GOVERNOR] Ventana W{w_num} leída: "
                        f"Baseline Net Return = {base_net:.2%}, Filtered Net Return = {filt_net:.2%}"
                    )
                except Exception as ex:
                    logger.warning(f"[FILTER-GOVERNOR] Error leyendo parquets de ventana W{w_num}: {ex}")

        # Calcular retornos acumulados netos
        r_baseline = sum(comp_base_rets)
        r_filtered = sum(comp_filt_rets)

        # Regla de Oro 3: Si la baseline está en pérdida o es cero, no relajar
        if r_baseline <= 0:
            logger.info(
                f"[FILTER-GOVERNOR] [BUG-FIX-LOG 2026-06-19] Baseline acumulada en pérdida o cero ({r_baseline:.2%}). "
                f"Manteniendo filtros en máxima restricción. Relaxation = 0.0"
            )
            return 0.0

        # Si el modelo filtrado ya tiene un desempeño >= 70% de la baseline, no relajar
        if r_filtered >= self.perf_threshold * r_baseline:
            logger.info(
                f"[FILTER-GOVERNOR] [BUG-FIX-LOG 2026-06-19] Desempeño filtrado ({r_filtered:.2%}) es adecuado "
                f"comparado con baseline ({self.perf_threshold * r_baseline:.2%}). Relaxation = 0.0"
            )
            return 0.0

        # Aplicar fórmula de relajación
        ratio = r_filtered / (self.perf_threshold * r_baseline)
        relaxation_factor = max(0.0, min(1.0, 1.0 - ratio))

        logger.info(
            f"[FILTER-GOVERNOR] [BUG-FIX-LOG 2026-06-19] Relajación Calculada para {current_w}: "
            f"factor = {relaxation_factor:.4f} | R_baseline = {r_baseline:.2%}, R_filtered = {r_filtered:.2%}, Ratio = {ratio:.4f}"
        )
        return relaxation_factor

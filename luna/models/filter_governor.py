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

        # [FIX-GOVERNOR-PATH-01 2026-06-26] Resolver sufijo de dirección dual-bot (_long/_short).
        # BUG previo (SOLID): se buscaba oos_trades(_xgb_baseline)_W{w}_seed{seed}.parquet SIN sufijo
        # de dirección -> file-not-found en dual-bot -> r_baseline=0.00% -> JAMÁS relajaba. Fallback
        # silencioso que viola R16 / addendum SOP A1. Ver docs/hallazgos_run_baseline_20260626.md §6.3.
        import os as _os_gov
        _gov_dir = _os_gov.environ.get("LUNA_DIRECTION", "").lower().strip()
        if _gov_dir not in ("long", "short"):
            try:
                _gov_dir = str(cfg.fase2.direction_mode).lower().strip()
            except Exception:
                _gov_dir = ""
        _dir_sfx = f"_{_gov_dir}" if _gov_dir in ("long", "short") else ""
        print(f"[FIX-GOVERNOR-PATH-01] FilterGovernor: sufijo dirección='{_dir_sfx or '(ninguno)'}' seed={self.seed} | resolviendo baselines dual-bot")  # RULE[fixbugsprints.md]

        def _resolve_gov_parquet(_base_dir, _stem):
            # intenta CON sufijo de dirección primero, luego SIN (compat runs antiguas)
            for _sfx in ([_dir_sfx, ""] if _dir_sfx else [""]):
                _cand = _base_dir / f"{_stem}{_sfx}.parquet"
                if _cand.exists():
                    return _cand
            return None

        # Recopilar trades de ventanas completadas
        for w_num in completed_nums:
            # Buscar en 2 ubicaciones posibles (run activa, reportes)
            p_filt = None
            p_base = None

            # Ubicación 1: Carpeta de run activa (dir per-window, sin sufijo de dirección)
            candidate_parent = self.models_dir.parent
            filt_run = candidate_parent / f"W{w_num}" / "oos_trades.parquet"
            base_run = candidate_parent / f"W{w_num}" / "oos_trades_xgb_baseline.parquet"
            if filt_run.exists() and base_run.exists():
                p_filt, p_base = filt_run, base_run
            else:
                # Ubicación 2: wfb reports — [FIX-GOVERNOR-PATH-01] CON sufijo de dirección (+ fallback)
                _rep_dir = project_root / "data" / "reports" / "wfb"
                p_filt = _resolve_gov_parquet(_rep_dir, f"oos_trades_W{w_num}_seed{self.seed}")
                p_base = _resolve_gov_parquet(_rep_dir, f"oos_trades_xgb_baseline_W{w_num}_seed{self.seed}")
            
            if p_filt and p_base:
                try:
                    df_base = pd.read_parquet(p_base)
                    df_filt = pd.read_parquet(p_filt)
                    
                    # [FIX-GOVERNOR-COST-01 2026-06-26] return_raw YA es neto de coste RT
                    # (predict_oos.py:1815 y :1948 -> ret_bruto = ret - _GLOBAL_COST_RT). Restar
                    # self.cost_rt aquí era DOBLE-COSTE: hundía el baseline real (+7.9% -> -4.07% con
                    # N=48) -> rama "baseline <=0" -> nunca relajaba. Ver hallazgos §6.3.
                    base_net = df_base["return_raw"].sum() if not df_base.empty else 0.0
                    filt_net = df_filt["return_raw"].sum() if not df_filt.empty else 0.0
                    
                    comp_base_rets.append(base_net)
                    comp_filt_rets.append(filt_net)
                    logger.debug(
                        f"[FILTER-GOVERNOR] Ventana W{w_num} leída: "
                        f"Baseline Net Return = {base_net:.2%}, Filtered Net Return = {filt_net:.2%}"
                    )
                except Exception as ex:
                    logger.warning(f"[FILTER-GOVERNOR] Error leyendo parquets de ventana W{w_num}: {ex}")

        # [FIX-GOVERNOR-PATH-01 2026-06-26] Fail-loud (R16/A1): distinguir "datos ausentes" de
        # "baseline real <=0". Antes ambos caían en la rama silenciosa "baseline 0.00%".
        if len(comp_base_rets) == 0 and len(completed_nums) > 0:
            logger.warning(
                f"[FILTER-GOVERNOR][FIX-GOVERNOR-PATH-01] DATOS AUSENTES: 0 baselines encontrados para "
                f"{len(completed_nums)} ventana(s) previa(s) (seed={self.seed}, sufijo='{_dir_sfx}'). "
                f"NO es un baseline real <=0 — revisar naming dual-bot. Relaxation = 0.0 por seguridad."
            )
            print(f"[FIX-GOVERNOR-PATH-01] WARNING: 0 baselines de {len(completed_nums)} ventanas (sufijo='{_dir_sfx}') -> revisar paths.")  # RULE[fixbugsprints.md]
            return 0.0

        # Calcular retornos acumulados netos
        r_baseline = sum(comp_base_rets)
        r_filtered = sum(comp_filt_rets)
        print(f"[FIX-GOVERNOR-COST-01] r_baseline={r_baseline:.4f} r_filtered={r_filtered:.4f} (return_raw ya neto; sin doble-coste) | ventanas={len(comp_base_rets)}")  # RULE[fixbugsprints.md]

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

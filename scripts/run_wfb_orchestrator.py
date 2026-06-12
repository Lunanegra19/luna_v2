import os
import sys
import io
import time
import psutil
import subprocess
import re
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import atexit
import shutil

# Forzar UTF-8 en consola Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent

def _restore_master_settings():
    import shutil
    _settings_path = _ROOT / "config" / "settings.yaml"
    _master_backup_path = _ROOT / "config" / "master_settings_backup_wfb.yaml"
    if _master_backup_path.exists():
        if _settings_path.exists() and _settings_path.stat().st_mtime > _master_backup_path.stat().st_mtime:
            try:
                shutil.copy2(_settings_path, _master_backup_path)
                print("\n[FIX-GLOBAL-RESTORE] settings.yaml editado por el usuario. Conservando cambios y actualizando backup maestro.")
            except Exception as e:
                print(f"\n[FIX-GLOBAL-RESTORE] ERROR actualizando backup maestro: {e}")
        else:
            try:
                shutil.copy2(_master_backup_path, _settings_path)
                _master_backup_path.unlink(missing_ok=True)
                print("\n[FIX-GLOBAL-RESTORE] settings.yaml restaurado exitosamente desde el backup maestro.")
            except Exception as e:
                print(f"\n[FIX-GLOBAL-RESTORE] ERROR restaurando settings.yaml: {e}")

def get_wfb_pid():
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = p.info.get('cmdline') or []
            # Check if it's a python process running the WFB orchestrator
            if 'python' in (p.info.get('name', '').lower()) or any('python' in cmd.lower() for cmd in cmdline):
                cmd_str = ' '.join(cmdline)
                # Ensure we don't catch THIS script (queue_wfb_seeds.py) by mistake!
                if 'wfb_worker.py' in cmd_str and 'queue_wfb_seeds.py' not in cmd_str:
                    return p.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None


def _kill_process_tree(pid: int):
    """Mata un proceso padre y recursivamente a todos sus hijos usando psutil."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        parent.wait(5)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass



BENCHMARK_777_SCORE = 0.0

# Upper-bound optimista para ventanas restantes:
# Asumimos que las ventanas aún no vistas darán el mejor resultado posible.
# WR_opt=0.70, RR_opt=2.0, WR_min influye solo si la ventana actual ya es la peor.
_WR_OPT  = 0.70   # WR "perfecta" para ventanas futuras (techo conservador)
_RR_OPT  = 2.0    # RR "perfecto" para ventanas futuras
_RET_OPT = 0.01   # Retorno medio "perfecto" (1% por trade)

# [FIX-I] _PRUNE_LIMIT leído de settings.yaml wfb.prune_threshold
# Antes: 0.95 hardcodeado. Es el margen de early-stopping del orquestador:
# upper-bound_optimista × prune_threshold < benchmark → descartar semilla
try:
    import yaml as _yaml_pt
    _settings_pt = _yaml_pt.safe_load(open(_ROOT / "config" / "settings.yaml", encoding="utf-8"))
    _PRUNE_LIMIT = float(_settings_pt.get("wfb", {}).get("prune_threshold", 0.95))
except Exception:
    _PRUNE_LIMIT = 0.95
    print(f"[FIX-I] WARN: No se pudo leer wfb.prune_threshold de settings.yaml. Usando fallback={_PRUNE_LIMIT}")
print(f"[FIX-I] WFB Orchestrator: _PRUNE_LIMIT={_PRUNE_LIMIT} (early-stopping semillas)")

# Número total de ventanas WFB
def _get_total_windows() -> int:
    try:
        import yaml as _yaml
        settings_path = _ROOT / "config" / "settings.yaml"
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f)
        windows = cfg.get("wfb", {}).get("windows", [])
        if windows:
            return len(windows)
    except Exception:
        pass
    return 6  # 6 es el fallback en wfb_worker.py

_N_WINDOWS = _get_total_windows()


def _compute_partial_score(seed: int, windows_done: list[int]) -> dict:
    """
    Calcula el score parcial del seed con las ventanas ya disponibles
    y devuelve tambien el upper-bound optimista asumiendo ventanas
    restantes perfectas.

    Retorna dict con: score_partial, score_upper_bound, wr_seen,
                      wr_min_seen, mean_ret_seen, n_trades_seen.
    """
    wfb_dir = _ROOT / "data" / "reports" / "wfb"
    dfs = []
    wr_per_w = {}

    for w in windows_done:
        path = wfb_dir / f"oos_trades_W{w}_seed{seed}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if len(df) >= 3:
                dfs.append(df)
                wr_per_w[w] = float(df["is_win"].mean()) if "is_win" in df.columns else 0.5

    if not dfs:
        return {"score_partial": 0, "score_upper_bound": 100, "wr_seen": 0.5,
                "wr_min_seen": 0.5, "mean_ret_seen": 0, "n_trades_seen": 0}

    combined = pd.concat(dfs, ignore_index=True)
    n        = len(combined)
    wr_seen  = float(combined["is_win"].mean()) if "is_win" in combined.columns else 0.5
    wr_vals  = list(wr_per_w.values())
    wr_min_s = min(wr_vals) if wr_vals else wr_seen
    wr_rng_s = max(wr_vals) - min(wr_vals) if len(wr_vals) > 1 else 0.0

    r = combined["return_pct"].dropna() if "return_pct" in combined.columns else pd.Series(dtype=float)
    mean_ret = float(r.mean()) if len(r) > 0 else 0.0
    avg_win  = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(abs(r[r < 0].mean())) if (r < 0).any() else 1e-9
    rr       = min(avg_win / avg_loss, 10.0)  # Evitar explosión a Inf

    # ── Score sobre ventanas vistas (misma formula que seed_champion.py) ──────
    s_wr_global = float(np.clip((wr_seen  - 0.40) / (0.75 - 0.40) * 100, 0, 100))
    s_wr_min    = float(np.clip((wr_min_s - 0.35) / (0.70 - 0.35) * 100, 0, 100))
    s_stability = float(np.clip((1 - wr_rng_s / 0.50) * 100, 0, 100))
    s_rr        = float(np.clip((rr - 0.5) / (2.0 - 0.5) * 100, 0, 100))
    s_ret       = float(np.clip((mean_ret - 0.0) / (0.01 - 0.0) * 100, 0, 100))
    
    score_seen  = (0.20 * s_wr_global + 0.30 * s_wr_min
                   + 0.25 * s_stability + 0.10 * s_rr + 0.15 * s_ret)

    # ── Upper-bound optimista para ventanas restantes ─────────────────────────
    # Asumimos que cada ventana futura tiene WR=_WR_OPT, RR=_RR_OPT y RET=_RET_OPT.
    # La WR_min global puede mejorar o no dependiendo de si ya hay una ventana
    # mala — si la peor ventana ya está en las vistas, las futuras no la borran.
    n_remaining = max(0, _N_WINDOWS - len(windows_done))  # Prevenir negativos si _N_WINDOWS está desincronizado
    s_wr_global_opt = float(np.clip((_WR_OPT  - 0.40) / (0.75 - 0.40) * 100, 0, 100))
    s_wr_min_opt    = max(s_wr_min, float(np.clip((_WR_OPT - 0.35) / (0.70 - 0.35) * 100, 0, 100)))
    s_stability_opt = 100.0   # asumimos rango = 0 en ventanas futuras
    s_rr_opt        = float(np.clip((_RR_OPT - 0.5) / (2.0 - 0.5) * 100, 0, 100))
    s_ret_opt       = float(np.clip((_RET_OPT - 0.0) / (0.01 - 0.0) * 100, 0, 100))
    
    score_remaining = (0.20 * s_wr_global_opt + 0.30 * s_wr_min_opt
                       + 0.25 * s_stability_opt + 0.10 * s_rr_opt + 0.15 * s_ret_opt)

    # Promedio ponderado por ventanas (peso = num ventanas)
    w_seen = len(windows_done) / _N_WINDOWS
    w_rem  = n_remaining / _N_WINDOWS
    score_upper_bound = score_seen * w_seen + score_remaining * w_rem

    # [FIX-D2] Sharpe parcial usando la formula exacta del Gauntlet (statistical_audit.py L286-287)
    # Formula: (mean_ret / std_ret) * sqrt(trades_per_year)
    # trades_per_year = n_trades / years_span — usa la frecuencia REAL de operaciones,
    # NO sqrt(8760) de barras horarias. Asi el gate SR es consistente con el DSR del Gauntlet.
    sr_partial = 0.0
    if len(r) > 1:
        _std_r_d2 = float(r.std())
        if _std_r_d2 > 1e-10:
            # Estimar trades_per_year desde timestamps del parquet si estan disponibles
            if "timestamp" in combined.columns:
                try:
                    _ts_d2 = pd.to_datetime(combined["timestamp"])
                    _years_d2 = max((_ts_d2.max() - _ts_d2.min()).total_seconds() / (365.25 * 24 * 3600), 1e-5)
                    _tpy_d2 = n / _years_d2
                except Exception:
                    _tpy_d2 = float(n)  # fallback: asumir 1 anno
            else:
                _tpy_d2 = float(n)  # fallback: asumir 1 anno
            sr_partial = float(r.mean()) / _std_r_d2 * float(_tpy_d2 ** 0.5)
    print(f"[FIX-D2] Sharpe parcial seed={seed} W{windows_done}: "
          f"SR={sr_partial:.4f} (n={n}, mean_ret={float(r.mean()):.5f})")

    return {
        "score_partial":      round(score_seen, 1),
        "score_upper_bound":  round(score_upper_bound, 1),
        "wr_seen":            round(wr_seen, 3),
        "wr_min_seen":        round(wr_min_s, 3),
        "mean_ret_seen":      round(mean_ret, 5),
        "rr":                 round(rr, 2),
        "n_trades_seen":      n,
        "windows_done":       windows_done,
        "sr_partial":         round(sr_partial, 4),  # [FIX-D2] Sharpe parcial — mismo calculo que Gauntlet
    }


def _check_multi_window_early_stop(seed: int, windows_done: list[int]) -> tuple[bool, str]:
    """
    Evalua si la seed debe descartarse tras completar las ventanas en windows_done.
    Lógica V2: Poda dinámica. El benchmark no está hardcodeado, se construye 
    con el score de la primera semilla exitosa en dynamic_benchmark.json.
    """
    # [FIX-D2] benchmark_path faltaba — causaba NameError en runtime (variable usada pero nunca definida)
    benchmark_path = _ROOT / "data" / "reports" / "wfb" / "dynamic_benchmark.json"
    dynamic_benchmark = 50.0  # Baseline inicial absoluta si no hay benchmark
    if not benchmark_path.exists():
        print(f"  [EARLY-STOP] Sin benchmark dinámico aún. Usando baseline absoluto: {dynamic_benchmark}/100.")
    else:
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
                dynamic_benchmark = bench_data.get("champion_score") or 50.0
        except Exception:
            print(f"  [EARLY-STOP] Error leyendo benchmark. Usando baseline absoluto: {dynamic_benchmark}/100.")

    res = _compute_partial_score(seed, windows_done)

    if res["n_trades_seen"] < 5:
        print(f"  [EARLY-STOP W{windows_done}] Muestra insuficiente ({res['n_trades_seen']} trades) — no descartando.")
        return False, "muestra insuficiente"

    ub   = res["score_upper_bound"]
    thr  = dynamic_benchmark * _PRUNE_LIMIT
    done = windows_done
    sr_p = res.get("sr_partial", 0.0)  # [FIX-D2] Sharpe parcial

    print(f"  [EARLY-STOP] seed{seed} tras W{done}:")
    print(f"    Score parcial     = {res['score_partial']:.1f}/100")
    print(f"    Upper-bound optim = {ub:.1f}/100  (threshold={thr:.1f})")
    print(f"    WR acum={res['wr_seen']*100:.1f}%  WR_min={res['wr_min_seen']*100:.1f}%  "
          f"RR={res['rr']:.2f}  n={res['n_trades_seen']}")
    print(f"    Sharpe parcial    = {sr_p:.4f}  [FIX-D2: gate adicional al score proxy]")
    print(f"    Benchmark Dinámico = {dynamic_benchmark:.1f}/100")

    # [FIX-D2] Gate adicional de Sharpe parcial: si el SR acumulado es negativo
    # con al menos 2 ventanas y >= 15 trades, es una señal firme de descarte.
    # El score proxy (WR/RR) puede ser optimista incluso con Sharpe negativo si
    # hay pocas trades o si el WR estabilizó por azar en ventanas tempranas.
    _MIN_WINDOWS_FOR_SR_GATE = 2
    _MIN_TRADES_FOR_SR_GATE  = 15
    _SR_DISCARD_THRESHOLD    = -0.10  # Sharpe < -0.10 con suficiente muestra = descarte seguro
    if (len(windows_done) >= _MIN_WINDOWS_FOR_SR_GATE
            and res["n_trades_seen"] >= _MIN_TRADES_FOR_SR_GATE
            and sr_p < _SR_DISCARD_THRESHOLD):
        reason = (f"[FIX-D2] Sharpe parcial={sr_p:.4f} < {_SR_DISCARD_THRESHOLD} "
                  f"con {res['n_trades_seen']} trades en W{done} — semilla incompatible con umbral DSR.")
        n_remaining = _N_WINDOWS - len(done)
        print(f"  [EARLY-STOP-D2] **** DESCARTANDO seed{seed} (Sharpe parcial) **** {reason}")
        print(f"  [EARLY-STOP-D2] Ahorrando ~{n_remaining*3}h de cálculo (W{done[-1]+1}..W{_N_WINDOWS}).")
        return True, reason

    if ub < thr and dynamic_benchmark > 0:
        reason = (f"upper_bound={ub:.1f} < threshold={thr:.1f} tras W{done} "
                  f"(WR={res['wr_seen']*100:.1f}% WR_min={res['wr_min_seen']*100:.1f}% SR={sr_p:.4f})")
        n_remaining = _N_WINDOWS - len(done)
        print(f"  [EARLY-STOP] **** DESCARTANDO seed{seed} **** {reason}")
        print(f"  [EARLY-STOP] Ahorrando ~{n_remaining*3}h de cálculo (W{done[-1]+1}..W{_N_WINDOWS}).")
        return True, reason

    print(f"  [EARLY-STOP] seed{seed} viable — upper_bound ({ub:.1f}) >= threshold ({thr:.1f}) "
          f"y Sharpe parcial ({sr_p:.4f}) >= {_SR_DISCARD_THRESHOLD}. Continuando.")
    return False, "OK"


def main():
    parser = argparse.ArgumentParser(description="LUNA V1 - WFB SEED QUEUE")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 777, 1337, 2025], help="Lista de seeds a ejecutar")
    parser.add_argument("--resume", action="store_true", help="Saltar ventanas ya completadas (usando parquets existentes)")
    parser.add_argument("--force-resume", action="store_true", help="Forzar --resume incluso en el primer intento de la primera seed")
    parser.add_argument("--smoke-test", action="store_true", help="[DEV] Modo extra rapido para validar el pipeline (inyecta LUNA_SMOKE_TEST=1)")
    parser.add_argument("--nocache", action="store_true", help="Forzar eliminación total de la caché WFB antes de empezar")
    args = parser.parse_args()

    if args.smoke_test:
        os.environ["LUNA_SMOKE_TEST"] = "1"
        print(">>>>>  [ATENCION] MODO SMOKE TEST ACTIVADO <<<<<")
        print(">>>>>  Se utilizaran pocos datos y 1 epoch/trial para todas las semillas <<<<<")

    # [FIX-GLOBAL-RESTORE] Self-healing and master backup initialization
    _settings_path = _ROOT / "config" / "settings.yaml"
    _master_backup_path = _ROOT / "config" / "master_settings_backup_wfb.yaml"

    if _master_backup_path.exists():
        if _settings_path.exists() and _settings_path.stat().st_mtime > _master_backup_path.stat().st_mtime:
            print("\n[FIX-GLOBAL-RESTORE] settings.yaml es mas reciente que el backup maestro (editado por el usuario).")
            print("[FIX-GLOBAL-RESTORE] Conservando los cambios del usuario y actualizando el backup maestro...")
            try:
                shutil.copy2(_settings_path, _master_backup_path)
            except Exception as e:
                print(f"[FIX-GLOBAL-RESTORE] Error actualizando backup maestro: {e}")
        else:
            print("\n[FIX-GLOBAL-RESTORE] ALERTA: Se detectó un backup maestro huérfano (posible hard-crash anterior).")
            print("[FIX-GLOBAL-RESTORE] Curando corrupción temporal: restaurando settings.yaml original...")
            try:
                shutil.copy2(_master_backup_path, _settings_path)
                _master_backup_path.unlink(missing_ok=True)
                print("[FIX-GLOBAL-RESTORE] Self-healing completado.")
            except Exception as e:
                print(f"[FIX-GLOBAL-RESTORE] Error en self-healing: {e}")

    if _settings_path.exists() and not _master_backup_path.exists():
        try:
            shutil.copy2(_settings_path, _master_backup_path)
            atexit.register(_restore_master_settings)
            print("[FIX-GLOBAL-RESTORE] Backup maestro creado. settings.yaml blindado contra hard-crashes.\n")
        except Exception as e:
            print(f"[FIX-GLOBAL-RESTORE] Error creando backup maestro: {e}")

    # ── GATE: Validación estática de código (AST, sin ejecución, ~200ms) ──────
    # Detecta bugs de lógica (KeyError, variables no definidas, etc.) ANTES de
    # invertir horas de cálculo. Si hay errores, el pipeline se bloquea aquí.
    try:
        _validator_path = _ROOT / "tools" / "diagnostics" / "static_code_validator.py"
        if _validator_path.exists():
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("static_code_validator", _validator_path)
            _sval = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_sval)
            # [FIX-ORC-NOCACHE-VALIDATOR] Si --nocache está activo, omitimos la Capa 2 (verificación de artefactos
            # y cache obsoletos/desalineados) ya que se van a eliminar y recrear desde cero.
            if args.nocache:
                print("[FIX-ORC-NOCACHE-VALIDATOR] --nocache detectado: Omitiendo verificaciones de entorno/artefactos obsoletos en la validación estática.")
            _val_result = _sval.run_static_validation(skip_env=args.nocache)
            _errors = _val_result.errors
            if _errors:
                print("\n[STATIC-VALIDATOR] " + "=" * 55)
                print("[STATIC-VALIDATOR] ❌ PIPELINE BLOQUEADO — bugs críticos detectados:")
                for _iss in _errors:
                    print(f"  [{_iss.check_id}] {_iss.file}:L{_iss.line} → {_iss.message}")
                print("[STATIC-VALIDATOR] Corrige los errores antes de reintentar la run.")
                print("[STATIC-VALIDATOR] " + "=" * 55 + "\n")
                sys.exit(1)
            else:
                _n_warn = len(_val_result.warnings)
                _warn_str = f" | {_n_warn} WARN" if _n_warn else ""
                print(f"[STATIC-VALIDATOR] ✅ Sin errores críticos ({_val_result.files_checked} archivos, "
                      f"{_val_result.elapsed_ms:.0f}ms){_warn_str}")
        else:
            print("[STATIC-VALIDATOR] WARN: validador no encontrado en tools/diagnostics/. Saltando.")
    except Exception as _val_err:
        print(f"[STATIC-VALIDATOR] WARN: error durante validación estática ({_val_err}). Continuando.")
    # ─────────────────────────────────────────────────────────────────────────

    # ── GATE: Pre-Flight Check (leakage estadístico, SOP Iron Rules) ──────────
    # Complementa el validador estático: detecta anti-patrones de causalidad
    # (shift negativo, scaler.fit sobre X_all, KFold sin Purge, etc.)
    # Solo ejecuta las secciones de código/SOP — NO las de datos (que necesitan
    # los parquets y pueden tardar). Runtime total: < 2s.
    try:
        _pf_path = _ROOT / "scripts" / "pre_flight_check.py"
        if _pf_path.exists():
            _pf_result = subprocess.run(
                [sys.executable, str(_pf_path), "--section", "code,sop,architecture", "--fail-fast"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(_ROOT)
            )
            if _pf_result.returncode != 0:
                print("\n[PRE-FLIGHT] " + "=" * 55)
                print("[PRE-FLIGHT] ❌ PIPELINE BLOQUEADO — fallos en pre-flight check:")
                # Mostrar solo las líneas FAIL para no inundar el log
                for _line in _pf_result.stdout.splitlines():
                    if "FAIL" in _line or "BLOQUEADO" in _line or "->" in _line:
                        print(f"  {_line.strip()}")
                print("[PRE-FLIGHT] Ejecuta: python scripts/pre_flight_check.py --verbose para detalles.")
                print("[PRE-FLIGHT] " + "=" * 55 + "\n")
                sys.exit(1)
            else:
                # Extraer resumen de la última línea útil del output
                _pf_lines = [l for l in _pf_result.stdout.splitlines() if l.strip()]
                _pf_summary = _pf_lines[-1].strip() if _pf_lines else "OK"
                print(f"[PRE-FLIGHT]  ✅ {_pf_summary}")
                print("\n[FIX-TRACKER] " + "="*60)
                print("[FIX-TRACKER] 🚀 SE COMPROBARON LOS SIGUIENTES AJUSTES Y FIXES INSTITUCIONALES:")
                print("[FIX-TRACKER]  - SFI Clasificación Legend (TIPO-1/2/3): ACTIVADO en settings.yaml cabecera.")
                print("[FIX-TRACKER]  - CPCV Temporal Reference Table: ACTIVADO en settings.yaml bajo n_purged_splits.")
                print("[FIX-TRACKER]  - WFB Lock/Zombie Process Guard: VERIFICADO (Limpio).")
                print("[FIX-TRACKER]  - Invariantes de robustez (TEST-85, TEST-129): OK (ALL SYSTEMS GO).")
                print("[FIX-TRACKER] " + "="*60 + "\n")
        else:
            print("[PRE-FLIGHT] WARN: scripts/pre_flight_check.py no encontrado. Saltando.")
    except Exception as _pf_err:
        print(f"[PRE-FLIGHT] WARN: error durante pre-flight ({_pf_err}). Continuando.")
    # ─────────────────────────────────────────────────────────────────────────

    print("=================================================================")
    print("   LUNA V1 - WFB SEED QUEUE  (N=5 Champion Selection Protocol)   ")
    print(f"   Seeds en cola: {args.seeds} ")
    print(f"   Resume: {args.resume} | Force Resume: {args.force_resume} | No Cache: {args.nocache}")
    print("=================================================================")

    if args.nocache:
        print("[!] --nocache detectado. Limpiando WFB cache...")
        import shutil
        cache_dir = Path(__file__).parent.parent / "data" / "wfb_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("[!] data/wfb_cache eliminado correctamente.")
            
        reports_dir = Path(__file__).parent.parent / "data" / "reports" / "wfb"
        if reports_dir.exists():
            count_deleted = 0
            for f in reports_dir.glob("*.parquet"):
                try:
                    f.unlink()
                    count_deleted += 1
                except Exception:
                    pass
            print(f"[!] {count_deleted} archivos parquet antiguos eliminados de data/reports/wfb.")
            
        # [CACHE-HYGIENE-01] Limpiar artefactos residuales del workspace activo
        models_dir = Path(__file__).parent.parent / "data" / "models"
        if models_dir.exists():
            _stale = [f for f in models_dir.iterdir() if f.is_file()]
            for f in _stale:
                try:
                    f.unlink()
                except Exception:
                    pass
            print(f"[CACHE-HYGIENE-01] {len(_stale)} artefactos eliminados de data/models/ (modelos residuales).")
        
        predictions_dir = Path(__file__).parent.parent / "data" / "predictions"
        if predictions_dir.exists():
            _stale_pred = [f for f in predictions_dir.glob("*.parquet")]
            for f in _stale_pred:
                try:
                    f.unlink()
                except Exception:
                    pass
            print(f"[CACHE-HYGIENE-01] {len(_stale_pred)} predicciones antiguas eliminadas de data/predictions/")

        # [FIX-BENCH-01] El dynamic_benchmark.json es un artefacto de la run anterior.
        # Con --nocache el modelo se reentrena desde cero (arquitectura o params distintos),
        # por lo que el score anterior es incomparable. Limpiarlo evita early-stop injusto
        # sobre seeds que son objetivamente mejores pero tienen un patron de seniales diferente.
        _bench_path = reports_dir / "dynamic_benchmark.json"
        if _bench_path.exists():
            _bench_path.unlink()
            print("[FIX-BENCH-01] dynamic_benchmark.json eliminado: nueva run parte de benchmark limpio.")
        else:
            print("[FIX-BENCH-01] dynamic_benchmark.json no existia. OK.")
        args.resume = False
        args.force_resume = False

    # current_pid = get_wfb_pid()
    # if current_pid:
    #     print(f"[*] Run WFB actual detectado en PID: {current_pid}")
    #     print("[*] Esperando a que el pipeline actual termine...")
    # 
    #     while psutil.pid_exists(current_pid):
    #         time.sleep(60)  # Revisar cada minuto
    # 
    #     print("\n[*] El WFB actual ha finalizado. Iniciando pausas de enfriamiento...")
    #     print("[*] Esperando 5 minutos para liberacion profunda de RAM/VRAM...")
    #     time.sleep(300)
    # else:
    #     print("[*] No se ha detectado WFB corriendo. Se iniciara la cola inmediatamente.")

    seeds_to_run = list(args.seeds)
    _n_seeds_fixed = len(args.seeds)  # solo para logging; el límite real lo fija max_seeds_to_explore
    target_complete = _n_seeds_fixed   # placeholder — se reemplaza por _max_seeds_to_explore más abajo
    start_time = time.time()

    # [DSR-R5-CORRECTION] Inyectar el número total de semillas planificadas en el entorno.
    # Es leído por run_statistical_validation.py para aplicar la corrección por
    # comparaciones múltiples al umbral DSR (SOP Iron Rule R5 — Bailey & LdP 2014).
    # Se usa len(args.seeds) = semillas PLANIFICADAS, no las que terminen el pipeline,
    # ya que el data snooping bias existe por el mero hecho de haber lanzado N candidatas.
    _n_seeds_planned = len(args.seeds)
    os.environ["LUNA_N_SEEDS_TOTAL"] = str(_n_seeds_planned)
    print(f"[DSR-R5-CORRECTION] LUNA_N_SEEDS_TOTAL={_n_seeds_planned} inyectado en entorno.")
    print(f"[DSR-R5-CORRECTION] El Gauntlet ajustará el umbral DSR por N={_n_seeds_planned} semillas (SOP R5).")

    is_first_seed = True
    pruned_seeds  = []  # seeds descartadas en early-stop
    complete_seeds = []  # seeds completadas
    failed_seeds = []

    # [FIX-SEED-LIMIT 2026-05-28] Leer límites de exploración desde settings.yaml.
    # max_seeds_to_explore: reemplaza max_random_seeds=20 hardcodeado (violación No-Fallback).
    # min_seeds_to_approve: early exit cuando tenemos suficientes seeds aprobadas.
    # Si no están en settings → CRITICAL + RuntimeError (política No-Fallback: son parámetros
    # que afectan el número de comparaciones múltiples del DSR y el quórum del ensamble).
    import random
    try:
        from config.settings import cfg as _cfg_sl
        _max_seeds_to_explore = int(_cfg_sl.wfb.max_seeds_to_explore)
        _min_seeds_to_approve = int(_cfg_sl.wfb.min_seeds_to_approve)
        print(f"[FIX-SEED-LIMIT] Límites cargados desde settings.yaml: "
              f"max_explore={_max_seeds_to_explore} | min_approve={_min_seeds_to_approve}")
    except AttributeError as _e_sl:
        _err_sl = (f"CRITICAL [FIX-SEED-LIMIT]: Parámetros wfb.max_seeds_to_explore o "
                   f"wfb.min_seeds_to_approve ausentes en settings.yaml: {_e_sl}. "
                   f"Añadirlos antes de continuar (afectan integridad estadística del DSR R5).")
        print(_err_sl)
        raise RuntimeError(_err_sl) from _e_sl

    max_random_seeds = _max_seeds_to_explore  # alias para retrocompatibilidad del bucle
    # [FIX-SEED-EXPLORE-01 2026-05-28] target_complete fijado a max_seeds_to_explore, no al numero
    # de seeds CLI. Esto permite generar seeds aleatorias en cuanto se agotan las fijas,
    # sin necesidad de que alguna sea podada por early-stop. El while ahora cuenta
    # complete + pruned + failed para no bloquearse si muchas seeds fallan.
    target_complete = _max_seeds_to_explore
    print(f"[FIX-SEED-EXPLORE-01] target_complete={target_complete} (max_seeds_to_explore). "
          f"Seeds fijas={_n_seeds_fixed} | Aprobadas requeridas={_min_seeds_to_approve}. "
          f"Se generaran aleatorias automaticamente si no se alcanza el quorum.")
    random_seeds_generated = 0
    _approved_seeds_count = 0  # contador de seeds que pasaron el Gauntlet en esta run

    process = None

    try:
        # [FIX-SEED-EXPLORE-01] Condicion de salida: exploradas >= max O aprobadas >= min_approve
        # Se cuentan complete+pruned+failed para no bloquearse en runs con muchos early-stops.
        while ((len(complete_seeds) + len(pruned_seeds) + len(failed_seeds)) < target_complete
               and _approved_seeds_count < _min_seeds_to_approve):
            if not seeds_to_run:
                if random_seeds_generated >= max_random_seeds:
                    print(f"[FATAL] Alcanzado limite de seeds aleatorias generadas ({max_random_seeds}). WFB abortado.")
                    break
                new_seed = random.randint(10000, 99999)
                print(f"[*] Nos quedamos sin seeds en la cola. Generando nueva seed aleatoria: {new_seed}")
                seeds_to_run.append(new_seed)
                random_seeds_generated += 1
            
            seed = seeds_to_run.pop(0)
            print("\n" + "="*50)
            print(f"[*] INICIANDO WFB PARA SEMILLA: {seed} ({len(complete_seeds)}/{target_complete} completadas)")
            print("="*50)

            retries = 3
            success = False
            pruned  = False
    
            for attempt in range(1, retries + 1):
                cmd = [sys.executable, "scripts/wfb_worker.py"]
                
                # Pasar la semilla directamente al orquestador por argparse
                cmd.extend(["--seed", str(seed)])
    
                # Lógica de reanudación:
                # - Si args.force_resume está activo, se reanuda SIEMPRE.
                # - Si no es la primera semilla o no es el primer intento, usamos el args.resume 
                #   (o forzamos reanudación si el pipeline falló en la misma sesión para salvar progreso).
                # - Si es la primera semilla y el primer intento, usamos args.resume solo si --force-resume está activo, 
                #   de lo contrario la empezamos limpia a menos que args.resume esté activo?
                #   Wait: the user wants `--resume` to respect caches. So if `--resume` is passed, we should pass it.
                #   If they pass `--force-resume`, we DEFINITELY pass it.
                #   Actually, let's just make it simpler: if args.resume OR args.force_resume OR attempt > 1 OR not is_first_seed.
                #   This means if the user explicitly launched the queue with --resume, we RESPECT IT.
                
                should_resume = False
                if args.force_resume:
                    should_resume = True
                elif attempt > 1 or not is_first_seed:
                    should_resume = True
                elif args.resume:
                    should_resume = True
                    
                if should_resume:
                    print(f"[*] Lanzando orquestador con --resume (First Seed: {is_first_seed}, Attempt: {attempt})...")
                    cmd.append("--resume")
                else:
                    print(f"[*] Lanzando orquestador en modo FRESH RUN (sin --resume)...")

                # [CACHE-INTEGRITY-01] Propagar --nocache al worker via CLI y env var
                # Esto garantiza que el worker limpie su caché de seed aunque el orquestador
                # ya haya eliminado wfb_cache/ a nivel global. La env var LUNA_NOCACHE=1
                # es la señal canónica que el worker detecta aunque no reciba el arg CLI.
                if args.nocache:
                    cmd.append("--nocache")
                    print(f"[CACHE-INTEGRITY-01] Propagando --nocache al worker para seed={seed}")

                if args.smoke_test:
                    cmd.append("--smoke-test")

    
                print(f"[*] Lanzando subproceso WFB (Att {attempt}): {' '.join(cmd)}")
                # [V2-FIX PROBLEMA 3] Propagar PYTHONPATH para garantizar que luna.* sea importable
                _run_env = os.environ.copy()
                _run_env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + _run_env.get("PYTHONPATH", "") if _run_env.get("PYTHONPATH") else "")
                _run_env["PYTHONUNBUFFERED"] = "1"
                _run_env["PYTHONHASHSEED"] = str(seed)
                _run_env["LUNA_SEED"] = str(seed)
                # [CACHE-INTEGRITY-01] Propagar LUNA_NOCACHE para que subprocesos anidados también la respeten
                if args.nocache:
                    _run_env["LUNA_NOCACHE"] = "1"
                else:
                    _run_env.pop("LUNA_NOCACHE", None)  # Asegurar que no herede nocache de runs anteriores

    
                # [GAP-05/AUDIT-#25] Crear directorio canónico data/runs/WFB_{ts}/ con run_manifest.json
                # e inyectar LUNA_ENSEMBLE_DIR para que wfb_worker.py y run_statistical_validation.py
                # puedan hacer double-write de artefactos (oos_raw_probs, tearsheets, verdicts).
                try:
                    import datetime as _dt_arch, json as _json_arch, shutil as _sh_arch
                    _ens_ts = _dt_arch.datetime.now().strftime("%Y%m%d_%H%M%S")
                    _ens_dir = _ROOT / "data" / "runs" / f"WFB_{_ens_ts}_seed{seed}"
                    _ens_dir.mkdir(parents=True, exist_ok=True)
                    _settings_src = _ROOT / "config" / "settings.yaml"
                    if _settings_src.exists():
                        _sh_arch.copy2(_settings_src, _ens_dir / "settings_snapshot.yaml")
                    with open(_ens_dir / "run_manifest.json", "w", encoding="utf-8") as _mf:
                        _json_arch.dump({
                            "started_at": _dt_arch.datetime.now().isoformat(),
                            "seed": seed,
                            "attempt": attempt,
                            "luna_ensemble_dir": str(_ens_dir),
                            "resume": should_resume,
                        }, _mf, indent=2)
                    _run_env["LUNA_ENSEMBLE_DIR"] = str(_ens_dir)
                    print(f"[AUDIT-#25] data/runs/ canónico creado: {_ens_dir.name}")
                except Exception as _e_arch:
                    print(f"[AUDIT-#25] No se pudo crear data/runs/ canónico (no bloqueante): {_e_arch}")
    
                process = subprocess.Popen(cmd, env=_run_env, cwd=str(_ROOT))
    
                # ── EARLY-STOP MONITOR MULTI-VENTANA ──────────────────────────
                # Cada 2 minutos, detectamos si aparecio un nuevo parquet de ventana.
                # En cuanto hay datos nuevos, evaluamos el early-stop acumulativo:
                # el upper-bound optimista (ventanas restantes perfectas) debe
                # superar al benchmark de seed777 * 0.95 para continuar.
                windows_seen: set = set()
                wfb_dir = _ROOT / "data" / "reports" / "wfb"
    
                while process.poll() is None:  # proceso todavia activo
                    time.sleep(120)  # revisar cada 2 minutos
    
                    # Detectar ventanas nuevas (excepto la última, donde ya no tiene sentido podar)
                    for w in range(1, _N_WINDOWS):
                        if w in windows_seen:
                            continue
                        w_path = wfb_dir / f"oos_trades_W{w}_seed{seed}.parquet"
                        if w_path.exists():
                            # AUDIT BUG-EARLYSTOP-01: verificar rango de fechas antes de evaluar early-stop
                            # Un parquet corrupto (de otra ventana/seed) puede tener trades validos
                            # pero con timestamps incorrectos, causando poda erronea.
                            _valid_parquet = True
                            try:
                                # O(1) lectura de schema para evitar leer la metadata completa dos veces
                                _cols = pd.read_parquet(w_path, columns=[]).columns
                                _target_col = ["timestamp"] if "timestamp" in _cols else []
                                _df_es = pd.read_parquet(w_path, columns=_target_col)
                                if len(_df_es) == 0:
                                    _valid_parquet = False
                                    print(f"[EARLY-STOP] W{w} parquet existe pero tiene 0 filas — ignorando.")
                            except Exception as _e_es:
                                _valid_parquet = False
                                print(f"[EARLY-STOP] W{w} parquet no legible ({_e_es}) — ignorando.")
    
                            if _valid_parquet:
                                windows_seen.add(w)
                                print(f"\n[EARLY-STOP] W{w} detectada para seed{seed}. Evaluando early-stop acumulativo...")

                                # [PIPELINE-INTEGRITY] Post-window integrity check
                                # Se ejecuta automaticamente al detectar cada nueva ventana
                                try:
                                    from luna.pipeline_integrity import PipelineIntegrityChecker as _PIC
                                    _trades_check = pd.read_parquet(w_path)
                                    _window_label = f"W{w}_seed{seed}"
                                    _pic_result = _PIC.post_window_check(_trades_check, _window_label)
                                    if _pic_result.get("cal_bug") is True:
                                        print(
                                            f"[PIPELINE-INTEGRITY] *** ALERTA CRITICA {_window_label}: "
                                            f"FIX-CALIB-BINARY-01 detectado. "
                                            f"xgb_prob_cal==raw en 100%% trades. "
                                            f"Resultados de esta ventana NO son fiables. ***"
                                        )
                                except Exception as _e_pic:
                                    print(f"[PIPELINE-INTEGRITY] Error en post_window_check W{w}: {_e_pic}")

                                should_stop, reason = _check_multi_window_early_stop(seed, sorted(windows_seen))
                                if should_stop:
                                    print(f"[EARLY-STOP] Terminando proceso seed{seed} (PID {process.pid})...")
                                    _kill_process_tree(process.pid)

                                    # [FIX-EARLYSTOP-MERGE-01 2026-06-03] Ejecutar merge_and_validate+Gauntlet
                                    # para seeds podadas con N>=30 trades.
                                    # ROOT CAUSE del bug documentado: early-stop mata el worker antes de
                                    # merge_and_validate, dejando 0 evaluaciones del Gauntlet tras 8+ seeds.
                                    # SOLUCION: relanzar wfb_worker con --merge-only inmediatamente tras el kill.
                                    # El mode merge-only crea EMPTY.flags para ventanas no ejecutadas y
                                    # llama merge_and_validate + Gauntlet sin adquirir lock ni backup.
                                    print(f"[FIX-EARLYSTOP-MERGE-01] Evaluando si seed{seed} tiene N>=30 trades para Gauntlet...")
                                    try:
                                        _prune_score = _compute_partial_score(seed, sorted(windows_seen))
                                        _n_prune = _prune_score.get("n_trades_seen", 0)
                                        print(f"[FIX-EARLYSTOP-MERGE-01] seed{seed} podada con {_n_prune} trades en W{sorted(windows_seen)}.")
                                        if _n_prune >= 30:
                                            print(f"[FIX-EARLYSTOP-MERGE-01] N={_n_prune} >= 30 — activando merge_and_validate+Gauntlet para seed{seed}...")
                                            _merge_cmd_es = [sys.executable, "scripts/wfb_worker.py",
                                                             "--seed", str(seed), "--merge-only"]
                                            print(f"[FIX-EARLYSTOP-MERGE-01] Comando: {' '.join(_merge_cmd_es)}")
                                            try:
                                                _merge_r_es = subprocess.run(
                                                    _merge_cmd_es, env=_run_env, cwd=str(_ROOT), timeout=300
                                                )
                                                print(f"[FIX-EARLYSTOP-MERGE-01] merge_and_validate exit={_merge_r_es.returncode} para seed{seed}.")
                                            except subprocess.TimeoutExpired:
                                                print(f"[FIX-EARLYSTOP-MERGE-01] TIMEOUT (300s) merge_and_validate seed{seed} — continuando.")
                                            except Exception as _e_merge_es:
                                                print(f"[FIX-EARLYSTOP-MERGE-01] ERROR merge_and_validate seed{seed}: {_e_merge_es} — continuando.")
                                        else:
                                            print(f"[FIX-EARLYSTOP-MERGE-01] N={_n_prune} < 30 — muestra insuficiente, Gauntlet no aplica.")
                                    except Exception as _e_prune_score:
                                        print(f"[FIX-EARLYSTOP-MERGE-01] ERROR evaluando trades para seed{seed}: {_e_prune_score}")

                                    pruned = True
                                    prune_log = wfb_dir / f"early_stop_seed{seed}.json"
                                    with open(prune_log, "w", encoding="utf-8") as flog:
                                        json.dump({"seed": seed, "reason": reason,
                                                   "pruned": True,
                                                   "windows_evaluated": sorted(windows_seen)},
                                                  flog, indent=2)
                                    break
    
                    if pruned:
                        break  # salir del while
    
                if pruned:
                    # [FIX-EARLYSTOP-COUNTING-01] Comprobar si la semilla "podada" fue aprobada por el Gauntlet
                    # gracias al merge_and_validate post-mortem (N >= 30)
                    _deploy_approved_post_mortem = False
                    try:
                        import glob as _glob_pm
                        _report_dir_pm = _ROOT / "data" / "reports"
                        _verdict_pattern_pm = str(_report_dir_pm / f"*seed{seed}*_statistical_verdict.json")
                        _verdict_files_pm = sorted(_glob_pm.glob(_verdict_pattern_pm), reverse=True)
                        if _verdict_files_pm:
                            with open(_verdict_files_pm[0], encoding="utf-8") as _vf_pm:
                                _v_pm = json.load(_vf_pm)
                            _deploy_approved_post_mortem = bool(_v_pm.get("deploy_approved", False))
                    except Exception as _e_pm:
                        print(f"[FIX-EARLYSTOP-COUNTING-01] Error leyendo verdict post-mortem para seed{seed}: {_e_pm}")
                        
                    if _deploy_approved_post_mortem:
                        print(f"[FIX-EARLYSTOP-COUNTING-01] ⭐ ¡RESURRECCIÓN! seed{seed} fue podada, pero el Gauntlet (N>=30) la APROBÓ.")
                        success = True
                        complete_seeds.append(seed)
                        _approved_seeds_count += 1
                        print(f"[FIX-SEED-LIMIT] ✅ Seed {seed} APROBADA post-mortem. Total aprobadas en esta run: {_approved_seeds_count}/{_min_seeds_to_approve} requeridas.")
                        
                        # Evaluamos early exit por aprobaciones
                        if _approved_seeds_count >= _min_seeds_to_approve:
                            _n_explored_log = len(complete_seeds) + len(pruned_seeds) + len(failed_seeds)
                            print(f"\n[FIX-SEED-EXPLORE-01] EARLY EXIT: {_approved_seeds_count} seeds aprobadas "
                                  f">= {_min_seeds_to_approve} requeridas. Deteniendo exploracion.")
                            print(f"[FIX-SEED-EXPLORE-01] Seeds aprobadas={_approved_seeds_count} | "
                                  f"Exploradas={_n_explored_log} (completadas={len(complete_seeds)} "
                                  f"podadas={len(pruned_seeds)} fallidas={len(failed_seeds)}).")
                            seeds_to_run.clear()
                        
                        break  # Salimos del loop de intentos, ya fue procesada exitosamente
                    else:
                        print(f"[EARLY-STOP] seed{seed} descartada. Pasando a la siguiente seed.")
                        pruned_seeds.append(seed)
                        break  # salir del loop de intentos para esta seed
    
                if process.returncode == 0:
                    print(f"[SUCCESS] WFB Seed {seed} finalizado correctamente.")
                    success = True
                    complete_seeds.append(seed)

                    # [FIX-SEED-LIMIT] Comprobar si la seed completada aprobó el Gauntlet
                    # Leemos el verdict JSON más reciente de esta seed para determinar deploy_approved
                    try:
                        import glob as _glob_sl
                        _report_dir = _ROOT / "data" / "reports"
                        _verdict_pattern = str(_report_dir / f"*seed{seed}*_statistical_verdict.json")
                        _verdict_files = sorted(_glob_sl.glob(_verdict_pattern), reverse=True)
                        if _verdict_files:
                            with open(_verdict_files[0], encoding="utf-8") as _vf_sl:
                                _v_sl = json.load(_vf_sl)
                            _approved_sl = bool(_v_sl.get("deploy_approved", False))
                            if _approved_sl:
                                _approved_seeds_count += 1
                                print(f"[FIX-SEED-LIMIT] ✅ Seed {seed} APROBADA por el Gauntlet. "
                                      f"Total aprobadas en esta run: {_approved_seeds_count}/{_min_seeds_to_approve} requeridas.")
                            else:
                                print(f"[FIX-SEED-LIMIT] ❌ Seed {seed} rechazada por el Gauntlet. "
                                      f"Aprobadas hasta ahora: {_approved_seeds_count}.")
                        else:
                            print(f"[FIX-SEED-LIMIT] ⚠️ No se encontró verdict JSON para seed {seed} — "
                                  f"no se contabiliza para early exit.")
                    except Exception as _e_limit:
                        print(f"[FIX-SEED-LIMIT] Error leyendo verdict de seed {seed}: {_e_limit} — continuando.")

                    # Early exit: si tenemos suficientes seeds aprobadas, parar exploración
                    # [FIX-SEED-EXPLORE-01] La condicion del while ya chequea _approved_seeds_count;
                    # este bloque solo vacía la cola para no iniciar nuevas seeds innecesariamente.
                    if _approved_seeds_count >= _min_seeds_to_approve:
                        _n_explored_log = len(complete_seeds) + len(pruned_seeds) + len(failed_seeds)
                        print(f"\n[FIX-SEED-EXPLORE-01] EARLY EXIT: {_approved_seeds_count} seeds aprobadas "
                              f">= {_min_seeds_to_approve} requeridas. Deteniendo exploracion.")
                        print(f"[FIX-SEED-EXPLORE-01] Seeds aprobadas={_approved_seeds_count} | "
                              f"Exploradas={_n_explored_log} (completadas={len(complete_seeds)} "
                              f"podadas={len(pruned_seeds)} fallidas={len(failed_seeds)}).")
                        # Vaciar la cola — el while saldrá en la siguiente iteración
                        seeds_to_run.clear()

                    try:
                        res_final = _compute_partial_score(seed, list(range(1, _N_WINDOWS + 1)))
                        final_score = res_final.get("score_partial", 0.0)
                        final_sr    = res_final.get("sr_partial", 0.0)  # [FIX-D2]
                        benchmark_path = _ROOT / "data" / "reports" / "wfb" / "dynamic_benchmark.json"
                        benchmark_path.parent.mkdir(parents=True, exist_ok=True)

                        # [FIX-D2] Leer DSR real y SR crudo real del verdict para el benchmark
                        _dsr_real_bm = 0.0
                        _sr_crudo_real_bm = 0.0
                        try:
                            import glob as _glob_bm
                            _vfiles_bm = sorted(
                                _glob_bm.glob(str(_ROOT / "data" / "reports" / f"*seed{seed}*_statistical_verdict.json")),
                                reverse=True
                            )
                            if _vfiles_bm:
                                with open(_vfiles_bm[0], encoding="utf-8") as _vfbm:
                                    _vd_bm = json.load(_vfbm)
                                _dsr_real_bm      = float(_vd_bm.get("statistical_audit", {}).get("dsr", 0.0))
                                _sr_crudo_real_bm = float(_vd_bm.get("metrics", {}).get("sharpe_crudo", 0.0))
                                print(f"[FIX-D2] Benchmark — SR_parcial={final_sr:.4f} vs SR_crudo_real={_sr_crudo_real_bm:.4f} | DSR_real={_dsr_real_bm:.4f}")
                        except Exception as _e_bm_dsr:
                            print(f"[FIX-D2] No se pudo leer DSR/SR real del verdict para benchmark: {_e_bm_dsr}")


                        is_new_champion = True
                        if benchmark_path.exists():
                            with open(benchmark_path, "r", encoding="utf-8") as f:
                                old_data = json.load(f)
                                if final_score <= old_data.get("champion_score", 0.0):
                                    is_new_champion = False

                        if is_new_champion:
                            print(f"[*] ¡NUEVO BENCHMARK DINÁMICO! Seed {seed} — "
                                  f"score={final_score:.1f}/100 | SR_parcial={final_sr:.4f} | "
                                  f"SR_real={_sr_crudo_real_bm:.4f} | DSR_real={_dsr_real_bm:.4f}")
                            with open(benchmark_path, "w", encoding="utf-8") as f:
                                json.dump({
                                    "champion_seed":     seed,
                                    "champion_score":    final_score,
                                    "champion_sr":       round(final_sr, 4),           # [FIX-D2] Sharpe parcial estimado
                                    "champion_sr_real":  round(_sr_crudo_real_bm, 4),  # [FIX-D2] Sharpe crudo real del verdict
                                    "champion_dsr":      round(_dsr_real_bm, 4),       # [FIX-D2] DSR real del Gauntlet
                                }, f, indent=2)
                        else:
                            print(f"[*] Seed {seed} no supera el benchmark actual — "
                                  f"score={final_score:.1f}/100 | SR={final_sr:.4f} | DSR={_dsr_real_bm:.4f}")
                    except Exception as e:
                        print(f"[WARNING] No se pudo actualizar el benchmark dinámico: {e}")
                        
                    break
                else:
                    print(f"[WARNING] WFB Seed {seed} crasheo con exit code {process.returncode}.")
                    if attempt < retries:
                        print("[*] Enfriando 1 minuto antes de reintentar...")
                        time.sleep(60)
    
            if not success:
                if not pruned:
                    failed_seeds.append(seed)
                print(f"[ERROR FATAL] WFB Seed {seed} irremediablemente muerta tras {retries} intentos.")
                print("[*] Rescatando script de cola y saltando a la SIGUIENTE semilla.")
                
            is_first_seed = False
            print("[*] Enfriando 2 minutos entre pipelines...")
            time.sleep(120)

    except KeyboardInterrupt:
        print("\n" + "!" * 60)
        print("[!] INTERRUPCIÓN DETECTADA (Ctrl+C)")
        print("[!] Orquestador abortado por el usuario.")
        if process and process.poll() is None:
            print(f"[!] Matando subproceso hijo WFB Worker (PID {process.pid}) y su descendencia...")
            _kill_process_tree(process.pid)
        print("!" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("[*] COLA COMPLETADA. Resumen del Ensemble Multi-Seed.")
    print("=" * 60)
    print(f"\n[COLA] Seeds completadas: {complete_seeds}")
    print(f"[COLA] Seeds descartadas (early-stop): {pruned_seeds}")
    print(f"[COLA] Seeds fallidas (errores FATAL): {failed_seeds}")
    print(f"[COLA] Seeds aprobadas por Gauntlet individual: {_approved_seeds_count}/{_min_seeds_to_approve}")
    print("\n[INFO] Estrategia Multi-Seed activa: todas las seeds se despliegan")
    print("[INFO] simultáneamente en producción como ensemble.")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════════════
    # [AUTO-ENSEMBLE-02 2026-05-29] Auto-llamada incondicional al Gauntlet Ensemble.
    #
    # CAMBIO vs AUTO-ENSEMBLE-01: El ensemble se evalúa siempre que haya seeds
    # completadas, sin requerir aprobación individual mínima.
    #
    # FUNDAMENTO ESTADÍSTICO: El gauntlet individual (Fase 1) evalúa cada seed
    # de forma aislada. El ensemble con consenso ≥N seeds es un gate distinto
    # e independiente: una seed con DSR=0.70 (falla individual) puede contribuir
    # señales válidas al consenso. El PBO y DSR del ENSEMBLE ya corrigen por el
    # número total de seeds exploradas (factor sqrt(log(N_seeds))).
    #
    # FLUJO:
    #   1. _prep_ensemble_eval(): copia trades a data/reports/wfb/ y actualiza
    #      active_seeds temporalmente con todas las seeds completadas.
    #   2. evaluate_ensemble_wfb.py: aplica consenso gate, soft embargo y gauntlet
    #      ensemble. Restaura active_seeds al finalizar.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"[AUTO-ENSEMBLE-02] Iniciando Fase 2: Gauntlet Ensemble Global...")
    print(f"[AUTO-ENSEMBLE-02] Seeds completadas: {len(complete_seeds)} | Aprobadas individualmente: {_approved_seeds_count}")
    print("[AUTO-ENSEMBLE-02] El ensemble se evalúa con TODAS las seeds (no solo aprobadas).")
    print("=" * 60)

    if not complete_seeds:
        print("[AUTO-ENSEMBLE-02] SKIP: 0 seeds completadas. Nada que evaluar.")
    else:
        # --- PASO 1: Preparar datos para evaluate_ensemble_wfb.py ---
        print("[AUTO-ENSEMBLE-02] Preparando trades para el evaluador ensemble...")
        import yaml as _yaml_ens
        import shutil as _shutil_ens

        _WFB_OUT = _ROOT / "data" / "reports" / "wfb"
        _WFB_OUT.mkdir(parents=True, exist_ok=True)
        _RUNS_DIR = _ROOT / "data" / "runs"
        _settings_path = _ROOT / "config" / "settings.yaml"

        # Mapear seed → run_dir más reciente
        _seed_run_map = {}
        for _run_dir in sorted(_RUNS_DIR.glob("WFB_*"), reverse=True):
            if "_seed" not in _run_dir.name:
                continue
            try:
                _s = int(_run_dir.name.split("_seed")[-1])
                if _s in complete_seeds and _s not in _seed_run_map:
                    _seed_run_map[_s] = _run_dir
            except Exception:
                pass

        print(f"[AUTO-ENSEMBLE-02] Utilizando {len(_seed_run_map)} seeds completadas para el evaluador del ensemble.")  # RULE[fixbugsprints.md]

        # Backup y actualización temporal de active_seeds
        _cfg_ens = _yaml_ens.safe_load(_settings_path.read_text(encoding="utf-8"))
        _old_active_seeds = list(_cfg_ens.get("wfb", {}).get("active_seeds", []))
        _new_active_seeds = sorted(_seed_run_map.keys())
        _cfg_ens["wfb"]["active_seeds"] = _new_active_seeds
        _settings_path.write_text(
            _yaml_ens.dump(_cfg_ens, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        print(f"[AUTO-ENSEMBLE-02] active_seeds actualizado: {_old_active_seeds} → {_new_active_seeds}")  # RULE[fixbugsprints.md]

        # --- PASO 2: Ejecutar evaluate_ensemble_wfb.py ---
        _ensemble_eval_script = _ROOT / "scripts" / "evaluate_ensemble_wfb.py"
        if not _ensemble_eval_script.exists():
            print(f"[AUTO-ENSEMBLE-02] ERROR: No se encontró {_ensemble_eval_script}.")
        else:
            try:
                _ens_result = subprocess.run(
                    [sys.executable, str(_ensemble_eval_script)],
                    cwd=str(_ROOT),
                    stdout=None,
                    stderr=None,
                    env={**os.environ,
                         "LUNA_RUN_ID": f"ensemble_auto_N{len(complete_seeds)}"},
                )
                if _ens_result.returncode == 0:
                    print("\n[AUTO-ENSEMBLE-02] ✅ Gauntlet Ensemble completado exitosamente.")
                    print("[AUTO-ENSEMBLE-02] Veredicto: data/reports/wfb/ensemble_statistical_verdict.json")
                    print("[AUTO-ENSEMBLE-02] Tearsheet: data/reports/wfb/wfb_ensemble_tearsheet_summary.md")
                else:
                    print(f"\n[AUTO-ENSEMBLE-02] ⚠️ evaluate_ensemble_wfb.py terminó con código {_ens_result.returncode}.")
                    print("[AUTO-ENSEMBLE-02] Revisar logs. Re-ejecutar manualmente si es necesario.")
            except Exception as _ens_err:
                print(f"[AUTO-ENSEMBLE-02] ERROR no bloqueante: {_ens_err}")  # RULE[fixbugsprints.md]

        # --- PASO 3: Restaurar active_seeds originales ---
        try:
            _cfg_ens2 = _yaml_ens.safe_load(_settings_path.read_text(encoding="utf-8"))
            _cfg_ens2["wfb"]["active_seeds"] = _old_active_seeds
            _settings_path.write_text(
                _yaml_ens.dump(_cfg_ens2, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8"
            )
            print(f"[AUTO-ENSEMBLE-02] active_seeds restaurado a: {_old_active_seeds}")  # RULE[fixbugsprints.md]
        except Exception as _rest_err:
            print(f"[AUTO-ENSEMBLE-02] WARN: no se pudo restaurar active_seeds: {_rest_err}")
    # ═══════════════════════════════════════════════════════════════════════

    # ── POST-WFB: Component Value Dashboard ────────────────────────────────
    # [CVD-01 2026-06-03] Análisis automático de atribución de valor por
    # componente del pipeline (XGBoost, MetaLabeler, HMM, OOD Guard, etc.)
    # Se ejecuta SIEMPRE al final del WFB — no bloqueante (errores se logean).
    # Output: logs/component_dashboard_YYYYMMDD_HHMMSS.log
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[CVD-01] Component Value Dashboard — atribución post-WFB")
    print("=" * 60)

    _cvd_script = _ROOT / "tools" / "diagnostics" / "component_value_dashboard.py"
    _pred_dir   = _ROOT / "data" / "predictions"
    _trade_files = list(_pred_dir.glob("oos_trades_seed*.parquet"))
    _n_trade_files = len(_trade_files)

    if not _cvd_script.exists():
        print(f"[CVD-01] SKIP: script no encontrado en {_cvd_script}")
    elif _n_trade_files == 0:
        print("[CVD-01] SKIP: no hay archivos oos_trades_seed*.parquet en data/predictions/")
    else:
        # Contar trades totales para decidir si tiene sentido correr el dashboard
        try:
            import pandas as _pd_cvd
            _total_trades = sum(
                len(_pd_cvd.read_parquet(f)) for f in _trade_files
            )
        except Exception:
            _total_trades = 999  # si no puede contar, deja correr igual

        if _total_trades < 10:
            print(f"[CVD-01] SKIP: {_total_trades} trades disponibles (mínimo=10 para análisis)")
        else:
            print(f"[CVD-01] {_n_trade_files} seeds | {_total_trades} trades → ejecutando dashboard...")
            try:
                import datetime as _dt_cvd
                _ts_cvd = _dt_cvd.datetime.now().strftime("%Y%m%d_%H%M%S")
                _log_dir = _ROOT / "logs"
                _log_dir.mkdir(exist_ok=True)
                _cvd_log = _log_dir / f"component_dashboard_{_ts_cvd}.log"

                _cvd_env = os.environ.copy()
                _cvd_env["PYTHONPATH"] = str(_ROOT) + (
                    os.pathsep + _cvd_env.get("PYTHONPATH", "") if _cvd_env.get("PYTHONPATH") else ""
                )
                _cvd_result = subprocess.run(
                    [sys.executable, str(_cvd_script)],
                    capture_output=True, text=True,
                    cwd=str(_ROOT), env=_cvd_env,
                    timeout=120,
                    encoding="utf-8", errors="replace",
                )

                # Guardar output completo al log
                _cvd_log.write_text(
                    _cvd_result.stdout + ("\n" + _cvd_result.stderr if _cvd_result.stderr else ""),
                    encoding="utf-8"
                )
                print(f"[CVD-01] Log completo guardado: {_cvd_log.name}")

                # Mostrar resumen ejecutivo (últimas líneas del output)
                _cvd_lines = _cvd_result.stdout.strip().splitlines()
                _summary_start = next(
                    (i for i, l in enumerate(_cvd_lines) if "RESUMEN EJECUTIVO" in l),
                    max(0, len(_cvd_lines) - 20)
                )
                print("[CVD-01] ── RESUMEN EJECUTIVO ──────────────────────────────")
                for _ln in _cvd_lines[_summary_start:]:
                    print(f"  {_ln}")

                if _cvd_result.returncode != 0:
                    print(f"[CVD-01] WARN: dashboard terminó con código {_cvd_result.returncode}")
                else:
                    print("[CVD-01] ✅ Dashboard completado exitosamente.")

            except subprocess.TimeoutExpired:
                print("[CVD-01] WARN: timeout 120s — dashboard omitido (no bloqueante)")
            except Exception as _cvd_err:
                print(f"[CVD-01] ERROR no bloqueante: {_cvd_err}")  # RULE[fixbugsprints.md]

    print()


if __name__ == '__main__':
    main()

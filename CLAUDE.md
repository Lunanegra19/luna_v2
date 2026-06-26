# CLAUDE.md — Luna V2

> Instrucciones de proyecto para Claude Code. Las reglas SOP viven en `.agents/rules/` (fuente única, también usada por Antigravity) y se importan abajo. Workflows on-demand en `.agents/workflows/`.

## Qué es Luna V2
Sistema de trading cuantitativo (BTC, OKX). Pipeline: **WFB (Walk-Forward Backtest)** sobre ~29 seeds → XGBoost meta-modelos por régimen → MetaLabelerV2 (LSTM-32 + RF-300) → calibración → **Gauntlet** (validación estadística: DSR/PBO/binomial). Arquitectura **dual-bot** (squads `long`/`short`, `fase2.direction_mode: long|short|both`). Orquestador: `scripts/run_wfb_orchestrator.py`. Config única: `config/settings.yaml`.

## Estado actual (2026-06-24)
- **Objetivo en curso:** mejorar el squad LONG. Plan paso a paso (volumen-primero) en `docs/plan_mejora_long_paso_a_paso.md`. Baseline congelado: `data/reports/baseline_long_20260624.json`.
- **Entorno reconstruido** tras corrupción por crash de doble-orquestador (ver memoria `luna-v2-env-recovery`). Config = 2026 dual-bot, `direction_mode: long` (fase de testeo). Fresh runs requieren `LUNA_SKIP_ARTIFACT_CHECKS=1`.
- **Hallazgos clave** (memoria): el silenciador Kelly corta correctamente la cola perdedora (NO aflojar threshold/OOD); el cuello de volumen real es la **banda DVOL Guardian** (`guardian_dvol_*`), no `ood_guard.contamination`; el dual-split degradó el modelo long (83% de la caída de Sharpe).
- **Tooling de diagnóstico nuevo:** `tools/diagnostics/{measure_long_run.py, monitor_run.py, invalidate_oos_cache.py}`.

## ⚠️ Reglas críticas (TL;DR — detalle completo en los imports de abajo)
- **No-Fallback Silencioso (R16):** nada de valores hardcodeados; parámetros críticos se leen de `settings.yaml`; si falta uno → excepción `CRITICAL`, no `.get(x, default)` silencioso.
- **El orquestador RESTAURA `settings.yaml` al terminar.** Editar settings SIEMPRE con runs detenidas y relanzar con `--nocache`/`--sficache` (ver `settings_restore_protection`). Causa raíz del último incidente.
- **Un fix a la vez**, estudiado, testeado y confirmado dentro del pipeline (no código muerto). **Siempre añadir `print()` con tag `[FIX-NOMBRE fecha]`** para trazar en logs.
- **Al iniciar una run:** verificar que no hay runs activas/zombies → lanzar → vigilar los logs los primeros 30s.
- **Triple Frontera (R4):** el holdout 2025+ se toca UNA vez. **DSR no Sharpe bruto (R5).** Embargo ≥96H general / 24H solo en alto consenso (R3).
- Tras tocar params: `python tools/diagnostics/audit_parametros_fijos.py` + `python scripts/pre_flight_check.py`.

## Reglas SOP (always-on, importadas de .agents/rules/)
@.agents/rules/sop_v10_rules.md
@.agents/rules/parameters.md
@.agents/rules/settings_restore_protection.md
@.agents/rules/fixaplly.md
@.agents/rules/fixbugsprints.md
@.agents/rules/inciorun.md
@.agents/rules/estructuracarpetas.md
@.agents/rules/windowstats.md
@.agents/rules/diagnostico_cuantitativo.md
@.agents/rules/graphify.md

> Nota: `.agents/rules/settingsyfallvack.md` es duplicado de `parameters.md` (No-Fallback) — no se importa.

## Workflows on-demand (.agents/workflows/)
Procedimientos invocables manualmente (no always-on): `run_sentinel` (arrancar+monitorizar run final), `logsreview`/`logreview02` (revisión de logs), `testprofundidad` / `Anlisisprofundidadrun` (análisis profundo de run), `runfonrensics` (forense de run), `graphify` (mapa 3D del código).

## Comandos clave
```powershell
# Lanzar WFB (editar settings con runs detenidas primero)
python scripts/run_wfb_orchestrator.py --seeds 42 100 777 ... --nocache    # o --sficache (reusa SFI) / --resume
# direction_mode en settings.yaml (fase2). Fresh run: $env:LUNA_SKIP_ARTIFACT_CHECKS=1

# Validación / auditoría
python scripts/pre_flight_check.py --verbose
python tools/diagnostics/audit_parametros_fijos.py

# Medición / seguimiento (tooling propio)
python tools/diagnostics/measure_long_run.py --run <run_id> --json <out.json>
python tools/diagnostics/monitor_run.py <log>            # fase/ventana/banda DVOL/errores
python tools/diagnostics/invalidate_oos_cache.py --seeds ...   # fast-path: reusa entrenamiento, re-corre solo OOS

# Mapa de código (preferir sobre grep para arquitectura)
python graphify/run_offline.py        # regenerar el grafo tras cambios
```

## Entorno
- Windows + PowerShell (primario) / Git-Bash. Python en venv. GPU: RTX 5070 (CUDA).
- `data/` ignorado en git (data lake, modelos, caches). Logs en `logs/` (loguru).
- MCP: Antigravity tiene `~/.gemini/antigravity-ide/mcp_config.json` (9 servers, actualmente ROTOS — apuntan a `luna_*.py` inexistentes en rutas viejas). Claude Code no tiene MCP configurados; si se necesitan, crear `.mcp.json` (sin secretos hardcodeados — usar `env`).

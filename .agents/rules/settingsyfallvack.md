---
trigger: always_on
---

las implementaciones, configuraciones y arreglos de código deben respetar estrictamente la política institucional de **No-Fallback Silencioso** para evitar sesgos estadísticos devastadores (como el bug PBO_N_BLOCKS):
1. **Evitar Valores Hardcodeados o Mágicos:** Toda constante o parámetro debe leerse dinámicamente de `config/settings.yaml`.
2. **Política No-Fallback en Parámetros Críticos:**
   - Para Gates del Gauntlet (`min_dsr`, `max_pbo`, `min_trades`, `max_drawdown`, `pbo_n_blocks`), parámetros de riesgo (`embargo_hours`, `purge_hours`) e integridad de base de datos: **Prohibido el fallback silencioso (ej. `.get("param", default)` dentro de bloques `except` sin advertir)**. Si la lectura falla o el parámetro falta, se debe lanzar un error `CRITICAL` + `KeyError` o `RuntimeError` para forzar la parada visible.
   - Para parámetros menores de diagnóstico o informes, se permite un aviso `WARNING` o fallback silencioso `DEBUG`.
3. **Registro y Trazabilidad:** Todo número o parámetro fijo justificado debe registrarse y documentarse formalmente en el archivo [parametros_fijos.md](file:///g:/Mi%20unidad/ia/luna_v2/docs/parametros_fijos.md) en la carpeta `docs/`.
4. **Verificación de Auditoría:** Tras alterar parámetros o configs, ejecutar la auditoría estática con `python tools/diagnostics/audit_parametros_fijos.py` y actualizar el documento de control en docs.
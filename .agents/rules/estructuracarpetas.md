---
trigger: always_on
---

📂 config/ → Configuración: Archivos institucionales .yaml (settings.yaml).
📂 dashboard/ → Quantitative Web Dashboard: Servidor local/producción, interfaz interactiva premium, telemetría en vivo de la VPS y visualizadores.
📂 data/ → Data Lake: Base de datos persistente, descargas, cachés de ventanas temporales, modelos entrenados, logs de transacciones OOS y features canónicas. (Ignorado en repositorios git para no subirlos).
📂 docs/ → Documentación: Manuales operativos, parámetros fijos canónicos, propuestas de mejora cuantitativa e informes de testing.
📂 graphify/ → Motor Graphify: Código del visualizador 3D AST, analizador de cohesión estructural y dependencias funcionales.
📂 graphify-out/ o graphify/out/ → Salidas del Mapa 3D: Contiene el visualizador interactive graph.html, la base de datos estructural graph.json y GRAPH_REPORT.md.
📂 logs/ → Registro: Bitácoras de operaciones estructuradas producidas por loguru.
📂 luna/ → Core Package: Todo el código principal, modelos matemáticos, ingeniería de variables (features), conectores (OKX), utilidades de datos y configuración del pipeline.
📂 scripts/ → Orquestadores: Scripts de ejecución principal que inician el sistema (ej. train_production_model.py, run_wfb_orchestrator.py, sync_data_lake.py, evaluate_ensemble_wfb.py).
📂 tests/ → Validación: Pruebas unitarias e integración de la librería core.
📂 tools/ → Mantenimiento y Diagnóstico:
📂 tools/diagnostics/ → Scripts de análisis (ej. audit_parquet.py, check_vars.py, search.py, fix_tests.py, find_missing_config.py, audit_parametros_fijos.py).
📂 tools/refactor/ → Herramientas de utilidad técnica.
📂 tools/dumps/ → Volcados en crudo de texto para depuración (ej. error.txt).
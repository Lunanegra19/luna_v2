# Plan de Integración de Graphify en Luna V2

Este documento detalla el plan estratégico e institucional para encapsular **Graphify** (`graphifyy`) dentro de su propia carpeta en la raíz del proyecto. Explica cómo esta herramienta se integra directamente con el agente **Antigravity** mediante reglas en `.agents/` para optimizar el desarrollo, prevenir la introducción de "código muerto" (lógica huérfana) y auditar la cohesión estructural antes de cualquier despliegue.

---

## 1. Arquitectura de Encapsulación

Para evitar el desorden en el repositorio y mantener la coherencia con **`RULE[estructuracarpetas.md]`**, todo lo relacionado con Graphify se aísla en una carpeta propia en la raíz:

```text
📂 luna_v2/
├── 📂 graphify/                 # Carpeta propia en la raíz (100% aislada)
│   ├── 📄 run_offline.py        # Script ejecutor optimizado offline
│   └── 📂 out/                  # Directorio de salidas de Graphify
│       ├── 📄 graph.json        # Base de datos del Grafo de Conocimiento (AST)
│       ├── 📄 graph.html        # Mapa interactivo visual de dependencias 3D
│       └── 📄 GRAPH_REPORT.md   # Reporte analítico con God Nodes y comunidades
```

---

## 2. Cómo Antigravity usa Graphify automáticamente (Sin fricción)

Como agente **Antigravity**, he configurado e integrado las directivas de Graphify en nuestro sistema interno de agentes en la carpeta de la raíz **`.agents/`**. 

Cada vez que inicias un chat o me pides realizar una modificación estructural, yo sigo el protocolo definido en **`.agents/rules/graphify.md`**:

```text
Triggers: always_on
Reglas de Operación:
1. Ante preguntas sobre arquitectura, cohesión o refactorizaciones, consulto graphify/out/graph.json antes de proponer cambios.
2. Utilizo "graphify query <pregunta>" para buscar conexiones indirectas (Community Detection).
3. Utilizo "graphify path <componenteA> <componenteB>" para trazar dependencias y evitar acoplamientos circulares nocivos.
4. Tras cualquier refactorización, ejecuto el actualizador offline para regenerar el mapa y comprobar que no hay "código muerto".
```

---

## 3. Protocolo de Prevención de Errores y Calidad de Código

El Grafo nos permite implementar un **filtro de calidad proactivo** en tres puntos críticos:

### A. Detección de Código Muerto (Lógica Huérfana)
* **El Problema:** Al crear sistemas modulares complejos, es común definir lógica en `luna/` (ej. un nuevo filtro, calibrador o sizer) que el orquestador nunca llega a invocar.
* **La Solución:** Graphify detecta **nodos huérfanos** (grado de entrada = 0, sin llamadas entrantes). En cada run, filtramos los nodos con 0 dependencias entrantes que pertenezcan a producción para alertar al usuario antes de subir cambios.

### B. Evitar Acoplamientos Circulares (Circular Imports)
* **El Problema:** Importaciones cruzadas entre `luna/live/risk_monitor.py` y `luna/live/okx_connector.py` que causan fallos en tiempo de ejecución.
* **La Solución:** Analizamos los ciclos del grafo dirigidos mediante algoritmos de detección de ciclos en `graph.json`. Si un cambio introduce un ciclo, el script de validación aborta.

### C. Identificación de "God Nodes" Congestionados
* **El Problema:** Clases o módulos que acumulan demasiadas responsabilidades (alto grado de salida/entrada), volviéndose cuellos de botella de mantenimiento.
* **La Solución:** El reporte `GRAPH_REPORT.md` identifica automáticamente los "God Nodes" para planificar refactorizaciones preventivas.

---

## 4. Instrucciones de Ejecución y Sincronización

Para actualizar el Grafo de Conocimiento de forma manual o automatizada sin coste de API:

```bash
# Ejecutar desde la raíz del proyecto para actualizar el grafo estructural AST
python graphify/run_offline.py
```

El script procesará los archivos de forma local, aplicará el clustering de comunidades para detectar módulos funcionales y regenerará el reporte y mapa HTML.

---

## 5. Visualización del Grafo de Cohesión de Luna V2

Una vez generado, puedes abrir **`graphify/out/graph.html`** directamente en cualquier navegador web para explorar de forma visual e interactiva la red de dependencias en 3D. Las comunidades detectadas por clustering (ej. *Live Inference, OKX Execution, Feature Pipelines*) se colorearán de forma armoniosa para que comprendas la salud estructural del bot al instante.

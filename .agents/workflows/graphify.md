---
name: graphify
description: Guía de integración de Graphify para análisis y mapeo estructural en 3D
---

# Workflow: Graphify (Mapeo Estructural y Visualización 3D)

Graphify es la herramienta principal de navegación, control de cohesión y auditoría arquitectónica de **Luna V2**. Extrae de forma estática (AST) las dependencias, importaciones y jerarquías de llamadas del código fuente, organizándolo en comunidades funcionales mediante algoritmos de agrupamiento de grafos (Louvain Clustering).

---

## 🗺️ El Mapa Interactivo 3D (`graph.html`)

El **Mapa Interactivo en 3D no es solo un nombre comercial o una abstracción**, es un lienzo tridimensional completamente real y dinámico basado en WebGL que se puede abrir localmente en cualquier navegador web moderno (Chrome, Edge, Firefox, Brave).

### ¿Cómo visualizarlo y usarlo?
1. **Abrir el Archivo:** Abre en tu navegador el archivo ubicado en:
   `graphify/out/graph.html` (o haz clic en el archivo desde tu explorador).
2. **Controles de Navegación 3D:**
   * **Rotar Cámara:** Haz clic izquierdo y arrastra el ratón para girar alrededor del grafo en el espacio 3D.
   * **Desplazar (Pan):** Haz clic derecho (o mantén presionada la tecla `Ctrl` + clic izquierdo) y arrastra para mover el lienzo horizontal o verticalmente.
   * **Zoom:** Utiliza la rueda de desplazamiento del ratón (scroll wheel) para acercarte o alejarte del núcleo del sistema.
3. **Interacción con Nodos (Clases, Funciones y Módulos):**
   * **Hover (Pasar el cursor):** Al posar el cursor sobre cualquier nodo, se mostrarán sus metadatos (nombre cualificado, grado de centralidad, comunidad asignada y métrica de cohesión).
   * **Física en Tiempo Real (Grab & Drag):** Puedes hacer clic izquierdo sobre cualquier nodo, mantener presionado y arrastrarlo. Verás cómo responde el sistema de partículas 3D en tiempo real bajo la simulación de fuerzas (Force-Directed Graph), reorganizando dinámicamente los enlaces.

---

## 📂 Estructura de Salidas en `graphify/out/`

Cada vez que se ejecuta el mapeo, se generan y actualizan de forma autónoma los siguientes archivos en la ruta encapsulada:

* **`graph.html`**: El visualizador interactivo 3D de WebGL con física integrada y códigos de color por comunidad.
* **`graph.json`**: El mapa de base de datos estructural del AST (nodos y enlaces) con metadatos completos, utilizado por el Agente para consultas algorítmicas de arquitectura.
* **`GRAPH_REPORT.md`**: Un reporte analítico completo y estructurado en Markdown que expone de forma directa la salud del codebase, dependencias circulares, llamadas sorprendentes y nodos con alta centralidad.
* **Archivos ocultos de caché (`.graphify_*`)**: Archivos temporales de indexación interna que optimizan los tiempos de respuesta para ejecuciones posteriores.

---

## 📊 Insights y Resumen Estructural de Luna V2

En la última run ejecutada con éxito (completada localmente en menos de 3 segundos), el analizador mapeó el estado actual de Luna V2 arrojando los siguientes datos de control:

* **Mapeo Global:** **2,220 Nodos** y **3,065 Enlaces (Edges)** con un 93% de extracción AST directa.
* **Comunidades Funcionales Detectadas:** **265 comunidades** basadas en cohesión e importaciones cruzadas.

### 👑 Nodos Estrella ("God Nodes" - Puntos Críticos de Acoplamiento)
Son los componentes que más conexiones acumulan en todo el codebase. Cualquier cambio en ellos afecta transversalmente a múltiples capas:
1. `_read()` — **94 conexiones** (Capa Base de Carga de Datos)
2. `_cfg()` — **61 conexiones** (Configuración de Pipeline y Settings Globales)
3. `HMMRegimeModel` — **37 conexiones** (Motor Matemático/Estadístico de Regímenes HMM)
4. `FeaturePipeline` — **31 conexiones** (Ingeniería de Variables Canónicas)
5. `OnchainFetcher` — **30 conexiones** (Capa de Ingesta On-chain)
6. `DataCollector` — **29 conexiones** (Orquestador de Datos Históricos y Live)

### 🧩 Comunidades Clave de la Arquitectura
Las 8 familias de módulos más cohesionadas mapeadas por el algoritmo:
* **Live Ensemble Inference**: Motor de inferencia y predicción concurrente que integra XGBoost, LightGBM, MetaLabeler y Platt Scaling.
* **Walk-Forward Engine**: Motor WFB, orquestación de ventanas móviles, CPCV y generación de tearsheets de validación.
* **Data & Feature Pipelines**: Pipelines de descarga y procesamientos históricos de features on-chain, macro y derivados.
* **OKX Connector & Exec**: Control y ejecución de órdenes en Broker Live, cálculo de Sizing, Kelly Fraccionario y TP/SL.
* **Seed Optimization**: Diagnóstico y minería avanzada de semillas estadísticas bajo modelos Krogh & Vedelsby.
* **System Configuration**: Verificación cruzada de variables congeladas, settings estructurados de invariants y control de esquemas.
* **Unit Validation Tests**: Pruebas unitarias integradas de la librería core, simuladores y gauntlets de validación.
* **Diagnostics & Audit Tools**: Scripts de herramientas forenses, auditoría de parquets y diagnóstico dinámico de metalabeler.

---

## 🛠️ Cómo Utilizar e Integrar Graphify en el Desarrollo Diario

Tanto el **Agente de IA** como el **Desarrollador Humano** deben seguir esta disciplina para mantener la cohesión del sistema libre de código muerto, acoplamiento cíclico e importaciones rotas:

### 1. Actualización Automática del Grafo
Cada vez que se añadan nuevos archivos de código, modifiquen imports, o se altere la estructura modular en `/luna` o `/scripts`, debe ejecutarse:
```bash
python graphify/run_offline.py
```
*Este comando se ejecuta de manera local y ultra rápida al estar optimizado para saltarse directorios voluminosos de datos/logs, garantizando que el grafo permanezca siempre actualizado.*

### 2. Uso del Grafo por la IA (Instrucciones Permanentes de Contexto)
El Agente de IA tiene una regla activa (`always_on`) que le obliga a consultar `graphify/out/graph.json` antes de responder preguntas de arquitectura compleja o refactorización:
* **Detectar código huérfano:** Permite comprobar qué scripts u orquestadores no tienen llamadas entrantes hacia sus módulos.
* **Visualizar Rutas:** Rastrea las llamadas indirectas y las jerarquías de herencia y llamadas usando el algoritmo de camino más corto (`shortest_path`).
* **Prevenir Dependencias Circulares:** Al planificar una nueva feature, el Agente simula la inserción en el grafo para detectar y bloquear la introducción de importaciones cíclicas peligrosas.

---
💡 **Consejo Profesional:** Mantén siempre una pestaña del navegador con `graphify/out/graph.html` abierta para supervisar de forma visual la evolución, limpieza y orden modular de tu infraestructura cuantitativa en tiempo real.


---
description: Sentinel continuo para monitoreo profundo de runs activas. Usar con /goal o como directiva principal durante un entrenamiento.
---

**[DIRECTIVA DE MONITOREO CONTINUO Y RESOLUCIÓN DE INCIDENCIAS - RUN SENTINEL]**

Tu objetivo a partir de ahora es ejecutar un bucle de monitoreo ininterrumpido sobre la run activa hasta su finalización total. Debes actuar como un "Sentinel" técnico y cuantitativo, siguiendo estrictamente este protocolo cíclico:

### FASE 1: Monitoreo Activo (Bucle de 3 Minutos)
1. Revisa el estado del proceso en background (`command_status`) o lee el archivo de log principal activamente cada 3 minutos (espera el tiempo necesario en background entre verificaciones).
2. Extrae bloques sustanciales de logs y léelos **línea por línea**. 
3. **Cero Análisis Superficiales:** No asumas que todo está bien solo porque no hay un "Traceback". Busca proactivamente:
   - Anomalías cuantitativas en los prints (ej. Sharpe Ratios imposibles, caídas súbitas en el número de features, métricas evaporadas).
   - "Silent Failures" o lógicas matemáticas que estén produciendo advertencias repetitivas (warnings).
   - Posibles fallas de lógica en los estados de transición o embargos.

### FASE 2: Protocolo de Investigación Profunda (Ante Errores o Anomalías)
Si la run se detiene, crashea, o detectas resultados numéricos sin sentido:
1. Detén el bucle de lectura pasivo e inicia un diagnóstico de **Máxima Profundidad**.
2. Rastrea el origen de la anomalía o el error hacia la función específica en el código.
3. Comprende exhaustivamente por qué falló. **Prohibido realizar arreglos rápidos (chapuzas) o envolver bloques en `try-except` sin justificación matemática.** La solución debe ser estructural y profesional.

### FASE 3: Obligatoriedad de Testeo de Hipótesis (Iron Rule del Bug-Fixing)
**NUNCA IMPLEMENTES UN CAMBIO SIN TESTEARLO PRIMERO.**
1. Antes de modificar el core del pipeline, plantea tu hipótesis sobre por qué falló y cómo solucionarlo.
2. Escribe un script aislado en `tools/diagnostics/` o en `scratch/` para reproducir el bug, aislar el contexto y validar que tu propuesta soluciona el problema de raíz de manera matemática y computacional.
3. Solo tras validar el script con éxito, aplica la modificación en el código fuente de producción.

### FASE 4: Relanzamiento y Continuación del Bucle
1. Deten la run si no esta detenida, verifica que no queden procesos zombie de la run anterior, aplica el fix testeado, asi la toda la run tendra en cuenta el nuevo fix.
2. Relanza la run completa (añadiendo `--nocache` si el error involucró ajustes de parámetros).
3. **Vuelve inmediatamente a la FASE 1.** El bucle de monitoreo debe continuar iterando ininterrumpidamente hasta que la run reporte su finalización definitiva (100% completado).
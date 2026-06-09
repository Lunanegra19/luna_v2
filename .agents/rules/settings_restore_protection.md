---
trigger: always_on
---

## Política de Persistencia de Parámetros en settings.yaml

**Contexto del problema:** El orquestador WFB (`run_wfb_orchestrator.py`) hace un backup de `settings.yaml` al arrancar y lo **restaura automáticamente al terminar**. Cualquier parámetro añadido a `settings.yaml` DURANTE una run activa será **eliminado** cuando esa run termine y restaure su backup.

**Bug documentado (2026-05-29):** Los parámetros de LIFECYCLE-01 y WEIGHTED-STABILITY-01 fueron añadidos mientras el task-1310 estaba activo. Cuando task-1310 terminó a las 15:33, restauró el settings de las 08:44, borrando los 9 parámetros recién añadidos. El código funcionaba con defaults internos pero violaba la política No-Fallback.

---

### Regla Obligatoria: Añadir Parámetros SIEMPRE Fuera de Runs Activas

**Antes de añadir cualquier parámetro a `settings.yaml`:**

1. **Verificar que no hay runs activas:**
   ```powershell
   Get-Process python | Where-Object {$_.CPU -gt 1}
   # O consultar task activos con manage_task list
   ```

2. **Si hay una run activa:** esperar a que termine, o detenerla conscientemente antes de añadir parámetros.

3. **Después de añadir los parámetros:** relanzar la run con `--nocache` para que el nuevo settings sea el que se respalda y restaura.

4. **Verificar que el backup generado POR LA NUEVA RUN ya incluye los parámetros:**
   ```powershell
   Select-String -Path "g:\Mi unidad\ia\luna_v2\config\settings_backup_wfb_*.yaml" -Pattern "stability_variance_threshold" | Select -Last 1
   ```

---

### Checklist Obligatorio al Implementar Nuevos Parámetros

- [ ] No hay runs activas antes de editar settings.yaml
- [ ] Parámetros añadidos con comentario `# [NOMBRE-FIX fecha]` para trazabilidad
- [ ] Run lanzada DESPUÉS de añadir los parámetros (el backup incluirá los nuevos params)
- [ ] Verificar con el script de integridad que los parámetros son leídos correctamente
- [ ] Documentar los valores en `docs/parametros_fijos.md`

---

### Patrón de Comentario Obligatorio en settings.yaml

Cada bloque de parámetros nuevos debe llevar su etiqueta identificativa:

```yaml
  # [NOMBRE-FIX YYYY-MM-DD] Descripcion breve del fix
  nombre_parametro: valor   # Justificacion del valor
```

Ejemplo correcto:
```yaml
  # [LIFECYCLE-01 2026-05-29] Evaluacion consciente del ciclo de vida de features
  stability_variance_threshold: 1.0e-6  # STD por debajo = datos inexistentes/constantes
  stability_min_real_years: 2           # minimo anos con varianza real para evaluar
```

---

### Si se detecta que parámetros fueron borrados por un restore:

1. Re-añadirlos ANTES de que el SFI (o el componente que los usa) arranque
2. Verificar con script de integridad que son leídos
3. Documentar el incidente en el log de la sesión
4. La run activa los leerá correctamente si el SFI importa el módulo DESPUÉS de la corrección (el módulo lee `_cfg_sfi` en tiempo de importación, no en tiempo de inicio del orquestador)

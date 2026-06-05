"""
Global Telemetry Interceptor - Luna V1
===================================================
Captura eventos a nivel de sistema operativo y runtime de Python (excepciones no manejadas,
warnings de librerías como Pandas, Scikit-Learn y XGBoost) y los enruta hacia Loguru.

Esto evita los "fallos ciegos" (silent failures) donde un proceso aborta y su output va a
stderr sin ser capturado en los archivos de log estructurados del sistema.
"""

import sys
import warnings
import traceback
from loguru import logger

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Intercepción de excepciones no manejadas.
    En lugar de imprimir a sys.stderr, las envía a logger.critical.
    Ignora KeyboardInterrupt para permitir salidas manuales limpias.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Formatear la traza del error
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = "".join(tb_lines)

    logger.critical(f"[CRITICAL_CRASH] Excepción no manejada detectada a nivel global:\n{tb_text}")


def handle_warning(message, category, filename, lineno, file=None, line=None):
    """
    Intercepción de warnings.
    Enruta warnings emitidos por warnings.warn() hacia logger.warning.
    """
    # Filtramos warnings muy ruidosos si es necesario, pero por defecto lo guardamos todo
    warning_text = f"{category.__name__} en {filename}:{lineno} - {message}"
    logger.warning(f"[GLOBAL_WARNING] {warning_text}")


def activate_global_telemetry():
    """
    Inyecta los hooks de captura global en sys y warnings.
    Debe ser llamado al principio del entry point (orquestador).
    """
    sys.excepthook = handle_exception
    warnings.showwarning = handle_warning
    logger.success("[GLOBAL_TELEMETRY] Interceptores de excepciones y warnings activados correctamente.")

if __name__ == "__main__":
    activate_global_telemetry()
    warnings.warn("Probando intercepción de warning.")
    raise RuntimeError("Probando intercepción de crash.")

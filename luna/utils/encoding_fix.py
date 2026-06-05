"""
encoding_fix.py — FIX-ENCODING-01 (2026-03-21)
================================================
Fuerza UTF-8 en stdout/stderr para compatibilidad con Windows cp1252.

Sin esto, PowerShell (codepage 1252) genera:
  - UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'
  - Espanol roto: 'regimen' -> 'r\u00e0\u0152gimen'
  - Caracteres escapados: \\u2501, \\u2192 en lugar de ━, →

Uso — añadir al inicio de cualquier script Python:
    from luna.utils.encoding_fix import fix_stdout_encoding
    fix_stdout_encoding()
"""

import sys
import os
import io


def fix_stdout_encoding() -> None:
    """Fuerza UTF-8 en stdout/stderr y en subprocesos hijos via PYTHONIOENCODING."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # stdout
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "") != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    # stderr
    if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "") != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

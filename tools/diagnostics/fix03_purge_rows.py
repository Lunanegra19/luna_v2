"""
fix03_purge_rows.py — Fix #3: replace hardcoded purge_rows=336 with dynamic VBH-based calculation
Run: python tools/diagnostics/fix03_purge_rows.py
"""
import pathlib

TARGET = pathlib.Path(r'g:\Mi unidad\ia\luna_v2\luna\models\predict_oos.py')

# Read raw bytes to avoid encoding issues
raw = TARGET.read_bytes()

# Exact bytes found by inspection (file uses UTF-8 with mojibake in comments)
OLD = b'            total_rows = len(df_clean)\r\n            purge_rows = 336  # 14 d\xc3\xadas \xc3\x83\xc6\x92\xc3\xa2\xe2\x82\xac\xe2\x80\x9d 24H\r\n            split_idx  = int(total_rows * 0.80) + purge_rows'

NEW = b"""            total_rows = len(df_clean)
            # [FIX-03] purge_rows calculado dinamicamente (antes: 336 = 14d x 24H hardcodeado, asume velas 1H)
            # Ahora: int(vertical_barrier_hours * 1.5) proporcional al horizonte real del TBM
            try:
                from config.settings import cfg as _cfg_pr
                _vbh_pr = int(_cfg_pr.xgboost.vertical_barrier_hours)
            except Exception:
                _vbh_pr = 72
                print('[FIX-03] WARN: No se pudo leer vertical_barrier_hours. Usando fallback vbh=72H')
            purge_rows = int(_vbh_pr * 1.5)
            print('[FIX-03] Fallback OOS split: purge_rows=' + str(purge_rows) + ' filas (vbh=' + str(_vbh_pr) + 'H x1.5, antes hardcode 336)')
            split_idx  = int(total_rows * 0.80) + purge_rows"""

NEW = NEW.replace(b'\n', b'\r\n')

if OLD in raw:
    fixed = raw.replace(OLD, NEW, 1)
    TARGET.write_bytes(fixed)
    print('[FIX-03] OK — purge_rows=336 reemplazado por calculo dinamico VBH*1.5')
else:
    print('[FIX-03] ERROR — patron no encontrado. Verificar bytes exactos.')
    idx = raw.find(b'purge_rows = 336')
    if idx >= 0:
        print(f'  purge_rows offset: {idx}')
        print(f'  Contexto bytes: {repr(raw[idx-80:idx+60])}')

# Also fix the error message below that references 336H
OLD_ERR = b'                logger.error(\"Dataset insuficiente para generar per\xc3\xadodo OOS con purge de 336H.\")'
NEW_ERR = b'                logger.error(f\"Dataset insuficiente para OOS: purge_rows={purge_rows} filas (vbh_pr x1.5). [FIX-03]\")  # antes: 336H hardcode'

raw2 = TARGET.read_bytes()
if OLD_ERR in raw2:
    fixed2 = raw2.replace(OLD_ERR, NEW_ERR, 1)
    TARGET.write_bytes(fixed2)
    print('[FIX-03] OK — mensaje de error 336H tambien actualizado')
else:
    print('[FIX-03] WARN — mensaje de error 336H no encontrado (puede ser encoding distinto)')
    idx2 = raw2.find(b'purge de 336')
    if idx2 >= 0:
        print(f'  Contexto: {repr(raw2[idx2-10:idx2+60])}')

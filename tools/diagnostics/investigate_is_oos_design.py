"""
INVESTIGACIÓN ESTRUCTURAL: Diseño IS/OOS de las ventanas WFB
=============================================================
Pregunta: ¿Son los períodos IS demasiado cortos para capturar todos los regímenes?
Si el IS es solo 2 meses y el mercado no presenta un régimen en ese período,
el agente especializado de ese régimen no puede aprender nada.

Investigar:
1. ¿Cuánto dura el IS de cada ventana actualmente?
2. ¿Qué regímenes existen en el histórico COMPLETO?
3. ¿Cuántos bear bars habría con IS de 6/12 meses?
4. ¿Cuál es el diseño WFB en settings.yaml?
5. ¿Qué hace la industria (expanding vs rolling window)?
"""
import sys
sys.path.insert(0, 'g:/Mi unidad/ia/luna_v2')
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

SEP = '─'*68
cfg_raw = yaml.safe_load(Path('g:/Mi unidad/ia/luna_v2/config/settings.yaml').read_text(encoding='utf-8', errors='replace'))

print(SEP)
print('FASE 1: Configuración actual del WFB en settings.yaml')
print(SEP)
wfb_cfg = cfg_raw.get('wfb', {})
for k, v in wfb_cfg.items():
    print(f'  {k}: {v}')

print()
print(SEP)
print('FASE 2: Duración real de cada ventana IS/OOS (de los parquets)')
print(SEP)
cache = Path('g:/Mi unidad/ia/luna_v2/data/wfb_cache/W1/features')
total_is_hours = []
total_oos_hours = []
for w in ['W1','W2','W3','W4','W5']:
    val = cache / f'features_validation_{w}.parquet'
    hld = cache / f'features_holdout_{w}.parquet'
    for label, fp in [('IS', val), ('OOS', hld)]:
        if not fp.exists(): continue
        df = pd.read_parquet(fp)
        idx = df.index
        t_min, t_max = idx.min(), idx.max()
        n_hours = len(df)
        n_months = n_hours / (24*30)
        if label == 'IS': total_is_hours.append(n_hours)
        else: total_oos_hours.append(n_hours)
        print(f'  {w}/{label}: {str(t_min)[:10]} → {str(t_max)[:10]} = {n_hours}h = {n_months:.1f} meses')

print()
print(f'  Media IS: {np.mean(total_is_hours):.0f}h = {np.mean(total_is_hours)/(24*30):.1f} meses')
print(f'  Media OOS: {np.mean(total_oos_hours):.0f}h = {np.mean(total_oos_hours)/(24*30):.1f} meses')

print()
print(SEP)
print('FASE 3: Distribución histórica COMPLETA de regímenes HMM')
print(SEP)
# Buscar el parquet de features completo con HMM
data_dir = Path('g:/Mi unidad/ia/luna_v2/data')
hmm_feature_files = []
for pattern in ['features_full*.parquet', 'features_sfi*.parquet', 'features_post_sfi*.parquet']:
    hmm_feature_files += list(data_dir.rglob(pattern))

df_full = None
for ff in sorted(hmm_feature_files, key=lambda x: x.stat().st_size, reverse=True)[:5]:
    try:
        df = pd.read_parquet(ff)
        if 'HMM_Semantic' in df.columns and len(df) > 10000:
            df_full = df
            print(f'  Usando: {ff.name} ({len(df)} filas)')
            break
    except: pass

if df_full is not None:
    idx = df_full.index
    print(f'  Período completo: {str(idx.min())[:10]} → {str(idx.max())[:10]}')
    print(f'  Total horas: {len(df_full)} = {len(df_full)/(24*365):.1f} años')
    print()
    print('  Regímenes en el histórico completo:')
    regime_counts = df_full['HMM_Semantic'].value_counts()
    total = len(df_full)
    for regime, n in regime_counts.items():
        pct = n/total*100
        months_equiv = n/(24*30)
        print(f'    {regime:30s}: {n:6d} ({pct:5.1f}%) = {months_equiv:.1f} meses equiv.')
    
    print()
    BEAR_STATES = ['3_BEAR_CRASH', '3_BEAR_CRASH_B', '4_BEAR_FORCED']
    CALM_BEAR = ['3_CALM_BEAR']
    
    print('  ── Bear regimes totales ──')
    for state in BEAR_STATES + CALM_BEAR:
        n = (df_full['HMM_Semantic'] == state).sum()
        if n > 0:
            pct = n/total*100
            print(f'    {state}: {n} ({pct:.1f}%)')
    
    print()
    print('FASE 4: ¿Cuántos bear bars habría con IS de 6 y 12 meses?')
    print(SEP)
    print('  Comparativa: IS actual (2m) vs IS extendido (6m, 12m)')
    print()
    
    # Simular IS de distintas longitudes para W4 (Ago-Sep 2025 = el problemático)
    w4_oos_start = pd.Timestamp('2025-10-01', tz='UTC')
    
    for is_months in [2, 4, 6, 12, 24]:
        is_start = w4_oos_start - pd.DateOffset(months=2 + is_months)  # embargo 2m + IS
        is_end   = w4_oos_start - pd.DateOffset(months=2)
        df_is_sim = df_full[(df_full.index >= is_start) & (df_full.index < is_end)]
        n_bear_sim = df_is_sim['HMM_Semantic'].isin(BEAR_STATES).sum()
        n_calm_sim = df_is_sim['HMM_Semantic'].isin(CALM_BEAR).sum()
        print(f'    IS={is_months:2d}m ({str(is_start)[:10]}→{str(is_end)[:10]}): n={len(df_is_sim)} | bear_crash={n_bear_sim} | calm_bear={n_calm_sim}')

else:
    print('  No se encontró dataset completo con HMM_Semantic')
    # Intentar con los parquets de IS concatenados
    frames = []
    for w in ['W1','W2','W3','W4','W5']:
        fp = cache / f'features_validation_{w}.parquet'
        if fp.exists():
            df = pd.read_parquet(fp)
            frames.append(df)
    if frames:
        df_all = pd.concat(frames).sort_index()
        df_all = df_all[~df_all.index.duplicated()]
        print(f'  Usando concatenación de IS: {len(df_all)} filas')
        if 'HMM_Semantic' in df_all.columns:
            print(df_all['HMM_Semantic'].value_counts().to_string())

print()
print(SEP)
print('FASE 5: ¿Qué usa la industria? Expanding vs Rolling Window')
print(SEP)
print("""
  ROLLING WINDOW (actual Luna V2):
    IS siempre es un bloque fijo de ~2 meses
    ✓ Evita overfitting a datos antiguos
    ✓ El modelo es "fresco" y reciente
    ✗ Si el régimen no aparece en 2 meses → sin datos
    ✗ Vulnerable a regímenes raros (bear crash, forced sell)

  EXPANDING WINDOW (alternativa):
    IS crece con cada ventana (W1=2m, W2=4m, W3=6m, W4=8m, W5=10m)
    ✓ Siempre hay datos históricos de TODOS los regímenes
    ✓ El agente bear_long siempre tiene ejemplos (del 2022 crash, etc.)
    ✗ Riesgo de non-stationarity (datos de 2022 ¿son relevantes en 2025?)
    ✗ IS muy grande = entrenamiento lento

  HÍBRIDO (solución profesional):
    IS rolling DE MÍNIMO X meses + pool histórico de regímenes raros
    bear_long entrena sobre: IS reciente + todos los BEAR_CRASH históricos
    ✓ Evita el problema de régimen ausente
    ✓ Mantiene relevancia temporal
    ✓ Es lo que hacen fondos como AQR, Two Sigma
""")

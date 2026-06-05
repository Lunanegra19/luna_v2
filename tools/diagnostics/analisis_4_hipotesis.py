# -*- coding: utf-8 -*-
"""
Analisis de 4 hipotesis sobre falta de edge en 2025:
H1: Feature set inadecuado para 2025
H2: Periodo IS 2017-2024 no generaliza a 2025
H3: W4 (Oct-Dic 2025) puede tener mas edge
H4: Ensemble 12 seeds puede generar N suficiente
"""
import pandas as pd
import json
from pathlib import Path

BASE = Path("g:/Mi unidad/ia/luna_v2")

# ============================================================
# H1: Feature set - cuales features tienen PSI estable en OOS
# ============================================================
print("=== H1: FEATURE SET - QUE FEATURES SON ESTABLES EN 2025 ===")
sf_found = False
for pattern in ["**/seed42/W3/selected_features.json", "**/W3/seed42/selected_features.json", 
                "**/wfb_cache/seed42/W3/selected_features.json"]:
    sf_paths = list(BASE.glob(pattern))
    if sf_paths:
        sf = json.loads(sf_paths[0].read_text(encoding="utf-8"))
        print("Features SFI seleccionadas W3/seed42:")
        for f in sf.get("selected_features", []):
            print(f"  - {f}")
        print("Pass-through:")
        for f in sf.get("pass_through_features", []):
            print(f"  - {f}")
        sf_found = True
        break

if not sf_found:
    print("selected_features.json no encontrado. Listando cache structure:")
    for p in sorted(BASE.glob("**/selected_features.json"))[:8]:
        print(f"  {p.relative_to(BASE)}")

print()

# ============================================================
# H2: Distribucion regimenes IS vs OOS
# ============================================================
print("=== H2: REGIMEN IS vs OOS DISTRIBUTION ===")
hmm_paths = list(BASE.glob("**/hmm_regime_labels.parquet"))
feat_train_paths = list(BASE.glob("**/features_train.parquet"))

if hmm_paths:
    hmm = pd.read_parquet(hmm_paths[0])
    hmm.index = pd.to_datetime(hmm.index, utc=True, errors="coerce")
    print(f"Total filas HMM: {len(hmm)} | Rango: {hmm.index.min().date()} -> {hmm.index.max().date()}")
    
    col = "HMM_Semantic" if "HMM_Semantic" in hmm.columns else ("HMM_Regime" if "HMM_Regime" in hmm.columns else None)
    if col:
        # IS: todo antes de 2025-01-01
        hmm_is  = hmm[hmm.index < pd.Timestamp("2025-01-01", tz="UTC")]
        hmm_oos = hmm[hmm.index >= pd.Timestamp("2025-01-01", tz="UTC")]
        
        print(f"\nIS (hasta 2024-12-31): {len(hmm_is)} barras")
        dist_is = hmm_is[col].value_counts()
        for reg, n in dist_is.items():
            pct = n / len(hmm_is) * 100
            print(f"  {str(reg):<35}: {n:>6} ({pct:>5.1f}%)")
        
        print(f"\nOOS 2025: {len(hmm_oos)} barras")
        dist_oos = hmm_oos[col].value_counts()
        for reg, n in dist_oos.items():
            pct = n / len(hmm_oos) * 100
            print(f"  {str(reg):<35}: {n:>6} ({pct:>5.1f}%)")
        
        # Comparacion: que regimenes cambian mas
        print("\nCAMBIO IS -> OOS (diferencia de distribucion):")
        all_regs = set(dist_is.index) | set(dist_oos.index)
        for reg in sorted(all_regs, key=str):
            pct_is  = dist_is.get(reg, 0) / len(hmm_is) * 100
            pct_oos = dist_oos.get(reg, 0) / len(hmm_oos) * 100
            delta = pct_oos - pct_is
            arrow = "^^ MAS" if delta > 5 else ("vv MENOS" if delta < -5 else "~  igual")
            print(f"  {str(reg):<35}: IS={pct_is:>5.1f}% OOS={pct_oos:>5.1f}% delta={delta:>+6.1f}% {arrow}")
else:
    print("hmm_regime_labels.parquet no encontrado. Buscando...")
    for p in list(BASE.glob("**/*hmm*.parquet"))[:5]:
        print(f"  {p.relative_to(BASE)}")

print()

# ============================================================
# H3: W4 - que datos hay en Oct-Dic 2025
# ============================================================
print("=== H3: W4 (OCT-DIC 2025) - QUE MERCADO VERIA EL MODELO ===")
# Leer precio de BTC en el periodo Oct-Dic 2025 para entender el contexto
feat_paths = list(BASE.glob("**/features_train.parquet"))
if not feat_paths:
    feat_paths = list(BASE.glob("**/features/*.parquet"))

if feat_paths:
    # Leer solo columna close y HMM si disponible
    try:
        avail = pd.read_parquet(feat_paths[0], columns=["close"]).iloc[-8760:]  # ultimo anno
        avail.index = pd.to_datetime(avail.index, utc=True, errors="coerce")
        
        # Q4 2025 (periodo W4 OOS)
        w4 = avail[(avail.index >= "2025-10-01") & (avail.index <= "2025-12-31")]
        if len(w4) > 0:
            ret_w4 = (w4["close"].iloc[-1] / w4["close"].iloc[0] - 1) * 100
            max_w4 = w4["close"].max()
            min_w4 = w4["close"].min()
            dd_w4 = (min_w4 / max_w4 - 1) * 100
            vol_w4 = w4["close"].pct_change().std() * 100
            print(f"BTC Oct-Dic 2025:")
            print(f"  Precio inicio Oct: {w4['close'].iloc[0]:,.0f}")
            print(f"  Precio fin Dic:    {w4['close'].iloc[-1]:,.0f}")
            print(f"  Retorno periodo:   {ret_w4:+.1f}%")
            print(f"  Maximo:            {max_w4:,.0f}")
            print(f"  Minimo:            {min_w4:,.0f}")
            print(f"  Drawdown max:      {dd_w4:.1f}%")
            print(f"  Volatilidad hora:  {vol_w4:.4f}%")
            
            # Por mes
            for month in [10, 11, 12]:
                m = w4[w4.index.month == month]
                if len(m) > 0:
                    ret_m = (m["close"].iloc[-1] / m["close"].iloc[0] - 1) * 100
                    print(f"  {['Oct','Nov','Dic'][month-10]} 2025: {ret_m:+.1f}% | close_fin={m['close'].iloc[-1]:,.0f}")
        else:
            print("Datos Oct-Dic 2025 no disponibles en features_train.parquet")
            print(f"Ultimo dato disponible: {avail.index.max().date()}")
    except Exception as e:
        print(f"Error leyendo features: {e}")
else:
    print("features_train.parquet no encontrado")

print()

# ============================================================
# H4: Estimacion trades ensemble completo
# ============================================================
print("=== H4: ESTIMACION TRADES ENSEMBLE 12 SEEDS ===")
# Leer todos los oos_trades disponibles ya generados
trades_files = list(BASE.glob("**/oos_trades_W*.parquet"))
if trades_files:
    print(f"Archivos de trades OOS disponibles: {len(trades_files)}")
    resumen = {}
    for tf in sorted(trades_files):
        try:
            t = pd.read_parquet(tf)
            name = tf.name
            resumen[name] = len(t)
        except Exception:
            resumen[tf.name] = "ERROR"
    
    for name, n in sorted(resumen.items()):
        print(f"  {name}: {n} trades")
    
    total = sum(v for v in resumen.values() if isinstance(v, int))
    seeds_con_datos = len([k for k, v in resumen.items() if isinstance(v, int)])
    print(f"\nTotal trades OOS generados hasta ahora: {total}")
    print(f"Seeds/ventanas con datos: {seeds_con_datos}")
    if seeds_con_datos > 0:
        media = total / seeds_con_datos
        print(f"Media por seed/ventana: {media:.1f} trades")
        print(f"Estimacion 12 seeds x 4 ventanas = 48 combinaciones: ~{int(media*48)} trades totales")
else:
    print("No se encontraron archivos oos_trades OOS aun")
    print("Buscando en reports...")
    for p in list(BASE.glob("**/oos_trades*.parquet"))[:5]:
        print(f"  {p.relative_to(BASE)}")

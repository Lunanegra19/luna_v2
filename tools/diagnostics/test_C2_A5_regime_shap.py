"""
test_C2_regime_transitions.py
==============================
H-C2: ¿La transición de régimen predice el WR del trade?
  A: señales en el primer bar de nuevo régimen → alta convicción → WR alto
  B: señales en transición → inestabilidad HMM → WR bajo

test_A5_shap_drivers.py (integrado)
H-A5: ¿Los trades ganadores y perdedores son explicados por features distintas?
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json, ast

WFB = Path("data/reports/wfb")
dfs = []
for f in sorted(WFB.glob("oos_trades_W*_seed*.parquet")):
    parts = f.stem.split("_")
    df = pd.read_parquet(f)
    df["_w"]    = parts[2]
    df["_seed"] = parts[3].replace("seed", "")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined["entry_dt"] = pd.to_datetime(combined["entry_time"], utc=True, errors="coerce")
combined = combined.sort_values("entry_dt").reset_index(drop=True)
baseline_wr = combined["is_win"].mean()

# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("TEST C2 — Transicion de Regimen HMM vs WR")
print("=" * 65)
print(f"\nN trades: {len(combined)} | WR baseline: {baseline_wr:.4f}")

if "hmm_regime" not in combined.columns:
    print("  [SKIP] hmm_regime no disponible")
else:
    # Detectar cambios de régimen por seed y ventana
    combined["prev_regime"] = combined.groupby(["_seed","_w"])["hmm_regime"].shift(1)
    combined["regime_changed"] = (
        combined["hmm_regime"] != combined["prev_regime"]
    ) & combined["prev_regime"].notna()

    n_transitions = combined["regime_changed"].sum()
    n_stable      = (~combined["regime_changed"] & combined["prev_regime"].notna()).sum()
    print(f"\n  Trades tras transicion: {n_transitions} ({n_transitions/len(combined)*100:.1f}%)")
    print(f"  Trades en regimen estable: {n_stable} ({n_stable/len(combined)*100:.1f}%)")

    # WR comparativo
    trans_wr  = combined[combined["regime_changed"]]["is_win"].mean()
    stable_wr = combined[~combined["regime_changed"] & combined["prev_regime"].notna()]["is_win"].mean()
    print(f"\n  WR tras transicion:    {trans_wr:.4f} ({trans_wr-baseline_wr:+.4f})")
    print(f"  WR regimen estable:    {stable_wr:.4f} ({stable_wr-baseline_wr:+.4f})")

    # Test estadistico
    g1 = combined[combined["regime_changed"]]["is_win"].dropna()
    g2 = combined[~combined["regime_changed"] & combined["prev_regime"].notna()]["is_win"].dropna()
    if len(g1) >= 10 and len(g2) >= 10:
        chi2, p_chi2 = stats.chi2_contingency([
            [int(g1.sum()), len(g1)-int(g1.sum())],
            [int(g2.sum()), len(g2)-int(g2.sum())]
        ])[:2]
        print(f"  Chi² (transicion vs estable): chi2={chi2:.3f}  p={p_chi2:.4f}")
        if p_chi2 < 0.05:
            direction = "TRANSICION MEJOR" if trans_wr > stable_wr else "ESTABLE MEJOR"
            print(f"  → SIGNIFICATIVO: {direction}")
        else:
            print(f"  → No significativo")

    # Por tipo de transicion
    print("\n─" * 65)
    print("Tipos de transicion y WR")
    print("─" * 65)
    valid_trans = combined[combined["regime_changed"]].copy()
    valid_trans["trans_key"] = valid_trans["prev_regime"].astype(str) + " → " + valid_trans["hmm_regime"].astype(str)
    for key in valid_trans["trans_key"].value_counts().head(10).index:
        sub = valid_trans[valid_trans["trans_key"] == key]
        wr  = sub["is_win"].mean()
        print(f"  N={len(sub):3d} WR={wr:.3f} ({wr-baseline_wr:+.3f}) | {key}")

    # Veredicto C2
    print(f"\nVEREDICTO C2:")
    if p_chi2 < 0.05:
        print(f"  CONFIRMADA: transicion de regimen predice WR (p={p_chi2:.4f})")
        if trans_wr > stable_wr:
            print(f"  → Primera señal tras transicion es MAS rentable")
            print(f"  ACCION: Añadir feature 'bars_in_regime' al meta-modelo")
        else:
            print(f"  → Señal en regimen estable es MAS rentable")
            print(f"  ACCION: Añadir filtro: no operar en los primeros 2 bars tras transicion")
    else:
        print(f"  DESCARTADA: la transicion no predice WR (p={p_chi2:.4f})")

# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST A5 — SHAP Drivers: Wins vs Losses")
print("=" * 65)

if "shap_drivers" not in combined.columns:
    print("  [SKIP] shap_drivers no en parquet")
else:
    # Intentar parsear shap_drivers (puede ser JSON string, dict, o lista)
    sample = combined["shap_drivers"].dropna().iloc[0] if combined["shap_drivers"].notna().any() else None
    print(f"  Tipo de shap_drivers: {type(sample).__name__}")
    print(f"  Muestra: {str(sample)[:200]}")

    def parse_shap(x):
        if pd.isna(x) or x is None:
            return {}
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except:
                try:
                    return ast.literal_eval(x)
                except:
                    return {}
        return {}

    combined["shap_parsed"] = combined["shap_drivers"].apply(parse_shap)
    valid_shap = combined[combined["shap_parsed"].apply(len) > 0]
    print(f"\n  Trades con SHAP valido: {len(valid_shap)}")

    if len(valid_shap) > 10:
        # Feature más importante por trade
        def top_feature(d):
            if not d:
                return None
            return max(d, key=lambda k: abs(d[k]))

        valid_shap = valid_shap.copy()
        valid_shap["top_feat"] = valid_shap["shap_parsed"].apply(top_feature)

        wins   = valid_shap[valid_shap["is_win"] == 1]
        losses = valid_shap[valid_shap["is_win"] == 0]

        print("\n  Top features en WINS:")
        for feat, cnt in wins["top_feat"].value_counts().head(8).items():
            pct = cnt / len(wins) * 100
            print(f"    {str(feat)[:40]:40s}: {cnt:3d} ({pct:.1f}%)")

        print("\n  Top features en LOSSES:")
        for feat, cnt in losses["top_feat"].value_counts().head(8).items():
            pct = cnt / len(losses) * 100
            print(f"    {str(feat)[:40]:40s}: {cnt:3d} ({pct:.1f}%)")

        # Chi² de distribución de top features
        all_feats = valid_shap["top_feat"].value_counts().head(10).index.tolist()
        w_counts  = [wins["top_feat"].eq(f).sum() for f in all_feats]
        l_counts  = [losses["top_feat"].eq(f).sum() for f in all_feats]
        if sum(w_counts) > 0 and sum(l_counts) > 0:
            chi2_shap, p_shap = stats.chi2_contingency([w_counts, l_counts])[:2]
            print(f"\n  Chi² distribución features WINS vs LOSSES: p={p_shap:.4f}")
            if p_shap < 0.05:
                print(f"  → SIGNIFICATIVO: features distintas en wins vs losses")
            else:
                print(f"  → No significativo: mismas features en wins y losses")

        print(f"\nVEREDICTO A5:")
        if len(valid_shap) < 50:
            print(f"  N insuficiente para test estadístico robusto ({len(valid_shap)} trades con SHAP)")
        elif p_shap < 0.05:
            print(f"  CONFIRMADA: features distintas en wins vs losses (p={p_shap:.4f})")
        else:
            print(f"  DESCARTADA: mismas features en wins y losses (p={p_shap:.4f})")
    else:
        print("  N insuficiente para análisis SHAP")
        print("  ACCION: Verificar que shap_drivers se guarda en predict_oos.py")

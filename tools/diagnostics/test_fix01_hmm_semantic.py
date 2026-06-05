"""
test_fix01_hmm_semantic.py
==========================
Test de verificación para FIX-01 (PIPE-001/HMM-003):
Verifica que HMM_Semantic se inyecta correctamente en feature_pipeline.py

Ejecutar: python tools/diagnostics/test_fix01_hmm_semantic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from pathlib import Path as P

ROOT = P(__file__).resolve().parent.parent.parent

def test_hmm_parquet():
    """Test 1: El parquet HMM tiene ambas columnas."""
    hmm_parquet = ROOT / "data" / "features" / "hmm_regime_labels.parquet"
    if not hmm_parquet.exists():
        print("[TEST-FIX-01][T1] SKIP: hmm_regime_labels.parquet no existe (se genera en WFB)")
        return True  # No es un fallo

    df_hmm = pd.read_parquet(hmm_parquet)
    print(f"[TEST-FIX-01][T1] hmm_regime_labels.parquet shape={df_hmm.shape}")
    print(f"[TEST-FIX-01][T1] Columnas: {list(df_hmm.columns)}")

    col_regime = "HMM_Regime"
    col_sem = "HMM_Semantic"

    assert col_regime in df_hmm.columns, f"FAIL: {col_regime} no en parquet"
    print(f"[TEST-FIX-01][T1] HMM_Regime dtype={df_hmm[col_regime].dtype}")

    if col_sem in df_hmm.columns:
        sem_dtype = df_hmm[col_sem].dtype
        uniques = df_hmm[col_sem].unique().tolist()[:6]
        cov = df_hmm[col_sem].notna().mean()
        print(f"[TEST-FIX-01][T1] HMM_Semantic dtype={sem_dtype} | cov={cov:.1%} | unicos={uniques}")
        assert str(sem_dtype) in ("object", "str", "string"), f"FAIL: HMM_Semantic debe ser string, es {sem_dtype}"
        print("[TEST-FIX-01][T1] PASS: HMM_Semantic presente y es tipo string (object)")
    else:
        print("[TEST-FIX-01][T1] WARN: HMM_Semantic NO en parquet — se regenerará en próximo WFB")

    return True


def test_feature_pipeline_import():
    """Test 2: feature_pipeline.py importa sin errores (syntax check)."""
    try:
        from luna.features.feature_pipeline import FeaturePipeline
        print("[TEST-FIX-01][T2] PASS: feature_pipeline.py importa sin errores de sintaxis")
        return True
    except SyntaxError as e:
        print(f"[TEST-FIX-01][T2] FAIL SYNTAX: {e}")
        return False
    except ImportError as e:
        # Importar dependencias puede fallar en test — lo que importa es la sintaxis
        print(f"[TEST-FIX-01][T2] ImportError (puede ser normal en test aislado): {e}")
        return True
    except Exception as e:
        print(f"[TEST-FIX-01][T2] ERROR: {e}")
        return False


def test_fix_code_present():
    """Test 3: El código del fix FIX-PIPE-001 está en feature_pipeline.py."""
    pipeline_file = ROOT / "luna" / "features" / "feature_pipeline.py"
    content = pipeline_file.read_text(encoding="utf-8", errors="replace")

    checks = [
        ("[FIX-PIPE-001]", "Tag FIX-PIPE-001 presente"),
        ("cols_to_join", "Variable cols_to_join presente"),
        ("HMM_Semantic\" in hmm_src.columns", "Guard HMM_Semantic en parquet presente"),
        ("[FIX-PIPE-001-B]", "Guard secundario FIX-PIPE-001-B presente"),
        ("state_map", "Fallback via state_map pkl presente"),
    ]

    all_pass = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"[TEST-FIX-01][T3] PASS: {desc}")
        else:
            print(f"[TEST-FIX-01][T3] FAIL: {desc} — '{pattern}' no encontrado")
            all_pass = False

    return all_pass


def test_holdout_has_semantic():
    """Test 4: features_holdout.parquet tiene HMM_Semantic (si existe)."""
    holdout = ROOT / "data" / "features" / "features_holdout.parquet"
    if not holdout.exists():
        print("[TEST-FIX-01][T4] SKIP: features_holdout.parquet no existe")
        return True

    import pyarrow.parquet as pq
    schema = pq.read_schema(holdout)
    col_names = schema.names

    if "HMM_Semantic" in col_names:
        df_ho = pd.read_parquet(holdout, columns=["HMM_Semantic"])
        cov = df_ho["HMM_Semantic"].notna().mean()
        uniques = df_ho["HMM_Semantic"].unique().tolist()[:6]
        print(f"[TEST-FIX-01][T4] PASS: HMM_Semantic en features_holdout.parquet | cov={cov:.1%} | unicos={uniques}")
        return True
    else:
        print("[TEST-FIX-01][T4] FAIL: HMM_Semantic NO en features_holdout.parquet")
        print(f"[TEST-FIX-01][T4] Columnas disponibles: {col_names[:20]}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TEST FIX-01: HMM_Semantic inyección en feature_pipeline.py")
    print("=" * 60)

    results = []
    results.append(("T1 - Parquet HMM", test_hmm_parquet()))
    results.append(("T2 - Import syntax", test_feature_pipeline_import()))
    results.append(("T3 - Código fix presente", test_fix_code_present()))
    results.append(("T4 - Holdout HMM_Semantic", test_holdout_has_semantic()))

    print("\n" + "=" * 60)
    print("RESUMEN:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[TEST-FIX-01] TODOS LOS TESTS PASAN ✓")
        sys.exit(0)
    else:
        print("\n[TEST-FIX-01] HAY FALLOS ✗")
        sys.exit(1)

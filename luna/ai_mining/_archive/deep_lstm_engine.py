"""
Deep LSTM Engine — Luna V1 AI Mining (Engine 8/8)  [PyTorch]
=============================================================
Implementación de la "Opción 2" de la Fase 4 de Correlaciones:
  Redes Neuronales Profundas para BTC con PyTorch.

Arquitectura: Bidirectional LSTM (BiLSTM)
  - Entrada: features supervivientes del RFE (≤20 "features de titanio")
  - BiLSTM: 2 capas × 128 hidden units (bidireccional)
  - Dropout: 0.3 entre capas
  - Capa de salida: Linear → sigmoid (probabilidad alcista 24H)

Validación: Walk-Forward Anchored (SOP R4)
  - Ventana mínima de entrenamiento: 12 meses
  - Paso WF: 4 semanas (semanal)
  - Sin contaminación futuro → no look-ahead (R1)

Output:
  - data/ai_mining/reports/lstm_report.md
  - data/ai_mining/reports/lstm_predictions.csv
  - data/ai_mining/models/lstm_model.pt     (checkpoint)
  - engine_lstm_wf.png                      (WF equity curve + prob)
  - engine_lstm_features.png                (feature importance via ablation)

Referencia: Correlaciones fase_4_produccion_vision.md "Opción 2"
  "LSTM (Long Short-Term Memory): red recurrente para identificar
   secuencias a mediano/largo plazo en los features supervivientes."
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FEATURES  = PROJECT_ROOT / "data" / "features"
REPORTS_DIR    = PROJECT_ROOT / "data" / "ai_mining" / "reports"
MODELS_DIR     = PROJECT_ROOT / "data" / "ai_mining" / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# —— Lee parámetros desde settings.yaml (R: "ningún número mágico en scripts") ——
try:
    from config.settings import cfg
    _lm = cfg.ai_mining.lstm
    HIDDEN_SIZE      = int(_lm.hidden_size)
    NUM_LAYERS       = int(_lm.num_layers)
    DROPOUT          = float(_lm.dropout)
    SEQUENCE_LEN     = int(_lm.sequence_len)
    HORIZON_H        = int(_lm.horizon_hours)
    LEARNING_RATE    = float(_lm.learning_rate)
    BATCH_SIZE       = int(_lm.batch_size)
    EPOCHS_PER_FOLD  = int(_lm.epochs_per_fold)
    PATIENCE         = int(_lm.patience)
    WF_TRAIN_MONTHS  = int(_lm.wf_train_months)
    WF_STEP_WEEKS    = int(_lm.wf_step_weeks)
    MAX_FEATURES     = int(_lm.max_features)
except Exception:
    # Fallback si cfg no disponible (ejecución aislada / tests)
    HIDDEN_SIZE      = 128
    NUM_LAYERS       = 2
    DROPOUT          = 0.3
    SEQUENCE_LEN     = 168
    HORIZON_H        = 24
    LEARNING_RATE    = 1e-3
    BATCH_SIZE       = 64
    EPOCHS_PER_FOLD  = 30
    PATIENCE         = 7
    WF_TRAIN_MONTHS  = 18
    WF_STEP_WEEKS    = 4
    MAX_FEATURES     = 20

# ── Features a usar (RFE survivors priority, luego fallback) ─────────────────
RFE_CSV = REPORTS_DIR.parent / "ai_mining" / "optuna" / "deep_best_params.json"
PRIORITY_FEATURES = [
    # On-chain y derivados (mayor peso estadístico probado)
    "MVRV_Proxy", "SSR", "Whale_Vol_ZScore", "Tx_Fees_USD",
    "FundingRate", "DangerZone", "DVOL", "OI_BTC",
    # Macro causal significativo
    "GlobalM2_Index", "Fed_Net_Liquidity", "VIX", "DXY",
    "YieldCurve_10Y3M", "Stablecoin_Cap", "DeFi_WBTC_TVL",
    # Cross-asset
    "eth_btc_corr_24h", "Gold_Ret", "NASDAQ_Ret",
    # Outputs de engines previos (señales sintetizadas)
    "Master_Causal_Signal", "K_Shape_Cluster_ID",
]


# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
def _build_bilstm(n_features: int, hidden: int, layers: int, drop: float):
    """Construye modelo BiLSTM con PyTorch."""
    import torch
    import torch.nn as nn

    class BiLSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=layers,
                dropout=drop if layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
            self.dropout = nn.Dropout(drop)
            self.fc1 = nn.Linear(hidden * 2, 64)  # ×2 por bidireccional
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            out, _ = self.bilstm(x)
            # Tomar el último timestep
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.sigmoid(self.fc2(out))
            return out.squeeze(-1)

    return BiLSTMModel()


# ─────────────────────────────────────────────────────────────────────────────
class DeepLSTMEngine:
    """Motor BiLSTM con Walk-Forward anchored para AI Mining."""

    def __init__(self):
        self.device      = None
        self.features    = []
        self.model       = None
        self.scaler      = None

    # ── Carga de datos ────────────────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        for name in ["features_train_causal.parquet",
                     "features_train_kshape.parquet",
                     "features_train.parquet"]:
            p = DATA_FEATURES / name
            if p.exists():
                df = pd.read_parquet(p)
                df.index = pd.to_datetime(df.index, utc=True)
                logger.info(f"LSTM: cargado {df.shape} desde {name}")
                return df
        raise FileNotFoundError("No dataset en data/features/")

    # ── Selección de features ────────────────────────────────────────────────────
    def select_features(self, df: pd.DataFrame) -> list[str]:
        """
        Prioridad de selección de features:
          1. Granger *** features del advanced_engine_results.csv  (NUEVO)
          2. RFE survivors del deep_discovery_report.md
          3. PRIORITY_FEATURES como fallback genérico
          Cap final: MAX_FEATURES
        """
        # —— Nivel 1: features Granger *** del Advanced Engine ——
        adv_csv = REPORTS_DIR / "advanced_engine_results.csv"
        granger_feats: list[str] = []
        if adv_csv.exists():
            try:
                adv_df = pd.read_csv(adv_csv)
                # Top features por evidence_score (Granger *** + SHAP + Coherencia)
                top = adv_df.sort_values("evidence_score", ascending=False)
                candidates = top["variable"].tolist()
                granger_feats = [f for f in candidates if f in df.columns]
                if granger_feats:
                    # Enriquecer con features macro de alta confianza statística
                    extra_macro = [f for f in [
                        "YieldCurve_10Y3M", "T10Y2Y", "M2_USA_raw", "GlobalM2_Index",
                        "Fed_Net_Liquidity", "FedFundsRate", "VIX", "DXY",
                        "FundingRate", "MVRV_Proxy", "Stablecoin_Cap",
                        "Master_Causal_Signal", "K_Shape_Cluster_ID",
                    ] if f in df.columns and f not in granger_feats]
                    granger_feats = (granger_feats + extra_macro)[:MAX_FEATURES]
                    logger.info(f"LSTM: Nivel-1 Granger top features ({len(granger_feats)}): {granger_feats[:8]}...")
            except Exception as e:
                logger.warning(f"LSTM: error leyendo advanced_engine_results.csv: {e}")

        if granger_feats:
            logger.info(f"LSTM: features finales [{len(granger_feats)}] (fuente: Granger/SHAP CSV)")
            return granger_feats

        # —— Nivel 2: RFE survivors del deep_discovery ——
        rfe_report = REPORTS_DIR / "deep_discovery_report.md"
        rfe_feats: list[str] = []
        if rfe_report.exists():
            try:
                import re
                text = rfe_report.read_text(encoding="utf-8")
                # Buscar varias variantes del formato en el .md
                for pattern in [
                    r"supervivientes[:\s]+\[([^\]]+)\]",
                    r"RFE survivors[:\s]+\[([^\]]+)\]",
                    r"Top features[:\s]+\[([^\]]+)\]",
                ]:
                    m = re.search(pattern, text, re.IGNORECASE)
                    if m:
                        raw = m.group(1)
                        rfe_feats = [f.strip().strip("'\"") for f in raw.split(",")]
                        rfe_feats = [f for f in rfe_feats if f in df.columns]
                        if rfe_feats:
                            logger.info(f"LSTM: Nivel-2 RFE survivors ({len(rfe_feats)} features)")
                            break
            except Exception:
                pass

        if rfe_feats:
            logger.info(f"LSTM: features finales [{len(rfe_feats)}] (fuente: RFE survivors)")
            return rfe_feats[:MAX_FEATURES]

        # —— Nivel 3: fallback PRIORITY_FEATURES ——
        feats = [f for f in PRIORITY_FEATURES if f in df.columns][:MAX_FEATURES]
        logger.warning(f"LSTM: Nivel-3 fallback a PRIORITY_FEATURES ({len(feats)} features). "
                       f"Ejecuta el Advanced Engine primero para mejorar la selección.")
        return feats


    # ── Preparación de secuencias ─────────────────────────────────────────────
    def _build_sequences(
        self, df: pd.DataFrame, feats: list[str]
    ) -> tuple:
        """
        Construye (X, y) como tensores:
          X: (N, SEQUENCE_LEN, n_features)
          y: (N,) binario — 1 si BTC+24H > 0
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        # Target: retorno BTC 24H causal (shift -HORIZON_H)
        if "close" not in df.columns:
            raise ValueError("'close' no encontrada")

        target = (df["close"].shift(-HORIZON_H) / df["close"] - 1 > 0).astype(float)

        # Feature matrix
        X_raw = df[feats].ffill().fillna(0).values.astype(np.float32)
        y_raw = target.values.astype(np.float32)

        # StandardScaler solo en fit (no look-ahead)
        # Se aplicará fold a fold en WF
        sequences_X, sequences_y = [], []
        for i in range(SEQUENCE_LEN, len(X_raw) - HORIZON_H):
            sequences_X.append(X_raw[i - SEQUENCE_LEN:i])
            sequences_y.append(y_raw[i])

        X_arr = np.array(sequences_X, dtype=np.float32)
        y_arr = np.array(sequences_y, dtype=np.float32)

        # Índices de tiempo para los labels
        idx_labels = df.index[SEQUENCE_LEN: SEQUENCE_LEN + len(y_arr)]

        return X_arr, y_arr, idx_labels

    # ── Entrenamiento de un fold ──────────────────────────────────────────────
    def _train_fold(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        n_features: int,
    ):
        """Entrena BiLSTM en un fold de WF. Retorna el modelo entrenado."""
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        # Normalizar X en el fold de train
        scaler = StandardScaler()
        Xt = X_train.reshape(-1, n_features)
        Xt = scaler.fit_transform(Xt)
        X_train_s = Xt.reshape(X_train.shape)
        self.scaler = scaler  # guardar para aplicar en test

        # Tensores
        Xt = torch.tensor(X_train_s, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        # Modelo
        model = _build_bilstm(n_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

        # Dataset y DataLoader
        dataset  = torch.utils.data.TensorDataset(Xt, yt)
        loader   = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Entrenamiento con early stopping
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        model.train()
        for epoch in range(EPOCHS_PER_FOLD):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss - 1e-4:
                best_loss  = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model

    # ── Optuna NAS (Neural Architecture Search) ────────────────────────────
    def _optuna_nas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 20,
    ) -> dict:
        """
        Optuna NAS para hiperparámetros del BiLSTM.
        Busca: hidden_size, num_layers, dropout, sequence_len, learning_rate.
        Usa los primeros WF_TRAIN_MONTHS * 720 barras como IS y el siguiente
        mes como validation interna (sin contaminar el WF real).
        Guarda best_params en lstm_best_params.json y retorna el dict.
        """
        import torch
        import torch.nn as nn
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("LSTM NAS: optuna no instalado — saltando NAS")
            return {}

        OPTUNA_DIR = MODELS_DIR
        best_params_file = OPTUNA_DIR / "lstm_best_params.json"

        # Datos para NAS: primer bloque IS + 1 mes OOS
        min_is  = int(WF_TRAIN_MONTHS * 720)
        val_sz  = int(4 * 7 * 24)  # 4 semanas
        if len(X) < min_is + val_sz:
            logger.warning("LSTM NAS: datos insuficientes")
            return {}

        X_nas_tr = X[:min_is]
        y_nas_tr = y[:min_is]
        X_nas_val = X[min_is:min_is + val_sz]
        y_nas_val = y[min_is:min_is + val_sz]

        n_base_features = X.shape[2]  # número de features original

        logger.info(f"LSTM NAS: iniciando Optuna ({n_trials} trials)...")

        def objective(trial):
            hidden = trial.suggest_categorical("hidden_size", [64, 128, 192, 256])
            layers = trial.suggest_int("num_layers", 1, 3)
            drop   = trial.suggest_float("dropout", 0.1, 0.5)
            lr     = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            # sequence_len afecta el número de muestras disponibles — ya está fijo en X

            # Mini-entrenamiento (5 épocas rápidas)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(X_nas_tr.reshape(-1, n_base_features)).reshape(X_nas_tr.shape)
            Xval_s = scaler.transform(X_nas_val.reshape(-1, n_base_features)).reshape(X_nas_val.shape)

            model = _build_bilstm(n_base_features, hidden, layers, drop).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCELoss()

            Xtt = torch.tensor(Xtr_s, dtype=torch.float32).to(self.device)
            ytt = torch.tensor(y_nas_tr, dtype=torch.float32).to(self.device)
            ds  = torch.utils.data.TensorDataset(Xtt, ytt)
            loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

            model.train()
            for _ in range(5):  # rápido: 5 épocas
                for Xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Evaluar en validation
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(Xval_s, dtype=torch.float32).to(self.device)
                probs = model(Xv).cpu().numpy()
            wr = float(((probs >= 0.5).astype(int) == y_nas_val.astype(int)).mean())
            return wr

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        best["best_val_wr"] = round(study.best_value, 4)
        logger.info(f"LSTM NAS completado — best WR val: {best['best_val_wr']:.1%} | params: {best}")

        # Guardar en JSON
        import json
        with open(best_params_file, "w") as f:
            json.dump(best, f, indent=2)
        logger.success(f"LSTM NAS: best params guardados en {best_params_file}")
        return best

    # ── Walk-Forward Anchored ─────────────────────────────────────────────────
    def walk_forward_analysis(
        self, X: np.ndarray, y: np.ndarray, idx_labels: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Walk-Forward Anchored:
          - Ventana mínima IS: WF_TRAIN_MONTHS meses
          - Paso OOS: WF_STEP_WEEKS semanas
          - Cada fold: entrena en [0, t], predice en [t, t+paso]
        """
        import torch
        n_features  = X.shape[2]
        results     = []
        n_total     = len(X)

        # Ventana mínima en barras (aprox 720H/mes)
        min_train   = int(WF_TRAIN_MONTHS * 720)
        step_bars   = int(WF_STEP_WEEKS * 7 * 24)

        fold_n = 0
        test_start = min_train

        while test_start < n_total:
            test_end = min(test_start + step_bars, n_total)

            X_tr, y_tr = X[:test_start], y[:test_start]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            if len(X_tr) < 1000 or len(X_te) < 10:
                test_start = test_end
                continue

            fold_n += 1
            logger.info(
                f"  WF Fold {fold_n}: IS={len(X_tr)} barras, OOS={len(X_te)} barras "
                f"| {str(idx_labels[test_start])[:10]} → {str(idx_labels[min(test_end-1, len(idx_labels)-1)])[:10]}"
            )

            # Entrenar
            model = self._train_fold(X_tr, y_tr, n_features)
            self.model = model

            # Predecir en OOS
            model.eval()
            with torch.no_grad():
                X_te_s = X_te.reshape(-1, n_features)
                X_te_s = self.scaler.transform(X_te_s)
                X_te_s = X_te_s.reshape(X_te.shape)
                Xtt = torch.tensor(X_te_s, dtype=torch.float32).to(self.device)
                probs = model(Xtt).cpu().numpy()

            # Calcular métricas OOS del fold
            preds_bin = (probs >= 0.5).astype(int)
            wr        = float((preds_bin == y_te.astype(int)).mean())
            avg_prob  = float(probs.mean())

            for i in range(len(probs)):
                if test_start + i < len(idx_labels):
                    results.append({
                        "timestamp":  idx_labels[test_start + i],
                        "fold":       fold_n,
                        "prob_bull":  float(probs[i]),
                        "pred_dir":   int(preds_bin[i]),
                        "actual":     int(y_te[i]),
                        "correct":    int(preds_bin[i] == int(y_te[i])),
                    })

            logger.info(f"    Fold {fold_n}: OOS WR={wr*100:.1f}% | avg_prob={avg_prob:.3f}")
            test_start = test_end

        return pd.DataFrame(results)

    # ── Feature Importance (Ablation) ─────────────────────────────────────────
    def feature_importance_ablation(
        self, X: np.ndarray, y: np.ndarray, feats: list[str]
    ) -> pd.Series:
        """
        Ablation study: elimina cada feature de a una, mide la caída del accuracy.
        La que más daña al eliminarla = más importante.
        """
        import torch

        if self.model is None:
            return pd.Series(dtype=float)

        logger.info("LSTM: calculando importancia por ablación...")
        n_features = X.shape[2]

        # Baseline accuracy en últimas 2000 barras
        test_X = X[-2000:]
        test_y = y[-2000:]

        self.model.eval()
        with torch.no_grad():
            Xs = test_X.reshape(-1, n_features)
            Xs = self.scaler.transform(Xs)
            Xs = Xs.reshape(test_X.shape)
            preds = self.model(
                torch.tensor(Xs, dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        baseline_acc = float(((preds >= 0.5).astype(int) == test_y.astype(int)).mean())

        importances = {}
        for i, feat in enumerate(feats):
            X_ablated = test_X.copy()
            X_ablated[:, :, i] = 0.0  # zeroing feature i

            with torch.no_grad():
                Xs = X_ablated.reshape(-1, n_features)
                Xs = self.scaler.transform(Xs)
                Xs = Xs.reshape(X_ablated.shape)
                preds_abl = self.model(
                    torch.tensor(Xs, dtype=torch.float32).to(self.device)
                ).cpu().numpy()
            abl_acc = float(((preds_abl >= 0.5).astype(int) == test_y.astype(int)).mean())
            importances[feat] = round(baseline_acc - abl_acc, 4)

        result = pd.Series(importances).sort_values(ascending=False)
        logger.info(f"LSTM Ablation top5: {result.head(5).to_dict()}")
        return result

    # ── Visual Analytics ──────────────────────────────────────────────────────
    def _plot_wf_results(self, preds_df: pd.DataFrame) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

            # Panel 1: Probabilidad alcista a lo largo del tiempo
            ax1.plot(preds_df["timestamp"], preds_df["prob_bull"],
                     color="#3498db", linewidth=0.7, alpha=0.8)
            ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax1.fill_between(preds_df["timestamp"], 0.5, preds_df["prob_bull"],
                             where=preds_df["prob_bull"] >= 0.5, alpha=0.3, color="#2ecc71")
            ax1.fill_between(preds_df["timestamp"], preds_df["prob_bull"], 0.5,
                             where=preds_df["prob_bull"] < 0.5, alpha=0.3, color="#e74c3c")
            ax1.set_ylabel("P(BTC +24H > 0)")
            ax1.set_title("BiLSTM Walk-Forward — Probabilidad Alcista OOS", fontweight="bold")
            ax1.set_ylim(0, 1)

            # Panel 2: Accuracy rolling 7 días
            preds_df["correct_f"] = preds_df["correct"].astype(float)
            rolling_acc = preds_df.set_index("timestamp")["correct_f"].rolling("7D").mean()
            ax2.plot(rolling_acc.index, rolling_acc.values, color="#9b59b6", linewidth=1.2)
            ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax2.axhline(rolling_acc.mean(), color="#e67e22", linestyle="-.", linewidth=1.5,
                        label=f"Media={rolling_acc.mean():.3f}")
            ax2.set_ylabel("Accuracy 7D rolling")
            ax2.set_title("Win Rate Rolling 7 Días")
            ax2.legend(loc="lower right")
            ax2.set_ylim(0, 1)

            # Panel 3: Accuracy por fold
            fold_stats = preds_df.groupby("fold")["correct"].mean()
            colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in fold_stats.values]
            ax3.bar(fold_stats.index, fold_stats.values, color=colors, alpha=0.85)
            ax3.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax3.set_xlabel("Walk-Forward Fold")
            ax3.set_ylabel("OOS Accuracy")
            ax3.set_title("Accuracy por Fold WF")

            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "engine_lstm_wf.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.success("engine_lstm_wf.png generado")
        except Exception as e:
            logger.warning(f"LSTM WF plot: {e}")

    def _plot_feature_importance(self, importance: pd.Series) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 7))
            top = importance.head(15)
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top.values[::-1]]
            ax.barh(range(len(top)), top.values[::-1], color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index[::-1], fontsize=9)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel("Accuracy drop (ablation)")
            ax.set_title("BiLSTM — Feature Importance por Ablación",
                         fontweight="bold")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "engine_lstm_features.png", dpi=120, bbox_inches="tight")
            plt.close()
            logger.success("engine_lstm_features.png generado")
        except Exception as e:
            logger.warning(f"LSTM features plot: {e}")

    # ── Guardar checkpoint ────────────────────────────────────────────────────
    def _save_checkpoint(self) -> None:
        try:
            import torch
            if self.model is not None:
                torch.save({
                    "model_state": self.model.state_dict(),
                    "features":    self.features,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers":  NUM_LAYERS,
                    "dropout":     DROPOUT,
                    "seq_len":     SEQUENCE_LEN,
                }, MODELS_DIR / "lstm_model.pt")
                logger.success(f"LSTM: checkpoint guardado en {MODELS_DIR / 'lstm_model.pt'}")
        except Exception as e:
            logger.warning(f"LSTM checkpoint error: {e}")

    # ── Reporte Markdown ──────────────────────────────────────────────────────
    def _save_report(
        self, preds_df: pd.DataFrame, importance: pd.Series,
        overall_wr: float, n_folds: int
    ) -> None:
        report = [
            "# Deep LSTM Engine — BiLSTM Walk-Forward Report",
            f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC",
            f"**Arquitectura:** BiLSTM {NUM_LAYERS}L×{HIDDEN_SIZE}H | Seq={SEQUENCE_LEN}H | "
            f"Horizon={HORIZON_H}H | Dropout={DROPOUT}",
            f"**WF:** IS mínimo={WF_TRAIN_MONTHS}M | Paso={WF_STEP_WEEKS}W | Folds={n_folds}",
            "",
            "## Métricas Globales OOS",
            "",
            f"| Métrica | Valor |",
            "|---|---|",
            f"| **OOS Win Rate** | **{overall_wr*100:.1f}%** |",
            f"| Total predicciones | {len(preds_df)} |",
            f"| Media P(Bull) | {preds_df['prob_bull'].mean():.3f} |",
            f"| P(Bull) actual | {preds_df['prob_bull'].iloc[-1]:.3f} |",
            f"| Dirección actual | {'🟢 BULLISH' if preds_df['prob_bull'].iloc[-1] >= 0.5 else '🔴 BEARISH'} |",
            "",
        ]

        # Stats por fold
        fold_stats = preds_df.groupby("fold")["correct"].mean()
        n_folds_real = len(fold_stats)
        report += [
            "## Accuracy por Fold WF",
            "",
            "| Fold | OOS Accuracy | Estado |",
            "|---|---|---|",
        ]
        for fold, acc in fold_stats.items():
            emoji = "✅" if acc >= 0.5 else "❌"
            report.append(f"| {fold} | {acc*100:.1f}% | {emoji} |")

        # Feature importance
        if not importance.empty:
            report += [
                "",
                "## Feature Importance (Ablation Study)",
                "",
                "| Feature | Accuracy Drop |",
                "|---|---|",
            ]
            for feat, val in importance.head(10).items():
                impact = "⬆️" if val > 0.01 else "➡️" if val > 0 else "⬇️"
                report.append(f"| `{feat}` | {val:+.4f} {impact} |")

        # Features utilizadas
        report += [
            "",
            "## Features BiLSTM",
            "",
            f"**{len(self.features)} features** (hasta {MAX_FEATURES} por RFE cap):",
            "",
            ", ".join([f"`{f}`" for f in self.features]),
            "",
            "---",
            "📊 Ver: `engine_lstm_wf.png` · `engine_lstm_features.png`",
        ]

        (REPORTS_DIR / "lstm_report.md").write_text("\n".join(report), encoding="utf-8")
        logger.success(f"LSTM: reporte guardado")

    # ── Pipeline principal ────────────────────────────────────────────────────
    def run(self) -> dict:
        logger.info("=" * 60)
        logger.info("Deep LSTM Engine — INICIO (BiLSTM Walk-Forward)")
        logger.info("=" * 60)

        # ── Verificar PyTorch ─────────────────────────────────────────────────
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"PyTorch {torch.__version__} | device={self.device}")
        except ImportError:
            logger.error("PyTorch no instalado. Ejecuta: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            return {}

        # ── Cargar datos ────────────────────────────────────────────────────
        df = self.load_data()

        # ── Seleccionar features ───────────────────────────────────────────────
        self.features = self.select_features(df)
        if not self.features:
            logger.error("LSTM: no hay features disponibles")
            return {}

        # ── Construir secuencias ───────────────────────────────────────────────
        logger.info(f"LSTM: construyendo secuencias (len={SEQUENCE_LEN}, n_feats={len(self.features)})...")
        X, y, idx_labels = self._build_sequences(df, self.features)
        logger.info(f"LSTM: X.shape={X.shape}, y.shape={y.shape}")

        # ── Optuna NAS — busca mejores hiperparámetros antes del WF ─────────────
        global HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LEARNING_RATE
        nas_params_file = MODELS_DIR / "lstm_best_params.json"
        if nas_params_file.exists():
            # Reutilizar params de ejecución anterior si existen
            import json
            with open(nas_params_file) as f:
                nas = json.load(f)
            logger.info(f"LSTM NAS: reutilizando params previos: {nas}")
        else:
            nas = self._optuna_nas(X, y, n_trials=20)

        if nas:
            HIDDEN_SIZE   = int(nas.get("hidden_size",   HIDDEN_SIZE))
            NUM_LAYERS    = int(nas.get("num_layers",    NUM_LAYERS))
            DROPOUT       = float(nas.get("dropout",     DROPOUT))
            LEARNING_RATE = float(nas.get("learning_rate", LEARNING_RATE))
            logger.info(f"LSTM: hiperparámetros Optuna aplicados — "
                        f"H={HIDDEN_SIZE} L={NUM_LAYERS} D={DROPOUT:.2f} LR={LEARNING_RATE:.5f}")

        # ── Walk-Forward ────────────────────────────────────────────────────
        logger.info("LSTM: iniciando Walk-Forward Anchored...")
        preds_df = self.walk_forward_analysis(X, y, idx_labels)

        if preds_df.empty:
            logger.warning("LSTM: no se generaron predicciones WF")
            return {}

        # ── Métricas globales ─────────────────────────────────────────────────
        overall_wr  = float(preds_df["correct"].mean())
        n_folds     = preds_df["fold"].nunique()
        last_prob   = float(preds_df["prob_bull"].iloc[-1])
        direction   = "BULLISH" if last_prob >= 0.5 else "BEARISH"

        logger.info(f"\n{'='*50}")
        logger.info(f"LSTM OOS Win Rate global: {overall_wr*100:.1f}%")
        logger.info(f"Folds completados: {n_folds}")
        logger.info(f"P(Bull) actual: {last_prob:.3f} → {direction}")
        logger.info(f"{'='*50}\n")

        # ── Feature Importance ────────────────────────────────────────────────
        logger.info("LSTM: ablation study...")
        importance = self.feature_importance_ablation(X, y, self.features)

        # ── Guardar resultados ────────────────────────────────────────────────
        preds_df.to_csv(REPORTS_DIR / "lstm_predictions.csv", index=False)
        self._save_checkpoint()
        self._plot_wf_results(preds_df)
        self._plot_feature_importance(importance)
        self._save_report(preds_df, importance, overall_wr, n_folds)

        logger.info("Deep LSTM Engine — COMPLETADO")
        return {
            "overall_wr":   round(overall_wr, 4),
            "n_folds":      n_folds,
            "last_prob":    round(last_prob, 4),
            "direction":    direction,
            "n_features":   len(self.features),
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DeepLSTMEngine().run()

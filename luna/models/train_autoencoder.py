import os
import sys
import json
import argparse
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import Tuple, List

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from config.settings import cfg
from loguru import logger

# Disable PyTorch warnings about number of workers on Windows
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DenoisingAutoEncoder(nn.Module):
    """
    AutoEncoder paramétrico para comprimir características crudas y ruidosas de OOS
    en un "Latent Space" ortogonal y estable antes de alimentar a las capas superiores.
    Implementación Denoising (añade Dropout a las entradas para forzar resiliencia).
    """
    def __init__(self, input_dim: int, latent_dim: int = 32, dropout_rate: float = 0.2):
        super(DenoisingAutoEncoder, self).__init__()
        
        # Cuello de Botella Dinámico Rígido/Predecible
        dim_h1 = max(64, input_dim // 2)
        dim_h2 = max(32, input_dim // 4)
        
        # Asegurarnos de que el Latent siempre sea <= h2
        latent = min(latent_dim, dim_h2)
        if latent < 8: latent = 16  # Fallback seguro
        self.latent_dim = latent
        
        # Encoder (Compresión)
        self.encoder = nn.Sequential(
            nn.Dropout(dropout_rate),  # Denoising: corrompe aleatoriamente la entrada
            nn.Linear(input_dim, dim_h1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(dim_h1),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(dim_h1, dim_h2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(dim_h2),
            nn.Linear(dim_h2, latent),
            nn.Tanh()  # Bloquea el espacio latente entre [-1, 1], domando valores atípicos
        )
        
        # Decoder (Reconstrucción)
        self.decoder = nn.Sequential(
            nn.Linear(latent, dim_h2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(dim_h2),
            nn.Linear(dim_h2, dim_h1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(dim_h1),
            nn.Linear(dim_h1, input_dim)
            # Salida lineal ya que reconstruimos Standard Scaled Normal data
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el espacio latente vectorizado"""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

def train_autoencoder(data_path: str, output_dir: str, epochs: int = 50, batch_size: int = 128, lr: float = 1e-3, latent_dim: int = 32):
    """
    Entrena el AutoEncoder contra el dataset origen de variables crudas.
    """
    # [FIX-AE-DETERMINISM-01 2026-06-26] Seedear ANTES de crear modelo y DataLoader. El OOD
    # AutoEncoder (torch) estaba sin seed -> filtrado OOD no reproducible. Ver hallazgos 6.6.
    from luna.utils.determinism import seed_everything as _seed_ood, seeded_generator as _gen_ood
    _ood_seed = _seed_ood()
    p_data = Path(data_path)
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)
    
    logger.info("  [1] Cargando raw features del parquet: {}", p_data.name)
    df = pd.read_parquet(p_data)
    
    # Excluir columnas objetivo, timestamps y categoricas HMM
    drop_cols = [c for c in df.columns if c.startswith("target") or c.startswith("meta") or c.startswith("HMM") or "time" in c.lower() or c in ["open", "high", "low", "close", "volume"]]
    features = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    # Sort for deterministic sequence
    features = sorted(features)
    
    logger.info("  [2] Total Features Iniciales para AE: {}", len(features))

    # --- FIX: Anchor the feature_cols list for stable WARM-START across WFB windows ---
    # Para evitar que W2, W3, etc., tengan un número distinto de features y rompan el
    # load_state_dict del ancla (causando un fallo silencioso de Warm-Start).
    _window_id_anchor = os.environ.get("LUNA_WINDOW_ID", "")
    _anchor_file = p_out / "ae_valid_features.json"
    
    if _window_id_anchor == "W1":
        # W1 dicta la geometría inicial (filtra features constantes en su propia ventana)
        constant_features = [f for f in features if df[f].nunique() <= 1]
        if constant_features:
            features = [f for f in features if f not in constant_features]
            print(f"[H-AE-VAL-01-FIX] W1 Constant filter: {len(features) + len(constant_features)} -> {len(features)} features.")
        try:
            with open(_anchor_file, "w", encoding="utf-8") as _af:
                json.dump(features, _af)
            print(f"📌 [AE-ANCHOR] W1: Guardando {len(features)} variables ancla en ae_valid_features.json")
        except Exception as _e_anchor:
            logger.warning(f"[AE-ANCHOR] W1: Error al guardar ae_valid_features.json: {_e_anchor}")
            
    elif _window_id_anchor.startswith("W"):
        # W2+ fuerza el uso exacto de las variables ancladas en W1
        _root_dir_anchor = Path(data_path).resolve().parents[2]
        _cached_w1_anchor = _root_dir_anchor / "data" / "wfb_cache" / "W1" / "models" / "ae_valid_features.json"
        _target_anchor = _cached_w1_anchor if _cached_w1_anchor.exists() else _anchor_file
        
        if _target_anchor.exists():
            try:
                with open(_target_anchor, "r", encoding="utf-8") as _af:
                    _anchored_features = json.load(_af)
                
                # Rellenar con 0.0 las variables ancladas que ya no existen o fueron descartadas por SFI en esta ventana
                _missing_anchor = [c for c in _anchored_features if c not in df.columns]
                for _mc in _missing_anchor:
                    df[_mc] = 0.0
                
                features = _anchored_features
                print(f"📌 [AE-ANCHOR] {_window_id_anchor}: Forzando {len(features)} variables ancla de W1 para Warm-Start perfecto.")
            except Exception as _e_anchor:
                logger.warning(f"[AE-ANCHOR] {_window_id_anchor}: Error leyendo ae_valid_features.json: {_e_anchor}")
        else:
            logger.warning(f"[AE-ANCHOR] No se encontró el ancla en {_target_anchor}, usando variables locales (Riesgo de size mismatch)")

    logger.info("  [2b] Features Finales para AE tras protocolo de anclaje: {}", len(features))

    X_raw = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    
    # Scaler
    logger.info("  [3] Ejecutando FIT del StandardScaler sobre data In-Sample")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Partición Pseudo/Validation para Early Stopping 
    # (El 20% final cronológico de este batch)
    split_idx = int(len(X_scaled) * 0.8)
    X_train = torch.tensor(X_scaled[:split_idx], dtype=torch.float32)
    X_val   = torch.tensor(X_scaled[split_idx:], dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True, generator=_gen_ood(_ood_seed))  # [FIX-AE-DETERMINISM-01] shuffle seedeado
    val_loader   = DataLoader(TensorDataset(X_val, X_val), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cpu")
    logger.info("  [4] Entrenando red vía Device: {}", device)
    
    # [MEJORA-MATH-C] Anchored AE Drift Loss
    from config.settings import cfg as _cfg_ae
    ae_anchored = bool(_cfg_ae.autoencoder.ae_anchored_kl_loss)
    kl_lambda = float(_cfg_ae.autoencoder.ae_kl_lambda)
    kl_alarm = float(_cfg_ae.autoencoder.ae_kl_drift_alarm_threshold)

    anchored_model = None
    if ae_anchored:
        _prev_model_path = p_out / "autoencoder_state.pt"
        if _prev_model_path.exists():
            try:
                anchored_model = DenoisingAutoEncoder(input_dim=len(features), latent_dim=latent_dim).to(device)
                anchored_model.load_state_dict(torch.load(_prev_model_path, map_location=device, weights_only=False))
                anchored_model.eval()
                logger.info("  [MEJORA-MATH-C] AE ancla cargado para Drift Loss.")
            except Exception as e:
                logger.warning("  [MEJORA-MATH-C] Error al cargar AE ancla: {}", e)
                anchored_model = None
                
    
    model = DenoisingAutoEncoder(input_dim=len(features), latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Regularización L2
    
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    
    best_model_state = None
    
    # [BUG-1-FIX 2026-06-18] Confirmación de forward pass única activa
    _anchored_status = "ACTIVO (Anchored Drift Loss ON)" if anchored_model is not None else "NO (primer entrenamiento o ancla no encontrada)"
    print(f"[BUG-1-FIX][AE] Forward pass única por batch. Modo ancla: {_anchored_status}. "
          f"Latent dim={latent_dim}, features={len(features)}, epochs={epochs}")
    logger.info("[BUG-1-FIX][AE] Training loop configurado con forward pass única. Anchored={}", anchored_model is not None)
    

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # [BUG-1-FIX 2026-06-18] Forward ÚNICA: se extrae curr_latent de la misma
            # pasada que produce reconstructed. Llamar model(batch_data) dos veces en
            # modo train() sesga BatchNorm (actualiza stats 2x por batch) y aplica
            # Dropout con máscaras distintas, produciendo gradientes incorrectos.
            curr_latent, reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            
            # [MEJORA-MATH-C] Anchored Drift Loss (usa curr_latent ya computado)
            l_drift = 0.0
            if anchored_model is not None:
                with torch.no_grad():
                    prev_latent, _ = anchored_model(batch_data)
                # Pseudo-KL Loss (MSE over Tanh bounded latent space)
                l_drift_tensor = criterion(curr_latent, prev_latent)
                l_drift = l_drift_tensor.item()
                loss = loss + kl_lambda * l_drift_tensor
                
            # Backward
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(device)
                _, reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_data)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            if anchored_model is not None:
                logger.info("  [5] Epoch {:03d}/{:03d} | Train Loss (MSE+Drift): {:.5f} | Val Loss: {:.5f} | Drift: {:.5f}", epoch+1, epochs, train_loss, val_loss, l_drift)
                if l_drift > kl_alarm:
                    logger.warning("  [MEJORA-MATH-C] Latent Drift ALARM! Drift {:.4f} > umbral {:.4f}", l_drift, kl_alarm)
            else:
                logger.info("  [5] Epoch {:03d}/{:03d} | Train Loss (MSE): {:.5f} | Val Loss: {:.5f}", epoch+1, epochs, train_loss, val_loss)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  [!] Early stopping alcanzado en Epoch {} (Paciencia: {})", epoch+1, patience)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Persistencia ACID
    logger.info("  [6] Guardando persistencia (Model, Scaler, Signatures) en {}", p_out)
    
    torch.save(model.state_dict(), p_out / "autoencoder_state.pt")
    joblib.dump(scaler, p_out / "autoencoder_scaler.joblib")
    
    with open(p_out / "autoencoder_config.json", "w") as f:
        json.dump({
            "features": features,
            "input_dim": len(features),
            "latent_dim": model.latent_dim,
            "train_loss": float(train_loss),
            "val_loss": float(best_val_loss)
        }, f, indent=4)
        
    logger.info("  [7] Inyectando features LATENT_AE en {}", p_data.name)
    try:
        model.eval()
        with torch.no_grad():
            latent_tensor = model.encode(torch.tensor(X_scaled, dtype=torch.float32)).cpu().numpy()
        for idx in range(latent_tensor.shape[1]):
            df[f"LATENT_AE_{idx}"] = latent_tensor[:, idx]
        df.to_parquet(p_data)
        logger.info("  [SUCCESS] {} features LATENT_AE inyectadas en parquet.", latent_tensor.shape[1])
    except Exception as e:
        logger.error("  [ERROR] Fallo inyectando LATENT_AE: {}", e)

    logger.info("  [SUCCESS] AutoEncoder V2 ensamblado y sellado paramétricamente. Latent Features Extraídas: {}", model.latent_dim)

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Entrenamiento del Denoising AutoEncoder para comprimir features OOS.")
    parser.add_argument("--data", type=str, default="data/features/features_train.parquet", help="Ruta al dataset parquet crudo")
    parser.add_argument("--out", type=str, default="data/models/", help="Directorio destino para volcado")
    parser.add_argument("--latent", type=int, default=int(cfg.autoencoder.ae_bottleneck_dim), help="Tamaño del Latent Space (features resultantes tras cuello de botella)")
    parser.add_argument("--epochs", type=int, default=80, help="Neuronal pass epochs")
    
    args = parser.parse_args()
    
    train_autoencoder(args.data, args.out, epochs=args.epochs, latent_dim=args.latent)

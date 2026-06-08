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
    
    logger.info("  [2] Total Features Seleccionadas para AE: {}", len(features))

    # [H-AE-VAL-01-FIX 2026-05-30] Filtrar features OOD-bloqueadas antes del AE fit.
    # PROBLEMA: El AE entrena sobre TODAS las features (incluidas las 84 bloqueadas por OOD Guard
    # que son CONSTANTES en validación). El 20% final cronológico usado como val interna
    # corresponde al período de validación OOS donde esas features son constantes.
    # CONSECUENCIA: MSE_val = (c - ŷ)² para features OOD constantes nunca converge a 0
    # → Val Loss explota (ratio Train/Val = 8.6x en epoch 11) → early stopping artificial.
    # El AE se detiene en epoch 12 con un modelo subentrenado en lugar de los 30-40 epochs óptimos.
    # SOLUCIÓN: Cargar selected_features.json del SFI y filtrar features con 
    _sfi_json = p_out.parent / "features" / "selected_features.json"
    _ood_filtered = 0
    if _sfi_json.exists():
        try:
            import json as _json_ae
            with open(_sfi_json, "r", encoding="utf-8") as _f:
                _sfi_data = _json_ae.load(_f)
            # selected_features.json tiene claves: 'selected_features' (list) y 'pass_through_features' (list)
            _approved = set()
            if isinstance(_sfi_data, list):
                _approved = set(_sfi_data)
            elif isinstance(_sfi_data, dict):
                # Formato estándar: 'selected_features' + 'pass_through_features'
                _sf = _sfi_data.get("selected_features", [])
                _pt = _sfi_data.get("pass_through_features", [])
                if isinstance(_sf, list):
                    _approved.update(_sf)
                if isinstance(_pt, list):
                    _approved.update(_pt)
                # Fallback: cualquier valor de tipo lista en el dict
                if not _approved:
                    for _v in _sfi_data.values():
                        if isinstance(_v, list) and _v and isinstance(_v[0], str):
                            _approved.update(_v)

            if _approved:
                _features_before = len(features)
                features = [f for f in features if f in _approved]
                _ood_filtered = _features_before - len(features)
                print(  # RULE[fixbugsprints.md]
                    f"[H-AE-VAL-01-FIX] OOD filter aplicado al AE: {_features_before} → {len(features)} features "
                    f"({_ood_filtered} features OOD-bloqueadas excluidas del AE fit). "
                    f"Previene Val Loss divergente (ratio 8.6x) y early stopping artificial."
                )
                logger.info(
                    "[H-AE-VAL-01-FIX] AE features filtradas por SFI OOD: {} → {} ({} excluidas).",
                    _features_before, len(features), _ood_filtered
                )
                print(f"[BUG-FIX-LOG 2026-06-05] [H-AE-VAL-01-FIX] AE features filtradas por SFI OOD: {_features_before} → {len(features)} ({_ood_filtered} excluidas)")
            else:
                print("[H-AE-VAL-01-FIX] selected_features.json sin features aprobadas — usando todas (sin filtrado OOD).")
        except Exception as _e_ood:
            print(f"[H-AE-VAL-01-FIX] ERROR leyendo selected_features.json: {_e_ood} — usando todas las features.")
            logger.warning("[H-AE-VAL-01-FIX] No se pudo aplicar OOD filter al AE: {}", _e_ood)
            print(f"[BUG-FIX-LOG 2026-06-05] [H-AE-VAL-01-FIX] No se pudo aplicar OOD filter al AE: {_e_ood}")
    else:
        print(f"[H-AE-VAL-01-FIX] selected_features.json no encontrado en {_sfi_json} — usando todas las features (retrocompatible).")
        logger.debug("[H-AE-VAL-01-FIX] selected_features.json ausente. AE sin filtrado OOD.")

    logger.info("  [2b] Features para AE tras filtro OOD: {} ({} excluidas)", len(features), _ood_filtered)

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
    
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, X_val), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cpu")
    logger.info("  [4] Entrenando red vía Device: {}", device)
    
    model = DenoisingAutoEncoder(input_dim=len(features), latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Regularización L2
    
    best_val_loss = float('inf')
    patience = 10
    no_improve = 0
    
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Forward
            _, reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            
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
    parser.add_argument("--latent", type=int, default=32, help="Tamaño del Latent Space (features resultantes tras cuello de botella)")
    parser.add_argument("--epochs", type=int, default=80, help="Neuronal pass epochs")
    
    args = parser.parse_args()
    
    train_autoencoder(args.data, args.out, epochs=args.epochs, latent_dim=args.latent)

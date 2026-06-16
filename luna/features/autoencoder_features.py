"""
autoencoder_features.py
=======================
Luna V2 — Dimensionality Reduction & Meta-Feature Extraction
Comprime el dataset continuo mediante Deep AutoEncoders para evitar 
el Curse of Dimensionality antes de inyectar los datos en el árbol XGBoost.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

_LIVE_AE_CACHE = None

class DeepFeatureAutoEncoder(nn.Module):
    """
    Arquitectura de cuello de botella (Bottleneck) simétrica.
    Proyecta N dimensiones hacia bottleneck_size dimensiones, reteniendo geometrías no lineales.
    """
    def __init__(self, input_dim: int, bottleneck_size: int = 32):
        super().__init__()
        
        # [LIVE-AE-FIX] Soporte dinámico para compatibilidad de checkpoints de V1 y V2.
        # Los checkpoints preentrenados usan proporciones (input_dim // 2, input_dim // 4) para las capas ocultas.
        # Para input_dim=492, esto da h1=246 y h2=123, coincidiendo exactamente con los pesos guardados.
        h1 = input_dim // 2
        h2 = input_dim // 4
        
        logger.info(f"✨ [LIVE-AE-FIX] AutoEncoder dinámico inicializado. Arquitectura: {input_dim} -> {h1} -> {h2} -> {bottleneck_size}")
        print(f"✨ [LIVE-AE-FIX] AutoEncoder dinámico inicializado. Arquitectura: {input_dim} -> {h1} -> {h2} -> {bottleneck_size}")
        
        # Reduciendo [Dropout -> Linear -> LeakyReLU -> BatchNorm] para coincidir exactamente con el checkpoint guardado
        self.encoder = nn.Sequential(
            nn.Dropout(0.2),                     # index 0 (sin parámetros)
            nn.Linear(input_dim, h1),            # index 1 (pesos de la capa lineal)
            nn.LeakyReLU(0.2),                   # index 2 (sin parámetros)
            nn.BatchNorm1d(h1),                  # index 3 (pesos de batchnorm)
            
            nn.Dropout(0.1),                     # index 4 (sin parámetros)
            nn.Linear(h1, h2),                   # index 5 (pesos de la capa lineal)
            nn.LeakyReLU(0.2),                   # index 6 (sin parámetros)
            nn.BatchNorm1d(h2),                  # index 7 (pesos de batchnorm)
            
            nn.Linear(h2, bottleneck_size)       # index 8 (pesos de cuello de botella)
        )
        
        # Expandiendo [Linear -> LeakyReLU -> BatchNorm] para coincidir exactamente con el checkpoint guardado
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, h2),      # index 0 (pesos de la capa lineal)
            nn.LeakyReLU(0.2),                   # index 1 (sin parámetros)
            nn.BatchNorm1d(h2),                  # index 2 (pesos de batchnorm)
            
            nn.Linear(h2, h1),                   # index 3 (pesos de la capa lineal)
            nn.LeakyReLU(0.2),                   # index 4 (sin parámetros)
            nn.BatchNorm1d(h1),                  # index 5 (pesos de batchnorm)
            
            nn.Linear(h1, input_dim)             # index 6 (pesos de la capa de reconstrucción)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

def apply_autoencoder(df: pd.DataFrame, train_end_date: str, bottleneck_size: int = 32, epochs: int = 30, live_mode: bool = False) -> pd.DataFrame:
    """
    Aplica una reducción de dimensionalidad AutoEncoder a las variables continuas.
    
    Regla Crítica (SOP R1): 
    El AutoEncoder se entrena *sólo* con datos anteriores a train_end_date para evitar Data Leakage.
    
    Args:
        df: DataFrame crudo proveniente de feature_pipeline.py
        train_end_date: Fecha de corte del conjunto Train (ej: "2023-12-31")
        bottleneck_size: El número de representaciones densas a recuperar.
        epochs: iteraciones del pase no-supervisado.
        live_mode: si es True, usa modo producción bypass en caliente cargando pesos preentrenados.
    """
    import os as _os_ae
    from pathlib import Path
    import json
    import joblib

    is_live = live_mode or (_os_ae.environ.get('LUNA_LIVE_PRODUCTION') == '1')
    if is_live:
        logger.info("✨ [LIVE-AE-FIX] Ejecutando AutoEncoder en modo de producción (Bypass de entrenamiento)")
        print("✨ [LIVE-AE-FIX] Modo LIVE/Bypass de AutoEncoder activado.")
        
        global _LIVE_AE_CACHE
        root_dir = Path(__file__).resolve().parents[2]
        models_dir = root_dir / "data" / "models"
        
        try:
            if '_LIVE_AE_CACHE' not in globals() or _LIVE_AE_CACHE is None:
                config_path = models_dir / "autoencoder_config.json"
                scaler_path = models_dir / "autoencoder_scaler.joblib"
                state_path = models_dir / "autoencoder_state.pt"
                
                logger.info(f"[LIVE-AE-FIX] Cargando checkpoints desde {models_dir}")
                if not config_path.exists():
                    raise FileNotFoundError(f"[LIVE-AE-FIX] Config file missing: {config_path}")
                if not scaler_path.exists():
                    raise FileNotFoundError(f"[LIVE-AE-FIX] Scaler file missing: {scaler_path}")
                if not state_path.exists():
                    raise FileNotFoundError(f"[LIVE-AE-FIX] Model state file missing: {state_path}")
                    
                with open(config_path, "r", encoding="utf-8") as f:
                    ae_config = json.load(f)
                feature_cols_loaded = ae_config.get("features", [])
                
                scaler_loaded = joblib.load(scaler_path)
                
                device_loaded = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # Test CUDA kernel freshness
                if device_loaded.type == 'cuda':
                    try:
                        _test = torch.zeros(1).to(device_loaded) + 1
                    except Exception as e:
                        logger.warning(f"[LIVE-AE-FIX] CUDA test failed: {e}. Fallback to CPU.")
                        device_loaded = torch.device('cpu')
                        
                autoencoder_loaded = DeepFeatureAutoEncoder(input_dim=len(feature_cols_loaded), bottleneck_size=bottleneck_size).to(device_loaded)
                autoencoder_loaded.load_state_dict(torch.load(state_path, map_location=device_loaded, weights_only=True))
                autoencoder_loaded.eval()
                
                _LIVE_AE_CACHE = {
                    "feature_cols": feature_cols_loaded,
                    "scaler": scaler_loaded,
                    "device": device_loaded,
                    "autoencoder": autoencoder_loaded
                }
                logger.success("✨ [LIVE-AE-FIX] Carga de checkpoints de producción exitosa en cache global.")
                print("✨ [LIVE-AE-FIX] Carga de checkpoints de producción exitosa.")
            else:
                logger.debug("[LIVE-AE-FIX] Usando cache en memoria para AutoEncoder.")
                
            cache = _LIVE_AE_CACHE
            feature_cols = cache["feature_cols"]
            scaler = cache["scaler"]
            device = cache["device"]
            autoencoder = cache["autoencoder"]

            # [FIX-ALIAS-TRAIN-LIVE] Bridge de nombres train→live para el Autoencoder OOD Guard.
            # BUG: El autoencoder fue entrenado con columnas que el live pipeline renombró en una
            # actualización posterior. Sin este bridge, esas 6 features llegan como 0.0, degradando
            # los scores de OOD y permitiendo/bloqueando trades incorrectamente.
            # IMPORTANTE: Solo aplica alias si la columna train NO existe y la live SÍ existe.
            # No sobreescribe datos reales. No hay fallback silencioso — cada alias se loguea.
            _AE_ALIAS_MAP = {
                'FundingRate_EMA3':  'funding_ema_3',
                'FundingRate_Pct90d': 'dv_funding_pct_90d',
                'OI_Open_USD':       'Coinglass_oi_open',
                'OI_High_USD':       'Coinglass_oi_high',
                'OI_Low_USD':        'Coinglass_oi_low',
                'ETF_Flow_Proxy':    'etf_flow_proxy',
            }
            _aliased = []
            _alias_missing = []
            for _train_col, _live_col in _AE_ALIAS_MAP.items():
                if _train_col not in df.columns:
                    if _live_col in df.columns:
                        df[_train_col] = df[_live_col]
                        _aliased.append(f"{_live_col}→{_train_col}")
                    else:
                        _alias_missing.append(f"{_train_col} (live equiv '{_live_col}' tampoco existe)")
            if _aliased:
                print(f"[FIX-ALIAS-TRAIN-LIVE] {len(_aliased)} aliases aplicados al Autoencoder: {_aliased}")
                logger.info(f"[FIX-ALIAS-TRAIN-LIVE] OOD Guard: {len(_aliased)} aliases restaurados: {_aliased}")
            if _alias_missing:
                logger.warning(f"[FIX-ALIAS-TRAIN-LIVE] {len(_alias_missing)} features sin equivalente live: {_alias_missing}")
                print(f"[FIX-ALIAS-TRAIN-LIVE] WARN: features sin equivalente vivo: {_alias_missing}")

            # 2. Alinear y rellenar DataFrame para evitar KeyErrors (después de alias bridge)
            missing_cols = []
            for c in feature_cols:
                if c not in df.columns:
                    missing_cols.append(c)
                    df[c] = 0.0
                    
            if missing_cols:
                logger.warning(f"[LIVE-AE-FIX] Columnas faltantes tras alias bridge: {len(missing_cols)}/{len(feature_cols)}. Rellenadas con 0.0: {missing_cols[:5]}")
                print(f"[LIVE-AE-FIX] Columnas alineadas post-alias (faltaban {len(missing_cols)}): {missing_cols[:5]}")
                
            # 3. Escalar y codificar
            X_raw = df[feature_cols].copy().ffill().fillna(0).values.astype(np.float32)
            X_scaled = scaler.transform(X_raw)
            
            full_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                meta_features = autoencoder.encode(full_tensor).cpu().numpy()
                
            for i in range(bottleneck_size):
                df[f'ae_feat_{i}'] = meta_features[:, i]
                
            logger.success(f"[LIVE-AE-FIX] Inferencia rápida del AutoEncoder exitosa. {bottleneck_size} variables añadidas.")
            print(f"✨ [LIVE-AE-FIX] Inferencia rápida exitosa. {bottleneck_size} features añadidas.")
            return df
        except Exception as e:
            logger.critical(f"[LIVE-AE-FIX] Fallo crítico en bypass de AutoEncoder en vivo: {e}. Fallback a modo normal.")
            print(f"❌ [LIVE-AE-FIX] Error crítico: {e}")

    logger.info(f"🔮 Inicializando AutoEncoder Compression [Bottleneck={bottleneck_size}]")
    
    import os as _os_ae
    if _os_ae.environ.get('LUNA_SMOKE_TEST') == '1':
        epochs = 1
        logger.warning(f"[SMOKE TEST] AutoEncoder epochs reducido a 1")
    
    from sklearn.preprocessing import StandardScaler
    
    # 1. Aislar las variables numéricas que no sean booleanas (no tienen sentido continuo) ni labels.
    ignore_cols = [c for c in df.columns if c.lower() in ["close", "open", "high", "low", "volume", "date", "timestamp"]]
    ignore_cols += [c for c in df.columns if c.startswith("target_") or c.startswith("HMM_") or "rule" in c or "bin" in c]
    ignore_cols += [c for c in df.columns if df[c].nunique() <= 2] # ignora flags/booleanas
    
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    if len(feature_cols) == 0:
        logger.warning("[AE] No hay features continuas para auto-codear. Omitiendo.")
        return df

    # [H-01-FIX 2026-05-30] Filtrar features con >NaN_THRESHOLD% de NaN en el período IS.
    # PROBLEMA: DVOL (80.26% NaN) y FundingRate (49.51% NaN) entran al AE porque ffill().fillna(0)
    # convierte sus NaN en CEROS — el AE aprende a reconstruir ceros que son padding, no datos reales.
    # En OOS, estas features tienen distribuciones distintas (o seguirán siendo NaN→0).
    # El espacio latente ae_feat_0..31 queda contaminado por estos patrones espurios.
    # SOLUCIÓN: excluir features con >NaN_THRESHOLD% NaN en training antes del fit del scaler.
    try:
        from config.settings import cfg as _cfg_ae_nan
        _nan_thr = float(int(getattr(_cfg_ae_nan.autoencoder), 'nan_threshold_pct', 0.40))
    except Exception:
        _nan_thr = 0.40  # 40% máximo NaN permitido en IS para incluir en AE

    _train_mask_nan = df.index <= pd.to_datetime(train_end_date, utc=True)
    _df_train_nan   = df.loc[_train_mask_nan, feature_cols]
    _nan_fracs      = _df_train_nan.isnull().mean()
    _features_before_nan = len(feature_cols)
    feature_cols    = [c for c in feature_cols if _nan_fracs.get(c, 0.0) <= _nan_thr]
    _nan_excluded   = _features_before_nan - len(feature_cols)
    if _nan_excluded > 0:
        _excluded_names = [c for c in (_df_train_nan.columns.tolist()) if _nan_fracs.get(c, 0.0) > _nan_thr][:10]
        print(  # RULE[fixbugsprints.md]
            f"[H-01-FIX] AE inline NaN filter: {_features_before_nan} → {len(feature_cols)} features "
            f"({_nan_excluded} excluidas por >{_nan_thr*100:.0f}% NaN en IS). "
            f"Ejemplos excluidos: {_excluded_names}. "
            f"Previene contaminación del espacio latente con padding de ceros."
        )
        logger.info(
            f"[H-01-FIX] AE inline: {_nan_excluded} features excluidas por NaN>{_nan_thr*100:.0f}% en IS ({len(feature_cols)} restantes)."
        )
    else:
        print(f"[H-01-FIX] AE inline NaN filter: todas las {len(feature_cols)} features OK (<={_nan_thr*100:.0f}% NaN).")

    # --- FIX: Anchor the feature_cols list for stable WARM-START across WFB windows ---
    from pathlib import Path
    import json
    import os as _os_ae
    
    _window_id_anchor = _os_ae.environ.get("LUNA_WINDOW_ID", "")
    _root_dir_anchor = Path(__file__).resolve().parents[2]
    _anchor_file = _root_dir_anchor / "data" / "models" / "ae_valid_features.json"
    
    if _window_id_anchor == "W1":
        # W1 dictates the geometry. Save the features.
        try:
            _anchor_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_anchor_file, "w", encoding="utf-8") as _af:
                json.dump(feature_cols, _af)
            print(f"📌 [AE-ANCHOR] W1: Guardando lista de {len(feature_cols)} variables válidas en ae_valid_features.json para anclar el espacio latente.")
        except Exception as _e_anchor:
            logger.warning(f"[AE-ANCHOR] W1: Error al guardar ae_valid_features.json: {_e_anchor}")
    elif _window_id_anchor.startswith("W"):
        # W2+ forces the geometry of W1. Since W1's output was cached, read from wfb_cache/W1.
        _cached_w1_anchor = _root_dir_anchor / "data" / "wfb_cache" / "W1" / "models" / "ae_valid_features.json"
        
        # Retrocompatibilidad por si aún está en data/models o estamos en debug
        _target_anchor = _cached_w1_anchor if _cached_w1_anchor.exists() else _anchor_file
        
        if _target_anchor.exists():
            try:
                with open(_target_anchor, "r", encoding="utf-8") as _af:
                    _anchored_features = json.load(_af)
                # Ensure all anchored features exist in the current df
                _missing_anchor = [c for c in _anchored_features if c not in df.columns]
                for _mc in _missing_anchor:
                    df[_mc] = 0.0
                feature_cols = _anchored_features
                print(f"📌 [AE-ANCHOR] {_window_id_anchor}: Forzando uso de las {len(feature_cols)} variables ancladas en W1 para mantener compatibilidad Warm-Start.")
            except Exception as _e_anchor:
                logger.warning(f"[AE-ANCHOR] {_window_id_anchor}: Error leyendo ae_valid_features.json: {_e_anchor}")

    logger.info(f"[AE] Seleccionadas {len(feature_cols)} features continuas crudas para compresión.")
    
    # Manejar NaNs con cuidado (Llenar con forward fill, resto con 0)
    data_feat = df[feature_cols].copy().ffill().fillna(0)

    
    # 2. Partición estricta SOP R1
    train_mask = data_feat.index <= pd.to_datetime(train_end_date, utc=True)
    if train_mask.sum() == 0:
        logger.warning("[AE] No hay datos de Train disponibles. Omitiendo AutoEncoder para prevenir leakage.")
        return df
        
    X_train_raw = data_feat[train_mask].values.astype(np.float32)
    X_all_raw   = data_feat.values.astype(np.float32)
    
    # StandardScaler (ajustado SÓLO en train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_all_scaled   = scaler.transform(X_all_raw)
    
    # 3. Preparar DataLoader y Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [FIX-CUDA-KERNEL] Testear si la arquitectura de la GPU está soportada por la versión de PyTorch
    if device.type == 'cuda':
        try:
            _test = torch.zeros(1).to(device) + 1
        except Exception as e:
            logger.warning(f"[AE] CUDA disponible pero falló kernel (ej. RTX 5070 no soportada en esta build): {e}. Fallback a CPU.")
            device = torch.device('cpu')

    autoencoder = DeepFeatureAutoEncoder(input_dim=len(feature_cols), bottleneck_size=bottleneck_size).to(device)
    
    # [WARM-START AE] Fase 3: Estabilidad del AutoEncoder
    # Buscar modelo de ventana anterior para evitar mutación del espacio latente entre ventanas WFB
    import re
    from pathlib import Path
    window_id = _os_ae.environ.get("LUNA_WINDOW_ID", "")
    seed_id = _os_ae.environ.get("LUNA_SEED", "")
    root_dir = Path(__file__).resolve().parents[2]
    
    m = re.match(r"W(\d+)", window_id)
    logger.info(f"[AE-DEBUG] LUNA_WINDOW_ID={window_id}, match={bool(m)}")
    # [FIX] Fase AI Mining es Shared Step (Agnóstico a la semilla).
    # seed_id viene vacío. El AE se comparte entre ventanas en la carpeta base.
    if m:
        w_idx = int(m.group(1))
        if w_idx > 1:
            prev_window = f"W{w_idx - 1}"
            prev_model_path = root_dir / "data" / "wfb_cache" / prev_window / "models" / "autoencoder_state.pt"
            if prev_model_path.exists():
                try:
                    autoencoder.load_state_dict(torch.load(prev_model_path, map_location=device, weights_only=True))
                    logger.info(f"[AE-WARM-START] Pesos inyectados desde {prev_window}. Mutación del espacio latente mitigada.")
                except Exception as e:
                    logger.warning(f"[AE-WARM-START] Fallo al inyectar pesos previos: {e}")
                    print(f"❌ [AE-WARM-START] CRÍTICO: Fallo al cargar pesos desde {prev_window}. El AutoEncoder empezará de cero. Error: {e}")

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    dataset = TensorDataset(train_tensor, train_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 4. Entrenamiento No-Supervisado (Solo In-Sample)
    autoencoder.train()
    # [FALLA-06-FIX 2026-05-30] Aumentar epochs a 60 con early stopping patience=8
    # Problema: 30 epochs dejaban MSE=0.121 (solo -8% en ultimas 20 epochs = sub-convergencia)
    # Fix: mas epochs + early stopping real (si no mejora >1e-5 en 8 epochs consecutivos -> parar)
    max_epochs = max(epochs, 60)  # minimo 60, respetando si se pasan mas desde pipeline
    patience_ae = 8
    min_delta_ae = 1e-3  # [H9-AE-FIX 2026-05-30] Cambiado de 1e-5 a 1e-3
    # 1e-5 era demasiado pequeño: mejora epoch55->60 fue 0.000363 >> 1e-5 -> nunca activaba
    # Con 1e-3: cualquier mejora < 0.1% por bloque de 8 epochs = plateau real -> parar
    # Esto ahorra ~20 epochs (~80s) sin perdida practica de calidad de reconstruccion

    best_loss_ae = float('inf')
    patience_counter_ae = 0
    logger.info(f"[AE] Entrenando modelo AE sobre {X_train_scaled.shape[0]} muestras de Train | "
                f"max_epochs={max_epochs} | early_stop patience={patience_ae} | min_delta={min_delta_ae}")
    print(f"[FALLA-06-FIX] AE training: max_epochs={max_epochs}, early_stopping=True (patience={patience_ae})")

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            reconstructed = autoencoder(batch_x)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            
            # Anti-Gradient Exploding guard
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.debug(f"  └ Epoch {epoch+1:02d}/{max_epochs} | MSE Loss: {avg_loss:.6f}")

        # [FALLA-06-FIX] Early stopping
        if best_loss_ae - avg_loss > min_delta_ae:
            best_loss_ae = avg_loss
            patience_counter_ae = 0
        else:
            patience_counter_ae += 1
            if patience_counter_ae >= patience_ae:
                logger.info(f"[FALLA-06-FIX] AE early stopping en epoch {epoch+1} | best_loss={best_loss_ae:.6f}")
                print(f"[FALLA-06-FIX] AE convergido: early stopping epoch {epoch+1}/{max_epochs} | MSE={best_loss_ae:.6f}")
                break

    # 4.5 Guardar pesos serializados para la siguiente ventana WFB
    models_dir = root_dir / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_pt = models_dir / "autoencoder_state.pt"
    try:
        torch.save(autoencoder.state_dict(), out_pt)
        logger.debug(f"[AE] Pesos exportados a {out_pt.name} para WFB Cache.")
    except Exception as e:
        logger.warning(f"[AE] Error exportando pesos: {e}")

    # 5. Proyectar Universo Total
    logger.info("[AE] Inyectando Representaciones Densas sobre DataFrame global.")
    full_tensor = torch.tensor(X_all_scaled).to(device)
    meta_features = autoencoder.encode(full_tensor).cpu().numpy()
    
    # 6. Añadir al DF
    for i in range(bottleneck_size):
        df[f'ae_feat_{i}'] = meta_features[:, i]
        
    logger.success(f"[AE] Dimensión reducida extraída: {bottleneck_size} variables (ae_feat_0 ... ae_feat_{bottleneck_size-1}) añadidas.")
    
    # Liberar VRAM si CUDA
    del autoencoder, full_tensor, train_tensor, X_all_scaled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return df

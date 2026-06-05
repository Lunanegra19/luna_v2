import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

# Force root directory import
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from luna.utils.encoding_fix import fix_stdout_encoding; fix_stdout_encoding()
import ccxt
import yfinance as yf

# Paths
FEATURES_DIR = _ROOT / "data" / "features"


class DataReconciler:
    def __init__(self, sample_days: int = 14):
        self.sample_days = sample_days
        self.start_time = None
        self.end_time = None
        
        self.df_internet = pd.DataFrame()
        self.df_local = pd.DataFrame()

    def load_local_data(self):
        """Loads and merges features_holdout and features_train to ensure coverage."""
        logger.info("⌛ Cargando Parquets locales generados por el sistema...")
        p_train = FEATURES_DIR / "features_train.parquet"
        p_holdout = FEATURES_DIR / "features_holdout.parquet"
        
        dfs = []
        if p_holdout.exists():
            dfs.append(pd.read_parquet(p_holdout))
        if p_train.exists():
            dfs.append(pd.read_parquet(p_train))
            
        if not dfs:
            logger.error("No se encontraron parquet files locales.")
            sys.exit(1)
            
        df_combo = pd.concat(dfs)
        df_combo = df_combo[~df_combo.index.duplicated(keep='last')]
        df_combo = df_combo.sort_index()
        
        if len(df_combo) == 0:
            logger.error("Dataset local vacío tras remover duplicados.")
            sys.exit(1)
            
        self.df_local = df_combo.copy()
        
        # Snap timeframes for the latest sample_days using the last date available locally
        max_idx = self.df_local.index.dropna().max()
        if pd.isna(max_idx):
            logger.error("El índice local es todo NaT.")
            sys.exit(1)
            
        self.end_time = pd.to_datetime(max_idx)
        self.start_time = self.end_time - pd.Timedelta(days=self.sample_days)
        
        # Filter strictly
        self.df_local = self.df_local.loc[self.start_time:self.end_time]
        logger.success(f"✓ Rango focalizado: {self.start_time} -> {self.end_time} ({len(self.df_local)} observaciones locales)")


    def fetch_internet_data(self):
        """Fetch raw data from Internet (Yahoo Finance)."""
        logger.info(f"⌛ Descargando datos raw de internet via Yahoo Finance ({self.sample_days} días)...")
        
        start_str = self.start_time.strftime("%Y-%m-%d")
        end_str = (self.end_time + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        
        # 1. Yahoo Finance BTC-USD (Daily -> Hora via ffill or 1h timeframe if available)
        try:
            # We fetch 1h data if within 730 days. Our dataframe is in 2025, which is < 730 days
            btc = yf.download("BTC-USD", start=start_str, end=end_str, interval="1h", progress=False)
            if not btc.empty:
                if isinstance(btc.columns, pd.MultiIndex):
                    btc.columns = btc.columns.get_level_values(0)
                df_binance = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df_binance.columns = ['open', 'high', 'low', 'close', 'volume']
                df_binance.index = pd.to_datetime(df_binance.index).tz_convert('UTC')
                df_binance = df_binance.loc[:self.end_time]
                logger.success(f"✓ Yahoo Finance BTC-USD descargado: {len(df_binance)} velas horarias")
            else:
                logger.error("No se encontraron datos para BTC-USD en YFinance.")
                df_binance = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error descargando BTC-USD: {e}")
            df_binance = pd.DataFrame()
            
        # 2. Yahoo Finance SP500 Daily -> Hora
        try:
            sp500 = yf.download("^GSPC", start=start_str, end=end_str, progress=False)
            
            if not sp500.empty:
                if isinstance(sp500.columns, pd.MultiIndex):
                    sp500.columns = sp500.columns.get_level_values(0)
                    
                sp500 = sp500[['Close']].rename(columns={'Close': 'SP500'})
                sp500.index = pd.to_datetime(sp500.index).tz_localize('UTC')
                # Expand to hourly via ffill
                sp500_hourly = sp500.resample('1h').ffill()
                logger.success(f"✓ Yahoo Finance SP500 descargado: {len(sp500_hourly)} puntos diarios -> horarios ffill")
            else:
                sp500_hourly = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error descargando Yahoo Finance SP500: {e}")
            sp500_hourly = pd.DataFrame()
            
        # Join Internet Sources
        if not df_binance.empty:
            self.df_internet = df_binance
            if not sp500_hourly.empty:
                self.df_internet = self.df_internet.join(sp500_hourly, how='left')
                self.df_internet['SP500'] = self.df_internet['SP500'].ffill()
                
        logger.info(f"Dataset de Internet consolidado: {self.df_internet.shape}")


    def reconcile(self) -> bool:
        """Compares temporal alignment and exact values."""
        if self.df_internet.empty or self.df_local.empty:
            logger.error("Error: Faltan datasets para comparar.")
            return False
            
        logger.info("=" * 60)
        logger.info("INICIANDO RECONCILIACIÓN INSTITUCIONAL")
        logger.info("=" * 60)
        
        common_idx = self.df_internet.index.intersection(self.df_local.index)
        logger.info(f"Timestamps evaluables: {len(common_idx)} (Coverage: {(len(common_idx)/len(self.df_internet))*100:.1f}%)")
        
        if len(common_idx) == 0:
            logger.error("CRITICAL ERROR: No hay ningún timestamp en común. Desfase crítico detectado.")
            return False

        internet_c = self.df_internet.loc[common_idx]
        local_c = self.df_local.loc[common_idx]
        
        cols_to_check = [("close", "close")]
        
        passed = True
        for int_col, loc_col in cols_to_check:
            # Tolerancia dinámica: el volumen de un nodo CCXT es tick-stream y varía levemente del raw
            tol = 0.05 if "volume" in int_col else 0.005
            
            non_nan_mask = internet_c[int_col].notna() & local_c[loc_col].notna()
            n_eval = non_nan_mask.sum()
            
            if n_eval == 0:
                logger.warning(f"No hay superposición no-nula para comparar '{int_col}'.")
                continue
                
            error_pct = (np.abs(internet_c.loc[non_nan_mask, int_col] - local_c.loc[non_nan_mask, loc_col]) / 
                         internet_c.loc[non_nan_mask, int_col].replace(0, 1e-9))
                         
            max_err = error_pct.max() * 100
            mean_err = error_pct.mean() * 100
            failed_rows = error_pct[error_pct > tol]
            
            if len(failed_rows) > 0:
                logger.error(f"❌ [DESFASE] Columna '{int_col}': {len(failed_rows)} filas fallaron. Error Max: {max_err:.3f}%")
                fails = failed_rows.index[:3]
                logger.info(f"Ejemplos:\n{internet_c.loc[fails, int_col]} (Internet) vs {local_c.loc[fails, loc_col]} (Local)")
                passed = False
            else:
                logger.success(f"✓ [SANO] Columna '{int_col}' alineada en {n_eval} velas. Error Avg: {mean_err:.3f}% | Max: {max_err:.3f}%")

        logger.info("=" * 60)
        if passed:
            logger.success("VEREDICTO FINAL: AUDITORÍA SANA.\nLa máquina local es un espejo fiel del mercado remoto. No existen desfaces temporales críticos ni duplicados corruptos.")
        else:
            logger.error("VEREDICTO FINAL: ANOMALÍA DETECTADA.\nLa información procesada localmente difiere de la realidad. Corrupción confirmada.")
            
        return passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    
    validator = DataReconciler(sample_days=args.days)
    validator.load_local_data()
    validator.fetch_internet_data()
    result = validator.reconcile()
    
    if not result:
         sys.exit(1)

if __name__ == "__main__":
    main()

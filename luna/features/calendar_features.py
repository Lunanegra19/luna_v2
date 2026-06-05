"""
calendar_features.py
====================
Luna V1 — Features de Calendario (Fase C)
Genera variables binarias y numéricas basadas en eventos de mercado conocidos.

Features que replica de Luna v2 (features_v10_train_with_regime.parquet):
- cal_days_to_fomc         (días hasta próxima reunión FOMC)
- cal_is_fomc_week         (1 si la semana tiene FOMC)
- cal_is_us_session        (1 si hora actual es sesión US 14:30-21:00 UTC)
- cal_deribit_expiry_days  (días hasta próximo vencimiento Deribit — último viernes mes)
- cal_is_expiry_week       (1 si es la semana de vencimiento)
- cal_quarter_end          (1 si es la última semana del trimestre → rebalancing)
- cal_btc_halving_days     (días desde/hasta el halving más cercano)
- cal_day_of_week          (0=Lunes, 6=Domingo — crypto tiene estacionalidad semanal)
- cal_hour_of_day          (0-23 — importante para datos 1H)
- cal_is_weekend           (Sábado/Domingo — menor liquidez)

Fuente de fechas FOMC: hardcodeadas 2020-2026 (Fed publica con años de antelación).
Actualizables vía config/settings.yaml → fomc_dates.

Reglas de causalidad (R1):
- Todas estas features son conocidas EN EL MOMENTO DE LA VELA → sin lag necesario.
- Sono información temporal pura, sin look-ahead.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


# ── Fechas FOMC 2020-2026 (fuente: FederalReserve.gov) ──────────────────────
# Solo se incluyen las reuniones con potencial de subida/bajada de tipos
# (con/sin conferencia de prensa post-2019, todas son relevantes)
FOMC_DATES: list[str] = [
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]

# ── Halvings BTC (épocas) ────────────────────────────────────────────────────
BTC_HALVING_DATES: list[str] = [
    "2012-11-28",   # Bloque 210,000
    "2016-07-09",   # Bloque 420,000
    "2020-05-11",   # Bloque 630,000
    "2024-04-20",   # Bloque 840,000
    "2028-03-15",   # Estimación siguiente halving
]


class CalendarFeatures:
    """
    Genera features de calendario para el feature pipeline de Luna V1.
    Compatible con índices DatetimeIndex UTC en 1H o diario.
    """

    def __init__(self):
        self._fomc_dates = pd.to_datetime(FOMC_DATES, utc=True)
        self._halving_dates = pd.to_datetime(BTC_HALVING_DATES, utc=True)

    # ── FOMC Features ────────────────────────────────────────────────────────

    def _add_fomc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        cal_days_to_fomc: días hasta el próximo FOMC (negativo = pasado).
        cal_is_fomc_week: 1 si la vela cae en la semana de un FOMC (Lun-Dom).
        """
        # PRINT FOR TRACEABILITY (RULE[fixbugsprints.md])
        print("[FIX-CALENDAR-VEC] Vectorized FOMC feature calculation active.")
        
        ts_series = pd.Series(df.index, index=df.index).dt.tz_convert("UTC")
        # Adjust FOMC dates to 23:59:59 to treat the entire day of announcement as Day 0
        fomc_dates_adjusted = pd.to_datetime(FOMC_DATES, utc=True) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        fomc_df = pd.DataFrame({'fomc_date': fomc_dates_adjusted}).sort_values('fomc_date')
        
        df_temp = pd.DataFrame({'ts': ts_series}, index=df.index)
        
        # [BUG-FIX-CAL-PRECISION] Cast keys to identical types to prevent pandas merge_asof precision incompatibility
        fomc_df['fomc_date'] = fomc_df['fomc_date'].astype(df_temp['ts'].dtype)
        print(f"[BUG-FIX-CAL-PRECISION] cast right key 'fomc_date' to match left key 'ts' dtype: {df_temp['ts'].dtype}")
        
        merged_fomc = pd.merge_asof(df_temp, fomc_df, left_on='ts', right_on='fomc_date', direction='forward')
        
        days_to_fomc = (merged_fomc['fomc_date'] - merged_fomc['ts']).dt.days
        df["cal_days_to_fomc"] = pd.Series(days_to_fomc.values, index=df.index).fillna(365).astype(int)
        
        # Is FOMC week
        ts_mondays = ts_series.dt.normalize() - pd.to_timedelta(ts_series.dt.weekday, unit='D')
        fomc_dates_series = pd.Series(self._fomc_dates)
        fomc_mondays = fomc_dates_series.dt.normalize() - pd.to_timedelta(fomc_dates_series.dt.weekday, unit='D')
        df["cal_is_fomc_week"] = ts_mondays.isin(fomc_mondays).astype(int).values
        
        return df

    # ── Deribit Expiry Features ───────────────────────────────────────────────

    @staticmethod
    def _get_deribit_expiry_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Deribit: opciones mensuales vencen el ÚLTIMO VIERNES del mes a las 08:00 UTC.
        """
        dates = []
        current = start.replace(day=1)
        while current <= end:
            # Último viernes del mes: ir al último día y retroceder hasta viernes
            if current.month == 12:
                last_day = current.replace(year=current.year + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                last_day = current.replace(month=current.month + 1, day=1) - pd.Timedelta(days=1)

            # Retroceder hasta viernes (weekday=4)
            days_back = (last_day.weekday() - 4) % 7
            last_friday = last_day - pd.Timedelta(days=days_back)
            # Deribit expiry: 08:00 UTC
            expiry_dt = pd.Timestamp(last_friday.year, last_friday.month, last_friday.day,
                                     8, 0, 0, tz="UTC")
            dates.append(expiry_dt)
            # Avanzar al mes siguiente
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return pd.DatetimeIndex(dates)

    def _add_deribit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        cal_deribit_expiry_days: días hasta el próximo vencimiento mensual Deribit.
        cal_is_expiry_week:      1 si es la semana del vencimiento.
        """
        # PRINT FOR TRACEABILITY (RULE[fixbugsprints.md])
        print("[FIX-CALENDAR-VEC] Vectorized Deribit feature calculation active.")
        
        ts_series = pd.Series(df.index, index=df.index).dt.tz_convert("UTC")
        df_temp = pd.DataFrame({'ts': ts_series}, index=df.index)
        
        start = df.index.min()
        end   = df.index.max()
        expiry_dates = self._get_deribit_expiry_dates(start, end + pd.Timedelta(days=40))
        expiry_df = pd.DataFrame({'expiry_date': expiry_dates}).sort_values('expiry_date')
        
        # [BUG-FIX-CAL-PRECISION] Cast keys to identical types to prevent pandas merge_asof precision incompatibility
        expiry_df['expiry_date'] = expiry_df['expiry_date'].astype(df_temp['ts'].dtype)
        print(f"[BUG-FIX-CAL-PRECISION] cast right key 'expiry_date' to match left key 'ts' dtype: {df_temp['ts'].dtype}")
        
        merged_expiry = pd.merge_asof(df_temp, expiry_df, left_on='ts', right_on='expiry_date', direction='forward')
        days_to_expiry = (merged_expiry['expiry_date'] - merged_expiry['ts']).dt.days
        df["cal_deribit_expiry_days"] = pd.Series(days_to_expiry.values, index=df.index).fillna(31).astype(int)
        
        df["cal_is_expiry_week"] = ((df["cal_deribit_expiry_days"] >= 0) & (df["cal_deribit_expiry_days"] <= 5)).astype(int)
        return df

    # ── Halving Features ─────────────────────────────────────────────────────

    def _add_halving_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        cal_days_since_halving: días desde el último halving.
        cal_halving_cycle_pct:  progreso en el ciclo de 4 años (0→1).
        """
        # PRINT FOR TRACEABILITY (RULE[fixbugsprints.md])
        print("[FIX-CALENDAR-VEC] Vectorized Halving feature calculation active.")
        
        ts_series = pd.Series(df.index, index=df.index).dt.tz_convert("UTC")
        df_temp = pd.DataFrame({'ts': ts_series}, index=df.index)
        
        halving_df = pd.DataFrame({'halving_date': self._halving_dates}).sort_values('halving_date')
        
        # [BUG-FIX-CAL-PRECISION] Cast keys to identical types to prevent pandas merge_asof precision incompatibility
        halving_df['halving_date'] = halving_df['halving_date'].astype(df_temp['ts'].dtype)
        print(f"[BUG-FIX-CAL-PRECISION] cast right key 'halving_date' to match left key 'ts' dtype: {df_temp['ts'].dtype}")
        
        merged_last = pd.merge_asof(df_temp, halving_df, left_on='ts', right_on='halving_date', direction='backward')
        
        df_temp_eps = pd.DataFrame({'ts_eps': ts_series + pd.Timedelta(microseconds=1)}, index=df.index)
        # Ensure df_temp_eps has exactly the same dtype resolution
        df_temp_eps['ts_eps'] = df_temp_eps['ts_eps'].astype(df_temp['ts'].dtype)
        
        halving_df_next = halving_df.rename(columns={'halving_date': 'next_halving_date'})
        halving_df_next['next_halving_date'] = halving_df_next['next_halving_date'].astype(df_temp['ts'].dtype)
        
        merged_next = pd.merge_asof(df_temp_eps, halving_df_next, left_on='ts_eps', right_on='next_halving_date', direction='forward')
        
        last_halving = pd.Series(merged_last['halving_date'].values, index=df.index).dt.tz_localize('UTC')
        next_halving = pd.Series(merged_next['next_halving_date'].values, index=df.index).dt.tz_localize('UTC')
        
        na_next_mask = next_halving.isna() & last_halving.notna()
        if na_next_mask.any():
            next_halving.loc[na_next_mask] = last_halving.loc[na_next_mask] + pd.Timedelta(days=1461)
            
        elapsed = pd.Series(-1, index=df.index, dtype=int)
        cycle = pd.Series(0.0, index=df.index, dtype=float)
        
        valid_mask = last_halving.notna()
        if valid_mask.any():
            elapsed.loc[valid_mask] = (ts_series.loc[valid_mask] - last_halving.loc[valid_mask]).dt.days
            cycle_days = (next_halving.loc[valid_mask] - last_halving.loc[valid_mask]).dt.days
            elapsed_cycle = (ts_series.loc[valid_mask] - last_halving.loc[valid_mask]).dt.days
            cycle.loc[valid_mask] = (elapsed_cycle / cycle_days.replace(0, 1)).round(4)
            
        df["cal_days_since_halving"] = elapsed
        df["cal_halving_cycle_pct"]  = cycle

        # [HALVING-HARMONIC-01 2026-06-03] Codificación armónica del ciclo de halving
        # Motivación: cal_halving_cycle_pct es lineal (0→1), XGBoost no aprende que
        # 0.99 y 0.01 son adyacentes (discontinuidad en el rollover).
        # Sin/cos eliminan esa discontinuidad y son OOS-estables (determinísticos).
        # Demostrado que el ciclo de halving (~4 años) es el predictor estructural
        # más estable de BTC: no depende de correlaciones macro que cambian de signo.
        df["cal_halving_cycle_sin"] = np.sin(2 * np.pi * cycle).round(6)
        df["cal_halving_cycle_cos"] = np.cos(2 * np.pi * cycle).round(6)

        # cal_days_to_next_halving: relevante para el mercado porque el "supply shock"
        # es anticipado por los traders con semanas/meses de antelación.
        # Sin look-ahead: los halvings son públicamente conocidos años antes.
        if valid_mask.any():
            days_to_next = pd.Series(-1, index=df.index, dtype=int)
            days_to_next.loc[valid_mask] = (
                next_halving.loc[valid_mask] - ts_series.loc[valid_mask]
            ).dt.days
            df["cal_days_to_next_halving"] = days_to_next
        else:
            df["cal_days_to_next_halving"] = -1

        print(
            f"[HALVING-HARMONIC-01] Halving features generadas: "
            f"cal_days_since_halving, cal_halving_cycle_pct, "
            f"cal_halving_cycle_sin, cal_halving_cycle_cos, cal_days_to_next_halving | "
            f"cycle_pct rango=[{cycle.min():.3f},{cycle.max():.3f}] "
            f"sin rango=[{df['cal_halving_cycle_sin'].min():.3f},{df['cal_halving_cycle_sin'].max():.3f}]"
        )
        return df

    # ── Temporal/Session Features ─────────────────────────────────────────────

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        cal_hour_of_day:    hora UTC (0-23) — para datos 1H
        cal_day_of_week:    0=Lunes … 6=Domingo
        cal_is_weekend:     1 si Sábado o Domingo
        cal_is_us_session:  1 si 13:30-21:00 UTC (mercado NYSE abierto)
        cal_is_asia_session:1 si 00:00-08:00 UTC (Tokio + Hong Kong)
        cal_is_eu_session:  1 si 07:00-15:30 UTC (Londres + Frankfurt)
        cal_quarter_end:    1 si es la última semana del trimestre
        """
        idx = df.index

        df["cal_hour_of_day"]     = idx.hour
        df["cal_day_of_week"]     = idx.dayofweek
        df["cal_is_weekend"]      = (idx.dayofweek >= 5).astype(int)

        # Sesiones de mercado
        df["cal_is_us_session"]   = (
            (idx.hour >= 13) & (idx.hour < 21)
        ).astype(int)

        df["cal_is_asia_session"] = (
            (idx.hour >= 0) & (idx.hour < 8)
        ).astype(int)

        df["cal_is_eu_session"]   = (
            (idx.hour >= 7) & (idx.hour < 16)
        ).astype(int)

        # Quarter end: última semana de cada trimestre (Mar, Jun, Sep, Dic)
        quarter_end_months = {3, 6, 9, 12}
        is_quarter_end = (
            idx.month.isin(quarter_end_months) &
            (idx.day >= 24)  # última semana del mes = días 24-31
        ).astype(int)
        df["cal_quarter_end"] = is_quarter_end

        return df

    # ── Método principal ─────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade todas las features de calendario al DataFrame.

        Args:
            df: DataFrame con DatetimeIndex UTC (1H o diario)

        Returns:
            DataFrame con columnas cal_* añadidas.
        """
        if df.empty:
            return df

        logger.info("Generando calendar features...")
        n_before = df.shape[1]

        # 1. Features temporales (vectorizadas — rápidas)
        df = self._add_temporal_features(df)

        # 2. FOMC (loop por timestamp — necesario para proximidad dinámica)
        df = self._add_fomc_features(df)

        # 3. Deribit Expiry
        df = self._add_deribit_features(df)

        # 4. Halving cycle
        df = self._add_halving_features(df)

        n_added = df.shape[1] - n_before
        cal_cols = [c for c in df.columns if c.startswith("cal_")]
        logger.success(f"Calendar features: +{n_added} columnas → {cal_cols}")

        return df


if __name__ == "__main__":
    # Test rápido
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    df_test = pd.DataFrame({"close": 42000.0}, index=idx)
    cf = CalendarFeatures()
    result = cf.transform(df_test)
    print(result[[c for c in result.columns if c.startswith("cal_")]].tail(10))
    print(f"\nTotal calendar features: {len([c for c in result.columns if c.startswith('cal_')])}")

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_date_column(df: pd.DataFrame):
    """Automatically detect the best date/time column for forecasting."""
    for candidate in ['Date', 'date', 'Timestamp', 'timestamp', 'Order Date', 'order_date', 'Transaction Date']:
        if candidate in df.columns:
            return candidate

    if 'Year' in df.columns and 'Month' in df.columns:
        return None

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.debug('Detected datetime64 date column: %s', col)
            return col
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.8:
                logger.debug('Detected parsable date column: %s', col)
                return col
        except Exception:
            continue
    return None


def build_forecast_dataframe(df: pd.DataFrame, target_col: str, date_col: str):
    df_copy = df[[date_col, target_col]].dropna().copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, target_col])
    df_copy = df_copy.rename(columns={date_col: 'ds', target_col: 'y'})
    return df_copy
    df_copy = df[[date_col, target_col]].dropna().copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col, target_col])
    df_copy = df_copy.rename(columns={date_col: 'ds', target_col: 'y'})
    return df_copy


def generate_forecast(df, periods: int = 30):
    try:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast
    except Exception as exc:
        raise RuntimeError(f'Forecast generation failed: {exc}')

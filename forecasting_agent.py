import logging
import pandas as pd
from prophet import Prophet
from utils.helper_functions import detect_date_column, build_forecast_dataframe

logger = logging.getLogger(__name__)


class ForecastingAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _infer_target_column(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'Global Sales' in numeric_cols:
            return 'Global Sales'
        known_targets = [col for col in numeric_cols if 'sales' in col.lower() or 'profit' in col.lower()]
        if known_targets:
            return known_targets[0]
        if numeric_cols:
            return numeric_cols[-1]
        raise ValueError('No numeric target column found for forecasting.')

    def prepare_forecast_data(self):
        df_copy = self.df.copy()
        date_col = detect_date_column(df_copy)
        if date_col:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            target = self._infer_target_column(df_copy)
            return build_forecast_dataframe(df_copy, target, date_col)

        if 'Year' in df_copy.columns and 'Month' in df_copy.columns:
            df_copy['YearMonth'] = pd.to_datetime(
                df_copy['Year'].astype(str) + '-' + df_copy['Month'].astype(str).str[:3],
                format='%Y-%b', errors='coerce'
            )
            target = self._infer_target_column(df_copy)
            return build_forecast_dataframe(df_copy, target, 'YearMonth')

        raise ValueError('No date column available for forecasting.')

    def create_forecast(self, periods=30):
        forecast_df = self.prepare_forecast_data()
        model = Prophet()
        model.fit(forecast_df)
        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)
        logger.info('Generated %s-day forecast with %s rows', periods, len(forecast))
        return forecast

    def create_quarterly_forecast(self):
        forecast_df = self.prepare_forecast_data()
        model = Prophet()
        model.fit(forecast_df)
        future = model.make_future_dataframe(periods=90, freq='D')
        forecast = model.predict(future)
        logger.info('Generated quarterly forecast with %s rows', len(forecast))
        return forecast

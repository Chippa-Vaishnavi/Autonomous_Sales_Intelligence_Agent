import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class VisualizationAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if df is not None else pd.DataFrame()

    def validate_dataframe(self):
        if self.df is None or self.df.empty:
            return False, 'Dataset is empty. Upload a valid sales CSV file before generating charts.'
        return True, None

    def _get_revenue_column(self):
        for col in ['Global Sales', 'National Sales', 'Revenue', 'Sales']:
            if col in self.df.columns:
                return col

        numeric_cols = [
            col for col in self.df.select_dtypes(include=['number']).columns
            if 'sales' in col.lower() or 'profit' in col.lower()
        ]
        if numeric_cols:
            return numeric_cols[0]
        raise ValueError('No revenue-like numeric column found for visualization.')

    def _prepare_time_series(self):
        revenue_col = self._get_revenue_column()
        if {'Year', 'Month'}.issubset(self.df.columns):
            ts_df = self.df[['Year', 'Month', revenue_col]].copy()
            ts_df['Month'] = ts_df['Month'].astype(str)
            ts_df['Date'] = pd.to_datetime(
                ts_df['Year'].astype(str) + '-' + ts_df['Month'].str[:3],
                format='%Y-%b',
                errors='coerce',
            )
            ts_df = ts_df.dropna(subset=['Date']).groupby('Date')[revenue_col].sum().reset_index()
            return ts_df

        date_candidates = ['Date', 'Order Date', 'Transaction Date', 'Timestamp']
        for col in date_candidates:
            if col in self.df.columns:
                ts_df = self.df[[col, revenue_col]].copy()
                ts_df[col] = pd.to_datetime(ts_df[col], errors='coerce')
                ts_df = ts_df.dropna(subset=[col]).rename(columns={col: 'Date'})
                ts_df = ts_df.groupby('Date')[revenue_col].sum().reset_index()
                return ts_df

        raise ValueError('No usable date column found for trend visualization.')

    def build_revenue_trend_chart(self):
        valid, msg = self.validate_dataframe()
        if not valid:
            logger.warning(msg)
            return None

        try:
            trend_df = self._prepare_time_series()
            fig = px.line(trend_df, x='Date', y=self._get_revenue_column(), markers=True)
            fig.update_layout(
                title='Revenue Trend',
                xaxis_title='Date',
                yaxis_title='Revenue',
                template='plotly_white',
            )
            return fig
        except Exception as exc:
            logger.exception('Failed to build revenue trend chart: %s', exc)
            return None

    def build_monthly_sales_trend_chart(self):
        valid, msg = self.validate_dataframe()
        if not valid:
            logger.warning(msg)
            return None

        try:
            revenue_col = self._get_revenue_column()
            monthly_df = self._prepare_time_series()
            monthly_df['Month'] = monthly_df['Date'].dt.to_period('M').astype(str)
            monthly_df = monthly_df.groupby('Month')[revenue_col].sum().reset_index()
            monthly_df = monthly_df.sort_values('Month')
            fig = px.bar(monthly_df, x='Month', y=revenue_col, color=revenue_col, color_continuous_scale='Blues')
            fig.update_layout(
                title='Monthly Sales Trend',
                xaxis_title='Month',
                yaxis_title='Revenue',
                template='plotly_white',
            )
            return fig
        except Exception as exc:
            logger.exception('Failed to build monthly sales trend chart: %s', exc)
            return None

    def build_regional_sales_chart(self):
        valid, msg = self.validate_dataframe()
        if not valid:
            logger.warning(msg)
            return None

        try:
            if 'Region' not in self.df.columns:
                raise ValueError('Region column is missing.')
            revenue_col = self._get_revenue_column()
            regional_df = self.df.groupby('Region')[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False)
            fig = px.bar(regional_df, x='Region', y=revenue_col, color='Region')
            fig.update_layout(
                title='Regional Sales Performance',
                xaxis_title='Region',
                yaxis_title='Revenue',
                template='plotly_white',
            )
            return fig
        except Exception as exc:
            logger.exception('Failed to build regional sales chart: %s', exc)
            return None

    def build_top_products_chart(self, top_n: int = 10):
        valid, msg = self.validate_dataframe()
        if not valid:
            logger.warning(msg)
            return None

        try:
            if 'Name' not in self.df.columns:
                raise ValueError('Product name column is missing.')
            revenue_col = self._get_revenue_column()
            product_df = self.df.groupby('Name')[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False).head(top_n)
            fig = px.bar(product_df, x='Name', y=revenue_col, color=revenue_col, color_continuous_scale='Viridis')
            fig.update_layout(
                title='Top Products',
                xaxis_title='Product',
                yaxis_title='Revenue',
                template='plotly_white',
            )
            fig.update_xaxes(tickangle=-35)
            return fig
        except Exception as exc:
            logger.exception('Failed to build top products chart: %s', exc)
            return None

    def build_forecast_visualization(self, forecast_df: pd.DataFrame):
        if forecast_df is None or forecast_df.empty:
            logger.warning('Forecast data is empty; skipping forecast visualization.')
            return None

        try:
            if 'ds' not in forecast_df.columns or 'yhat' not in forecast_df.columns:
                raise ValueError('Forecast dataframe is missing ds or yhat columns.')

            viz_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            if 'y' in forecast_df.columns:
                viz_df['actual'] = forecast_df['y']
            viz_df['ds'] = pd.to_datetime(viz_df['ds'], errors='coerce')
            viz_df = viz_df.dropna(subset=['ds'])

            fig = go.Figure()
            if 'actual' in viz_df.columns:
                fig.add_trace(go.Scatter(x=viz_df['ds'], y=viz_df['actual'], mode='markers', name='Actual', marker=dict(color='gray')))
            fig.add_trace(go.Scatter(x=viz_df['ds'], y=viz_df['yhat'], mode='lines', name='Forecast', line=dict(color='#2563eb', width=3)))
            if {'yhat_lower', 'yhat_upper'}.issubset(viz_df.columns):
                fig.add_trace(go.Scatter(
                    x=viz_df['ds'],
                    y=viz_df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(37, 99, 235, 0.15)',
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=viz_df['ds'],
                    y=viz_df['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(37, 99, 235, 0.15)',
                    showlegend=False,
                ))
            fig.update_layout(
                title='Forecast Visualization',
                xaxis_title='Date',
                yaxis_title='Forecast',
                template='plotly_white',
            )
            return fig
        except Exception as exc:
            logger.exception('Failed to build forecast visualization: %s', exc)
            return None

    def build_visual_analytics(self, forecast_df: pd.DataFrame = None):
        figures = {}
        figures['revenue_trend'] = self.build_revenue_trend_chart()
        figures['monthly_sales_trend'] = self.build_monthly_sales_trend_chart()
        figures['regional_sales'] = self.build_regional_sales_chart()
        figures['top_products'] = self.build_top_products_chart()
        figures['forecast'] = self.build_forecast_visualization(forecast_df)
        return figures

import pandas as pd


class EDAAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def revenue_summary(self):
        revenue_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'profit' in col.lower()]
        summary = self.df[revenue_cols].sum().to_dict()
        return summary

    def product_highlights(self):
        if 'Name' in self.df.columns and 'Global Sales' in self.df.columns:
            top_product = self.df.sort_values('Global Sales', ascending=False).iloc[0]
            return {'top_product': top_product['Name'], 'top_revenue': float(top_product['Global Sales'])}
        return {}

    def regional_performance(self):
        if 'Region' in self.df.columns and 'Global Sales' in self.df.columns:
            return self.df.groupby('Region')['Global Sales'].sum().sort_values(ascending=False).to_dict()
        return {}

    def customer_summary(self):
        metrics = {}
        if 'Country' in self.df.columns and 'Global Sales' in self.df.columns:
            metrics['top_country'] = self.df.groupby('Country')['Global Sales'].sum().idxmax()
            metrics['country_sales'] = self.df.groupby('Country')['Global Sales'].sum().sort_values(ascending=False).to_dict()
        return metrics

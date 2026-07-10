import pandas as pd


class RecommendationAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_recommendations(self):
        recommendations = []
        if 'Global Sales' in self.df.columns and 'Region' in self.df.columns:
            regional_sales = self.df.groupby('Region')['Global Sales'].sum().sort_values(ascending=False)
            if len(regional_sales) > 0:
                recommendations.append(f'Focus marketing investment in {regional_sales.index[0]} to capitalize on the strongest region.')
            if len(regional_sales) > 1:
                recommendations.append(f'Investigate underperforming region {regional_sales.index[-1]} for operational improvements.')

        if 'Genre' in self.df.columns and 'Global Sales' in self.df.columns:
            genres = self.df.groupby('Genre')['Global Sales'].sum().sort_values(ascending=False)
            recommendations.append(f'Expand product promotion around {genres.index[0]} titles, which show strongest revenue potential.')

        if 'Platform' in self.df.columns and 'Global Sales' in self.df.columns:
            platform_sales = self.df.groupby('Platform')['Global Sales'].sum().sort_values(ascending=False)
            recommendations.append(f'Optimize platform-specific campaigns for {platform_sales.index[0]}, the top revenue platform.')

        return recommendations

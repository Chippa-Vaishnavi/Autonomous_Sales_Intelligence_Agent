import pandas as pd


class InsightAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_executive_summary(self):
        summary = []
        if 'Global Sales' in self.df.columns:
            total_revenue = self.df['Global Sales'].sum()
            summary.append(f'Total global sales are ${total_revenue:,.2f} for the uploaded dataset.')

        if 'Region' in self.df.columns and 'Global Sales' in self.df.columns:
            top_region = self.df.groupby('Region')['Global Sales'].sum().idxmax()
            summary.append(f'The highest revenue region is {top_region}.')

        if 'Genre' in self.df.columns and 'Global Sales' in self.df.columns:
            top_genre = self.df.groupby('Genre')['Global Sales'].sum().idxmax()
            summary.append(f'The top performing genre is {top_genre}.')

        return ' '.join(summary)

    def detect_risks(self):
        risk_items = []
        if 'Global Sales' in self.df.columns and 'Region' in self.df.columns:
            low_regions = self.df.groupby('Region')['Global Sales'].sum().nsmallest(2).to_dict()
            for region, sales in low_regions.items():
                risk_items.append(f'Region {region} is at risk with lower revenue of ${sales:,.2f}.')
        return risk_items

    def performance_insights(self):
        insights = {}
        if 'Country' in self.df.columns and 'Global Sales' in self.df.columns:
            insights['top_country'] = self.df.groupby('Country')['Global Sales'].sum().idxmax()
            insights['bottom_country'] = self.df.groupby('Country')['Global Sales'].sum().idxmin()
        return insights

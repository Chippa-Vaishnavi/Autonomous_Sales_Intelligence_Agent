import pandas as pd


class ProfilingAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self):
        profile = {
            'row_count': len(self.df),
            'column_count': len(self.df.columns),
            'missing_values': self.df.isna().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'sample_head': self.df.head(5).to_dict(orient='records')
        }
        return profile

    def get_quality_summary(self):
        missing_count = self.df.isna().sum().sum()
        duplicate_count = int(self.df.duplicated().sum())
        return {
            'missing_count': int(missing_count),
            'duplicate_count': duplicate_count,
            'completeness': round(100.0 - (missing_count / (self.df.shape[0] * self.df.shape[1]) * 100.0), 2)
        }

import pandas as pd


class CleaningAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')

        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        return self.df

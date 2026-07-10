import os
import pandas as pd
import streamlit as st
from agents.profiling_agent import ProfilingAgent
from agents.cleaning_agent import CleaningAgent
from agents.eda_agent import EDAAgent
from agents.forecasting_agent import ForecastingAgent
from agents.insight_agent import InsightAgent
from agents.recommendation_agent import RecommendationAgent
from agents.chat_agent import ChatAgent
from agents.report_agent import ReportAgent


class SupervisorAgent:
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df
        self.cleaned_df = None
        self.profile_agent = ProfilingAgent(df)
        self.cleaning_agent = CleaningAgent(df)
        self.eda_agent = None
        self.forecasting_agent = None
        self.insight_agent = None
        self.recommendation_agent = None

    def run_cleaning_pipeline(self):
        self.cleaned_df = self.cleaning_agent.clean()
        self.eda_agent = EDAAgent(self.cleaned_df)
        self.forecasting_agent = ForecastingAgent(self.cleaned_df)
        self.insight_agent = InsightAgent(self.cleaned_df)
        self.recommendation_agent = RecommendationAgent(self.cleaned_df)
        return self.cleaned_df

    def render_dashboard(self):
        st.header('Sales Intelligence Dashboard')
        profile = self.profile_agent.profile()
        st.metric('Rows', profile['row_count'])
        st.metric('Columns', profile['column_count'])
        total_missing = sum(profile['missing_values'].values())
        st.metric('Total Missing Values', total_missing)

        if self.cleaned_df is None:
            self.run_cleaning_pipeline()

        revenue = self.cleaned_df['Global Sales'].sum() if 'Global Sales' in self.cleaned_df.columns else 0
        st.metric('Total Global Sales', f'${revenue:,.2f}')

    def render_profile(self):
        st.header('Dataset Profile')
        profile = self.profile_agent.profile()
        st.write(profile)

    def render_sales_analytics(self):
        st.header('Sales Analytics')
        if self.cleaned_df is None:
            self.run_cleaning_pipeline()

        st.write(self.eda_agent.revenue_summary())
        st.write(self.eda_agent.product_highlights())
        st.write(self.eda_agent.regional_performance())
        st.write(self.eda_agent.customer_summary())

    def render_forecasting(self):
        st.header('Forecasting')
        if self.cleaned_df is None:
            self.run_cleaning_pipeline()

        try:
            forecast_30 = self.forecasting_agent.create_forecast(periods=30)
            forecast_90 = self.forecasting_agent.create_quarterly_forecast()
            st.write(forecast_30.tail())
            st.write(forecast_90.tail())
        except Exception as exc:
            st.error(f'Forecasting error: {exc}')

    def render_insights(self):
        st.header('AI Insights')
        if self.cleaned_df is None:
            self.run_cleaning_pipeline()

        st.write(self.insight_agent.generate_executive_summary())
        st.write(self.insight_agent.detect_risks())
        st.write(self.insight_agent.performance_insights())

    def render_recommendations(self):
        st.header('Recommendations')
        if self.cleaned_df is None:
            self.run_cleaning_pipeline()

        recs = self.recommendation_agent.generate_recommendations()
        for rec in recs:
            st.write(f'- {rec}')

    def render_chat(self):
        st.header('Chat with Data')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            st.warning('Set OPENAI_API_KEY in environment variables to use chat.')
            return

        prompt = st.text_input('Ask a sales question', value='Which product generated highest revenue?')
        if st.button('Ask') and prompt:
            chat_agent = ChatAgent(self.cleaned_df if self.cleaned_df is not None else self.raw_df, openai_api_key)
            response = chat_agent.ask_question(prompt)
            st.write(response)

    def render_report(self):
        st.header('Download Report')
        if self.cleaned_df is None:
            self.run_cleaning_pipeline()

        summary = self.insight_agent.generate_executive_summary()
        metrics = self.eda_agent.revenue_summary()
        recs = self.recommendation_agent.generate_recommendations()
        report_agent = ReportAgent(summary, metrics, recs)
        output_path = os.path.join('reports', 'sales_intelligence_report.pdf')
        os.makedirs('reports', exist_ok=True)
        report_agent.create_pdf(output_path)
        with open(output_path, 'rb') as f:
            st.download_button('Download PDF report', f, file_name='sales_intelligence_report.pdf', mime='application/pdf')

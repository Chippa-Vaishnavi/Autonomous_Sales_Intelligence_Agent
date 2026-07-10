import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from langchain.llms import GoogleGemini
except ImportError:
    GoogleGemini = None

try:
    from google.generativeai import client as gemini_client
except ImportError:
    gemini_client = None


class GenerativeAIAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-pro'):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError('GEMINI_API_KEY is required for Generative AI functions.')
        self.model = model
        self.client = self._build_client()

    def _build_client(self):
        if GoogleGemini is not None:
            return GoogleGemini(api_key=self.api_key)
        if gemini_client is not None:
            gemini_client.configure(api_key=self.api_key)
            return gemini_client
        raise RuntimeError('Gemini client is not installed. Install google-generativeai or langchain.')

    def _query(self, prompt: str) -> str:
        if hasattr(self.client, 'generate'):
            response = self.client.generate(model=self.model, prompt=prompt)
            return response.text if hasattr(response, 'text') else str(response)
        return str(self.client(prompt))

    def generate_executive_summary(self, dataset_profile: dict, top_insights: dict) -> str:
        prompt = (
            'Using the dataset profile and top insights, write a polished executive summary for a sales intelligence report. '
            f'Dataset summary: {dataset_profile}. Insights: {top_insights}. '
            'Keep it concise and suitable for executive stakeholders.'
        )
        logger.info('Generating executive summary using Gemini.')
        return self._query(prompt)

    def explain_trends(self, trend_context: str) -> str:
        prompt = (
            'Provide a business-focused explanation of sales trends and what they mean for growth strategy. '
            f'Trend details: {trend_context}.'
        )
        return self._query(prompt)

    def product_performance_narrative(self, product_context: str) -> str:
        prompt = (
            'Generate a product performance narrative for sales leadership based on the following analysis: '
            f'{product_context}.'
        )
        return self._query(prompt)

    def regional_analysis_narrative(self, region_context: str) -> str:
        prompt = (
            'Produce a regional performance narrative that highlights strengths, weaknesses, and next-step recommendations. '
            f'{region_context}.'
        )
        return self._query(prompt)

    def identify_risks(self, risk_context: str) -> str:
        prompt = (
            'Identify top risks and mitigation actions for the sales organization based on the following details: '
            f'{risk_context}.'
        )
        return self._query(prompt)

    def recommend_actions(self, recommendation_context: str) -> str:
        prompt = (
            'Write actionable recommendations for growth, retention, and operational improvement. '
            f'{recommendation_context}.'
        )
        return self._query(prompt)

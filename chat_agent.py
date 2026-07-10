import logging
import pandas as pd

try:
    from pandasai import PandasAI
except Exception as exc:
    PandasAI = None
    pandasai_import_error = exc
else:
    pandasai_import_error = None

try:
    from pandasai.llm import OpenAI as PandasAIOpenAI
except ImportError:
    PandasAIOpenAI = None

logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(self, df: pd.DataFrame, api_key: str, use_gemini: bool = False):
        self.df = df
        self.api_key = api_key
        self.use_gemini = use_gemini
        self.llm = None
        self.pandas_ai = None
        self._initialize()

    def _initialize(self):
        if PandasAI is None:
            logger.warning('pandasai is unavailable: %s', pandasai_import_error)
            return

        self.llm = self._initialize_llm()
        self.pandas_ai = PandasAI(self.llm)

    def _initialize_llm(self):
        if self.use_gemini:
            try:
                from langchain.llms import GoogleGemini
                return GoogleGemini(api_key=self.api_key)
            except Exception as exc:
                logger.warning('Gemini client unavailable, falling back to OpenAI: %s', exc)
        if PandasAIOpenAI is None:
            raise RuntimeError('OpenAI support is unavailable in the current environment.')
        return PandasAIOpenAI(api_token=self.api_key)

    def ask_question(self, question: str):
        if self.pandas_ai is None:
            return 'Chat mode is unavailable because pandasai could not be imported in this environment.'

        try:
            response = self.pandas_ai.run(self.df, prompt=question)
            return str(response)
        except Exception as exc:
            logger.error('Chat query failed: %s', exc)
            return f'Error: {exc}'

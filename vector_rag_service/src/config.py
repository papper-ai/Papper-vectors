from pathlib import Path

from openai import AsyncClient
from pydantic_settings import BaseSettings
import logging

BASE_DIR = Path(__file__).parent

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class Settings(BaseSettings):
    URL_TO_LLM: str
    URL_TO_EMBEDDINGS: str
    _llm_client = None

    @property
    def openai_client(self) -> AsyncClient:
        return self._llm_client

    @openai_client.setter
    def openai_client(self, value):
        self._llm_client = value

    class Config:
        extra = "ignore"


settings = Settings()

openai_client = AsyncClient(base_url=settings.URL_TO_LLM, api_key="password")
settings.openai_client = openai_client

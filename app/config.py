from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI RAG Service"
    environment: str = "development"
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_device: int = -1
    llm_max_new_tokens: int = 256
    llm_temperature: float = 0.2
    llm_do_sample: bool = False

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

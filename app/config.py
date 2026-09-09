from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI RAG Service"
    environment: str = "development"
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    embedding_dimension: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

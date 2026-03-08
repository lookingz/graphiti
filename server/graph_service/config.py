from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore


class Settings(BaseSettings):
    openai_api_key: str | None = Field(None)
    openai_base_url: str | None = Field(None)
    model_name: str | None = Field(None)
    embedding_model_name: str | None = Field(None)
    embedding_dim: int = Field(1536)

    # Generic settings for OpenRouter or other providers
    api_key: str | None = Field(None)
    base_url: str | None = Field(None)

    embedding_api_key: str | None = Field(None)
    embedding_base_url: str | None = Field(None)

    # Database backend: "neo4j" or "falkordb"
    db_backend: str = Field('neo4j')

    # Neo4j configuration
    neo4j_uri: str = Field('bolt://localhost:7687')
    neo4j_user: str = Field('neo4j')
    neo4j_password: str = Field('password')

    # FalkorDB configuration
    falkordb_host: str = Field('localhost')
    falkordb_port: int = Field(6379)
    falkordb_user: str | None = Field(None)
    falkordb_password: str | None = Field(None)

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


@lru_cache
def get_settings():
    return Settings()  # type: ignore[call-arg]


ZepEnvDep = Annotated[Settings, Depends(get_settings)]

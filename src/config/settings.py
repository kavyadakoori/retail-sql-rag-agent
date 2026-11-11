from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "data" / "documents"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_ENV: str = Field(
        default="development",
        validation_alias=AliasChoices("APP_ENV", "app_env"),
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )
    API_BASE_URL: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias=AliasChoices("API_BASE_URL", "api_base_url"),
    )

    AI_GATEWAY_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AI_GATEWAY_URL", "ai_gateway_url"
        ),
    )
    AI_GATEWAY_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AI_GATEWAY_API_KEY", "ai_gateway_api_key"
        ),
    )
    AI_GATEWAY_MODEL: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AI_GATEWAY_MODEL", "ai_gateway_model"
        ),
    )

    MYSQL_HOST: str | None = Field(
        default=None, validation_alias=AliasChoices("MYSQL_HOST", "mysql_host")
    )
    MYSQL_PORT: int = Field(
        default=3306, validation_alias=AliasChoices("MYSQL_PORT", "mysql_port")
    )
    MYSQL_DATABASE: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MYSQL_DATABASE", "mysql_database"),
    )
    MYSQL_USER: str | None = Field(
        default=None, validation_alias=AliasChoices("MYSQL_USER", "mysql_user")
    )
    MYSQL_PASSWORD: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MYSQL_PASSWORD", "mysql_password"),
    )

    QDRANT_URL: str | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_URL", "qdrant_url")
    )
    QDRANT_API_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY", "qdrant_api_key"),
    )
    QDRANT_COLLECTION: str = Field(
        default="retailiq_documents",
        validation_alias=AliasChoices(
            "QDRANT_COLLECTION", "qdrant_collection"
        ),
    )
    QDRANT_VECTOR_SIZE: int = Field(
        default=1536,
        validation_alias=AliasChoices(
            "QDRANT_VECTOR_SIZE", "qdrant_vector_size"
        ),
    )
    QDRANT_LOCAL_PATH: Path = Field(
        default=BASE_DIR / ".local" / "qdrant",
        validation_alias=AliasChoices(
            "QDRANT_LOCAL_PATH", "qdrant_local_path"
        ),
    )

    MAX_FILE_SIZE_BYTES: int = Field(default=15 * 1024 * 1024)


Config = AppConfig()

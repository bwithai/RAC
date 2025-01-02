from typing import Annotated, Any, Literal, Union

from pydantic import (
    AnyUrl,
    BeforeValidator,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors(value: Any) -> Union[list[str], str]:
    """
    Parse CORS origins from a string or list.

    Args:
        value (Any): The input value to parse.

    Returns:
        Union[list[str], str]: A list of origins or a single origin string.

    Raises:
        ValueError: If the input value is invalid.
    """
    if isinstance(value, str) and not value.startswith("["):
        return [origin.strip() for origin in value.split(",")]
    elif isinstance(value, (list, str)):
        return value
    raise ValueError(f"Invalid CORS origins: {value}")


class Settings(BaseSettings):
    # Model configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file="../.env",  # Path to the environment file
        image_db="../imageDB",  # Default image database path
        env_ignore_empty=True,  # Ignore empty environment variables
        extra="ignore",  # Ignore extra fields
    )

    # Application settings
    API_V1_STR: str = "/api/v1"  # Base API endpoint
    FRONTEND_HOST: str = "http://localhost:5173"  # Frontend host
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"  # App environment

    # CORS settings
    BACKEND_CORS_ORIGINS: Annotated[
        Union[list[AnyUrl], str], BeforeValidator(parse_cors)
    ] = []

    def IMAGE_DB(self) -> str:
        """Return the image database path."""
        return self.model_config.get("image_db", "")

    # Computed fields
    @computed_field
    @property
    def all_cors_origins(self) -> list[str]:
        """
        Combine backend CORS origins with the frontend host.

        Returns:
            list[str]: List of all CORS origins.
        """
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    PROJECT_NAME: str
    SQLITE_DB_PATH: str = ""  # SQLite database file path (optional)

    @computed_field
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """
        Generate the SQLAlchemy database URI.

        Returns:
            str: Database URI for SQLAlchemy.
        """
        if self.SQLITE_DB_PATH:
            return f"sqlite:///{self.SQLITE_DB_PATH}"
        return ""  # Return an empty string if no database path is provided.

    def print_settings(self) -> None:
        """
        Print all settings for debugging purposes.
        """
        for key, value in self.dict().items():
            print(f"{key}: {value}")


# Instantiate settings
settings = Settings()
# Uncomment to debug settings
# settings.print_settings()
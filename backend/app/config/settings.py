"""
Application configuration settings.
Centralizes all configuration in one place for easy modification.
"""
import os
from typing import List
from pathlib import Path


class Settings:
    """Application settings with sensible defaults for prototype."""

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # API Configuration
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Company Name Standardizer"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "API for normalizing and grouping similar company names"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".csv"]

    # Matching Algorithm Configuration
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "85.0"))

    # Common corporate suffixes to normalize
    CORPORATE_SUFFIXES: List[str] = [
        "inc", "incorporated", "corp", "corporation", "ltd", "limited",
        "llc", "llp", "co", "company", "group", "holdings", "partners",
        "enterprises", "international", "global", "industries", "solutions",
        "plc", "ag", "sa", "nv", "bv", "gmbh", "spa"
    ]

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = Path("logs")
    LOG_FILE: str = "app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def get_log_path(cls) -> Path:
        """Get full path to log file, creating directory if needed."""
        cls.LOG_DIR.mkdir(exist_ok=True)
        return cls.LOG_DIR / cls.LOG_FILE


# Singleton instance
settings = Settings()

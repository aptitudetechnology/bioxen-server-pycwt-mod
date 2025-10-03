"""Server configuration using Pydantic settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server configuration settings."""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Job management
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT_SECONDS: int = 3600
    
    # Backends
    DEFAULT_BACKEND: str = "sequential"
    AVAILABLE_BACKENDS: list[str] = ["sequential", "joblib"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()


class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "WeatherOdds Pro"

    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "weatherodds")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours

    # NASA APIs
    NASA_EARTHDATA_TOKEN: Optional[str] = os.getenv("NASA_EARTHDATA_TOKEN")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    GIOVANNI_API_KEY: Optional[str] = os.getenv("GIOVANNI_API_KEY")

    # NOAA
    NOAA_API_TOKEN: Optional[str] = os.getenv("NOAA_API_TOKEN")

    # Data Processing
    MAX_YEARS_HISTORICAL: int = 30
    FORECAST_MONTHS_AHEAD: int = 2

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"


settings = Settings()

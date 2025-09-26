from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "pdfDeepseeker API"
    database_url: str
    redis_url: str
    gemini_api_key: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # OCR
    languages: str = "en"  # comma-separated: "en,ch_sim"
    gpu: bool = True        # attempt CUDA, falls back to CPU automatically

    # Security — leave empty to disable auth
    api_key: str = ""

    # Tuning
    confidence_threshold: float = 0.4
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Tunnel — read by start.sh, not used by the server itself
    ngrok_domain: str = ""
    cf_tunnel: str = ""

    class Config:
        env_file = ".env"

    @property
    def language_list(self) -> list[str]:
        return [l.strip() for l in self.languages.split(",") if l.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

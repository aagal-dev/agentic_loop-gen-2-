from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ollama provider configurations
    ollama_model: str = "beam:cloud"
    ollama_max_token_output: int = 4096
    ollama_temperature: float = 0.2
    ollama_host: str | None = None

    # pre-defined prompt paths
    tool_agent_prompt_path: Path = BASE_DIR / "prompts" / "tool-agent.md"

    # MCP server variable configurations
    zapier_url: str | None = None
    zapier_token: str | None = None
  
settings = Settings()

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

    # Optional episodic-memory indexing providers
    cohere_api_key: str | None = None
    cohere_embed_model: str = "embed-v4.0"
    cohere_output_dimension: int = 1024
    #cohere_embed_url: str = "https://api.cohere.com/v2/embed"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "agentic_loop_gen_2_ep"
    #qdrant_vector_name: str | None = None

    # pre-defined prompt paths
    tool_agent_prompt_path: Path = BASE_DIR / "prompts" / "tool-agent.md"

    # MCP server variable configurations
    zapier_url: str | None = None
    zapier_token: str | None = None
  
settings = Settings()

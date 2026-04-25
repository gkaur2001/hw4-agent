"""
Configuration — loaded from environment / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3.1")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

KB_DIR: Path = Path(os.getenv("KB_DIR", "data/kb"))
OUTPUTS_DIR: Path = Path(os.getenv("OUTPUTS_DIR", "outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIEVED_DOCS: int = int(os.getenv("MAX_RETRIEVED_DOCS", "3"))

# Vector DB settings
CHROMA_DB_DIR: Path = Path(os.getenv("CHROMA_DB_DIR", "data/chroma_db"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Optional OpenAI fallback (not required)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

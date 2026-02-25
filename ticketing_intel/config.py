"""
Central configuration. Reads from environment / .env file.

Usage:
    from ticketing_intel.config import cfg
    print(cfg.embedding_model)
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; export vars manually


@dataclass
class Config:
    # Embedding
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    voyage_api_key: str = field(default_factory=lambda: os.getenv("VOYAGE_API_KEY", ""))

    # Data
    jira_dump_path: str = field(default_factory=lambda: os.getenv("JIRA_DUMP_PATH", "jira_1000.jsonl"))

    # MongoDB
    mongo_uri: str = field(default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    mongo_db:  str = field(default_factory=lambda: os.getenv("MONGO_DB",  "jiradump"))

    # Storage
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", ".cache/tickets.db")))

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.db_path = Path(self.db_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embeddings_path(self) -> Path:
        return self.cache_dir / "embeddings.npz"

    def validate(self):
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to .env or export it.")
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY is not set.")


cfg = Config()

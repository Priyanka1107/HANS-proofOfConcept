# app/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env from the project root automatically (current working directory)
# If you run commands from the project root, this will pick up .env correctly.
load_dotenv()


def _default_embedding_provider() -> str:
    # Prefer Cohere if key exists, otherwise local
    return "cohere" if os.getenv("COHERE_API_KEY") else "local"


def _default_generation_provider() -> str:
    """
    Prefer Anthropic if ANTHROPIC_API_KEY exists (your preferred setup),
    otherwise fall back to Cohere if COHERE_API_KEY exists,
    otherwise use extractive fallback.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("COHERE_API_KEY"):
        return "cohere"
    return "extractive"


def _default_embedding_dimension(provider: str) -> int:
    # Common dims:
    # - Cohere embed-multilingual-v3.0 -> 1024
    # - Local MiniLM-L6-v2 -> 384
    provider = (provider or "").strip().lower()
    return 1024 if provider == "cohere" else 384


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # ------------------------------------------------------------------
    # Vector DB (Qdrant Cloud)
    # ------------------------------------------------------------------
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "htw_documents_cohere_1024")

    # ------------------------------------------------------------------
    # Elasticsearch (optional / stub)
    # ------------------------------------------------------------------
    ELASTICSEARCH_URL: str = os.getenv("ELASTICSEARCH_URL", "")
    ELASTICSEARCH_API_KEY: str = os.getenv("ELASTICSEARCH_API_KEY", "")
    ELASTICSEARCH_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "htw_documents_local_384")

    # ------------------------------------------------------------------
    # Logging DB (optional)
    # ------------------------------------------------------------------
    POSTGRES_URL: str = os.getenv("POSTGRES_URL", "")

    # ------------------------------------------------------------------
    # Knowledge base (fixed snapshot)
    # ------------------------------------------------------------------
    KB_VERSION: str = os.getenv("KB_VERSION", "htw_en_kb_2026-01-25")
    MANIFEST_PATH: str = os.getenv("MANIFEST_PATH", "output/objects_manifest.jsonl")

    # ------------------------------------------------------------------
    # Provider routing
    # ------------------------------------------------------------------
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", _default_embedding_provider())
    GENERATION_PROVIDER: str = os.getenv("GENERATION_PROVIDER", _default_generation_provider())

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    # Cohere embed model (used when EMBEDDING_PROVIDER="cohere")
    COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")

    # Cohere chat model (only used if GENERATION_PROVIDER="cohere")
    COHERE_CHAT_MODEL: str = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")

    # Legacy / backwards compatibility (not used in current flow but kept)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "local-minilm")

    # Embedding dimension:
    # - If EMBEDDING_DIMENSION is set in env -> use it
    # - else -> default based on EMBEDDING_PROVIDER
    EMBEDDING_DIMENSION: int = int(
        os.getenv(
            "EMBEDDING_DIMENSION",
            str(
                _default_embedding_dimension(
                    os.getenv("EMBEDDING_PROVIDER", _default_embedding_provider())
                )
            ),
        )
    )

    # Cohere rerank model (used if COHERE_API_KEY exists; rerank call uses ClientV2)
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-english-v3.0")

    # Generation model:
    # - Used for Anthropic if GENERATION_PROVIDER="anthropic"
    # - Used for Cohere if GENERATION_PROVIDER="cohere" (via COHERE_CHAT_MODEL above)
    #GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "claude-3-haiku-20240307")
    GENERATION_MODEL: str = os.getenv(
    "GENERATION_MODEL",
    "mistral-small-latest" if GENERATION_PROVIDER == "mistral" else "claude-3-haiku-20240307",
    )
    # ------------------------------------------------------------------
    # Streamlit / frontend
    # ------------------------------------------------------------------
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

    # ------------------------------------------------------------------
    # Object-type priority (used in conflict resolution)
    # ------------------------------------------------------------------
    OBJECT_TYPE_PRIORITY = {
        "degree_program": 100,
        "deadline_rule": 90,
        "application_route_rule": 80,
        "fees_funding_rule": 70,
        "language_proof_rule": 60,
        "general_info": 10,
    }


config = Config()
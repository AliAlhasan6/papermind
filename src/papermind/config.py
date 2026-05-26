"""Centralized config so we read env vars in one place."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

_PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CHROMA_PATH     = str(_PROJECT_ROOT / "chroma_db")
KG_PATH         = str(_PROJECT_ROOT / "kg.pkl")
COLLECTION_NAME = "papers"

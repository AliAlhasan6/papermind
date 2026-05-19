"""Centralized config so we read env vars in one place."""
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

CHROMA_PATH     = "./chroma_db"
KG_PATH         = "./kg.pkl"
COLLECTION_NAME = "papers"

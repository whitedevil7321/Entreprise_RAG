import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_FALLBACK_API_KEY: str = os.getenv("GEMINI_FALLBACK_API_KEY")
    QDRANT_URL: str=os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY: str=os.getenv("QDRANT_API_KEY")
    QDRANT_PROJECT: str="enterprise_rag"

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL="meta-llama/llama-prompt-guard-2-22m"
    GROQ_FALLBACK_API_KEY: str = os.getenv("GROQ_FALLBACK_API_KEY")

settings=Settings()
     
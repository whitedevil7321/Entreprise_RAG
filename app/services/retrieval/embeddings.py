import time
from typing import Optional

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings


BATCH_SIZE = 50

# These dimensions must match the actual vectors returned by each model.
GEMINI_DIM = 3072
FALLBACK_DIM = 768

_GEMINI_MODEL_NAME = "models/gemini-embedding-2-preview"
_FALLBACK_MODEL_NAME = "all-mpnet-base-v2"

_active_model = None
_model_type: Optional[str] = None
_dim: Optional[int] = None


def _probe_gemini():
    """Initialize Gemini embeddings and verify that an embedding call works."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model=_GEMINI_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY,
        )
        probe_vector = model.embed_query("embedding health check")

        if not probe_vector:
            raise RuntimeError("Gemini returned an empty embedding.")

        logfire.info(
            "Gemini embeddings are available",
            model=_GEMINI_MODEL_NAME,
            dimension=len(probe_vector),
        )
        return model

    except Exception as exc:
        logfire.warning(
            f"Gemini embeddings unavailable; using fallback model: {exc}"
        )
        return None


def _load_fallback():
    """Load the local Sentence Transformers fallback model."""
    from sentence_transformers import SentenceTransformer

    logfire.info(f"Loading fallback embedding model: {_FALLBACK_MODEL_NAME}")
    return SentenceTransformer(_FALLBACK_MODEL_NAME)


def _init() -> None:
    """Initialize the selected embedding model once per process."""
    global _active_model, _model_type

    if _active_model is not None:
        return

    gemini_model = _probe_gemini()

    if gemini_model is not None:
        _active_model = gemini_model
        _model_type = "Gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "Fallback"

    logfire.info(
        f"Active embedding model: {_model_type}",
        dimension=get_embedding_dim(),
    )


def get_embedding_model_name() -> str:
    """Return the name of the active embedding model."""
    _init()
    return _GEMINI_MODEL_NAME if _model_type == "Gemini" else _FALLBACK_MODEL_NAME


def get_embedding_dim() -> int:
    """Return the actual output dimension of the active embedding model."""
    global _dim
    _init()

    if _dim is None:
        if _model_type == "Gemini":
            # Derive the dimension from a real output once, then cache it.
            _dim = len(_active_model.embed_query("dimension check"))
        else:
            _dim = int(_active_model.get_sentence_embedding_dimension())
    return _dim


def embed_query(query: str) -> list[float]:
    """Embed one query using the active model."""
    _init()

    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    if _model_type == "Gemini":
        vector = _active_model.embed_query(query)
    else:
        vector = _active_model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

    expected_dim = get_embedding_dim()
    if len(vector) != expected_dim:
        raise ValueError(
            f"Query embedding dimension mismatch: expected {expected_dim}, "
            f"received {len(vector)}."
        )

    logfire.info(
        "Query embedded",
        model=_model_type,
        dimension=len(vector),
    )
    return vector


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a batch, retrying Gemini calls for transient rate-limit errors."""
    if not batch:
        return []

    if _model_type == "Gemini":
        for attempt in range(4):
            try:
                vectors = _active_model.embed_documents(batch)
                expected_dim = len(vectors[0]) if vectors else 0

                if any(len(vector) != expected_dim for vector in vectors):
                    raise ValueError("Gemini returned inconsistent vector dimensions.")

                return vectors

            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = any(
                    marker in message
                    for marker in ("429", "rate", "quota", "resource_exhausted")
                )

                if is_rate_limit and attempt < 3:
                    wait_seconds = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit; retrying in {wait_seconds}s."
                    )
                    time.sleep(wait_seconds)
                else:
                    logfire.error(f"Gemini batch embedding failed: {exc}")
                    raise

        raise RuntimeError("Gemini batch embedding failed after retries.")

    vectors = _active_model.encode(
        batch,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents in batches using the active embedding model."""
    _init()

    if not texts:
        return []

    cleaned_texts = [
        text.strip() for text in texts
        if isinstance(text, str) and text.strip()
    ]

    all_embeddings: list[list[float]] = []

    for start in range(0, len(cleaned_texts), BATCH_SIZE):
        batch = cleaned_texts[start : start + BATCH_SIZE]

        with logfire.span(
            "Embed Batch",
            model=_model_type,
            start=start,
            batch_size=len(batch),
        ):
            vectors = _embed_batch(batch)

        if vectors:
            expected_dim = len(vectors[0])
            if any(len(vector) != expected_dim for vector in vectors):
                raise ValueError("Batch contains inconsistent embedding dimensions.")

        all_embeddings.extend(vectors)

    return all_embeddings

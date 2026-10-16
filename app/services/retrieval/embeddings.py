import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import Settings as settings

BATCH_SIZE = 50
_GEMINI_DIM=3072
_FALLBACK_DIM=768   #all-mpnet-base-v2

_active_model = None
_model_type:str | None =None #Gemini or Fallback



def _probe_gemimi():
    """
    Try one embed call to verify Gemini is reacable. Returns model or None

    """
    try:
        model=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview",
                                           google_api_key=settings.GEMINI_API_KEY,)
        model.embed_query("probe")
        logfire.info(f"Gemini is active with model: gemini-embedding-2-preview")
        return model

    except Exception as e:
        logfire.error(f"Failed to initialize Gemini model: {e} will use fallback model instead.")
        return None

def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading fallback model: all-mpnet-base-v2")
    return SentenceTransformer('all-mpnet-base-v2')


def _init():
    global _active_model, _model_type

    if not _active_model:
        _active_model = _probe_gemimi()
        if not _active_model:
            _active_model = _load_fallback()
            _model_type="Fallback"
        else:
            _model_type="Gemini"
    return


def get_embedding_dim()->int:
    """
    Return the vector dimension for active model. If Gemini is active, return 3072, else return 768 for fallback model.
    call after _init() to ensure model is loaded.
    """
    _init()
    if _model_type == "Gemini":
        return _GEMINI_DIM
    else:
        return _FALLBACK_DIM


def _embed_batch(batch: list[str])->list[list[float]]:
    """
    Embed a batch of text using the active model. If Gemini is active, use it; otherwise, use the fallback model.
    """
    if _model_type == "Gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err=str(e).lower()
                is_rate_limit=any(x in err for x in ("429","rate","quota","resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait=2**attempt
                    logfire.warning(f"Gemini rate limit hit, retrying in {wait} seconds...")
                    time.sleep(wait)
                else:
                    logfire.error(f"Failed to embed batch with Gemini: {e}")
                    raise
        raise RuntimeError("Failed to embed batch with Gemini after retries.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()
    


def embed_query(query: str)-> list[float]:
    """
    This function is used to embed query for retrieval
    """
    _init()
    if _model_type == "Gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()

def embed_texts(texts:list[str])->list[list[float]]:
    """
    Embed a list of texts using the active model. If Gemini is active, use it; otherwise, use the fallback model.
    """
    _init()
    all_embeddings:list[list[float]] = []
    for i in range(0,len(texts),BATCH_SIZE):
        batch=texts[i:i+BATCH_SIZE]
        with logfire.span("Embed Batch",model=_model_type,start=i,batch_size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings









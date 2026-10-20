import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import Settings as settings
from app.services.retrieval.embeddings import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def search_enterprise_knowledge(query: str, limit: int = 15):
    """
    Search the Qdrant knowledge base and return the most relevant results.
    """
    try:
        
        logfire.info(f"Qdrant search query: {query}")

        query_vector = embed_query(query)

        info = client.get_collection(settings.QDRANT_COLLECTION)
        expected = info.config.params.vectors.size
        if len(query_vector) != expected:
            # Happens when Gemini embeddings fail and the 768-dim fallback is used
            # against a collection that was built with Gemini (or vice versa).
            logfire.error(
                f"Embedding size {len(query_vector)} does not match collection "
                f"size {expected}. Re-ingest with --wipe or fix the embedding model."
            )
            return []

        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results = []

        for res in response.points:
            payload = res.payload or {}
            content = payload.get("text", "")

            if not content.strip():
                continue

            results.append({
                "content": content,
                "source": payload.get("source", "Unknown"),
                "score": res.score,
            })

        # Log the actual results before reranking.
        for i, doc in enumerate(results, start=1):
            logfire.info(
                f"Qdrant result {i}: "
                f"score={doc['score']}, "
                f"source={doc['source']}, "
                f"content={doc['content'][:150]}"
            )

        return results

    except Exception:
        logfire.exception("Qdrant search failed")
        return []
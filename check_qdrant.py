
from app.config import settings
from app.services.retrieval.qdrant_service import (
    client,
    search_enterprise_knowledge,
)

query = "what is kubernetes"

print("\nQuery:", query)
print("\n--- Direct Qdrant Search Results ---")

results = search_enterprise_knowledge(query, limit=10)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Score:", result.get("score"))
    print("Source:", result.get("source"))
    print("Content:", result.get("content", "")[:500])

print("\n--- Collection Info ---")
info = client.get_collection(settings.QDRANT_COLLECTION)
print("Collection:", settings.QDRANT_COLLECTION)
print("Indexed points:", info.points_count)

import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state:AgentState):
    """
    Performs vector search and semantic reranking for technical Queries
    """
    query=state["current_query"]

    with logfire.span("Knowledge Retrival"):
        logfire.info(f"Searching Qdrant for {query}")
        raw_results=search_enterprise_knowledge(query,limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents=[doc['content'] for doc in raw_results]

        with logfire.span("Semantic Ranking"):
            ranked_contents=rerank_documents(query=query,documents=doc_contents,top_n=5)
            logfire.info("Reranking complete. kept top 5 most relevant documents")

        formatted_documents=[f"CONTENT: {doc}" for doc in ranked_contents]
        print("DOCUMENT TYPE:", type(formatted_documents))
        print("FIRST DOCUMENT:", type(formatted_documents[0])
            if formatted_documents else "EMPTY")
        print("DOCUMENT SAMPLE:", repr(formatted_documents[0])
            if formatted_documents else "EMPTY")
    return {
        "documents": formatted_documents,
        "status":f"Found Technical Content",
        "plan": state["plan"]+["Context Retrived"]
    }

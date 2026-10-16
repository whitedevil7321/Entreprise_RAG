import time
import logfire
from flashrank import Ranker,RerankRequest

_ranker=None

def _get_ranker()-> Ranker:
    """
    Initialize flashranker locally
    this Raranker uses local ONXX model for ultra fast reranking
    """
    global _ranker
    if _ranker is None:
        logfire.info("Initializing Flashrank model (Tinybert) Locally")
        try:
            _ranker=Ranker(cache_dir="/tmp/flashrank")

        except:
            _ranker=Ranker()
    return _ranker 

def rerank_documents(query:str,documents:list[str],top_n:int=5)-> list[str]:
    if not documents:
        return []
    start_time=time()

    logfire.log(f"Sending documents of Length: {len(documents)} for Reranking.")

    try:
        ranker=_get_ranker()
        passages=[
            {"id":i,"text":doc}
            for i ,doc in enumerate(documents)
        ]
        request=RerankRequest(query=query,passages=passages)
        results=ranker.rerank(request)

        reranked_documents=[]
        for res in results[:top_n]:
            reranked_documents.append(res['text'])

        duration=time.time()-start_time
        top_score=results[0]['score'] if results else 'N/A'
        logfire.info(f"Reranker Done in {duration:.2f}s. Top Semantic Score: {top_score}")
        return reranked_documents
    except Exception as e:
        logfire.error(f"[Reranker] semantic ranking Failed")
        return documents[:top_n]
    



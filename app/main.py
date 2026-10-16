import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))


from fastapi import FastAPI, Response
from app.agents.graph import rag_agent

from pydantic import BaseModel
from typing import Optional


app=FastAPI(title="Enterprise Rag Agent App")

class QueryRequrst(BaseModel):
    q:str
    thread_id: Optional[str]="default_user"

@app.get("/")
def home():
    return {"messages":"Enterprise Langgraph RAG api is live"}

@app.get("/graph")
def get_graph_image():
    """
    Return the mermaid image of the agent's workflow
    """
    png_bytes=rag_agent.get_graph().draw_mermaid_ong()
    return Response(content=png_bytes,)

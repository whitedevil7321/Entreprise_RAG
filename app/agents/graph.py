from langgraph.graph import StateGraph,END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.node.planner import planner_node
from app.agents.node.reponder import generate_node
from app.agents.node.retriever import retrieve_node


# Initialize StateGraph
workflow=StateGraph(AgentState)

# Define Nodes
workflow.add_node("planner",planner_node)
workflow.add_node("retriever",retrieve_node)
workflow.add_node("responder",generate_node)


# Define the Edges and route Planning
def route_planner(state:AgentState):
    """
    Route the Agent on the basis of planner's decision
        """
    if state["current_query"]=="CONVERSATIONAL":
        return "responder"
    return "retriever"

workflow.set_entry_point("planner")

# add conditional edges
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever":"retriever",
        "responder":"responder"
    }
)

workflow.add_edge("retriever","responder")
workflow.add_edge("responder",END)

checkpointer=MemorySaver()


rag_agent=workflow.compile(checkpointer=checkpointer)
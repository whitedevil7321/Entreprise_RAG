
from app.agents.state import AgentState
import logfire
from app.Gateway.Gateway_llm import Gateway_portkey


def planner_node(state: AgentState):
    """Decide whether retrieval is needed and generate a search query."""

    messages = state.get("messages", [])

    if not messages:
        return {
            "current_query": "CONVERSATIONAL",
            "status": "No message to process",
            "plan": [{"Intent": "Conversational/Memory", "Retrieval": "Skipped"}],
        }

    history = ""
    for msg in messages[:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history += f"{role}: {msg.get('content', '')}\n"

    user_message = str(messages[-1].get("content", "")).strip()

    prompt = f"""
You are a query planner for a technical knowledge assistant.

Conversation history:
{history}

Latest user message:
{user_message}

Rules:
- If the message is a greeting or can be answered using conversation history
  alone, output exactly: CONVERSATIONAL

- For technical questions about Kubernetes, Intel, or networking,
  output a concise search query that preserves the user's topic.
- For other factual or technical questions that need knowledge-base
  retrieval, output a concise search query.
- Never output a similarity score, numeric score, JSON, or explanation.
- Output only CONVERSATIONAL or the search query.
"""

    with logfire.span("Planner Decision"):
        # response = client.models.generate_content(
        #     model="gemini-3.7-flash",
        #     contents=prompt,
        #     )
        logfire.info("Portkey Planner Initiated")
        
        response =Gateway_portkey(prompt=prompt) 
        logfire.info("Portkey Planner Completed")
        # LLMs often wrap the answer in quotes or add a trailing period.
        decision = response.strip().strip('"\'`').rstrip('.').strip()
        print(f"Planner output: {decision}")
        logfire.info(f"Planner output: {decision}")

    if decision.upper() == "CONVERSATIONAL":
        print("PLANNER USER MESSAGE:", repr(user_message))
        print("PLANNER FINAL DECISION:", repr(decision))
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversation using memory",
            "plan": [
                {"Intent": "Conversational/Memory", "Retrieval": "Skipped"}
            ],
        }

    # Reject obviously malformed numeric-only queries.
    if not decision or not any(char.isalpha() for char in decision):
        decision = user_message
    print("PLANNER USER MESSAGE:", repr(user_message))
    print("PLANNER FINAL DECISION:", repr(decision))
    return {
        "current_query": decision,
        "status": f"Technical query required: {decision}",
        "plan": [{"Intent": "Technical", "Search Term": decision}],
    }


import logfire
from app.agents.state import AgentState
from app.Gateway.Gateway_llm import Gateway_portkey



def _get_message_parts(msg):
    """Extract role and content from supported message formats."""
    if isinstance(msg, dict):
        role = msg.get("role", msg.get("type", "assistant"))
        content = msg.get("content", "")
    elif isinstance(msg, str):
        role = "user"
        content = msg
    else:
        role = getattr(msg, "type", "assistant")
        content = getattr(msg, "content", str(msg))

    if role in ("human", "user"):
        role = "User"
    else:
        role = "Assistant"

    return role, content


def generate_node(state: AgentState):
    """
    Synthesizes a response using technical documentation
    and conversation history.
    """
    query = state["current_query"]
    messages = state.get("messages", [])

    history_lines = []
    for msg in messages:
        role, content = _get_message_parts(msg)
        history_lines.append(f"{role}: {content}")

    history_str = "\n".join(history_lines)

    if messages:
        _, user_msg = _get_message_parts(messages[-1])
    else:
        user_msg = query

    if query == "CONVERSATIONAL":
        logfire.info(
            "Generating conversational response using memory"
        )

        prompt = f"""
You are a friendly and helpful Enterprise AI Assistant.
Answer the user's latest message using the conversation history.


CONVERSATION HISTORY:
{history_str}

LATEST MESSAGE:
{user_msg}
"""
    else:
        logfire.info("Generating Technical RAG Response")

        max_context_chars = 25000
        full_context = ""

        for doc in state.get("documents", []):
            doc_text = str(doc)

            if len(full_context) + len(doc_text) <= max_context_chars:
                full_context += doc_text + "\n\n"
            else:
                logfire.warning(
                    "Context truncated to fit Groq TPM limits."
                )
                break

        prompt = f"""
You are a Senior Technical Architect.
Answer the question using the technical context provided.
If the context does not contain the answer, say so clearly.

TECHNICAL CONTEXT:
{full_context}

CONVERSATION HISTORY:
{history_str}

USER QUESTION:
{user_msg}
"""

    response = Gateway_portkey(prompt=prompt) 

    logfire.info(f"Final prompt characters: {len(prompt)}")
    logfire.info(f"Retrieved documents: {len(state.get('documents', []))}")
    logfire.info(f"History characters: {len(history_str)}")

    return {
        "final_answer": response,
        # Append the reply to memory so follow-up questions can see it.
        "messages": [{"role": "assistant", "content": response}],
        "status": "Response generated"
    }

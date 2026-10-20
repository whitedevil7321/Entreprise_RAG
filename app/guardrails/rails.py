import traceback
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import Settings as settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

from langchain_google_genai import ChatGoogleGenerativeAI
_rails: LLMRails | None = None


# Groq is much faster than Gemini for this short classification step.
# llama-3.1-8b-instant was shut down by Groq on 2026-08-16; gpt-oss-20b replaces it.
GUARDRAIL_GROQ_MODEL = "openai/gpt-oss-20b"
GUARDRAIL_GEMINI_MODEL = "gemini-3.7-flash"


def _build_guardrail_llm():
    """Groq if a key is set, otherwise Gemini. Both fail fast instead of hanging."""
    if settings.GROQ_API_KEY:
        logfire.info(f"Guardrails LLM: Groq {GUARDRAIL_GROQ_MODEL}")
        return ChatGroq(
            model=GUARDRAIL_GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            reasoning_effort="low",   # gpt-oss is a reasoning model; keep it quick
            reasoning_format="hidden",  # return only the answer text to NeMo
            max_retries=1,
            request_timeout=20,
        )

    logfire.info(f"Guardrails LLM: Gemini {GUARDRAIL_GEMINI_MODEL}")
    return ChatGoogleGenerativeAI(
        model=GUARDRAIL_GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
        thinking_level="low",
        max_retries=1,
        timeout=30,
    )



def initialize_rails() -> None:
    """Build the NeMo LLMRails singleton at app startup."""
    global _rails

    llm = _build_guardrail_llm()
    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=llm)
    logfire.info("🛡️ NeMo Guardrails initialised.")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:

            result = _rails.generate(messages=[{"role": "user", "content": message}])
            print(f"GuardRail Result:{result}")
        except Exception:
            # Don't take the whole API down if the guardrail LLM call fails.
            traceback.print_exc()
            logfire.exception("Guardrails check failed — letting request through")
            return False, None

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None

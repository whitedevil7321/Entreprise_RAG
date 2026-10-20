import logfire
from portkey_ai import Portkey
from langchain_groq import ChatGroq
from app.config import settings

# Primary: Gemini through Portkey.
# "@google" must match the provider slug in your Portkey Model Catalog.
MODEL = "@tripmytravels/openai/gpt-oss-120b"

# Fallback: Groq, used when Gemini is overloaded (503), times out (504) or errors.
FALLBACK_MODEL = "openai/gpt-oss-120b"

# request_timeout is in milliseconds; fail over to Groq after 45s.
_portkey = Portkey(api_key=settings.PORTKEY_API, request_timeout=45000)

_groq_keys = [k for k in (settings.GROQ_API_KEY, settings.GROQ_FALLBACK_API_KEY) if k]
_groq_clients = [
    ChatGroq(
        model=FALLBACK_MODEL,
        api_key=k,
        temperature=0.2,
        reasoning_effort="low",
        reasoning_format="hidden",  # return only the answer, not the reasoning
        max_retries=1,
        request_timeout=60,
    )
    for k in _groq_keys
]


def _call_gemini(prompt: str) -> str:
    response = _portkey.chat.completions.create(
        model=MODEL,
        # Gemini rejects requests with only a system message; send it as a user turn.
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def _call_groq(prompt: str) -> str:
    last_error = None
    for client in _groq_clients:
        try:
            content = client.invoke(prompt).content
            if isinstance(content, list):  # defensive: join text parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return (content or "").strip()
        except Exception as e:
            last_error = e
            logfire.warning(f"Groq fallback key failed: {e}")
    raise RuntimeError(f"All Groq fallback keys failed: {last_error}")


def Gateway_portkey(prompt: str) -> str:
    if settings.PORTKEY_API:
        try:
            return _call_gemini(prompt)
        except Exception as e:
            if not _groq_clients:
                raise
            logfire.warning(f"Gemini via Portkey failed, falling back to Groq: {e}")
    elif not _groq_clients:
        raise RuntimeError("Neither PORTKEY_API_KEY nor GROQ_API_KEY is set in .env")

    with logfire.span("Groq fallback", model=FALLBACK_MODEL):
        return _call_groq(prompt)

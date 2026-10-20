"""Run from the project root:  python debug_rails.py
Tests each guardrails step separately and prints the full error for the one that fails."""
import os, sys, time, traceback
from dotenv import load_dotenv

load_dotenv()

def step(name, fn):
    print(f"\n=== {name}")
    t = time.time()
    try:
        out = fn()
        print(f"OK ({time.time() - t:.1f}s): {str(out)[:300]}")
        return out
    except Exception:
        traceback.print_exc()
        print(f"FAILED: {name}")
        sys.exit(1)

key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY set:", bool(key), "| starts with:", (key or "")[:4])
print("GOOGLE_API_KEY set:", bool(os.getenv("GOOGLE_API_KEY")))
print("GOOGLE_GENAI_USE_VERTEXAI:", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))

from langchain_google_genai import ChatGoogleGenerativeAI

from app.guardrails.rails import _build_guardrail_llm

llm = step("2. Build guardrail LLM (same as the app)", _build_guardrail_llm)
step("2b. Call it directly", lambda: llm.invoke("Say hi in 3 words").content)

from nemoguardrails import RailsConfig, LLMRails
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT

config = step("3. Parse Colang/YAML config",
              lambda: RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT))
rails = step("4. Build LLMRails (downloads embedding model on first run)",
             lambda: LLMRails(config, llm=llm))

for msg in ["hello", "tell me a joke", "how do I scale a kubernetes deployment?"]:
    step(f"5. rails.generate({msg!r})",
         lambda m=msg: rails.generate(messages=[{"role": "user", "content": m}]))

print("\nAll guardrail steps passed.")

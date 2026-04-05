"""
Enhanced RAG assistant for PyXspec/XSPEC/CLOUDY/Chandra manuals.

Extends the basic rag_answer() approach with three capabilities:
  1. Manual lookup (your existing RAG) — primary source, always consulted first
  2. Python code execution — for physical calculations (unit conversions, derived quantities, etc.)
  3. Fallback general knowledge — for debugging Python errors not covered by the manual,
     with a hard constraint against inventing library-specific API functions

Usage
-----
  from rag_with_tools import tool_rag_answer

  answer = tool_rag_answer(
      question="Why is my PyXspec code hanging?",
      all_chunk_embeddings=all_chunk_embeddings,
      all_chunks=all_chunks,
      embedding_model=embedding_model,
      client=client,
  )
  print(answer)

Dependencies
------------
  pip install anthropic sentence-transformers numpy
"""

import json
import io
import sys
import traceback
import numpy as np
import anthropic


# ---------------------------------------------------------------------------
# Similarity helpers (same as your existing code)
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Dot product of two unit vectors == cosine similarity."""
    return float(np.dot(vec1, vec2))


def search_chunks(
    query: str,
    embedding_model,
    chunk_embeddings: list,
    chunks: list,
    top_k: int = 10,
) -> list[dict]:
    """Return the top_k most relevant chunks for *query*."""
    query_embedding = embedding_model.encode(query)
    similarities = [
        cosine_similarity(query_embedding, emb) for emb in chunk_embeddings
    ]
    ranked = sorted(
        enumerate(similarities), key=lambda x: x[1], reverse=True
    )[:top_k]
    return [
        {"chunk": chunks[idx], "similarity": sim, "index": idx}
        for idx, sim in ranked
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search_manual(
    query: str,
    top_k: int,
    min_similarity: float,
    embedding_model,
    chunk_embeddings: list,
    chunks: list,
) -> dict:
    """Called when Claude invokes the 'search_manual' tool."""
    results = search_chunks(query, embedding_model, chunk_embeddings, chunks, top_k=top_k)
    relevant = [r for r in results if r["similarity"] >= min_similarity]
    if not relevant:
        return {
            "found": False,
            "message": f"No manual sections found with similarity >= {min_similarity}.",
            "sections": [],
        }
    sections = []
    for r in relevant:
        text = r["chunk"][:2000]
        # End on a complete sentence
        end = max(text.rfind("."), text.rfind("?"), text.rfind("!"))
        if end > 0:
            text = text[: end + 1]
        sections.append({"similarity": round(r["similarity"], 3), "text": text})
    return {"found": True, "sections": sections}


def _tool_run_python(code: str) -> dict:
    """Execute *code* in a restricted sandbox and return stdout/result/error."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Allowed builtins — keep it minimal but useful for science calculations
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "bool": bool,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "True": True,
            "False": False,
            "None": None,
        },
        "np": np,
        # Common math shortcuts Claude is likely to use
        "pi": np.pi,
        "e": np.e,
        "sqrt": np.sqrt,
        "log": np.log,
        "log10": np.log10,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
    }
    local_vars: dict = {}

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    try:
        exec(compile(code, "<sandbox>", "exec"), safe_globals, local_vars)
        output = stdout_capture.getvalue()
        err = stderr_capture.getvalue()
        return {"success": True, "stdout": output, "stderr": err, "locals": {
            k: repr(v) for k, v in local_vars.items() if not k.startswith("_")
        }}
    except Exception:
        return {
            "success": False,
            "error": traceback.format_exc(),
            "stdout": stdout_capture.getvalue(),
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# Tool schemas for the Anthropic API
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_manual",
        "description": (
            "Search the loaded documentation manuals "
            "for sections relevant to the query. "
            "Always call this first before using any other knowledge. "
            "Returns the most similar manual sections with their similarity scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query to find relevant manual sections.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top sections to retrieve (default 8).",
                    "default": 8,
                },
                "min_similarity": {
                    "type": "number",
                    "description": (
                        "Minimum cosine similarity threshold (0–1). "
                        "Sections below this are excluded. Default 0.25."
                    ),
                    "default": 0.25,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute a small Python snippet and return its stdout output. "
            "Use this for physical calculations, unit conversions, or numerical derivations. "
            "numpy is available as `np`. Do NOT use this to simulate PyXspec/XSPEC/CLOUDY calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print() to produce output.",
                }
            },
            "required": ["code"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a scientific software assistant specialising in X-ray astronomy analysis
with PyXspec, XSPEC, CLOUDY, and the Chandra observatory.

Your response strategy — follow this order strictly:

1. ALWAYS call search_manual first to look for relevant documentation.
2. If the manual sections returned have similarity >= 0.3 and directly answer the
   question, base your answer on them. Cite section content explicitly.
3. If the manual sections are NOT relevant enough (similarity < 0.3, or simply do
   not address the question):
   a. For CALCULATIONS (unit conversions, derived quantities, numerical results):
      use run_python to compute the answer exactly. Show the code and result.
   b. For DEBUGGING Python errors (AttributeError, ImportError, hangs, etc.):
      apply general Python debugging knowledge. Describe likely causes and fixes
      clearly without referencing any PyXspec/XSPEC/CLOUDY API unless you confirmed
      the function/attribute name from the manual text you retrieved.
4. NEVER invent function names, method names, or parameter names for PyXspec,
   XSPEC, CLOUDY, or any other specialised library. If you are unsure whether an
   API exists, say so explicitly and suggest the user check the relevant manual
   section you found (or note that no manual section was found).

Be concise, precise, and always distinguish between information from the manual
and information from your general knowledge.
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tool_rag_answer(
    question: str,
    all_chunk_embeddings: list,
    all_chunks: list,
    embedding_model,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1500,
    max_tool_rounds: int = 6,
    verbose: bool = True,
) -> str:
    """
    Answer *question* using an agentic loop with three tools:
      - search_manual  (RAG over your loaded chunks)
      - run_python     (sandboxed calculation executor)

    Parameters
    ----------
    question            : the user's question
    all_chunk_embeddings: list of numpy embeddings (one per chunk)
    all_chunks          : list of text chunks (parallel to embeddings)
    embedding_model     : SentenceTransformer instance
    client              : anthropic.Anthropic client
    model               : Claude model string
    max_tokens          : max tokens per API call
    max_tool_rounds     : safety limit on agentic loop iterations
    verbose             : print tool call trace

    Returns
    -------
    str : Claude's final answer
    """
    messages = [{"role": "user", "content": question}]

    for round_num in range(max_tool_rounds):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text blocks for the final answer
        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # If Claude is done (no more tool calls), return its answer
        if response.stop_reason == "end_turn" or not tool_use_blocks:
            return "\n".join(text_blocks) if text_blocks else "(No answer generated)"

        # Append Claude's response (with tool use) to the message history
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call and collect results
        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input

            if verbose:
                print(f"[Tool call] {tool_name}({json.dumps(tool_input)[:120]}...)")

            if tool_name == "search_manual":
                result = _tool_search_manual(
                    query=tool_input.get("query", question),
                    top_k=tool_input.get("top_k", 8),
                    min_similarity=tool_input.get("min_similarity", 0.25),
                    embedding_model=embedding_model,
                    chunk_embeddings=all_chunk_embeddings,
                    chunks=all_chunks,
                )
            elif tool_name == "run_python":
                result = _tool_run_python(tool_input.get("code", ""))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            if verbose:
                preview = json.dumps(result)[:200]
                print(f"[Tool result] {preview}...")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        # Feed tool results back to Claude
        messages.append({"role": "user", "content": tool_results})

    return "Maximum tool-call rounds reached without a final answer."


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Paste your existing setup here ---
    # from sentence_transformers import SentenceTransformer
    # import numpy as np, json, anthropic
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # data = np.load('all_chunk_embeddings.npz')
    # all_chunk_embeddings = list(data['embeddings'])
    # with open('all_chunks_metadata.json') as f:
    #     all_chunks = json.load(f)['chunks']
    # client = anthropic.Anthropic()

    # Example questions that exercise all three capabilities:
    DEMO_QUESTIONS = [
        # Answered purely from manual
        "What does Fit.query = 'no' do in PyXspec?",
        # Requires physical calculation
        "What is the energy resolution in eV of Chandra HETG at 6.7 keV if E/dE = 800?",
        # Debugging: not in manual, but Claude should use general Python knowledge
        "My PyXspec loop hangs silently after a few iterations. The loop calls "
        "AllData.clear() and AllModels.clear() each time. What are common causes?",
    ]

    for q in DEMO_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"QUESTION: {q}")
        print("=" * 70)
        # answer = tool_rag_answer(q, all_chunk_embeddings, all_chunks, embedding_model, client)
        # print(answer)
        print("(Uncomment the lines above after loading your embeddings)")

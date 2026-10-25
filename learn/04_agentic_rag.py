"""
LESSON 4 — Agentic RAG with LangGraph (loops + branches)

THE BIG IDEA
------------
Lesson 3 was a STRAIGHT LINE: retrieve -> prompt -> answer, exactly once.
But what if the first retrieval is bad (low similarity)? A linear pipeline just
answers from weak context. A smarter app would notice, REPHRASE the question, and
search AGAIN before answering. That "notice -> go back and retry" is a LOOP, and a
loop can't be written as a simple `prompt | llm` pipe. That's what LangGraph adds.

LangGraph lets you wire steps ("nodes") into a GRAPH with conditional edges, so
flow can branch and cycle. The three concepts:
  - STATE : a shared dict passed between nodes; each node reads/updates it.
  - NODES : plain functions that do work (retrieve, grade, rewrite, generate).
  - EDGES : connections — including CONDITIONAL ones that decide where to go next.

THE GRAPH WE BUILD
------------------
                     ┌───────────────── (score too low & tries left) ──────────────┐
                     ▼                                                              │
    START → retrieve → grade ──(good enough OR out of tries)──→ generate → END      │
                     ▲                                                              │
                     └──────────────────── rewrite ◄────────────────────────────────┘

The building blocks (embed, cosine, retrieve) are the SAME hand-written ones from
lessons 1-3 — LangGraph only adds the control flow around them.

RUN IT
------
    uv run learn/04_agentic_rag.py
(Needs:  ollama pull nomic-embed-text  AND  ollama pull llama3.1:8b)
"""
import math
from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"

# If the best retrieved chunk scores below this, we consider retrieval "weak" and
# try rewriting the question. Tune this to see the loop trigger more/less often.
SCORE_THRESHOLD = 0.72
MAX_ATTEMPTS = 2  # how many times we'll rewrite+retry before giving up and answering

KNOWLEDGE_BASE = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language created by Guido van Rossum.",
    "The Great Wall of China is over 13,000 miles long.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Paris is the capital city of France.",
    "Mount Everest is the tallest mountain on Earth.",
]


# --- The familiar building blocks from lessons 1-3 --------------------------
def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# Index the knowledge base once, at import time (ingestion).
INDEX = [(chunk, embed(chunk)) for chunk in KNOWLEDGE_BASE]


# --- The shared STATE that flows through the graph --------------------------
class RAGState(TypedDict):
    question: str          # the (possibly rewritten) query used for retrieval
    original_question: str  # the user's first question, kept for the final prompt
    chunks: list           # retrieved chunk texts
    best_score: float      # similarity of the best retrieved chunk
    attempts: int          # how many rewrites we've done
    answer: str            # the final generated answer


# --- NODES: each is just a function (state in -> partial state out) ----------
def retrieve_node(state: RAGState) -> dict:
    """Embed the current question and grab the top-3 chunks + the best score."""
    query_vec = embed(state["question"])
    scored = [(cosine_similarity(query_vec, vec), chunk) for chunk, vec in INDEX]
    scored.sort(reverse=True)
    top = scored[:3]
    best = top[0][0]
    print(f"  [retrieve] query={state['question']!r}  best_score={best:.3f}")
    return {"chunks": [c for _, c in top], "best_score": best}


def rewrite_node(state: RAGState) -> dict:
    """Ask the LLM to rephrase the question to retrieve better, and count the try."""
    prompt = (
        "Rewrite the following question to be clearer and more specific, so it "
        "better matches an encyclopedia. Reply with ONLY the rewritten question.\n\n"
        f"Question: {state['original_question']}"
    )
    rewritten = ollama.chat(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}]
    )["message"]["content"].strip()
    print(f"  [rewrite]  attempt {state['attempts'] + 1}: {rewritten!r}")
    return {"question": rewritten, "attempts": state["attempts"] + 1}


def generate_node(state: RAGState) -> dict:
    """Build the grounded prompt from the retrieved chunks and answer."""
    context = "\n".join(f"- {c}" for c in state["chunks"])
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_question']}\n\nAnswer:"
    )
    answer = ollama.chat(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]
    print("  [generate] produced the final answer")
    return {"answer": answer}


# --- The CONDITIONAL EDGE: this is the "decision" that makes it agentic ------
def should_retry(state: RAGState) -> str:
    """After grading, decide the next node: loop back to 'rewrite' or go 'generate'.

    Returns a string that names the next node. This single function is what turns
    a straight line into a graph with a loop.
    """
    good_enough = state["best_score"] >= SCORE_THRESHOLD
    out_of_tries = state["attempts"] >= MAX_ATTEMPTS
    if good_enough or out_of_tries:
        reason = "score is good" if good_enough else "out of retries"
        print(f"  [decision] {reason} -> generate")
        return "generate"
    print(f"  [decision] score {state['best_score']:.3f} < {SCORE_THRESHOLD} -> rewrite & retry")
    return "rewrite"


def build_graph():
    """Wire the nodes and edges into a runnable graph."""
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "retrieve")            # always start by retrieving
    # After retrieve, the conditional edge picks the next node by name:
    g.add_conditional_edges("retrieve", should_retry,
                            {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")        # the LOOP: rewrite -> retrieve again
    g.add_edge("generate", END)              # generating ends the run
    return g.compile()


def ask(graph, question):
    print(f"\n{'=' * 70}\nQUESTION: {question}\n{'=' * 70}")
    final = graph.invoke({
        "question": question,
        "original_question": question,
        "chunks": [], "best_score": 0.0, "attempts": 0, "answer": "",
    })
    print(f"\nFINAL ANSWER: {final['answer']}")


if __name__ == "__main__":
    graph = build_graph()

    # Clear question — retrieval is strong on the first try, so NO loop (straight
    # through: retrieve -> generate). Same path as lesson 3.
    ask(graph, "What is the capital of France?")

    # Vague question — the first retrieval is likely weak, so watch the graph LOOP:
    # retrieve -> (low score) -> rewrite -> retrieve again -> generate.
    ask(graph, "where's that big iron tower everyone photographs?")

    print("\nTakeaway: the SAME embed/retrieve/generate pieces from lesson 3, but")
    print("LangGraph let us add a decision + a loop. That ability to branch and")
    print("cycle (and keep state across steps) is what turns a pipeline into an")
    print("'agent'. app.py is the linear version; this is the graph version.")

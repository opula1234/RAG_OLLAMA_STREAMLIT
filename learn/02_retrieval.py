"""
LESSON 2 — Retrieval: finding the most relevant text

THE BIG IDEA
------------
In lesson 1 we learned how to measure how related two texts are. "Retrieval" just
applies that at scale:

    1. Ahead of time, embed every piece of our knowledge ("chunks") into vectors.
    2. When a question comes in, embed the question too.
    3. Compare the question's vector to every chunk's vector.
    4. Return the top-k closest chunks.

That's a "vector search". Real systems use a vector DATABASE (Chroma, Pinecone...)
to do step 3 fast over millions of chunks — but the logic is identical to what we
write here by hand. A vector DB is just a fast version of this loop.

WHAT THIS SCRIPT DOES
---------------------
Builds a tiny in-memory "knowledge base" of facts, embeds them once, then lets a
query retrieve the most relevant ones.

RUN IT
------
    uv run learn/02_retrieval.py
"""
import math

import ollama

EMBED_MODEL = "nomic-embed-text"

# Our "knowledge base": just a list of short facts. In a real app these would be
# chunks extracted from your PDF. Starting with plain strings keeps the mechanics
# visible — no file loading to distract us.
KNOWLEDGE_BASE = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language created by Guido van Rossum.",
    "The Great Wall of China is over 13,000 miles long.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The mitochondria is the powerhouse of the cell.",
    "Paris is the capital city of France.",
    "JavaScript is mainly used for building interactive websites.",
    "Mount Everest is the tallest mountain on Earth.",
]


def embed(text):
    """Turn text into a vector (see lesson 1)."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def cosine_similarity(a, b):
    """Direction-closeness of two vectors; higher = more related (see lesson 1)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def build_index(chunks):
    """Embed every chunk ONCE and keep (text, vector) pairs.

    This is the "indexing"/"ingestion" step. Doing it ahead of time means queries
    only need to embed the question, not the whole knowledge base, every time.
    """
    print(f"Indexing {len(chunks)} chunks (embedding each one)...")
    return [(chunk, embed(chunk)) for chunk in chunks]


def retrieve(query, index, top_k=3):
    """Return the `top_k` chunks whose vectors are closest to the query's vector."""
    query_vec = embed(query)
    scored = [
        (cosine_similarity(query_vec, vec), chunk)
        for chunk, vec in index
    ]
    scored.sort(reverse=True)   # highest similarity first
    return scored[:top_k]


if __name__ == "__main__":
    index = build_index(KNOWLEDGE_BASE)

    for query in ["What is the capital of France?",
                  "Tell me about programming languages."]:
        print(f"\nQUERY: {query}")
        print("Top matches (score · chunk):")
        for score, chunk in retrieve(query, index, top_k=3):
            print(f"  {score:.3f}  {chunk}")

    print("\nTakeaway: notice the right facts float to the top by MEANING, not by")
    print("shared keywords ('capital'/'France' pulls in both Paris facts). Those")
    print("top chunks are what we'll feed to the LLM in lesson 3.")

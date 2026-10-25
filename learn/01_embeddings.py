"""
LESSON 1 — What is an embedding?

THE BIG IDEA
------------
A computer can't compare the *meaning* of two sentences directly. So we use an
"embedding model" to turn each piece of text into a list of numbers (a "vector").
The trick: the model is trained so that texts with SIMILAR MEANING get vectors
that point in a SIMILAR DIRECTION.

If we can measure how close two vectors point, we can measure how related two
texts are — without any keyword matching. That measurement is the whole basis of
"retrieval" in RAG (next lesson).

WHAT THIS SCRIPT DOES
---------------------
1. Turns a few sentences into vectors using a local Ollama embedding model.
2. Shows you what a vector actually looks like.
3. Measures similarity between sentences with "cosine similarity" — computed by
   hand in plain Python so nothing is hidden.

RUN IT
------
    uv run learn/01_embeddings.py
(Needs Ollama running and the model pulled:  ollama pull nomic-embed-text)
"""
import math

import ollama

EMBED_MODEL = "nomic-embed-text"


def embed(text):
    """Ask Ollama to turn `text` into a vector (a list of floats)."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def cosine_similarity(a, b):
    """How closely do two vectors point in the same direction? Range ~ -1..1.

    cosine = (a · b) / (|a| * |b|)
      - a · b   is the "dot product": multiply matching elements, add them up.
      - |a|     is the vector's length (sqrt of the sum of its squares).
    1.0 = identical direction (very similar meaning), 0 = unrelated.
    This is the entire math behind semantic search — that's it.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    # --- Step 1: look at a single embedding ---------------------------------
    sentence = "The cat sat on the mat."
    vector = embed(sentence)
    print(f"Sentence : {sentence!r}")
    print(f"Vector   : length={len(vector)} numbers")
    print(f"First 5  : {[round(x, 4) for x in vector[:5]]}")
    print("           ^ every sentence becomes a list of this many numbers.\n")

    # --- Step 2: compare meanings -------------------------------------------
    # Notice: none of these share many keywords, yet the model still knows which
    # pairs are related in MEANING.
    query = "A kitten is resting on a rug."
    candidates = [
        "The cat sat on the mat.",          # same meaning, different words
        "Dogs love to play fetch.",          # loosely related (animals)
        "The stock market fell sharply.",
        "The mat was sat on by the cat."# unrelated
    ]

    query_vec = embed(query)
    print(f"QUERY: {query!r}\n")
    print("Similarity of the query to each candidate (higher = more related):")

    scored = []
    for text in candidates:
        score = cosine_similarity(query_vec, embed(text))
        scored.append((score, text))

    # Sort most-similar first, so the ranking is obvious.
    for score, text in sorted(scored, reverse=True):
        print(f"  {score:.3f}  {text}")

    print("\nTakeaway: the kitten/cat line scores highest even though it shares")
    print("almost no words with the query. THAT is semantic similarity — and it")
    print("is exactly how RAG finds relevant text. On to lesson 2.")

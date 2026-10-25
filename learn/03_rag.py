"""
LESSON 3 — Full RAG: Retrieval-Augmented Generation

THE BIG IDEA
------------
An LLM on its own answers from its training data — it doesn't know YOUR documents,
and it can "hallucinate" (make things up). RAG fixes this by changing the question
we ask the model. Instead of:

    "What is the capital of France?"

we retrieve relevant facts first (lesson 2) and ask:

    "Using ONLY these facts: <retrieved chunks>, answer: What is the capital...?"

That's the whole trick. "Retrieval-Augmented" = we augment the prompt with
retrieved context. "Generation" = the LLM writes the answer. The model is now
grounded in OUR text, so answers are accurate and traceable to a source.

THE FULL PIPELINE (this is RAG, end to end)
-------------------------------------------
    question
      -> embed question                      (lesson 1)
      -> retrieve top-k relevant chunks      (lesson 2)
      -> build a prompt = instructions + chunks + question   (this file)
      -> LLM generates an answer from that prompt            (this file)

RUN IT
------
    uv run learn/03_rag.py
(Needs:  ollama pull nomic-embed-text  AND  ollama pull llama3.1:8b)
"""
import math

import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"

KNOWLEDGE_BASE = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language created by Guido van Rossum.",
    "The Great Wall of China is over 13,000 miles long.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Paris is the capital city of France.",
    "Mount Everest is the tallest mountain on Earth."
]


def embed(text):
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def retrieve(query, index, top_k=3):
    """Top-k most relevant chunks (see lesson 2)."""
    query_vec = embed(query)
    scored = [(cosine_similarity(query_vec, vec), chunk) for chunk, vec in index]
    scored.sort(reverse=True)
    return scored[:top_k]


def build_prompt(question, chunks):
    """Glue the retrieved chunks and the question into one grounded prompt.

    The 'ONLY the context' instruction is what keeps the model honest — if the
    answer isn't in our chunks, we want it to say so instead of inventing one.
    """
    context = "\n".join(f"- {c}" for c in chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate(prompt):
    """Send the prompt to the local chat model and return its answer."""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def ask(question, index):
    """One full RAG turn — and we print every intermediate step so you can SEE it."""
    print(f"\n{'=' * 70}\nQUESTION: {question}\n{'=' * 70}")

    # 1) RETRIEVE
    results = retrieve(question, index, top_k=3)
    print("\n[1] RETRIEVED CONTEXT (what we found, most relevant first):")
    for score, chunk in results:
        print(f"    {score:.3f}  {chunk}")
    chunks = [chunk for _, chunk in results]

    # 2) AUGMENT (build the grounded prompt)
    prompt = build_prompt(question, chunks)
    print("\n[2] PROMPT SENT TO THE LLM:")
    print("    " + prompt.replace("\n", "\n    "))

    # 3) GENERATE
    answer = generate(prompt)
    print("\n[3] LLM ANSWER:")
    print(f"    {answer}")


if __name__ == "__main__":
    # Index the knowledge base once (ingestion), then ask questions.
    print("Indexing knowledge base...")
    index = [(chunk, embed(chunk)) for chunk in KNOWLEDGE_BASE]

    ask("What is the capital of France?", index)
    # This one is NOT in our knowledge base — watch the model say it doesn't know,
    # instead of hallucinating. That honesty is the payoff of grounding.
    ask("Who won the 2022 World Cup?", index)
    
    ask("How long great wall",index)
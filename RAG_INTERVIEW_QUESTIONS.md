# RAG Interview Questions and Answers

This guide starts with Retrieval-Augmented Generation (RAG) fundamentals and
progresses to system design and production concerns. Examples and the tech-stack
section describe this repository: a local PDF question-answering app plus
step-by-step learning exercises.

## 1. Tech stack in this project

| Layer | Technology | How it is used here |
|---|---|---|
| Language | Python (3.10+) | Implements the app, pipeline helpers, and exercises. |
| UI | Streamlit | PDF upload, chat, streaming answers, and source display in [app.py](./app.py). |
| PDF text extraction | pypdf via LangChain's `PyPDFLoader` | Loads text and page metadata from uploaded PDFs. |
| Orchestration | LangChain | Connects document loading, splitting, embeddings, vector search, prompts, and chat models. |
| Chunking | `RecursiveCharacterTextSplitter` | Splits documents into 1,000-character chunks with 150-character overlap, via [rag_core.py](./rag_core.py). |
| Embeddings | Ollama `nomic-embed-text` | Embeds document chunks and user questions locally. |
| Chat LLM | Ollama `llama3.1:8b` | Generates answers from retrieved context locally. |
| Vector store | ChromaDB | Persists vectors locally in `chroma_db/` and performs similarity search. |
| Agent workflow exercise | LangGraph | Demonstrates retrieval grading, query rewriting, and retrying in [learn/04_agentic_rag.py](./learn/04_agentic_rag.py). This is a learning exercise, not the main app flow. |
| Dependency management | uv and `uv.lock` | Installs the declared dependencies and records a reproducible resolution. |
| Containerization | Docker and Docker Compose | Runs the Streamlit app and Ollama in separate containers, with persistent named volumes. |
| Tests | pytest | Tests the extracted pipeline helpers and logging; Ollama integration tests are optional. |

The main app uses a straightforward RAG pipeline. It does not currently include
OCR, hybrid keyword/vector retrieval, a reranker, or a production agent loop.
The `learn/` exercises teach embeddings, manual cosine similarity, basic RAG,
and agentic retry behavior separately from the app.

## 2. RAG fundamentals

### Q1. What is Retrieval-Augmented Generation?

**Answer:** RAG retrieves relevant information from an external knowledge source
and adds it to the prompt sent to a language model. The model can then answer
using current or private data that may not be part of its training. This project
extracts text from PDFs, indexes chunks in ChromaDB, retrieves relevant chunks
for each question, and asks the local LLM to answer from those chunks.

### Q2. What are the main stages of a RAG system?

**Answer:** There are two broad phases:

1. **Ingestion/indexing:** load documents, extract text, split it into chunks,
   create embeddings, and store vectors with metadata.
2. **Querying:** embed the question, retrieve relevant chunks, optionally
   transform or rerank them, assemble a prompt, generate an answer, and return
   citations or sources.

In this app, the ingestion and chat paths are in [app.py](./app.py), while
chunking and prompt/context helpers are in [rag_core.py](./rag_core.py).

### Q3. What is an embedding?

**Answer:** An embedding is a numeric vector representing text in a semantic
space. Texts with related meanings tend to have vectors that are close under a
chosen distance or similarity measure. The project uses Ollama's
`nomic-embed-text` model for both PDF chunks and questions.

### Q4. What is semantic search, and how is it different from keyword search?

**Answer:** Keyword search matches literal terms, while semantic search compares
meaning represented by embeddings. A semantic search can find a passage about
"car" for a question using "automobile," even without exact word overlap.
Semantic search may still miss exact identifiers, rare names, or precise
phrases, so production systems often combine it with keyword search.

### Q5. What does top-k mean?

**Answer:** `k` is the number of candidate chunks returned by retrieval. This
app retrieves four chunks (`TOP_K = 4`). A low value may omit needed evidence;
a high value can add irrelevant text, consume context-window capacity, and
distract the generator. Choose it by evaluating retrieval and answer quality
against representative questions.

### Q6. What does “grounding” mean in RAG?

**Answer:** Grounding means instructing the model to base its answer on
retrieved evidence, rather than relying only on its learned knowledge. This
project's prompt says to use only the provided context and to say when the
answer is not found. A prompt instruction helps, but does not guarantee that
the model will always obey it.

### Q7. Does RAG fine-tune or retrain the LLM?

**Answer:** No. Standard RAG supplies relevant information in the prompt at
inference time. Fine-tuning changes model weights and is useful for changing
behavior, style, or task patterns; RAG is generally the better first choice for
adding frequently changing or source-citable knowledge.

### Q8. Why split documents into chunks?

**Answer:** Embedding an entire long document can blur distinct topics, and
large passages may not fit efficiently into the model's context. Chunks let the
retriever select focused evidence. This app uses 1,000-character chunks and
150-character overlap; those are starting settings, not universal optima.

### Q9. Why use chunk overlap?

**Answer:** A fact or sentence can cross a chunk boundary. Overlap repeats some
text between neighboring chunks so that the meaning is less likely to be split
apart. Too much overlap increases storage and duplicate retrieval; too little
can lose boundary context.

### Q10. What is the role of a vector database?

**Answer:** A vector database stores embeddings and associated text/metadata,
then finds vectors similar to a query vector. ChromaDB is the local persistent
vector store in this project. It also avoids manually comparing a query against
every vector in application code as the collection grows.

## 3. Project-specific questions

### Q11. Walk me through a question in this app.

**Answer:** A user submits a question in Streamlit. The Chroma-backed
LangChain retriever embeds the question with the configured Ollama embedding
model and returns the four nearest chunks with scores. The app joins the chunk
text into context, formats the context and question with a grounded prompt,
streams a response from `ChatOllama`, displays the retrieved pages/chunks, and
logs the query, results, and answer.

### Q12. Walk me through PDF ingestion.

**Answer:** The uploaded PDF is written to a temporary file because
`PyPDFLoader` loads from a path. The loader extracts page documents, the
text-splitting helper creates overlapping chunks while preserving metadata, and
`Chroma.from_documents` embeds and persists those chunks. The temporary PDF is
removed in a `finally` block after processing.

### Q13. Why use the same embedding model at ingestion and query time?

**Answer:** Stored document vectors and query vectors must inhabit the same
embedding space and have compatible dimensions. Switching embedding models
without rebuilding the index can produce invalid or meaningless comparisons.
This app uses the same `OllamaEmbeddings` client for indexing and querying.

### Q14. Why is the pipeline logic in `rag_core.py`?

**Answer:** The Streamlit script executes UI code when imported, and a live
Ollama dependency makes direct tests less isolated. The pure helpers in
`rag_core.py` centralize splitting, context assembly, and prompt creation so
they can be unit-tested without starting the UI or calling Ollama.

### Q15. How does this project show sources?

**Answer:** Chroma's similarity search returns documents with metadata and
scores. The app displays each retrieved chunk's page metadata and score and
shows a text preview. These are useful evidence links, but they are not yet
precise quoted-span citations with PDF page links.

### Q16. What does a Chroma similarity score mean here?

**Answer:** The app calls `similarity_search_with_score`, and Chroma's default
distance behavior is L2 distance: lower values indicate closer vectors. Scores
are not probabilities and should not be interpreted as universal relevance
confidence. The manual embedding lessons instead calculate cosine similarity,
where a higher score indicates a closer direction; score direction depends on
the metric and API.

### Q17. Is the main application agentic RAG?

**Answer:** No. The main app retrieves once and generates an answer. The
separate `learn/04_agentic_rag.py` exercise uses LangGraph to grade retrieval,
rewrite the query, and retry up to a limit. It demonstrates agentic control flow
but is not wired into the Streamlit app.

### Q18. What PDF limitation should you mention?

**Answer:** `PyPDFLoader` extracts text from text-based PDFs. Image-only or
scanned PDFs generally need OCR, which this repository does not currently
implement. Extraction quality should be checked before investigating retrieval
or generation.

### Q19. How do you run and test the project?

**Answer:** Install dependencies with `uv sync`, pull the configured models
with Ollama, and launch with `uv run streamlit run app.py`. Run unit tests with
`uv run pytest`; Ollama-dependent integration tests can be selected with
`uv run pytest -m integration` and require a running local Ollama service.

## 4. Retrieval and generation design

### Q20. How would you choose a chunking strategy?

**Answer:** I would begin with the document structure and the target questions.
Prefer boundaries such as sections, paragraphs, or sentences where possible,
then tune chunk size and overlap on a representative evaluation set. I would
measure whether relevant evidence appears in the retrieved chunks and whether
the generator has enough context, rather than assuming a particular token or
character size is best.

### Q21. What is metadata filtering, and when would you use it?

**Answer:** Metadata filtering restricts retrieval to documents or chunks with
properties such as document ID, tenant, date, category, or access level. It is
useful for multi-document search, recency constraints, and authorization. This
project preserves page metadata but does not yet provide multi-document
selection or access-control filters.

### Q22. What is hybrid search?

**Answer:** Hybrid search combines lexical retrieval (for example BM25) with
vector retrieval. Lexical search is strong for exact terms, codes, and names;
vector search is strong for semantic matches. Results can be combined with
rank fusion or a learned method. Hybrid search is a potential improvement, not
a current feature in this app.

### Q23. What is reranking?

**Answer:** A reranker scores a smaller candidate set with a more precise model,
often comparing each query-document pair directly. The system can retrieve a
broader set cheaply, rerank it, and send only the strongest evidence to the
generator. This can improve relevance at the cost of additional latency and
compute. The current app does not rerank.

### Q24. What is MMR?

**Answer:** Maximal Marginal Relevance selects results that are both relevant
to the query and diverse from one another. It helps avoid returning several
near-duplicate chunks when the context budget is limited. The current app uses
similarity search, not MMR.

### Q25. How would you handle follow-up questions?

**Answer:** A follow-up such as "What about the second option?" may not be
searchable by itself. I would use conversation history to rewrite it into a
standalone retrieval query, then retrieve evidence and generate using the
original conversational context where appropriate. I would keep history
bounded or summarized so it does not overwhelm the context window. The current
app stores chat messages for display, but does not feed them into retrieval or
the LLM prompt.

### Q26. How would you make an answer more reliable?

**Answer:** Improve evidence retrieval first, then constrain generation to the
evidence, return citations, and allow abstention when evidence is missing.
Evaluate retrieval and generation separately. Additional options include
reranking, query rewriting, structured output validation, and checking that
each cited claim is supported. A “use only context” instruction alone does not
eliminate hallucinations.

### Q27. What is the difference between a retrieval failure and a generation
failure?

**Answer:** If the relevant evidence is absent from the retrieved chunks, the
problem is retrieval, ingestion, or document parsing. If the correct evidence
is present but the response is inaccurate or unsupported, the problem is more
likely prompt construction or generation. Inspecting the exact retrieved
chunks—something the app and `debug.py` support—helps isolate the layer.

## 5. Advanced and system-design questions

### Q28. How would you evaluate a RAG system?

**Answer:** I would create a representative test set of questions, expected
evidence, and acceptable answers, then evaluate components separately:

- **Retrieval:** Recall@k, Precision@k, MRR, or nDCG against relevant passages.
- **Generation:** correctness, completeness, groundedness, and appropriate
  abstention.
- **Citations:** whether cited passages actually support the claims.
- **Operations:** latency, throughput, error rate, and resource use.

Automated metrics should be supplemented with human review. This repository
tests helper behavior and has optional Ollama integration coverage, but does
not yet contain a RAG quality benchmark.

### Q29. How would you improve ingestion for large or varied documents?

**Answer:** I would build an idempotent ingestion pipeline with document IDs,
content hashes, versioning, metadata, and retryable stages. I would extract
document structure and tables where possible, add OCR for scanned content,
track extraction errors, batch embedding requests, and avoid re-embedding
unchanged content. I would also support deletion and re-indexing of a single
document rather than clearing the entire store.

### Q30. How would you scale beyond a local Chroma collection?

**Answer:** I would first measure the bottleneck. Possible steps include
choosing a vector store suited to the deployment, using approximate nearest
neighbor indexing, batching ingestion, caching query embeddings, and separating
the UI/API from ingestion workers. For multi-user workloads I would add
authentication, tenant-aware metadata filters, quotas, observability, backups,
and durable managed storage. A managed service is a trade-off against the
current fully local privacy and deployment model.

### Q31. How do you control latency and cost?

**Answer:** Measure each stage separately: query embedding, search, reranking,
prompt size, and generation. Retrieve only enough evidence, cap context size,
stream answers, cache safe reusable work, batch ingestion, and select models
based on measured quality/latency. In this repository the models run locally,
so there is no hosted-token charge, but CPU/GPU, memory, and response time
remain costs.

### Q32. What are the risks of putting retrieved text into an LLM prompt?

**Answer:** Retrieved documents are untrusted input. They may contain
instructions that attempt to override the system's intended behavior, sensitive
data, or malformed content. Treat retrieved text as data rather than
instructions, use explicit prompt boundaries, enforce authorization before
retrieval, avoid exposing secrets in context, and validate generated actions
and outputs. Prompt wording alone is not a security boundary.

### Q33. What privacy properties does this project have, and what should not be
overclaimed?

**Answer:** The configured Ollama models and Chroma store run locally, which
supports a local-data workflow without a hosted model API. However, privacy
depends on how the machine, Ollama service, logs, uploaded files, and backups
are configured. The project writes retrieval details and answers to `rag.log`,
so logs should be protected and managed appropriately. “Local” is not by itself
a complete privacy or security guarantee.

### Q34. What is agentic RAG, and when is it useful?

**Answer:** Agentic RAG adds decisions or loops around retrieval and
generation—for example, grading whether retrieval is useful, rewriting a query,
and searching again. It can help with ambiguous questions or weak first-pass
retrieval, but adds latency, model calls, complexity, and failure modes. It
should be introduced only when evaluation shows a benefit. This behavior is
illustrated in the LangGraph learning exercise, not the main application.

### Q35. How do you prevent a bad relevance threshold from causing poor answers
in an agentic flow?

**Answer:** Calibrate the threshold against labeled examples and the actual
score metric. A fixed threshold may not transfer between embedding models,
distance functions, or datasets. Set a maximum retry count, retain the original
question, log each attempt, and abstain or clearly report insufficient evidence
if retries still fail. The learning exercise uses a simple threshold and is
illustrative rather than a production-calibrated policy.

### Q36. How would you support multiple users securely?

**Answer:** Authenticate users, authorize document access before retrieval,
isolate collections or apply mandatory tenant filters, and test that filters
cannot be bypassed. Protect local files and logs, limit upload size and file
types, validate ingestion, and avoid returning another user's source snippets.
The current project is a local single-user learning app and does not implement
these multi-tenant controls.

## 6. A concise project explanation for interviews

> I built a local PDF question-answering application using Python, Streamlit,
> LangChain, Ollama, and ChromaDB. It extracts text and page metadata from PDFs,
> splits the text into overlapping chunks, embeds and stores them locally, then
> retrieves the four closest chunks for a question and streams a locally
> generated answer grounded in that context. I extracted chunking and prompt
> helpers into a testable module, expose retrieved sources in the UI, and log
> retrieval results. The `learn/` folder separately demonstrates embeddings,
> manual cosine similarity, basic RAG, and a LangGraph retry workflow. Current
> limitations include no OCR, hybrid retrieval, reranking, conversation-aware
> prompting, or production multi-user access control.

## 7. Quick revision checklist

- Explain ingestion separately from query-time retrieval.
- Distinguish embeddings from the LLM that generates the answer.
- Know what chunk size, overlap, top-k, and metadata do.
- Confirm whether a reported score is a distance or similarity.
- Inspect retrieved evidence before blaming the generator.
- Separate features in the main app from examples in `learn/`.
- State limitations honestly and describe improvements as future work, not
  implemented functionality.

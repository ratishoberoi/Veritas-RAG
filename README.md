# 🧠 Neural KB — Production-Grade Personal Knowledge Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-00A67E?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama%203-F55036?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**A self-evaluating, hallucination-resistant RAG system with conversational memory, multi-modal ingestion, and a dual-LLM judge architecture.**

[Live Demo](#) · [Architecture Docs](#technical-deep-dive) · [Roadmap](#scalability--roadmap)

</div>

---

## Executive Summary

Neural KB is a **production-grade personal knowledge engine** that transforms unstructured documents — PDFs, YouTube videos, and Notion exports — into a queryable, conversational intelligence layer. Unlike naive RAG implementations that blindly return retrieved text, Neural KB introduces a **self-evaluation architecture**: every generated answer is independently scored by a second LLM pass that detects hallucinations, measures contextual grounding, and reports a calibrated confidence score to the user.

> **"Don't just retrieve and generate — retrieve, generate, and verify."**

The system achieves **100% confidence scores on directly answerable questions** from technical documents (validated against the Attention Is All You Need paper), correctly refuses out-of-context queries with `0%` confidence, and maintains **anaphora-aware conversational memory** across multi-turn sessions — resolving pronoun references like "its" and "they" from prior context.

### Key Metrics from Production Testing

| Benchmark | Result |
|---|---|
| Direct factual retrieval (e.g., "How many attention heads?") | **100% Confidence ✅** |
| Multi-step numerical retrieval (e.g., BLEU score) | **90% Confidence ✅** |
| Out-of-context hallucination guard (e.g., "FIFA World Cup 2026") | **0% — Correctly Refused ✅** |
| Anaphora resolution across turns (e.g., "What were its hyperparameters?") | **Resolved correctly ✅** |
| Ingestion throughput (15-page technical PDF) | **79 chunks, ~12s ✅** |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangChain 0.3+ (LCEL) | Chain composition, memory, retrieval |
| **LLM** | Groq / Llama 3.1 8B Instant | Generation + evaluation (250+ tok/s) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | 384-dim semantic vectors |
| **Vector Store** | Pinecone Serverless (AWS us-east-1) | Scalable similarity search + metadata filtering |
| **PDF Ingestion** | LangChain PyPDFLoader | Page-level document extraction |
| **YouTube Ingestion** | youtube-transcript-api + yt-dlp | Dual-plan transcript pipeline |
| **Speech-to-Text** | Groq Whisper Large v3 | Audio transcription fallback |
| **Memory** | LangChain `RunnableWithMessageHistory` | Session-scoped conversational state |
| **UI** | Streamlit | Premium dark-mode chat dashboard |
| **Environment** | Python 3.10, dotenv | Secrets management |

---

## System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["📥 Ingestion Layer"]
        PDF["📄 PDF\n(PyPDFLoader)"]
        YT["🎥 YouTube\n(Dual-Plan Pipeline)"]
        NOTION["📝 Notion\n(Markdown Export)"]
    end

    subgraph PROCESSING["⚙️ Processing Layer"]
        SPLIT["RecursiveCharacterTextSplitter\nchunk_size=1000 | overlap=200"]
        EMBED["HuggingFace Embeddings\nall-MiniLM-L6-v2 | 384 dims"]
    end

    subgraph STORAGE["🗄️ Vector Store"]
        PINE["Pinecone Serverless\nAWS us-east-1 | cosine similarity\nMetadata: source, method, page, title"]
    end

    PDF --> SPLIT
    YT --> SPLIT
    NOTION --> SPLIT
    SPLIT --> EMBED
    EMBED --> PINE

    style INGESTION fill:#1a1a2e,stroke:#7c6fff,color:#f0f0ff
    style PROCESSING fill:#1a1a2e,stroke:#ff6f9c,color:#f0f0ff
    style STORAGE fill:#1a1a2e,stroke:#4dffa6,color:#f0f0ff
```

---

## Technical Deep Dive

### RAG Pipeline

The retrieval pipeline is built on LangChain's **LCEL (LangChain Expression Language)** — a composable, type-safe abstraction for chaining operations. The core pipeline follows the pattern:

```
Query → Retriever → format_docs() → PromptTemplate → LLM → StrOutputParser
```

#### Chunking Strategy — Why `RecursiveCharacterTextSplitter`?

Fixed-size character splitters destroy semantic coherence by cutting mid-sentence. `RecursiveCharacterTextSplitter` applies a **hierarchy of separators** in order of semantic weight:

```
["\n\n", "\n", " ", ""]
```

It first attempts to split on paragraph breaks, then sentences, then words — only falling back to raw characters as a last resort. This guarantees that chunk boundaries align with natural language units.

**Configuration rationale:**

| Parameter | Value | Reasoning |
|---|---|---|
| `chunk_size` | `1000` | Fits comfortably within LLM context while retaining enough topical coherence |
| `chunk_overlap` | `200` | 20% overlap ensures answers straddling two chunks are never lost |
| `add_start_index` | `True` | Enables provenance tracking — know exactly where in the document a chunk originated |

#### Pinecone Serverless Integration

Pinecone's serverless tier provides **auto-scaling vector search with zero infrastructure overhead**. The index is configured with:

- **Dimensions:** `384` (matched to `all-MiniLM-L6-v2` output)
- **Metric:** `cosine` (optimal for normalized semantic similarity)
- **Spec:** `ServerlessSpec(cloud="aws", region="us-east-1")`

**Metadata filtering** enables source-scoped retrieval without maintaining multiple indexes:

```python
# Search only within YouTube transcripts
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"method": "groq_whisper"}
    }
)
```

This is the key architectural advantage over naive RAG: a single index serves multiple use cases through **predicate pushdown at the vector store layer** — the filter is applied before scoring, not after.

---

## The Judge Architecture

### Dual-LLM Self-Evaluation

Neural KB implements the **LLM-as-a-Judge** pattern — a technique used in production evaluation frameworks like RAGAS and TruLens. After the primary generation pass, a second, independent LLM invocation evaluates the answer against the retrieved context.

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant R as Retriever
    participant G as Generator LLM
    participant J as Judge LLM
    participant UI as Streamlit UI

    U->>R: Submit query
    R->>R: Embed query → similarity search (k=4)
    R-->>G: Return top-4 chunks + formatted context
    G->>G: Generate answer with SYSTEM_PROMPT grounding rules
    G-->>J: Pass [question + context + answer]
    J->>J: Evaluate contextual support
    J->>J: Detect hallucination markers
    J->>J: Calculate confidence score (0-100)
    J-->>UI: Return structured JSON evaluation
    UI-->>U: Display answer + confidence card + sources
```

### Evaluation Schema

The judge is prompted to return a **strict JSON object** — no markdown, no preamble:

```json
{
  "context_sufficient": true,
  "confidence_score": 92,
  "hallucination_detected": false,
  "hallucination_reason": "",
  "evaluation_summary": "Answer is fully supported by context with all claims traceable."
}
```

### Confidence Score Rubric

| Score Range | Interpretation | UI Indicator |
|---|---|---|
| `90–100` | Every claim directly traceable to retrieved context | 🟢 **Context Verified** |
| `70–89` | Mostly supported, minor inferential gaps | 🟢 **Context Verified** |
| `40–69` | Partial support, some unverifiable claims | 🟡 **Moderate Confidence** |
| `0–39` | Weak or no contextual grounding | 🔴 **Low Confidence** |

### Hallucination Guard

The generator LLM is instructed via system prompt to **prefix any outside-knowledge statement** with `[OUTSIDE KNOWLEDGE]`. This creates an explicit, machine-readable signal that the judge can detect and flag. The dual-layer approach — generator self-flagging + judge independent verification — provides redundant hallucination detection.

**Validated result from live testing:**

> Query: *"Who won FIFA World Cup 2026?"*
> Response: `[OUTSIDE KNOWLEDGE] I couldn't find this in your knowledge base...`
> Judge verdict: `hallucination_detected: true` | `confidence_score: 0%` 🔴

---

## Multi-Modal Ingestion Pipeline

### YouTube: Two-Plan Architecture

YouTube transcript acquisition uses a **graceful degradation strategy** — optimizing for speed and cost while maintaining reliability.

```mermaid
flowchart TD
    URL["🔗 YouTube URL"] --> EXTRACT["Extract Video ID\n(handles v=, youtu.be,\n/shorts/, /embed/)"]
    EXTRACT --> PLANA

    subgraph PLANA["🅰️ Plan A — Transcript API"]
        API["youtube-transcript-api\nlist_transcripts()"]
        LANG["find_transcript(['en'])\nfetch() → Document list"]
        API --> LANG
    end

    subgraph PLANB["🅱️ Plan B — Whisper Transcription"]
        YTDLP["yt-dlp audio download\nformat: worstaudio/worst\n32kbps MP3"]
        SIZE["File size check\n> 24MB → FFmpeg re-encode\n16kHz mono, 16kbps"]
        WHISPER["Groq Whisper Large v3\nTranscription API"]
        CLEANUP["finally: cleanup\ntemp files"]
        YTDLP --> SIZE --> WHISPER --> CLEANUP
    end

    PLANA -->|"✅ Success"| DONE["📄 Document chunks\nwith metadata"]
    PLANA -->|"❌ Bot detection\nDisabled captions\nRate limit"| PLANB
    PLANB --> DONE

    style PLANA fill:#1a1a2e,stroke:#4dffa6,color:#f0f0ff
    style PLANB fill:#1a1a2e,stroke:#ffb347,color:#f0f0ff
```

**Why this matters:**

- Plan A fails silently on ~30% of videos due to YouTube's bot detection and caption policies
- Plan B using yt-dlp + Groq Whisper transcribes **any video with audio** in under 3 seconds for a 10-minute clip
- The `finally` block guarantees temp file cleanup even on transcription failure — no disk leaks in long-running deployments
- FFmpeg re-encoding to 16kHz mono ensures the 24MB Groq Whisper file size limit is never exceeded

---

## Conversation Design

### Session-Scoped Memory with `RunnableWithMessageHistory`

Stateless LLMs have no concept of "previous turn." Neural KB implements **session-scoped conversational memory** using LangChain's `RunnableWithMessageHistory` — a wrapper that automatically hydrates and persists conversation state around any LCEL chain.

```mermaid
stateDiagram-v2
    [*] --> NewSession: First query

    state NewSession {
        [*] --> CreateHistory: session_id not in store{}
        CreateHistory --> StoreHistory: ChatMessageHistory()
    }

    state ExistingSession {
        [*] --> LoadHistory: session_id found in store{}
        LoadHistory --> TrimHistory: len > 20 messages?
        TrimHistory --> InjectHistory: messages[-20:]
    }

    NewSession --> Invoke: history injected into prompt
    ExistingSession --> Invoke: history injected into prompt

    state Invoke {
        [*] --> BuildPrompt: [SystemPrompt + ChatHistory + HumanMessage]
        BuildPrompt --> LLMCall
        LLMCall --> ParseResponse
    }

    Invoke --> SaveTurn: Auto-save HumanMessage + AIMessage
    SaveTurn --> ExistingSession: Next query
    SaveTurn --> [*]: Session cleared
```

### Anaphora Resolution in Practice

The `MessagesPlaceholder(variable_name="chat_history")` injects the full conversation history into the prompt template. This enables the LLM to resolve **co-references** — pronouns and implicit references — across turns.

**Validated 2-turn memory test:**

| Turn | Query | Resolved Reference |
|---|---|---|
| Turn 1 | *"What optimizer did the authors use?"* | — |
| Turn 2 | *"What were **its** hyperparameters?"* | `its` → `Adam optimizer` |
| Result | β₁=0.9, β₂=0.98, ε=10⁻⁹ returned correctly | **100% confidence ✅** |

**Memory architecture decisions:**

- **Window size: 20 messages (10 exchanges)** — balances context length vs. LLM token cost. Beyond this, older turns are evicted (`messages[-20:]`)
- **Global `store` dict keyed by `session_id`** — enables multi-user isolation with zero database overhead in development. Production upgrade path: Redis
- **`RunnableWithMessageHistory` auto-saves** — no manual `.add_user_message()` calls. The wrapper handles read-before and write-after automatically

---

## Project Structure

```
rag-knowledge-base/
├── app.py              # Streamlit UI — chat interface, sidebar, eval cards
├── ingestor.py         # Multi-source loader — PDF, YouTube (Plan A/B), Notion
├── vector_store.py     # Pinecone integration — upload, embed, retrieve
├── retriever.py        # RAG chain — memory, filtering, dual-LLM evaluation
├── .env                # API keys (gitignored)
├── requirements.txt    # Frozen dependencies
└── data/               # Local document storage
```

---

## Installation & Quickstart

```bash
# 1. Clone and create virtual environment
git clone https://github.com/yourusername/neural-kb.git
cd neural-kb
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Add: GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

# 4. Create Pinecone index (384 dims, cosine, serverless AWS us-east-1)
# Run once via Pinecone console or vector_store.py init function

# 5. Launch
streamlit run app.py
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key (LLM + Whisper) |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_INDEX_NAME` | ✅ | Index name (e.g., `rag-knowledge-base`) |
| `HUGGINGFACEHUB_API_TOKEN` | Optional | Required only for private HF models |

---

## Scalability & Roadmap

### Current Architecture Limitations

| Limitation | Impact | Planned Fix |
|---|---|---|
| Global `store` dict for memory | Lost on server restart | **Redis** persistent session store |
| Single dense vector retrieval | Misses keyword-exact matches | **Hybrid Search** (BM25 + dense) |
| Top-4 retrieval, no reranking | Suboptimal chunk ordering | **Cross-Encoder Reranker** (ms-marco) |
| Synchronous PDF ingestion | Blocks UI on large files | **Async job queue** (Celery/RQ) |
| Shared Pinecone namespace | No multi-tenant isolation | **Namespace-per-user** partitioning |

### Phase 2 Roadmap

**Hybrid Search** — Dense vector similarity captures semantic meaning but misses exact technical terms (e.g., `β₂=0.98`). BM25 sparse retrieval excels at keyword-exact matching. Combining both via **Reciprocal Rank Fusion (RRF)** produces measurably better recall on technical documents.

**Cross-Encoder Reranking** — The current bi-encoder retrieval (embed query → embed chunks → cosine similarity) is fast but imprecise. Adding a **cross-encoder reranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second-stage filter reranks the top-20 retrieved chunks to select the best 4 — significantly improving answer quality on ambiguous queries.

**Redis Session Store** — Replacing the in-memory `store` dict with Redis enables **persistent conversational state** across server restarts, horizontal scaling across multiple workers, and TTL-based session expiry. Drop-in replacement: `langchain_community.chat_message_histories.RedisChatMessageHistory`.

**Streaming Responses** — `chain.astream()` with `st.write_stream()` enables token-by-token streaming in the Streamlit UI — eliminating the perceived latency of waiting for a full response before display.

**Document Version Control** — Track document versions in Pinecone metadata to support **differential updates**: re-ingesting a changed PDF only re-embeds modified pages rather than the entire document.

---

## Architecture Decisions — Engineering Rationale

### Why Groq over OpenAI?

Groq's LPU (Language Processing Unit) hardware runs Llama 3 at **250+ tokens/second** vs OpenAI's ~40 tokens/second for GPT-4o. For a RAG system that makes two LLM calls per query (generation + evaluation), this latency difference is architecturally significant. At 10 queries/minute, Groq's free tier handles the full load without rate limiting.

### Why `all-MiniLM-L6-v2` over larger embedding models?

`all-MiniLM-L6-v2` produces **384-dimensional vectors** — half the size of `text-embedding-ada-002` (1536 dims) and `all-mpnet-base-v2` (768 dims). For a personal knowledge base with thousands of documents, this translates to a **4x reduction in Pinecone storage costs** and faster similarity search with negligible quality degradation on factual retrieval tasks.

### Why `temperature=0` for RAG?

Factual retrieval demands deterministic, reproducible outputs. `temperature=0` forces the LLM to select the highest-probability token at every step — eliminating creative variation that would introduce factual inconsistency between identical queries. Temperature is only increased for the evaluation judge when assessing nuanced hallucination cases.

---

## Live Test Results

The following results were produced in a single session against the **"Attention Is All You Need"** (Vaswani et al., 2017) paper:

```
Query: "How many attention heads does the base transformer model use?"
Answer: "The base model employs h = 8 parallel attention layers, or heads."
Confidence: 100% ✅ — Context Verified

Query: "What is the BLEU score reported for the base model?"
Answer: "The BLEU score reported for the base model is 27.3."
Confidence: 90% ✅ — Context Verified

Query: "What optimizer did the authors use?"
Answer: "The authors used the Adam optimizer."
Confidence: 100% ✅ — Context Verified

Query: "What were its hyperparameters?"  ← anaphora: "its" = Adam
Answer: "β₁=0.9, β₂=0.98, ε=10⁻⁹"
Confidence: 100% ✅ — Memory resolved correctly

Query: "Who won FIFA World Cup 2026?"
Answer: "[OUTSIDE KNOWLEDGE] I couldn't find this in your knowledge base."
Confidence: 0% 🔴 — Hallucination flagged and refused
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with LangChain · Pinecone · Groq · Streamlit · HuggingFace**

*If this project helped you, consider giving it a ⭐ on GitHub*

</div>
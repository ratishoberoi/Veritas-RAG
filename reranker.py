"""
reranker.py — Cross-Encoder Reranking Stage

WHY RERANKING EXISTS:
─────────────────────────────────────────────────────────────────
Bi-encoders (like all-MiniLM-L6-v2) embed query and document
INDEPENDENTLY, then compare via cosine similarity. This is fast
but lossy — the encoder never "sees" query and document together.

A Cross-Encoder sees the query and document as a SINGLE input:
    input = [CLS] query [SEP] document [SEP]

It then outputs a single relevance score (not an embedding).
This is much more expensive but dramatically more accurate —
the model can capture exact term overlap, co-references, and
contextual relevance that cosine similarity misses entirely.

PIPELINE:
    Hybrid Retrieval (BM25 + Dense) → top 12 chunks (broad recall)
         ↓
    Cross-Encoder (BAAI/bge-reranker-base) → re-score all 12
         ↓
    Top 4 highest-scoring chunks → LLM context (precision)

HOW THIS REDUCES NOISE:
    Without reranking, LLM receives chunks ranked by embedding
    similarity — which may include topically related but answer-
    irrelevant chunks. Example: query "What is β₂?" might return
    chunks about "beta testing" (similar embeddings) alongside the
    actual optimizer hyperparameter chunk.

    The cross-encoder scores each chunk by asking: "Does THIS chunk
    directly answer THIS specific question?" — discarding all
    tangentially related noise before it reaches the LLM context.
"""

import os
import time
from typing import List, Tuple, Optional
from langchain_core.documents import Document


# ── Global model cache (prevent reload on every call) ──
_reranker_model = None
_reranker_model_name = None


def get_reranker(model_name: str = "BAAI/bge-reranker-base"):
    """
    Load and cache the CrossEncoder model.

    Model options ranked by speed/quality tradeoff:
    ┌─────────────────────────────────┬────────┬──────────┬───────────┐
    │ Model                           │ Size   │ Speed    │ Quality   │
    ├─────────────────────────────────┼────────┼──────────┼───────────┤
    │ BAAI/bge-reranker-base          │ 278MB  │ Fast     │ Good ✅   │
    │ BAAI/bge-reranker-large         │ 560MB  │ Medium   │ Better    │
    │ cross-encoder/ms-marco-MiniLM-L6│ 67MB   │ Fastest  │ OK        │
    │ BAAI/bge-reranker-v2-m3         │ 568MB  │ Slow     │ Best      │
    └─────────────────────────────────┴────────┴──────────┴───────────┘

    Default: bge-reranker-base — best balance for local CPU inference.
    """
    global _reranker_model, _reranker_model_name

    if _reranker_model is not None and _reranker_model_name == model_name:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] 🔄 Loading cross-encoder: {model_name}")
        print("[Reranker] ⏳ First load may take 30-60s (downloading ~278MB)...")
        start = time.time()

        _reranker_model = CrossEncoder(
            model_name,
            max_length=512,       # truncate long chunks to fit model context
            device="cpu"          # use "cuda" if GPU available
        )
        _reranker_model_name = model_name

        elapsed = time.time() - start
        print(f"[Reranker] ✅ Cross-encoder loaded in {elapsed:.1f}s")
        return _reranker_model

    except ImportError:
        raise ImportError(
            "[Reranker] ❌ sentence-transformers not installed.\n"
            "Run: pip install sentence-transformers"
        )


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int = 4,
    model_name: str = "BAAI/bge-reranker-base",
    score_threshold: Optional[float] = None,
    return_scores: bool = False
) -> List[Document]:
    """
    Rerank retrieved documents using a Cross-Encoder model.

    The cross-encoder scores each (query, document) pair jointly,
    capturing fine-grained relevance that embedding similarity misses.

    Args:
        query: The user's question
        docs: Candidate documents from hybrid retrieval (typically 10-15)
        top_k: Number of top documents to return after reranking
        model_name: HuggingFace cross-encoder model identifier
        score_threshold: Optional minimum score to include a document.
                         Documents below this score are dropped even if
                         they would be in top_k. Range: typically -10 to 10.
                         Recommended: -3.0 (aggressive filtering)
        return_scores: If True, inject rerank_score into doc metadata

    Returns:
        List of top_k most relevant Document objects, sorted by score desc.

    Example scores from bge-reranker-base:
        "What is multi-head attention?" + relevant chunk → score ~7.2
        "What is multi-head attention?" + unrelated chunk → score ~-4.1
    """
    if not docs:
        print("[Reranker] ⚠️ No documents to rerank.")
        return []

    if len(docs) <= top_k:
        print(f"[Reranker] ℹ️ {len(docs)} docs ≤ top_k={top_k}, skipping rerank.")
        return docs

    reranker = get_reranker(model_name)

    # Build (query, document_text) pairs for batch scoring
    pairs = [(query, doc.page_content) for doc in docs]

    print(f"[Reranker] 🔄 Scoring {len(pairs)} chunks for query: '{query[:60]}...'")
    start = time.time()

    scores: List[float] = reranker.predict(pairs).tolist()

    elapsed = time.time() - start
    print(f"[Reranker] ✅ Reranked {len(docs)} → keeping top {top_k} "
          f"(scored in {elapsed:.2f}s)")

    # Zip docs with scores, sort descending
    scored_docs: List[Tuple[float, Document]] = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    # Log score distribution for debugging
    score_vals = [s for s, _ in scored_docs]
    print(f"[Reranker] 📊 Score range: max={score_vals[0]:.2f} | "
          f"min={score_vals[-1]:.2f} | "
          f"cutoff={score_vals[min(top_k-1, len(score_vals)-1)]:.2f}")

    # Apply score threshold filter if provided
    if score_threshold is not None:
        pre_filter = len(scored_docs)
        scored_docs = [(s, d) for s, d in scored_docs if s >= score_threshold]
        if len(scored_docs) < pre_filter:
            print(f"[Reranker] 🔍 Threshold {score_threshold} filtered "
                  f"{pre_filter - len(scored_docs)} low-relevance chunks")

    # Take top_k
    top_docs = scored_docs[:top_k]

    result = []
    for score, doc in top_docs:
        if return_scores:
            # Inject score into metadata for transparency
            doc.metadata["rerank_score"] = round(score, 4)
        result.append(doc)

    return result


def rerank_with_explanation(
    query: str,
    docs: List[Document],
    top_k: int = 4,
    model_name: str = "BAAI/bge-reranker-base"
) -> Tuple[List[Document], List[dict]]:
    """
    Rerank documents and return detailed scoring explanation.
    Useful for debugging retrieval quality in the terminal.

    Returns:
        (reranked_docs, score_report) where score_report is a list of dicts:
        [{"rank": 1, "score": 7.2, "source": "PDF p.3", "preview": "..."}]
    """
    if not docs:
        return [], []

    reranker = get_reranker(model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs).tolist()

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    reranked_docs = [doc for _, doc in scored[:top_k]]

    score_report = []
    for i, (score, doc) in enumerate(scored):
        source = doc.metadata.get("source", "unknown")
        method = doc.metadata.get("method", "")
        page = doc.metadata.get("page", "")

        if method == "groq_whisper":
            src_label = f"YouTube: {doc.metadata.get('title', '')[:30]}"
        elif str(source).endswith(".pdf"):
            pg = f" p.{int(float(page))+1}" if page != "" else ""
            src_label = f"PDF:{os.path.basename(str(source))}{pg}"
        else:
            src_label = str(source)[:40]

        score_report.append({
            "rank": i + 1,
            "score": round(score, 3),
            "kept": i < top_k,
            "source": src_label,
            "preview": doc.page_content[:80].replace("\n", " ") + "..."
        })

    return reranked_docs, score_report
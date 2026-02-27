import os
import time
from typing import List, Optional, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone, ServerlessSpec
from pydantic import Field

load_dotenv()

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("[VectorStore] 🔄 Loading embedding model (all-MiniLM-L6-v2)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[VectorStore] ✅ Embedding model loaded.")
    return _embeddings


def init_pinecone_index():
    print("[VectorStore] 🔄 Connecting to Pinecone...")
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-knowledge-base")

    if not api_key:
        raise ValueError("[VectorStore] ❌ PINECONE_API_KEY not found in .env")

    pc = Pinecone(api_key=api_key)
    existing_indexes = [i.name for i in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"[VectorStore] 🔨 Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        timeout = 300
        start = time.time()
        while not pc.describe_index(index_name).status["ready"]:
            if time.time() - start > timeout:
                raise TimeoutError(f"[VectorStore] ❌ Index creation timed out after {timeout}s")
            print("[VectorStore] ⏳ Waiting for index to be ready...")
            time.sleep(5)
        print(f"[VectorStore] ✅ Index '{index_name}' created.")
    else:
        print(f"[VectorStore] ✅ Index '{index_name}' already exists.")

    return pc.Index(index_name)


def upload_documents(chunks: List[Document], namespace: Optional[str] = None) -> PineconeVectorStore:
    if not chunks:
        raise ValueError("[VectorStore] ❌ No chunks to upload.")

    print(f"[VectorStore] 🔄 Uploading {len(chunks)} chunks to Pinecone...")
    try:
        init_pinecone_index()
        embeddings = get_embeddings()
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-knowledge-base")

        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        vector_store.add_documents(chunks, batch_size=100)
        print(f"[VectorStore] ✅ Successfully uploaded {len(chunks)} chunks.")
        return vector_store

    except Exception as e:
        raise ConnectionError(f"[VectorStore] ❌ Upload failed: {str(e)}")


def load_vector_store(namespace: Optional[str] = None) -> PineconeVectorStore:
    print("[VectorStore] 🔄 Loading existing Pinecone vector store...")
    try:
        embeddings = get_embeddings()
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-knowledge-base")
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        print("[VectorStore] ✅ Vector store loaded.")
        return vector_store

    except Exception as e:
        raise ConnectionError(f"[VectorStore] ❌ Failed to load vector store: {str(e)}")


# ─────────────────────────────────────────────
# CUSTOM HYBRID RETRIEVER (no EnsembleRetriever needed)
# ─────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Custom Hybrid Retriever implementing Reciprocal Rank Fusion (RRF)
    over BM25 (sparse) + Pinecone dense vector search.

    No dependency on langchain.retrievers.EnsembleRetriever.

    RRF Formula:
        score(d) = Σ  weight_i / (rank_i(d) + 60)

    The constant 60 is a smoothing factor that prevents top-ranked
    documents from dominating. Documents appearing in BOTH result
    lists receive additive scores — consensus between retrieval
    methods naturally floats the best chunks to the top.
    """

    dense_retriever: Any = Field(description="Pinecone vector store retriever")
    sparse_retriever: Any = Field(description="BM25 in-memory retriever")
    dense_weight: float = Field(default=0.6)
    sparse_weight: float = Field(default=0.4)
    k: int = Field(default=12)
    rrf_k: int = Field(default=60, description="RRF smoothing constant")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Fetch from both retrievers, fuse with weighted RRF, return top-k.

        Steps:
          1. Run dense retrieval → ordered list of docs
          2. Run BM25 retrieval → ordered list of docs
          3. For each doc in each list, compute: weight / (rank + rrf_k)
          4. Sum scores for docs appearing in both lists
          5. Sort by final RRF score descending → return top k
        """
        # ── Fetch from both retrievers ──
        try:
            dense_docs = self.dense_retriever.invoke(query)
        except Exception as e:
            print(f"[HybridRetriever] ⚠️ Dense retrieval failed: {e}")
            dense_docs = []

        try:
            sparse_docs = self.sparse_retriever.invoke(query)
        except Exception as e:
            print(f"[HybridRetriever] ⚠️ BM25 retrieval failed: {e}")
            sparse_docs = []

        # ── Build RRF score map keyed by page_content ──
        scores: dict = {}      # content_hash → score
        doc_map: dict = {}     # content_hash → Document

        def content_key(doc: Document) -> str:
            return doc.page_content[:200]  # use first 200 chars as stable key

        # Score dense results
        for rank, doc in enumerate(dense_docs):
            key = content_key(doc)
            rrf_score = self.dense_weight / (rank + self.rrf_k)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        # Score BM25 results
        for rank, doc in enumerate(sparse_docs):
            key = content_key(doc)
            rrf_score = self.sparse_weight / (rank + self.rrf_k)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

        # ── Sort by RRF score descending ──
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        result = []
        for key in sorted_keys[:self.k]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(scores[key], 6)
            result.append(doc)

        print(f"[HybridRetriever] ✅ RRF fusion: "
              f"{len(dense_docs)} dense + {len(sparse_docs)} BM25 → top {len(result)}")
        return result


def build_bm25_retriever(
    chunks: List[Document],
    k: int = 12
) -> BM25Retriever:
    if not chunks:
        raise ValueError("[VectorStore] ❌ Cannot build BM25 retriever: no chunks provided.")
    print(f"[VectorStore] 🔄 Building BM25 index over {len(chunks)} chunks...")
    retriever = BM25Retriever.from_documents(chunks, k=k)
    print(f"[VectorStore] ✅ BM25 index ready ({len(chunks)} documents).")
    return retriever


def _apply_metadata_filter(
    chunks: List[Document],
    filter_dict: dict
) -> List[Document]:
    """
    Manually apply metadata filter to BM25 chunks (in-memory equivalent
    of Pinecone's server-side filter). Supports substring matching.
    """
    filtered = []
    for doc in chunks:
        match = True
        for key, value in filter_dict.items():
            doc_val = str(doc.metadata.get(key, "")).lower()
            filter_val = str(value).lower()
            if filter_val not in doc_val:
                match = False
                break
        if match:
            filtered.append(doc)
    return filtered


def build_hybrid_retriever(
    chunks: List[Document],
    namespace: Optional[str] = None,
    source_filter: Optional[dict] = None,
    k: int = 12,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6
) -> HybridRetriever:
    """
    Build a HybridRetriever combining BM25 + Pinecone via custom RRF.

    Args:
        chunks: Document chunks for BM25 in-memory index
        namespace: Pinecone namespace
        source_filter: Metadata filter for scoped retrieval
        k: Total candidates to return after fusion
        bm25_weight: BM25 contribution to RRF (default 0.4)
        vector_weight: Dense vector contribution to RRF (default 0.6)

    Returns:
        HybridRetriever instance (LangChain BaseRetriever compatible)
    """
    print("[VectorStore] 🔄 Building Hybrid Retriever (BM25 + Pinecone)...")
    print(f"[VectorStore] ⚖️  Weights → Vector: {vector_weight} | BM25: {bm25_weight}")

    # ── Dense retriever (Pinecone) ──
    vector_store = load_vector_store(namespace=namespace)
    search_kwargs = {"k": k}
    if source_filter:
        search_kwargs["filter"] = source_filter
        print(f"[VectorStore] 🔍 Source filter applied: {source_filter}")

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # ── Sparse retriever (BM25) with manual filter ──
    filtered_chunks = chunks
    if source_filter and chunks:
        filtered_chunks = _apply_metadata_filter(chunks, source_filter)
        print(f"[VectorStore] 🔍 BM25 filtered to {len(filtered_chunks)} chunks")

    if not filtered_chunks:
        print("[VectorStore] ⚠️ No BM25 chunks match filter — using dense only.")
        return dense_retriever

    sparse_retriever = build_bm25_retriever(filtered_chunks, k=k)

    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=vector_weight,
        sparse_weight=bm25_weight,
        k=k
    )

    print("[VectorStore] ✅ HybridRetriever ready (custom RRF fusion).")
    return hybrid
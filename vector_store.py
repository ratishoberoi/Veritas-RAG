import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from pinecone import Pinecone, ServerlessSpec

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
    """Upload document chunks to Pinecone vector store."""
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
    """Load existing Pinecone vector store for retrieval."""
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


def build_bm25_retriever(
    chunks: List[Document],
    k: int = 4
) -> BM25Retriever:
    """
    Build an in-memory BM25 retriever from document chunks.

    BM25 (Best Match 25) is a probabilistic keyword ranking algorithm.
    It excels at exact term matching — technical IDs, specific parameter
    names, numeric values — that dense embeddings often miss.

    Args:
        chunks: List of Document objects (same chunks uploaded to Pinecone)
        k: Number of documents to retrieve per query

    Returns:
        BM25Retriever instance ready for use in EnsembleRetriever
    """
    if not chunks:
        raise ValueError("[VectorStore] ❌ Cannot build BM25 retriever: no chunks provided.")

    print(f"[VectorStore] 🔄 Building BM25 index over {len(chunks)} chunks...")
    retriever = BM25Retriever.from_documents(chunks, k=k)
    print(f"[VectorStore] ✅ BM25 index ready ({len(chunks)} documents).")
    return retriever


def build_hybrid_retriever(
    chunks: List[Document],
    namespace: Optional[str] = None,
    source_filter: Optional[dict] = None,
    k: int = 4,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6
) -> EnsembleRetriever:
    """
    Build a Hybrid Retriever combining BM25 + Pinecone vector search
    via Reciprocal Rank Fusion (RRF).

    RRF score formula:
        RRF(d) = Σ 1 / (rank_i(d) + k)
    where rank_i(d) is the rank of document d in retriever i,
    and k=60 is a smoothing constant (LangChain default).

    Weights:
        - vector_weight=0.6 → semantic understanding dominates
        - bm25_weight=0.4  → keyword precision supplements

    Args:
        chunks: Document chunks (required for BM25 in-memory index)
        namespace: Pinecone namespace for isolation
        source_filter: Metadata filter dict for Pinecone (e.g. {"method": "groq_whisper"})
        k: Number of results per retriever (final output = merged RRF ranking)
        bm25_weight: Weight for BM25 results in RRF fusion
        vector_weight: Weight for vector search results in RRF fusion

    Returns:
        EnsembleRetriever combining both retrievers
    """
    print("[VectorStore] 🔄 Building Hybrid Retriever (BM25 + Pinecone)...")
    print(f"[VectorStore] ⚖️  Weights → Vector: {vector_weight} | BM25: {bm25_weight}")

    # ── Dense vector retriever (Pinecone) ──
    vector_store = load_vector_store(namespace=namespace)
    search_kwargs = {"k": k}
    if source_filter:
        search_kwargs["filter"] = source_filter
        print(f"[VectorStore] 🔍 Source filter applied: {source_filter}")

    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # ── Sparse keyword retriever (BM25) ──
    # Apply source filter manually to BM25 since it's in-memory
    filtered_chunks = chunks
    if source_filter and chunks:
        filtered_chunks = _apply_metadata_filter(chunks, source_filter)
        print(f"[VectorStore] 🔍 BM25 filtered to {len(filtered_chunks)} chunks")

    if not filtered_chunks:
        print("[VectorStore] ⚠️  No chunks match filter for BM25 — using dense only.")
        return dense_retriever

    sparse_retriever = build_bm25_retriever(filtered_chunks, k=k)

    # ── Ensemble with RRF fusion ──
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[vector_weight, bm25_weight]
    )

    print("[VectorStore] ✅ Hybrid retriever ready (RRF fusion active).")
    return hybrid_retriever


def _apply_metadata_filter(
    chunks: List[Document],
    filter_dict: dict
) -> List[Document]:
    """
    Apply metadata filtering to BM25 chunks manually.
    Mimics Pinecone's server-side metadata filtering for BM25 in-memory store.

    Supports:
        - Exact match: {"method": "groq_whisper"}
        - Substring match: {"source": "pdf"} matches any source containing "pdf"
    """
    filtered = []
    for doc in chunks:
        match = True
        for key, value in filter_dict.items():
            doc_val = str(doc.metadata.get(key, "")).lower()
            filter_val = str(value).lower()
            # Substring match for flexibility (e.g. "pdf" matches ".pdf" paths)
            if filter_val not in doc_val:
                match = False
                break
        if match:
            filtered.append(doc)
    return filtered


if __name__ == "__main__":
    from ingestor import ingest

    print("\n" + "="*50)
    print("FULL SYNC: PDF + YouTube → Pinecone + BM25 Test")
    print("="*50)

    all_chunks = []

    print("\n--- INGESTING: PDF ---")
    try:
        pdf_path = "data/transformer_paper.pdf"
        if not os.path.exists(pdf_path):
            print(f"[Sync] ⚠️ PDF not found at {pdf_path} — skipping.")
        else:
            pdf_chunks = ingest(source=pdf_path, source_type="pdf")
            print(f"[Sync] ✅ PDF → {len(pdf_chunks)} chunks.")
            all_chunks.extend(pdf_chunks)
    except Exception as e:
        print(f"[Sync] ❌ PDF ingestion failed: {e}")

    print(f"\n--- COMBINED: {len(all_chunks)} total chunks ---")

    if all_chunks:
        vector_store = upload_documents(all_chunks)

        print("\n--- HYBRID RETRIEVER TEST ---")
        hybrid = build_hybrid_retriever(chunks=all_chunks, k=4)

        test_queries = [
            "What is the BLEU score of the base model?",  # numeric — BM25 advantage
            "How does multi-head attention work?",          # semantic — vector advantage
            "β1 β2 epsilon optimizer hyperparameters",    # technical terms — BM25 advantage
        ]

        for query in test_queries:
            print(f"\n🔍 Hybrid Query: '{query}'")
            results = hybrid.invoke(query)
            for i, doc in enumerate(results[:2]):
                source = doc.metadata.get("source", "unknown")
                print(f"  Result {i+1}: {doc.page_content[:100]}...")

    print("\n" + "="*50)
    print("✅ Hybrid retriever test complete.")
    print("="*50)
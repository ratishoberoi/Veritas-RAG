import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
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


if __name__ == "__main__":
    from ingestor import ingest

    print("\n" + "="*50)
    print("FULL SYNC: PDF + YouTube → Pinecone")
    print("="*50)

    all_chunks = []

    print("\n--- INGESTING: PDF ---")
    try:
        pdf_path = "data/specrel.pdf"
        if not os.path.exists(pdf_path):
            print(f"[Sync] ⚠️ PDF not found at {pdf_path} — skipping.")
        else:
            pdf_chunks = ingest(source=pdf_path, source_type="pdf")
            print(f"[Sync] ✅ PDF → {len(pdf_chunks)} chunks.")
            all_chunks.extend(pdf_chunks)
    except Exception as e:
        print(f"[Sync] ❌ PDF ingestion failed: {e}")

    print("\n--- INGESTING: YouTube ---")
    try:
        yt_chunks = ingest(
            source="https://www.youtube.com/watch?v=aircAruvnKk",
            source_type="youtube"
        )
        print(f"[Sync] ✅ YouTube → {len(yt_chunks)} chunks.")
        all_chunks.extend(yt_chunks)
    except Exception as e:
        print(f"[Sync] ❌ YouTube ingestion failed: {e}")

    print(f"\n--- COMBINED: {len(all_chunks)} total chunks ---")

    if not all_chunks:
        print("[Sync] ❌ No chunks to upload. Exiting.")
    else:
        try:
            vector_store = upload_documents(all_chunks)
            print(f"\n[Sync] ✅ Upload complete — {len(all_chunks)} chunks in Pinecone!")

            print("\n--- SEARCH TEST ---")
            test_queries = [
                "What are this person's skills?",
                "What is a neural network?",
            ]
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                results = vector_store.similarity_search(query, k=2)
                for i, doc in enumerate(results):
                    source = doc.metadata.get("source", "unknown")
                    method = doc.metadata.get("method", "pdf")
                    print(f"  Result {i+1} | Source: {source} | Method: {method}")
                    print(f"  📦 {doc.page_content[:120]}...")

        except Exception as e:
            print(f"[Sync] ❌ Upload or search failed: {e}")

    print("\n" + "="*50)
    print("✅ Full sync complete. Check Pinecone dashboard.")
    print("="*50)
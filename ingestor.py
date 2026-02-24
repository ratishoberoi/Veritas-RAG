import os
import tempfile
import time
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, NotionDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import yt_dlp
from groq import Groq

load_dotenv()

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    elif "/embed/" in url or "/shorts/" in url:
        parts = url.split("/")
        return parts[-1].split("?")[0]
    else:
        raise ValueError(f"[Ingestor] ❌ Invalid YouTube URL format: {url}")


def load_pdf(file_path: str) -> List[Document]:
    print(f"[Ingestor] Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[Ingestor] ✅ Loaded {len(documents)} pages from PDF.")
    return documents


def load_notion(dir_path: str) -> List[Document]:
    print(f"[Ingestor] Loading Notion directory: {dir_path}")
    loader = NotionDirectoryLoader(dir_path)
    documents = loader.load()
    print(f"[Ingestor] ✅ Loaded {len(documents)} Notion page(s).")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 300
) -> List[Document]:
    print(f"[Ingestor] Chunking {len(documents)} document(s)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingestor] ✅ Created {len(chunks)} chunks.")
    return chunks


def fetch_via_transcript_api(video_id: str, url: str) -> Optional[List[Document]]:
    print("[Ingestor] 🅰️  Plan A: Trying transcript API (direct)...")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript_obj = transcript_list.find_transcript(["en"])
            transcript_type = "auto" if transcript_obj.is_generated else "manual"
            print(f"[Ingestor] ✅ Found {'auto-generated' if transcript_obj.is_generated else 'manual'} English transcript.")
        except NoTranscriptFound:
            transcript_obj = transcript_list.find_generated_transcript(["en"])
            transcript_type = "auto"
            print("[Ingestor] ⚡ Fallback to auto-generated English transcript.")

        transcript = transcript_obj.fetch()
        full_text = " ".join([entry["text"] for entry in transcript])

        return [Document(
            page_content=full_text,
            metadata={
                "source": url,
                "video_id": video_id,
                "transcript_type": transcript_type,
                "method": "transcript_api"
            }
        )]

    except TranscriptsDisabled:
        print("[Ingestor] ⚠️ Transcripts disabled — switching to Plan B.")
        return None
    except NoTranscriptFound:
        print("[Ingestor] ⚠️ No English transcript found — switching to Plan B.")
        return None
    except Exception as e:
        print(f"[Ingestor] ❌ Plan A failed: {str(e)[:100]}")
        return None


def download_audio(url: str, output_path: str) -> tuple:
    print("[Ingestor] 🎵 Downloading audio via yt-dlp (direct connection)...")

    ydl_opts = {
        "format": "worstaudio/worst",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "user_agent": BROWSER_UA,
        "retries": 3,
        "fragment_retries": 3,
        "http_headers": {
            "User-Agent": BROWSER_UA,
            "Accept-Language": "en-US,en;q=0.9",
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "32",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown Title")

    audio_file = output_path + ".mp3"

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"[Ingestor] ❌ Expected audio file not found: {audio_file}")

    size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    print(f"[Ingestor] ✅ Audio downloaded: {size_mb:.2f}MB | Title: {title}")

    if size_mb > 24:
        print(f"[Ingestor] ⚠️ File is {size_mb:.2f}MB — re-encoding to fit Groq 25MB limit...")
        compressed_path = output_path + "_compressed.mp3"
        os.system(
            f'ffmpeg -i "{audio_file}" -ar 16000 -ac 1 -b:a 16k "{compressed_path}" -y -loglevel quiet'
        )
        os.remove(audio_file)
        audio_file = compressed_path
        size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        print(f"[Ingestor] ✅ Re-encoded to {size_mb:.2f}MB.")

    return audio_file, title


def transcribe_via_groq(audio_path: str) -> str:
    print("[Ingestor] 🤖 Transcribing audio via Groq Whisper (whisper-large-v3)...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("[Ingestor] ❌ GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="text",
            language="en"
        )

    print(f"[Ingestor] ✅ Transcription complete ({len(transcription)} chars).")
    return transcription


def fetch_via_whisper(video_id: str, url: str) -> Optional[List[Document]]:
    print("[Ingestor] 🅱️  Plan B: yt-dlp + Groq Whisper...")
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio")
    audio_file = None
    title = "Unknown Title"

    try:
        try:
            audio_file, title = download_audio(url, output_path)
        except Exception as e:
            print(f"[Ingestor] ⚠️ Audio download failed: {str(e)[:100]}")
            print("[Ingestor] 🔄 Attempting metadata-only fallback...")
            try:
                with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title", "Unknown Title")
                    print(f"[Ingestor] ℹ️ Title retrieved: {title} — but no audio to transcribe.")
            except Exception:
                pass
            return None

        transcript_text = transcribe_via_groq(audio_file)

        return [Document(
            page_content=transcript_text,
            metadata={
                "source": url,
                "video_id": video_id,
                "title": title,
                "transcript_type": "whisper_transcription",
                "method": "groq_whisper"
            }
        )]

    except Exception as e:
        print(f"[Ingestor] ❌ Plan B failed: {str(e)}")
        return None

    finally:
        for f in [audio_file, output_path + ".mp3", output_path + "_compressed.mp3"]:
            if f and os.path.exists(f):
                os.remove(f)
                print(f"[Ingestor] 🗑️  Deleted temp file: {f}")
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


def load_youtube(url: str) -> List[Document]:
    print(f"\n[Ingestor] Loading YouTube: {url}")
    video_id = extract_video_id(url)

    documents = fetch_via_transcript_api(video_id, url)
    if documents:
        print("[Ingestor] ✅ Plan A succeeded.")
        return documents

    print("[Ingestor] ⚠️  Plan A failed — switching to Plan B...")
    documents = fetch_via_whisper(video_id, url)
    if documents:
        print("[Ingestor] ✅ Plan B succeeded.")
        return documents

    print("[Ingestor] ❌ Both plans failed for this video.")
    return []


def ingest(source: str, source_type: str) -> List[Document]:
    if source_type == "pdf":
        documents = load_pdf(source)
    elif source_type == "youtube":
        documents = load_youtube(source)
    elif source_type == "notion":
        documents = load_notion(source)
    else:
        raise ValueError(
            f"[Ingestor] ❌ Unknown source_type: '{source_type}'. "
            f"Use 'pdf', 'youtube', or 'notion'."
        )

    if not documents:
        raise ValueError(f"[Ingestor] ❌ No content loaded from: {source}")

    chunks = chunk_documents(documents)
    return chunks


if __name__ == "__main__":
    print("\n" + "="*50)
    print("FINAL TEST: PDF + YouTube")
    print("="*50)

    print("\n--- TEST 1: PDF ---")
    try:
        pdf_path = "data/sample.pdf"
        if not os.path.exists(pdf_path):
            print(f"[Test] ⚠️ PDF not found at {pdf_path} — skipping.")
        else:
            pdf_chunks = ingest(source=pdf_path, source_type="pdf")
            print(f"[Test] ✅ PDF → {len(pdf_chunks)} chunks created.")
            print(f"[Test] 📦 Sample: {pdf_chunks[0].page_content[:150]}...")
    except Exception as e:
        print(f"[Test] ❌ PDF test failed: {e}")

    print("\n--- TEST 2: YouTube ---")
    try:
        yt_chunks = ingest(
            source="https://www.youtube.com/watch?v=aircAruvnKk",
            source_type="youtube"
        )
        print(f"[Test] ✅ YouTube → {len(yt_chunks)} chunks created.")
        print(f"[Test] 📦 Sample: {yt_chunks[0].page_content[:150]}...")
        print(f"[Test] 📋 Method used: {yt_chunks[0].metadata.get('method', 'unknown')}")
    except Exception as e:
        print(f"[Test] ❌ YouTube test failed: {e}")

    print("\n" + "="*50)
    print("✅ Final ingestor.py test complete.")
    print("="*50)
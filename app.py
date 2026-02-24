import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from ingestor import ingest
from vector_store import upload_documents
from retriever import ask, build_rag_chain, store

load_dotenv()

st.set_page_config(
    page_title="Neural KB — Personal Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-hover: #1e1e2a;
    --accent-primary: #7c6fff;
    --accent-secondary: #ff6f9c;
    --text-primary: #f0f0ff;
    --text-secondary: #8888aa;
    --text-muted: #55556a;
    --border: #2a2a3a;
    --border-accent: #7c6fff44;
}

* { box-sizing: border-box; }

.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }

.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 280px !important;
    max-width: 320px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] > div {
    background: var(--bg-secondary) !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

.stTextInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px var(--border-accent) !important;
}

.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
    padding: 0.4rem 1rem !important;
}
.stButton > button:hover {
    border-color: var(--accent-primary) !important;
    background: var(--bg-hover) !important;
    color: var(--accent-primary) !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

.stRadio > div { gap: 0.5rem; }
.stRadio label { color: var(--text-secondary) !important; font-size: 0.82rem !important; }

hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

.stSpinner > div { border-top-color: var(--accent-primary) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────

def eval_card_html(evaluation: dict) -> str:
    if not evaluation:
        return ""
    score = evaluation.get("confidence_score", 0)
    sufficient = evaluation.get("context_sufficient", False)
    hallucination = evaluation.get("hallucination_detected", False)
    summary = evaluation.get("evaluation_summary", "")
    reason = evaluation.get("hallucination_reason", "")

    if score >= 70 and sufficient and not hallucination:
        accent = "#4dffa6"
        status = "Context Verified"
        icon = "✦"
    elif score >= 40:
        accent = "#ffb347"
        status = "Moderate Confidence"
        icon = "◈"
    else:
        accent = "#ff5f7e"
        status = "Low Confidence"
        icon = "⚠"

    hallucination_row = ""
    if hallucination and reason:
        # Sanitize reason to prevent HTML injection/truncation
        safe_reason = str(reason)[:200].replace('<', '&lt;').replace('>', '&gt;')
        hallucination_row = f"""
        <div style="margin-top:6px;padding:6px 8px;background:#ff5f7e08;
                    border-left:2px solid #ff5f7e;border-radius:0 4px 4px 0;
                    font-size:0.72rem;color:#ff5f7e;">⚡ {safe_reason}</div>"""
    return f"""
    <div style="background:#16161f;border:1px solid {accent}22;border-radius:10px;
                padding:12px 14px;margin:8px 0;border-left:3px solid {accent};">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="color:{accent};font-size:0.85rem;">{icon}</span>
            <span style="font-family:'Syne',sans-serif;font-size:0.8rem;
                         font-weight:600;color:{accent};">{status}</span>
            <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;
                         font-size:0.72rem;color:#55556a;">{score}%</span>
        </div>
        <div style="background:#1e1e2a;border-radius:3px;height:3px;overflow:hidden;margin-bottom:8px;">
            <div style="width:{score}%;height:100%;background:{accent};border-radius:3px;"></div>
        </div>
        <div style="font-size:0.75rem;color:#8888aa;line-height:1.4;">{summary}</div>
        {hallucination_row}
    </div>"""


def source_chip_html(source_label) -> str:
    if not isinstance(source_label, str):
        source_label = str(source_label)

    is_yt = "YouTube" in source_label or "youtube" in source_label.lower()
    icon = "🎥" if is_yt else "📄"
    color = "#ff6f9c" if is_yt else "#7c6fff"
    bg = "#ff6f9c10" if is_yt else "#7c6fff10"
    border = "#ff6f9c30" if is_yt else "#7c6fff30"
    clean_label = source_label.replace('🎥 YouTube: ', '').replace('📄 PDF: ', '')

    return f"""
    <span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
                 background:{bg};border:1px solid {border};border-radius:20px;
                 font-size:0.72rem;color:{color};margin:2px 3px 2px 0;
                 font-family:'Inter',sans-serif;">
        {icon} {clean_label}
    </span>"""


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_session_state():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "retriever": None,
        "ingested_sources": [],
        "active_filter": None,
        "filter_label": "All Sources",
        "kb_loaded": False,
        "turn_count": 0,
        "session_id": "streamlit_session"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_chain(source_filter=None):
    try:
        chain, retriever = build_rag_chain(source_filter=source_filter)
        st.session_state.rag_chain = chain
        st.session_state.retriever = retriever
        st.session_state.kb_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return False


# ─────────────────────────────────────────────
# SIDEBAR — only ingestion + filters + stats
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 8px 0;">
            <div style="font-family:'Syne',sans-serif;font-weight:800;
                        font-size:1.3rem;color:#f0f0ff;letter-spacing:-0.5px;">
                🧠 Neural KB
            </div>
            <div style="font-size:0.72rem;color:#55556a;
                        font-family:'JetBrains Mono',monospace;margin-top:2px;">
                PERSONAL KNOWLEDGE BASE
            </div>
        </div>
        <hr style="border-color:#2a2a3a;margin:8px 0 16px 0;">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:600;
                    color:#7c6fff;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            ⊕ Add Source
        </div>
        """, unsafe_allow_html=True)

        source_type = st.radio(
            "Source Type", ["PDF", "YouTube", "Notion"],
            horizontal=True, label_visibility="collapsed"
        )

        source_input = None
        uploaded_file = None

        if source_type == "PDF":
            uploaded_file = st.file_uploader(
                "Drop PDF here", type=["pdf"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uploaded_file.read())
                tmp.flush()
                source_input = tmp.name

        elif source_type == "YouTube":
            source_input = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
                label_visibility="collapsed"
            )

        elif source_type == "Notion":
            source_input = st.text_input(
                "Notion folder path",
                placeholder="/path/to/notion/export",
                label_visibility="collapsed"
            )

        if st.button("⚡  Ingest & Upload", use_container_width=True):
            if not source_input:
                st.warning("Provide a source first.")
            else:
                with st.spinner("Processing..."):
                    try:
                        chunks = ingest(
                            source=source_input,
                            source_type=source_type.lower()
                        )
                        upload_documents(chunks)
                        label = (
                            uploaded_file.name
                            if source_type == "PDF" and uploaded_file
                            else source_input
                        )
                        short_label = label[:35] + "..." if len(label) > 35 else label
                        st.session_state.ingested_sources.append(f"{source_type}: {short_label}")
                        load_chain(st.session_state.active_filter)
                        st.success(f"✅ {len(chunks)} chunks uploaded!")
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.markdown("<hr style='border-color:#2a2a3a;margin:16px 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:600;
                    color:#7c6fff;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            ◎ Filter Sources
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🌐 All", use_container_width=True):
                st.session_state.active_filter = None
                st.session_state.filter_label = "All Sources"
                load_chain(None)
                st.rerun()
        with col2:
            if st.button("📄 PDF", use_container_width=True):
                st.session_state.active_filter = {"source": "pdf"}
                st.session_state.filter_label = "PDFs Only"
                load_chain({"source": "pdf"})
                st.rerun()
        with col3:
            if st.button("🎥 YT", use_container_width=True):
                st.session_state.active_filter = {"method": "groq_whisper"}
                st.session_state.filter_label = "YouTube Only"
                load_chain({"method": "groq_whisper"})
                st.rerun()

        filter_color = "#7c6fff" if not st.session_state.active_filter else "#ff6f9c"
        st.markdown(f"""
        <div style="text-align:center;font-family:'JetBrains Mono',monospace;
                    font-size:0.7rem;color:{filter_color};margin:6px 0 0 0;">
            ◉ {st.session_state.filter_label}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#2a2a3a;margin:16px 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:600;
                    color:#7c6fff;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;">
            ◈ Session Stats
        </div>
        """, unsafe_allow_html=True)

        turns = st.session_state.turn_count
        sources_count = len(st.session_state.ingested_sources)

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
            <div style="background:#16161f;border:1px solid #2a2a3a;border-radius:8px;
                        padding:10px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;
                             font-weight:800;color:#7c6fff;">{turns}</div>
                <div style="font-size:0.65rem;color:#55556a;
                             font-family:'JetBrains Mono',monospace;margin-top:2px;">TURNS</div>
            </div>
            <div style="background:#16161f;border:1px solid #2a2a3a;border-radius:8px;
                        padding:10px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;
                             font-weight:800;color:#ff6f9c;">{sources_count}</div>
                <div style="font-size:0.65rem;color:#55556a;
                             font-family:'JetBrains Mono',monospace;margin-top:2px;">INGESTED</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.ingested_sources:
            st.markdown("""
            <div style="font-size:0.72rem;color:#55556a;
                        font-family:'JetBrains Mono',monospace;margin-bottom:6px;">
                THIS SESSION:
            </div>
            """, unsafe_allow_html=True)
            for src in st.session_state.ingested_sources:
                icon = "🎥" if "YouTube" in src else "📄"
                st.markdown(f"""
                <div style="font-size:0.72rem;color:#8888aa;padding:3px 0;
                             border-left:2px solid #2a2a3a;padding-left:8px;margin:2px 0;">
                    {icon} {src}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#2a2a3a;margin:16px 0;'>", unsafe_allow_html=True)

        kb_col, clear_col = st.columns(2)
        with kb_col:
            if st.button("🔄 Load KB", use_container_width=True):
                with st.spinner("Connecting..."):
                    if load_chain(st.session_state.active_filter):
                        st.success("✅ Loaded!")
        with clear_col:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.turn_count = 0
                session_id = st.session_state.session_id
                if session_id in store:
                    from langchain_community.chat_message_histories import ChatMessageHistory
                    store[session_id] = ChatMessageHistory()
                st.rerun()

        st.markdown("""
        <div style="margin-top:16px;text-align:center;font-size:0.65rem;
                    color:#2a2a3a;font-family:'JetBrains Mono',monospace;">
            LangChain · Pinecone · Groq · Streamlit
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT AREA — question input lives here only
# ─────────────────────────────────────────────

def render_chat():
    st.markdown("""
    <div style="padding:24px 0 8px 0;border-bottom:1px solid #2a2a3a;margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.8rem;
                    color:#f0f0ff;letter-spacing:-1px;">
            Personal Knowledge Base
        </div>
        <div style="font-size:0.82rem;color:#55556a;margin-top:4px;font-family:'Inter',sans-serif;">
            Chat with your PDFs, YouTube videos, and Notion pages — powered by Llama 3 + Groq
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.kb_loaded:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;padding:60px 20px;text-align:center;">
            <div style="font-size:3rem;margin-bottom:16px;">🧠</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        font-weight:700;color:#f0f0ff;margin-bottom:8px;">
                Knowledge Base Not Loaded
            </div>
            <div style="font-size:0.82rem;color:#55556a;max-width:360px;line-height:1.6;">
                Click <strong style="color:#7c6fff;">Load KB</strong> in the sidebar to connect
                to your existing Pinecone index, or ingest a new source to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("evaluation"):
                    st.markdown(eval_card_html(message["evaluation"]), unsafe_allow_html=True)
                if message.get("sources"):
                    with st.expander("📚 Sources Used"):
                        chips = "".join([source_chip_html(s) for s in message["sources"]])
                        st.markdown(f'<div style="padding:4px 0;">{chips}</div>', unsafe_allow_html=True)

    if question := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(
                        question=question,
                        session_id=st.session_state.session_id,
                        source_filter=st.session_state.active_filter
                    )

                    answer = result["answer"]
                    evaluation = result["evaluation"]
                    sources = result["sources"]

                    st.session_state.turn_count += 1
                    st.markdown(answer)

                    if evaluation:
                        st.markdown(eval_card_html(evaluation), unsafe_allow_html=True)

                    if sources:
                        with st.expander("📚 Sources Used"):
                            chips = "".join([source_chip_html(s) for s in sources])
                            st.markdown(f'<div style="padding:4px 0;">{chips}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "evaluation": evaluation,
                        "sources": sources
                    })

                except Exception as e:
                    import traceback
                    print(f"CHAT ERROR:\n{traceback.format_exc()}")
                    err = f"❌ Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err,
                        "evaluation": None,
                        "sources": []
                    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
"""
Meeting Intelligence - Streamlit UI
Run with: streamlit run app.py
Place this file next to main.py (project root).
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_questions

st.set_page_config(page_title="Meeting Intelligence", page_icon="🎙️", layout="centered")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
INK = "#12202B"
ACCENT = "#0E7C7B"
TINT = "#E6F2F1"
LINE = "#E3E7EA"
MUTED = "#5B6B76"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

    html, body, [class*="css"], .stApp {{ font-family: 'Instrument Sans', sans-serif; color: {INK}; }}
    .stApp {{ background: #F8F9FA; }}
    #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
    .block-container {{ max-width: 860px; padding-top: 3rem; padding-bottom: 5rem; }}

    h1, h2, h3 {{ font-family: 'Newsreader', serif !important; font-weight: 500 !important; letter-spacing: -0.01em; color: {INK}; }}

    .brand {{ font-weight: 600; font-size: 1rem; display: flex; align-items: center; gap: .5rem; }}
    .brand i {{ width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; display: inline-block; }}

    .hero h1 {{ font-size: clamp(2.4rem, 6vw, 3.9rem); line-height: 1.05; margin: 3.5rem 0 1rem; max-width: 14em; }}
    .hero p {{ font-size: 1.15rem; color: {MUTED}; max-width: 34em; line-height: 1.55; margin-bottom: 1.6rem; }}

    .wave {{ display: flex; align-items: center; gap: 4px; height: 96px; margin: 2.8rem 0 2.2rem; }}
    .wave span {{ flex: 1; border-radius: 3px; background: #CBD5DA; min-width: 3px; }}
    .wave span.hl {{ background: {ACCENT}; }}

    .features {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-top: 1rem; border-top: 1px solid {LINE}; padding-top: 1.6rem; }}
    .features h4 {{ font-size: 1rem; margin: 0 0 .35rem; font-weight: 600; }}
    .features p {{ font-size: .92rem; color: {MUTED}; margin: 0; line-height: 1.5; }}
    @media (max-width: 700px) {{ .features {{ grid-template-columns: 1fr; gap: 1.2rem; }} }}

    .page-title {{ font-family: 'Newsreader', serif; font-size: 2.3rem; margin: 2rem 0 .3rem; line-height: 1.15; }}
    .page-sub {{ color: {MUTED}; margin-bottom: 1.6rem; }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
      border-radius: 8px; border: 1px solid {LINE}; background: #fff; color: {INK};
      font-weight: 500; padding: .55rem 1.2rem; transition: border-color .15s, background .15s;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; background: #fff; }}
    .stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
      background: {ACCENT}; border-color: {ACCENT}; color: #fff;
    }}
    .stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
      background: #0B6665; border-color: #0B6665; color: #fff;
    }}

    /* Inputs, tabs */
    [data-testid="stFileUploader"] section {{ background: #fff; border: 1.5px dashed #B9C4CA; border-radius: 12px; padding: 1.6rem; }}
    .stTextInput input {{ border-radius: 8px; border: 1px solid {LINE}; background: #fff; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{ padding: .6rem 0; font-weight: 500; color: {MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {INK}; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}

    /* Result sections */
    .section-label {{ font-weight: 600; font-size: 1rem; margin: 1.6rem 0 .4rem; }}
    .panel {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: .8rem; line-height: 1.6; }}

    /* Chat */
    [data-testid="stChatMessage"] {{ background: transparent; padding: .6rem 0; }}
    [data-testid="stChatInput"] {{ border-radius: 12px; }}
    .src {{ font-size: .88rem; color: {MUTED}; border-left: 2px solid {LINE}; padding-left: .8rem; margin: .5rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
DEFAULTS = {
    "page": "landing",
    "result": None,
    "messages": [],
    "source_name": None,
    "analysis_running": False,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def go(page: str):
    st.session_state.page = page
    st.rerun()


def reset():
    st.session_state.result = None
    st.session_state.messages = []
    st.session_state.source_name = None
    st.session_state.analysis_running = False
    go("upload")


def show(content):
    """Render a string or a list returned by the extractors."""
    if not content:
        st.caption("Nothing found.")
    elif isinstance(content, (list, tuple)):
        st.markdown("\n".join(f"- {item}" for item in content))
    else:
        st.markdown(content)


def brand():
    st.markdown('<div class="brand"><i></i>Meeting Intelligence</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Pipeline (same steps as main.py, with progress reporting)
# --------------------------------------------------------------------------
def analyse(source: str, language: str) -> dict:
    with st.status("Analysing your video", expanded=True) as status:
        st.write("Extracting audio")
        chunks = process_input(source)

        st.write("Transcribing speech")
        transcript = transcribe_all(chunks, language)

        st.write("Writing summary")
        title = generate_title(transcript)
        summary = summarize(transcript)

        st.write("Finding action items, decisions and open questions")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        st.write("Preparing the recording for questions")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Analysis complete", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def landing_page():
    brand()

    # Waveform: grey bars are speech, teal bars are the moments worth keeping.
    heights = [
        18, 34, 52, 28, 64, 80, 44, 22, 38, 70, 90, 58, 30, 16, 42, 76, 60, 36, 24, 48,
        84, 66, 40, 20, 32, 56, 72, 46, 26, 54, 88, 62, 34, 18, 44, 68, 50, 28, 22, 36
    ]
    highlights = {5, 10, 15, 20, 30, 35}
    bars = "".join(
        f'<span class="{"hl" if i in highlights else ""}" style="height:{h}%"></span>'
        for i, h in enumerate(heights)
    )

    st.markdown(
        f"""
        <div class="hero">
        <h1>Turn a meeting recording into answers.</h1>
        <p>Upload a video or paste a YouTube link. You get a summary, the action items and decisions, then you can ask the recording anything.</p>
        </div>
        <div class="wave">{bars}</div>
        """,
        unsafe_allow_html=True,
    )

    col, _ = st.columns([1, 2])
    with col:
        if st.button("Analyse a video", type="primary", use_container_width=True):
            go("upload")

    st.markdown(
        """
        <div class="features" style="margin-top:3.5rem">
        <div><h4>Summary</h4><p>A short title and overview, so you know what was covered without watching it.</p></div>
        <div><h4>Actions and decisions</h4><p>Who needs to do what, what was agreed, and what is still open.</p></div>
        <div><h4>Ask the recording</h4><p>Ask in plain language. Every answer shows the part of the transcript it came from.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def upload_page():
    brand()
    st.markdown('<div class="page-title">Add a recording</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a file or paste a YouTube link. Longer videos take a few minutes.</div>',
        unsafe_allow_html=True,
    )

    tab_file, tab_url = st.tabs(["Upload a file", "YouTube link"])
    uploaded, url = None, ""

    with tab_file:
        uploaded = st.file_uploader(
            "Video or audio file",
            type=["mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a"],
            label_visibility="collapsed",
        )

    with tab_url:
        url = st.text_input(
            "YouTube link",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        ).strip()

    language = st.radio(
        "Spoken language",
        ["english", "hinglish"],
        horizontal=True,
        format_func=str.capitalize,
    )

    c1, c2, _ = st.columns([1.2, 1, 2])

    with c1:
        start = st.button(
            "Analyse recording",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
        )

    with c2:
        if st.button("Back", use_container_width=True):
            go("landing")

    if not start:
        return

    if not uploaded and not url:
        st.warning("Add a file or paste a YouTube link to continue.")
        return

    # Prevent duplicate analysis
    if st.session_state.analysis_running:
        st.warning("Analysis is already running.")
        return

    st.session_state.analysis_running = True

    tmp_path = None

    try:
        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1]

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            source, name = tmp_path, uploaded.name

        else:
            source, name = url, url

        st.session_state.result = analyse(source, language)
        st.session_state.source_name = name
        st.session_state.messages = []

    except Exception as e:
        st.error(f"Analysis failed: {e}. Check the file or link and try again.")
        return

    finally:
        st.session_state.analysis_running = False

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    go("workspace")


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------
def workspace_page():
    result = st.session_state.result

    if not result:
        go("upload")
        return

    top_l, top_r = st.columns([4, 1.3])

    with top_l:
        brand()

    with top_r:
        if st.button("New recording", use_container_width=True):
            reset()

    st.markdown(f'<div class="page-title">{result["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{st.session_state.source_name}</div>', unsafe_allow_html=True)

    tab_overview, tab_ask, tab_transcript = st.tabs(["Overview", "Ask", "Transcript"])

    with tab_overview:
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        with st.container(border=True):
            show(result["summary"])

        st.markdown('<div class="section-label">Action items</div>', unsafe_allow_html=True)
        with st.container(border=True):
            show(result["action_items"])

        st.markdown('<div class="section-label">Key decisions</div>', unsafe_allow_html=True)
        with st.container(border=True):
            show(result["key_decisions"])

        st.markdown('<div class="section-label">Open questions</div>', unsafe_allow_html=True)
        with st.container(border=True):
            show(result["open_questions"])

    with tab_ask:
        ask_tab()

    with tab_transcript:
        st.download_button(
            "Download transcript",
            result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )
        with st.container(border=True, height=520):
            st.write(result["transcript"])


def render_sources(sources):
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            score = s.get("score")
            score_txt = f" · relevance {score:.2f}" if isinstance(score, (int, float)) else ""
            st.markdown(
                f'<div class="src"><b>Part {s.get("chunk_index")}</b>{score_txt}<br>{s.get("content")}</div>',
                unsafe_allow_html=True,
            )


def ask_tab():
    messages = st.session_state.messages

    if not messages:
        st.caption("Ask anything about the recording, for example: “What did we decide about the deadline?”")

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources"))

    question = st.chat_input("Ask a question about this recording")

    if question:
        messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript"):
                try:
                    res = ask_questions(st.session_state.result["rag_chain"], question)
                    answer, sources = res["answer"], res.get("sources", [])
                except Exception as e:
                    answer, sources = f"I couldn't answer that: {e}", []

            st.markdown(answer)
            render_sources(sources)

        messages.append({"role": "assistant", "content": answer, "sources": sources})


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
PAGES = {"landing": landing_page, "upload": upload_page, "workspace": workspace_page}
PAGES[st.session_state.page]()
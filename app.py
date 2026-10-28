import os
from dotenv import load_dotenv

# Load environment variables BEFORE importing modules
load_dotenv()

import streamlit as st
import time

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from core.realtime_analyzer import (
    RealtimeAudioBuffer,
    transcribe_audio_bytes,
    update_live_summary,
    answer_live_question,
)


# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "realtime_buffer": None,
    "realtime_transcript": "",
    "realtime_summary": "",
    "realtime_pending_summary": "",
    "realtime_chunk_count": 0,
    "realtime_chat_history": [],
    "realtime_final_report": None,
    "realtime_error": None,
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    status = steps.get(key, "pending")
    if status == "active":
        return "dot-active"
    if status == "done":
        return "dot-done"
    return "dot-pending"


def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-dot {css}"></div>
            <span>{icon} {label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def append_realtime_transcript(text: str):
    """Append one recognized segment to the full and pending transcripts."""
    clean_text = (text or "").strip()
    if not clean_text:
        return

    if st.session_state.realtime_transcript:
        st.session_state.realtime_transcript += " " + clean_text
    else:
        st.session_state.realtime_transcript = clean_text

    if st.session_state.realtime_pending_summary:
        st.session_state.realtime_pending_summary += " " + clean_text
    else:
        st.session_state.realtime_pending_summary = clean_text


def transcribe_realtime_chunk(audio_bytes: bytes, language: str):
    """Transcribe a WAV chunk and update session state."""
    text = transcribe_audio_bytes(audio_bytes, language=language)
    if text:
        append_realtime_transcript(text)
        st.session_state.realtime_chunk_count += 1

        # Refresh the live summary after every 5 successfully transcribed chunks.
        if st.session_state.realtime_chunk_count % 5 == 0:
            pending_text = st.session_state.realtime_pending_summary.strip()
            if pending_text:
                st.session_state.realtime_summary = update_live_summary(
                    pending_text,
                    st.session_state.realtime_summary,
                )
                st.session_state.realtime_pending_summary = ""


def drain_realtime_audio(language: str, max_chunks: int | None = None):
    """Transcribe queued audio chunks. If max_chunks is None, drain the full queue."""
    buffer = st.session_state.realtime_buffer
    processed = 0

    while buffer is not None:
        if max_chunks is not None and processed >= max_chunks:
            break

        audio_bytes = buffer.get_chunk()
        if audio_bytes is None:
            break

        transcribe_realtime_chunk(audio_bytes, language)
        processed += 1


def flush_realtime_buffer(buffer):
    """Return any final partial audio, including when the helper method is absent."""
    if hasattr(buffer, "flush_pending"):
        return buffer.flush_pending()

    # Compatibility fallback for the realtime_analyzer.py version without flush_pending().
    with buffer.lock:
        if len(buffer.pending) == 0:
            return None
        samples = buffer.pending.copy()
        buffer.pending = buffer.pending[:0]
    return buffer._to_wav_bytes(samples)


@st.fragment(run_every="1s")
def realtime_processing_panel():
    """Poll the microphone audio queue without rerunning the entire app."""
    if st.session_state.realtime_buffer is None:
        st.info("Microphone buffer is not initialize.refresh the page and try again.")
        return

    try:
        # Limit each poll so the UI remains responsive if several chunks are queued.
        drain_realtime_audio(
            st.session_state.get("realtime_language", "english"),
            max_chunks=3,
        )
    except Exception as exc:
        st.session_state.realtime_error = str(exc)

    if st.session_state.realtime_error:
        st.error(f"Realtime processing error: {st.session_state.realtime_error}")

    st.markdown("### Live Transcript")
    transcript = st.session_state.realtime_transcript.strip()
    if transcript:
        st.code(transcript, language=None)
    else:
        st.info("start the mic and start speaking. Transcript will show here.")

    st.markdown("### Live Summary")
    summary = st.session_state.realtime_summary.strip()
    if summary:
        st.markdown(summary)
    else:
        st.caption("Summary will update after processing first five chunks.")


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    mode = st.radio(
        "Realtime Meeting Analyzer",
        ["Video Analysis", "Realtime Meeting Analyzer"],
        key="analysis_mode",
    )

    if mode == "Video Analysis":
        st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
        source = st.text_input(
            "YouTube URL or File Path",
            placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
        )
        language = st.selectbox("Language", ["english", "hinglish"], index=0)
        run_btn = st.button("⚡  Analyse", use_container_width=True)
    else:
        source = ""
        language = "english"
        run_btn = False

    if mode == "Video Analysis" and st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio", "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title", "🏷️", "Title Generation"),
            ("summary", "📋", "Summarisation"),
            ("extract", "🔍", "Extraction"),
            ("rag", "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)


# ─── Main Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─── Existing Video / Recorded Audio Pipeline ───────────────────────────────────
if mode == "Video Analysis" and run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see sidebar for live status…")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as exc:
            for key in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(key) == "active":
                    st.session_state.pipeline_steps[key] = "pending"
            progress_placeholder.error(f"❌ Error: {exc}")


# ─── Realtime Meeting Analyzer ───────────────────────────────────────────────────
if mode == "Realtime Meeting Analyzer":
    st.markdown("## 🎙️ Realtime Meeting Analyzer")
    st.write(
        "Browser's microphone will capture the meeting audio."
        "speech get transcribe into small chunks, live summary will be update"
        "now you can ask questions regarding meeting."
    )
    st.warning(
        "Browser microphone captures the sound."
        "if meeting is audible into the microphone, mic can capture that audio if possible."
    )

    realtime_language = st.selectbox(
        "Meeting language",
        ["english", "hinglish"],
        index=0,
        key="realtime_language_select",
    )
    st.session_state.realtime_language = realtime_language

    if st.session_state.realtime_buffer is None:
        st.session_state.realtime_buffer = RealtimeAudioBuffer()

    control_col1, control_col2 = st.columns([1, 2])
    with control_col1:
        reset_clicked = st.button("🧹 Start New Meeting / Reset", use_container_width=True)
    with control_col2:
        st.caption(
            "first click on start and then provide the microphones permission.after ending the meeting"
            " press WebRTC widget Stop,then click on the button finish meeting."
        )

    if reset_clicked:
        st.session_state.realtime_buffer.reset()
        st.session_state.realtime_transcript = ""
        st.session_state.realtime_summary = ""
        st.session_state.realtime_pending_summary = ""
        st.session_state.realtime_chunk_count = 0
        st.session_state.realtime_chat_history = []
        st.session_state.realtime_final_report = None
        st.session_state.realtime_error = None
        st.rerun()

    webrtc_ctx = webrtc_streamer(
        key="realtime-meeting-microphone",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=st.session_state.realtime_buffer.audio_frame_callback,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    st.caption("Microphone status: " + ("running" if webrtc_ctx.state.playing else "stopped"))
    realtime_processing_panel()

    st.markdown("---")
    finish_clicked = st.button("🏁 Finish Meeting & Generate Report", use_container_width=True)

    if finish_clicked:
        if webrtc_ctx.state.playing:
            st.warning("first press the microphone widgets stop button,then finish the meeting")
        elif not st.session_state.realtime_transcript.strip() and st.session_state.realtime_buffer.audio_queue.empty():
            st.warning("transcript not formed till now.start the microphone and speak something,then finish it.")
        else:
            try:
                with st.spinner("Remaining audio is processing and final report is getting ready"):
                    buffer = st.session_state.realtime_buffer

                    # Process all complete chunks first, in chronological order.
                    drain_realtime_audio(realtime_language, max_chunks=None)

                    # Process the final partial chunk, if any.
                    remaining_audio = flush_realtime_buffer(buffer)
                    if remaining_audio:
                        transcribe_realtime_chunk(remaining_audio, realtime_language)

                    full_transcript = st.session_state.realtime_transcript.strip()
                    if not full_transcript:
                        st.warning("speech not detected from audio.speak and try again.")
                    else:
                        final_title = generate_title(full_transcript)
                        final_summary = summarize(full_transcript)
                        final_actions = extract_action_items(full_transcript)
                        final_decisions = extract_key_decisions(full_transcript)
                        final_questions = extract_questions(full_transcript)

                        st.session_state.realtime_summary = final_summary
                        st.session_state.realtime_pending_summary = ""
                        st.session_state.realtime_final_report = {
                            "title": final_title,
                            "summary": final_summary,
                            "transcript": full_transcript,
                            "action_items": final_actions,
                            "key_decisions": final_decisions,
                            "open_questions": final_questions,
                        }
            except Exception as exc:
                st.session_state.realtime_error = str(exc)
                st.error(f"Final report error: {exc}")

    # Show final report if generated.
    final_report = st.session_state.realtime_final_report
    if final_report:
        st.markdown("---")
        st.markdown("## 📄 Final Meeting Report")
        st.markdown(f"### {final_report['title']}")

        report_col1, report_col2 = st.columns([3, 2], gap="medium")
        with report_col1:
            st.markdown("### 📋 Summary")
            st.markdown(final_report["summary"])
        with report_col2:
            st.markdown("### 📝 Full Transcript")
            st.text_area(
                "Final transcript",
                value=final_report["transcript"],
                height=260,
                key="realtime_final_transcript_display",
                disabled=True,
            )

        action_col, decision_col, question_col = st.columns(3, gap="medium")
        with action_col:
            st.markdown("### ✅ Action Items")
            st.markdown(final_report["action_items"])
        with decision_col:
            st.markdown("### 🔑 Key Decisions")
            st.markdown(final_report["key_decisions"])
        with question_col:
            st.markdown("### ❓ Open Questions")
            st.markdown(final_report["open_questions"])

    # Q&A works on the transcript captured so far, even before the final report.
    st.markdown("---")
    st.markdown("## 💬 Ask the Live Meeting")
    for message in st.session_state.realtime_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("realtime_meeting_question_form", clear_on_submit=True):
        realtime_question = st.text_input(
            "Ask a question",
            placeholder="What decisions have been discussed so far?",
        )
        ask_clicked = st.form_submit_button("Ask Meeting")

    if ask_clicked and realtime_question.strip():
        try:
            with st.spinner("searching answer from meeting transcript..."):
                answer = answer_live_question(
                    st.session_state.realtime_transcript,
                    realtime_question.strip(),
                    st.session_state.realtime_summary,
                )
            st.session_state.realtime_chat_history.append(
                {"role": "user", "content": realtime_question.strip()}
            )
            st.session_state.realtime_chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Question answer error: {exc}")


# ─── Recorded-Mode Results ──────────────────────────────────────────────────────
if mode == "Video Analysis" and st.session_state.result:
    r = st.session_state.result

    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(
                f'<div class="transcript-box">{r["transcript"]}</div>',
                unsafe_allow_html=True,
            )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """<div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>""",
        unsafe_allow_html=True,
    )

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""", unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input(
            "Your question",
            placeholder="What were the main decisions made?",
            label_visibility="collapsed",
            key="recorded_mode_question",
        )
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

elif mode == "Video Analysis":
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)

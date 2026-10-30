import os
from dotenv import load_dotenv

# Load environment variables BEFORE importing modules
load_dotenv()

import time
import streamlit as st

from core.auth import sign_up, sign_in, sign_out, get_current_user

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

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ────────────────────────────────────────────────────────────────────
# Tip: add .streamlit/config.toml with  [theme] base="light"  so native widgets match.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0f172a;
    --ink-2: #475569;
    --ink-3: #94a3b8;
    --paper: #f5f7fa;
    --white: #ffffff;
    --line: #e2e8f0;
    --brand: #2446d6;
    --brand-soft: #e8edff;
    --live: #e5484d;
    --ok: #12906b;
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Manrope', system-ui, sans-serif !important;
}
.stApp { background: var(--paper) !important; color: var(--ink); }
.block-container { padding-top: 1.6rem !important; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.01em; font-weight: 700 !important; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }
p, li, span, label { color: var(--ink); }
hr { border: none !important; border-top: 1px solid var(--line) !important; margin: 1.25rem 0 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--white) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

.brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .25rem; }
.brand-mark {
    width: 34px; height: 34px; border-radius: 9px; background: var(--brand);
    color: #fff; display: grid; place-items: center; font-size: 1.05rem;
}
.brand-name { font-weight: 800; font-size: 1.05rem; color: var(--ink); line-height: 1.1; }
.brand-sub { font-size: .75rem; color: var(--ink-3); }

/* Page header */
.page-title { font-size: 1.85rem; font-weight: 800; color: var(--ink); letter-spacing: -0.02em; margin: 0; }
.page-sub { color: var(--ink-2); font-size: .95rem; margin: .2rem 0 0; }

/* Bordered containers act as cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white);
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}

/* Meeting header */
.meeting-title { font-size: 1.5rem; font-weight: 800; color: var(--ink); margin: 0 0 .35rem; letter-spacing: -0.01em; }
.chip {
    display: inline-flex; align-items: center; gap: .35rem;
    padding: .18rem .6rem; border-radius: 999px; font-size: .76rem; font-weight: 600;
    background: var(--brand-soft); color: var(--brand); margin-right: .4rem;
}
.chip-ok { background: #e3f6ef; color: var(--ok); }
.chip-live { background: #fdeceb; color: var(--live); }
.chip-idle { background: #eef1f5; color: var(--ink-2); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.chip-live .dot { animation: blink 1.4s infinite; }
@keyframes blink { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) { .chip-live .dot { animation: none; } }

/* Pipeline steps */
.step { display: flex; align-items: center; gap: .6rem; padding: .35rem 0; font-size: .85rem; color: var(--ink-2); }
.step .dot { flex-shrink: 0; }
.step-done .dot { background: var(--ok); }
.step-active .dot { background: var(--brand); animation: blink 1.2s infinite; }
.step-pending .dot { background: var(--line); }

/* Inputs & buttons */
.stTextInput input, .stSelectbox > div > div, .stTextArea textarea {
    background: var(--white) !important; border: 1px solid var(--line) !important;
    border-radius: 9px !important; color: var(--ink) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--brand) !important; box-shadow: 0 0 0 3px var(--brand-soft) !important;
}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 9px !important; font-weight: 600 !important; padding: .5rem 1.1rem !important;
    border: 1px solid var(--line) !important; background: var(--white) !important; color: var(--ink) !important;
    transition: background .15s, border-color .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--brand) !important; color: var(--brand) !important; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {
    background: var(--brand) !important; border-color: var(--brand) !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #1b37ad !important; color: #fff !important;
}
.stButton > button:focus-visible { outline: 3px solid var(--brand-soft) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--ink-2); padding: .55rem 1rem; }
.stTabs [aria-selected="true"] { color: var(--brand) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--brand) !important; }

/* Transcript */
.transcript {
    font-size: .9rem; line-height: 1.85; color: var(--ink-2);
    max-height: 420px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
}
.empty { text-align: center; padding: 4.5rem 1rem; color: var(--ink-2); }
.empty h3 { margin-bottom: .35rem; }
.empty p { max-width: 420px; margin: 0 auto; font-size: .92rem; line-height: 1.6; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session state ──────────────────────────────────────────────────────────────
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


# ─── Authentication gate ────────────────────────────────────────────────────────
def _user_email(user):
    """Read the email from either a Supabase User object or a dictionary."""
    if user is None:
        return ""
    if isinstance(user, dict):
        return user.get("email", "") or ""
    return getattr(user, "email", "") or ""


def render_auth_page():
    """Show login/sign-up UI until a Supabase user is authenticated."""
    st.markdown(
        '<div class="page-title">Welcome to Meeting Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="page-sub">Sign in to access your meeting workspace, or create a new account.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    left, center, right = st.columns([1, 1.35, 1])
    with center:
        with st.container(border=True):
            login_tab, signup_tab = st.tabs(["Login", "Create account"])

            with login_tab:
                with st.form("login_form"):
                    email = st.text_input("Email address", placeholder="you@example.com")
                    password = st.text_input("Password", type="password")
                    login_clicked = st.form_submit_button(
                        "Login", type="primary", use_container_width=True
                    )

                if login_clicked:
                    if not email.strip() or not password:
                        st.error("Enter both your email address and password.")
                    else:
                        try:
                            response = sign_in(email.strip(), password)
                            user = getattr(response, "user", None)
                            session = getattr(response, "session", None)
                            if user is None or session is None:
                                st.error(
                                    "Login could not be completed. Check your email verification "
                                    "and credentials, then try again."
                                )
                            else:
                                st.session_state["authenticated_user"] = user
                                st.success("Login successful.")
                                st.rerun()
                        except Exception as exc:
                            st.error(f"Login failed: {exc}")

            with signup_tab:
                with st.form("signup_form"):
                    new_email = st.text_input(
                        "Email address", placeholder="you@example.com", key="signup_email"
                    )
                    new_password = st.text_input(
                        "Password", type="password", key="signup_password"
                    )
                    confirm_password = st.text_input(
                        "Confirm password", type="password", key="signup_confirm_password"
                    )
                    signup_clicked = st.form_submit_button(
                        "Create account", type="primary", use_container_width=True
                    )

                if signup_clicked:
                    if not new_email.strip() or not new_password or not confirm_password:
                        st.error("Complete all fields to create your account.")
                    elif new_password != confirm_password:
                        st.error("The passwords do not match.")
                    elif len(new_password) < 6:
                        st.error("Use a password with at least 6 characters.")
                    else:
                        try:
                            response = sign_up(new_email.strip(), new_password)
                            user = getattr(response, "user", None)
                            session = getattr(response, "session", None)
                            if user is not None and session is not None:
                                st.session_state["authenticated_user"] = user
                                st.success("Account created and signed in successfully.")
                                st.rerun()
                            else:
                                st.success(
                                    "Account created. Check your inbox and verify your email, "
                                    "then return here and log in."
                                )
                        except Exception as exc:
                            st.error(f"Could not create account: {exc}")


# The meeting workspace is available only after successful authentication.
if not get_current_user():
    render_auth_page()
    st.stop()


# ─── Helpers ────────────────────────────────────────────────────────────────────
def render_step(label: str, key: str):
    status = st.session_state.pipeline_steps.get(key, "pending")
    st.markdown(
        f'<div class="step step-{status}"><span class="dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def build_report_md(report: dict) -> str:
    return (
        f"# {report['title']}\n\n"
        f"## Summary\n{report['summary']}\n\n"
        f"## Action items\n{report['action_items']}\n\n"
        f"## Key decisions\n{report['key_decisions']}\n\n"
        f"## Open questions\n{report['open_questions']}\n\n"
        f"## Transcript\n{report['transcript']}\n"
    )


def render_report(report: dict, key_prefix: str):
    """Shared, tabbed layout for finished meeting reports."""
    with st.container(border=True):
        st.markdown(f'<div class="meeting-title">{report["title"]}</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="chip chip-ok"><span class="dot"></span>Analysis complete</span>',
            unsafe_allow_html=True,
        )

    tab_overview, tab_notes, tab_transcript = st.tabs(["Overview", "Notes", "Transcript"])

    with tab_overview:
        with st.container(border=True):
            st.markdown("### Summary")
            st.markdown(report["summary"])

    with tab_notes:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            with st.container(border=True):
                st.markdown("### ✅ Action items")
                st.markdown(report["action_items"])
        with c2:
            with st.container(border=True):
                st.markdown("### 🔑 Key decisions")
                st.markdown(report["key_decisions"])
        with c3:
            with st.container(border=True):
                st.markdown("### ❓ Open questions")
                st.markdown(report["open_questions"])

    with tab_transcript:
        with st.container(border=True):
            st.markdown(
                f'<div class="transcript">{report["transcript"]}</div>',
                unsafe_allow_html=True,
            )

    st.download_button(
        "Download report (.md)",
        data=build_report_md(report),
        file_name="meeting_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_download",
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
        st.info("The microphone buffer is not initialised. Refresh the page and try again.")
        return

    try:
        # Limit each poll so the UI stays responsive if several chunks are queued.
        drain_realtime_audio(
            st.session_state.get("realtime_language", "english"),
            max_chunks=3,
        )
    except Exception as exc:
        st.session_state.realtime_error = str(exc)

    if st.session_state.realtime_error:
        st.error(f"Realtime processing error: {st.session_state.realtime_error}")

    col_t, col_s = st.columns([3, 2], gap="medium")
    with col_t:
        with st.container(border=True):
            st.markdown("### Live transcript")
            transcript = st.session_state.realtime_transcript.strip()
            if transcript:
                st.markdown(f'<div class="transcript">{transcript}</div>', unsafe_allow_html=True)
            else:
                st.caption("Start the microphone and speak. The transcript appears here.")
    with col_s:
        with st.container(border=True):
            st.markdown("### Live summary")
            summary = st.session_state.realtime_summary.strip()
            if summary:
                st.markdown(summary)
            else:
                st.caption("The summary updates after the first five transcribed chunks.")


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    current_user = get_current_user()
    current_email = _user_email(current_user)
    if current_email:
        st.caption(f"Signed in as {current_email}")
    if st.button("Log out", use_container_width=True, key="logout_button"):
        try:
            sign_out()
        finally:
            st.session_state.pop("authenticated_user", None)
        st.rerun()

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🎙️</div>
            <div>
                <div class="brand-name">Meeting Assistant</div>
                <div class="brand-sub">Transcribe, summarise, ask</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    mode = st.radio(
        "Workspace",
        ["Video Analysis", "Realtime Meeting Analyzer"],
        key="analysis_mode",
    )

    if mode == "Video Analysis":
        source = st.text_input(
            "YouTube URL or file path",
            placeholder="https://youtube.com/watch?v=… or /path/to/file.mp4",
        )
        language = st.selectbox("Language", ["english", "hinglish"], index=0)
        run_btn = st.button("Analyse meeting", type="primary", use_container_width=True)
    else:
        source = ""
        language = "english"
        run_btn = False

    if mode == "Video Analysis" and st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown("**Pipeline**")
        for step, label in [
            ("audio", "Audio processing"),
            ("transcript", "Transcription"),
            ("title", "Title generation"),
            ("summary", "Summarisation"),
            ("extract", "Extraction"),
            ("rag", "Chat index"),
        ]:
            render_step(label, step)


# ─── Page header ────────────────────────────────────────────────────────────────
if mode == "Video Analysis":
    st.markdown('<h1 class="page-title">Video analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Turn a recording into a transcript, summary, action items and a chat you can query.</p>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<h1 class="page-title">Realtime meeting</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Capture audio from your microphone and follow the meeting as it happens.</p>',
        unsafe_allow_html=True,
    )
st.markdown("---")


# ─── Video / recorded audio pipeline ────────────────────────────────────────────
if mode == "Video Analysis" and run_btn:
    if not source.strip():
        st.error("Enter a YouTube URL or a file path to continue.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        status_box = st.status("Analysing your meeting…", expanded=True)

        try:
            status_box.write("Processing audio")
            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            status_box.write("Transcribing")
            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            status_box.write("Generating title")
            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            status_box.write("Writing summary")
            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            status_box.write("Extracting action items, decisions and questions")
            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done")

            status_box.write("Building chat index")
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
            status_box.update(label="Analysis complete", state="complete", expanded=False)
            time.sleep(0.5)
            st.rerun()

        except Exception as exc:
            for key in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(key) == "active":
                    st.session_state.pipeline_steps[key] = "pending"
            status_box.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Something went wrong: {exc}")


# ─── Realtime meeting analyzer ──────────────────────────────────────────────────
if mode == "Realtime Meeting Analyzer":
    realtime_language = st.selectbox(
        "Meeting language",
        ["english", "hinglish"],
        index=0,
        key="realtime_language_select",
    )
    st.session_state.realtime_language = realtime_language

    if st.session_state.realtime_buffer is None:
        st.session_state.realtime_buffer = RealtimeAudioBuffer()

    with st.container(border=True):
        head_col, reset_col = st.columns([3, 1], gap="medium")
        with head_col:
            st.markdown("### Microphone")
            st.caption(
                "Press Start below and allow microphone access. When the meeting ends, press Stop "
                "on the widget, then choose Finish meeting. To capture a call, make sure it is "
                "audible to your microphone."
            )
        with reset_col:
            reset_clicked = st.button("Start new meeting", use_container_width=True)

        webrtc_ctx = webrtc_streamer(
            key="realtime-meeting-microphone",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=st.session_state.realtime_buffer.audio_frame_callback,
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.markdown(
                '<span class="chip chip-live"><span class="dot"></span>Recording</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="chip chip-idle"><span class="dot"></span>Microphone off</span>',
                unsafe_allow_html=True,
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

    realtime_processing_panel()

    finish_clicked = st.button("Finish meeting and generate report", type="primary", use_container_width=True)

    if finish_clicked:
        if webrtc_ctx.state.playing:
            st.warning("Press Stop on the microphone widget first, then finish the meeting.")
        elif (
            not st.session_state.realtime_transcript.strip()
            and st.session_state.realtime_buffer.audio_queue.empty()
        ):
            st.warning("No transcript yet. Start the microphone, speak, then finish the meeting.")
        else:
            try:
                with st.spinner("Processing the remaining audio and preparing your report…"):
                    buffer = st.session_state.realtime_buffer

                    # Process all complete chunks first, in chronological order.
                    drain_realtime_audio(realtime_language, max_chunks=None)

                    # Process the final partial chunk, if any.
                    remaining_audio = flush_realtime_buffer(buffer)
                    if remaining_audio:
                        transcribe_realtime_chunk(remaining_audio, realtime_language)

                    full_transcript = st.session_state.realtime_transcript.strip()
                    if not full_transcript:
                        st.warning("No speech was detected. Speak clearly and try again.")
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
                st.error(f"Could not generate the report: {exc}")

    final_report = st.session_state.realtime_final_report
    if final_report:
        st.markdown("---")
        st.markdown("## Meeting report")
        render_report(final_report, "realtime")

    # Q&A works on the transcript captured so far, even before the final report.
    st.markdown("---")
    st.markdown("## Ask about this meeting")
    for message in st.session_state.realtime_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("realtime_meeting_question_form", clear_on_submit=True):
        realtime_question = st.text_input(
            "Your question",
            placeholder="What decisions have been made so far?",
        )
        ask_clicked = st.form_submit_button("Ask")

    if ask_clicked and realtime_question.strip():
        try:
            with st.spinner("Searching the transcript…"):
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
            st.error(f"Could not answer the question: {exc}")


# ─── Recorded-mode results ──────────────────────────────────────────────────────
if mode == "Video Analysis" and st.session_state.result:
    r = st.session_state.result
    render_report(r, "recorded")

    st.markdown("---")
    st.markdown("## Ask about this meeting")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.form("recorded_question_form", clear_on_submit=True):
        q_col, b_col = st.columns([6, 1], gap="small", vertical_alignment="bottom")
        with q_col:
            user_input = st.text_input(
                "Your question",
                placeholder="What were the main decisions?",
            )
        with b_col:
            send_btn = st.form_submit_button("Ask", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

elif mode == "Video Analysis" and not run_btn:
    st.markdown(
        """
        <div class="empty">
            <h3>No meeting analysed yet</h3>
            <p>Paste a YouTube link or a local file path in the sidebar, choose the language,
            and select <strong>Analyse meeting</strong>.</p>
        </div>""",
        unsafe_allow_html=True,
    )
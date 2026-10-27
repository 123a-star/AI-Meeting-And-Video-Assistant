
import io
import os
import queue
import threading
import wave

import av
import numpy as np
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3-turbo",
)
LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-20b")


class RealtimeAudioBuffer:
    """Collect microphone frames and produce fixed-length WAV chunks."""

    def __init__(self, chunk_seconds=CHUNK_SECONDS):
        self.chunk_samples = SAMPLE_RATE * chunk_seconds
        self.audio_queue = queue.Queue()
        self.pending = np.empty(0, dtype=np.int16)
        self.resampler = None
        self.lock = threading.Lock()
        self.last_error = None

    def audio_frame_callback(self, frame):
        """Called by streamlit-webrtc in its audio processing thread."""
        try:
            if self.resampler is None:
                self.resampler = av.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=SAMPLE_RATE,
                )

            output_frames = self.resampler.resample(frame)

            for output_frame in output_frames:
                samples = output_frame.to_ndarray().reshape(-1)
                samples = samples.astype(np.int16, copy=False)

                with self.lock:
                    self.pending = np.concatenate(
                        (self.pending, samples)
                    )

                    while len(self.pending) >= self.chunk_samples:
                        chunk = self.pending[:self.chunk_samples]
                        self.pending = self.pending[self.chunk_samples:]

                        self.audio_queue.put(self._to_wav_bytes(chunk))

        except Exception as exc:
            self.last_error = str(exc)

        return frame

    @staticmethod
    def _to_wav_bytes(samples):
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(samples.tobytes())

        return buffer.getvalue()

    def get_chunk(self):
        """Return the next completed WAV chunk, or None if none is ready."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    
    def flush_pending(self):
        """Return any remaining audio samples as a WAV chunk."""
        with self.lock:
            if len(self.pending) == 0:
                return None

            samples = self.pending.copy()
            self.pending = np.empty(0, dtype=np.int16)

        return self._to_wav_bytes(samples)


    def reset(self):
        """Clear buffered audio before a new meeting."""
        with self.lock:
            self.pending = np.empty(0, dtype=np.int16)

            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

        self.last_error = None


def transcribe_audio_bytes(audio_bytes, language="english"):
    """Transcribe one WAV chunk using Groq Whisper."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    client = Groq(api_key=api_key)

    options = {
        "file": ("realtime_chunk.wav", audio_bytes, "audio/wav"),
        "model": GROQ_STT_MODEL,
        "response_format": "json",
        "temperature": 0.0,
    }

    # Hinglish can contain both Hindi and English. Let the model
    # detect the language instead of forcing Hindi transcription.
    if language.lower() == "english":
        options["language"] = "en"

    response = client.audio.transcriptions.create(**options)
    return response.text.strip()


def get_realtime_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=api_key,
        temperature=0.2,
    )


def update_live_summary(transcript, previous_summary=""):
    """Update the running summary using the newly accumulated transcript."""
    llm = get_realtime_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a meeting assistant.
Update the existing meeting summary using the transcript provided.

Rules:
- Use only information supported by the transcript.
- Preserve important earlier points that remain relevant.
- Include new decisions, tasks, deadlines, and unresolved issues.
- Keep the result concise and organized into bullet points.
- Do not invent facts.

Existing summary:
{previous_summary}

New transcript:
{transcript}""",
        ),
        ("human", "Return the updated meeting summary."),
    ])

    return (prompt | llm | StrOutputParser()).invoke({
        "previous_summary": previous_summary or "No summary yet.",
        "transcript": transcript,
    })


def answer_live_question(transcript, question, summary=""):
    """Answer a question using the meeting content captured so far."""
    if not transcript.strip():
        return "Abhi meeting transcript available nahi hai."

    llm = get_realtime_llm()

    # Bound context size for long-running meetings.
    recent_transcript = transcript[-18000:]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an accurate meeting assistant.
Answer using only the supplied meeting summary and transcript.
If the answer is not available, say:
"I could not find this information in the meeting transcript."
Do not invent facts.

Meeting summary:
{summary}

Recent transcript:
{transcript}""",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "summary": summary or "No summary generated yet.",
        "transcript": recent_transcript,
        "question": question,
    })

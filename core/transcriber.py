import whisper
import os
import requests
from pydub import AudioSegment
from groq import Groq

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Sarvam sync API accepts <=30 seconds.
# We use 25 seconds with a safety margin.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# ---------------------------------------------------------
# Clients / Models
# ---------------------------------------------------------

_groq_client = None
_model = None


def get_groq_client():
    """Create Groq client only when it is needed."""
    global _groq_client

    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set in environment / .env"
            )

        _groq_client = Groq(api_key=GROQ_API_KEY)

    return _groq_client


def load_model():
    """Load local Whisper only when Groq is unavailable/fails."""
    global _model

    if _model is None:
        print(f"Loading local Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Local Whisper model loaded.")

    return _model


# ---------------------------------------------------------
# Groq Whisper
# ---------------------------------------------------------

def transcribe_chunk_groq(chunk_path: str) -> str:
    """
    Transcribe one audio chunk using Groq Whisper.
    """

    client = get_groq_client()

    print("  → Sending audio to Groq Whisper...")

    with open(chunk_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=GROQ_MODEL,
            response_format="json",
            temperature=0.0,
        )

    return transcription.text.strip()


# ---------------------------------------------------------
# Local Whisper fallback
# ---------------------------------------------------------

def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Fallback transcription using local Whisper."""

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"].strip()


# ---------------------------------------------------------
# Sarvam
# ---------------------------------------------------------

def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam."""

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:
        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio.
    Split the chunk into 25-second pieces.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_sv_{i}.wav"

        piece.export(
            piece_path,
            format="wav"
        )

        try:
            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces} ..."
            )

            full_text += (
                _send_to_sarvam(piece_path)
                + " "
            )

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# ---------------------------------------------------------
# Main transcription router
# ---------------------------------------------------------

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Routing:

    English:
        Groq Whisper → Local Whisper fallback

    Hinglish:
        Sarvam
    """

    # ---------------------------------------------
    # Hinglish → Sarvam
    # ---------------------------------------------

    if language.lower() == "hinglish":

        print("Using Sarvam for Hinglish transcription.")

        return transcribe_chunk_sarvam(
            chunk_path
        )

    # ---------------------------------------------
    # English → Groq
    # ---------------------------------------------

    try:

        return transcribe_chunk_groq(
            chunk_path
        )

    except Exception as e:

        print(
            f"\n⚠️ Groq transcription failed: {e}"
        )

        print(
            "→ Falling back to local Whisper..."
        )

        return transcribe_chunk_whisper(
            chunk_path
        )


# ---------------------------------------------------------
# Transcribe all chunks
# ---------------------------------------------------------

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    full_transcript = ""

    if language.lower() == "hinglish":
        engine = "Sarvam AI"
    else:
        engine = "Groq Whisper"

    print(
        f"Using {engine} for transcription."
    )

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()
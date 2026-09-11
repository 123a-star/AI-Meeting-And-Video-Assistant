
import os
import yt_dlp
from pydub import AudioSegment


# Folder where downloaded/conveted audio files will be stored
DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download YouTube audio and convert it to WAV.
    """

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],

        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)

        # Replace downloaded extension with .wav
        filename = os.path.splitext(filename)[0] + ".wav"

    return filename


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to a speech-friendly WAV format.

    Output:
        Mono audio
        16 kHz sample rate
    """

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    # Optimize audio for speech recognition
    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 2) -> list:
    """
    Split audio into small WAV chunks.

    We use 2-minute chunks so that each WAV file remains
    comfortably below Groq's audio upload limit.

    Returns:
        List of generated chunk file paths.
    """

    audio = AudioSegment.from_wav(wav_path)

    # Convert the complete audio to speech-friendly format
    audio = audio.set_channels(1).set_frame_rate(16000)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """
    Process either a YouTube URL or a local audio/video file.

    YouTube URL:
        Download → WAV → chunks

    Local file:
        Convert → WAV → chunks
    """

    if source.startswith("http://") or source.startswith("https://"):

        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:

        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks


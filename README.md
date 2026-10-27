# AI Meeting and Video Assistant

An AI-powered Meeting and Video Assistant built with Streamlit. It analyzes recorded YouTube videos and local audio/video files, generates meeting insights, and answers questions using Retrieval-Augmented Generation (RAG). It also supports live microphone-based meeting transcription and analysis through a separate Realtime Meeting Analyzer.

## Features

### 1. Video Analysis
- YouTube video processing
- Local audio and video file processing
- Audio conversion and chunking
- English speech-to-text using Groq Whisper
- Hinglish transcription using Sarvam AI
- Local Whisper fallback for English transcription
- AI-generated meeting title
- Meeting summarization
- Action item extraction
- Key decision extraction
- Important question extraction
- Transcript-based vector database creation
- RAG-based question answering
- Interactive chat with recorded video or audio content

### 2. Realtime Meeting Analyzer
- Live microphone audio capture through WebRTC
- Audio frame collection and chunk-based processing
- Live speech-to-text transcription using Groq Whisper
- Continuously updated meeting transcript
- Periodically refreshed live summary
- Meeting-related question answering
- Final meeting report generation, including available summary, decisions, action items, and questions

**Note:** The Realtime Meeting Analyzer currently uses the available transcript and summary for question answering. Its Q&A pipeline is separate from the RAG pipeline used by Video Analysis.

## Application Architecture

The application has two separate analysis modes.

### Mode 1: Video Analysis Pipeline

The Video Analysis mode processes recorded video or audio content and creates a RAG-based question-answering system.

```text
YouTube URL / Local Audio or Video File
                  |
                  v
          Audio Processing
          Download / Convert
                  |
                  v
          Audio Chunking
                  |
                  v
             Transcription
                  |
          +-------+--------+
          |                |
          v                v
      Groq Whisper      Sarvam AI
      English Audio     Hinglish Audio
          |                |
          +-------+--------+
                  |
                  v
           Full Transcript
                  |
          +-------+--------+
          |                |
          v                v
   AI-Generated Title   Meeting Summary
          |                |
          +-------+--------+
                  |
                  v
          Meeting Insights
          - Action Items
          - Key Decisions
          - Important Questions
                  |
                  v
         RAG Chain Creation
                  |
                  v
       Transcript Embeddings
                  |
                  v
              ChromaDB
                  |
                  v
        User Asks a Question
                  |
                  v
           Vector Retrieval
                  |
                  v
      Relevant Transcript Chunks
                  |
                  v
              Groq LLM
                  |
                  v
          Context-Aware Answer
```

The transcription provider depends on the selected language and configured application logic. Local Whisper can serve as a fallback for English transcription when supported by the implementation.

### How RAG Works in Video Analysis

1. The application generates a transcript from the recorded content.
2. The transcript is processed to build a RAG chain.
3. The RAG components make relevant transcript content available for retrieval.
4. When a user asks a question, the system retrieves relevant context.
5. The retrieved context is passed to the language model to generate an answer.

This allows users to ask questions about specific details discussed in a long video or recording without manually searching through the entire transcript.

The exact embedding and retrieval implementation is defined in `core/rag_engine.py` and the associated vector-store modules.

### Mode 2: Realtime Meeting Analyzer Pipeline

The Realtime Meeting Analyzer captures live audio from the browser microphone and processes it in short audio chunks.

```text
       Browser Microphone
                |
                v
       streamlit-webrtc
                |
                v
        Audio Frame Callback
                |
                v
         Realtime Audio Buffer
                |
                v
        Audio Chunks (~8 sec)
                |
                v
         Groq Whisper API
                |
                v
          Live Transcript
                |
                v
         Transcript Updates
                |
                v
       Periodic Live Summary
                |
                +-------------------+
                |                   |
                v                   v
      User Asks a Question    Meeting Finishes
                |                   |
                v                   v
     Transcript + Summary     Final Processing
                |                   |
                v                   v
             Groq LLM          Final Report
                |
                v
        Context-Based Answer
```

#### How Realtime Processing Works

1. The browser captures microphone audio.
2. WebRTC sends audio frames to the application's audio callback.
3. `RealtimeAudioBuffer` collects the audio frames and prepares chunks for processing.
4. The transcription function sends each available chunk to Groq Whisper.
5. The recognized text is appended to the running meeting transcript.
6. The live summary is updated periodically as new transcript content becomes available.
7. When the user asks a question, the application uses the available transcript and summary to generate an answer.
8. When the meeting ends, the final processing logic can generate the meeting report and process any remaining audio when the buffer-flushing functionality is available.

**Realtime implementation detail:** This is chunk-based, near-real-time processing rather than instant word-by-word transcription. Actual update times depend on audio chunking, API response time, and processing load.

**RAG distinction:** The current Realtime Meeting Analyzer does not automatically use the ChromaDB retrieval pipeline. Its question-answering function is separate from the recorded Video Analysis RAG chain.

## Technology Stack

### Frontend
- Streamlit
- streamlit-webrtc

### Programming Language
- Python

### AI and Language Models
- Groq API
- Groq Whisper for speech-to-text
- Groq-hosted LLM for summarization and question answering
- Sarvam AI for supported Hinglish transcription
- OpenAI Whisper for local English transcription fallback

### RAG and Vector Database
- LangChain
- LangChain Core
- LangChain Community
- LangChain Text Splitters
- LangChain Chroma
- ChromaDB
- Sentence Transformers

### Audio and Video Processing
- yt-dlp
- Pydub
- FFmpeg

### Supporting Libraries
- NumPy
- PyTorch
- Torchvision
- Pillow
- Requests
- python-dotenv
- tiktoken

## Project Structure

```text
AI-Meeting-And-Video-Assistant/
|
|-- app.py
|-- main.py
|-- test.py
|-- requirements.txt
|-- README.md
|-- .env
|-- .gitignore
|
|-- core/
|   |-- extractor.py
|   |-- rag_engine.py
|   |-- realtime_analyzer.py
|   |-- summarizer.py
|   |-- transcriber.py
|   |-- vector_store.py
|
|-- utils/
|   |-- audio_processor.py
|
|-- myenv/
    |-- Python virtual environment
```

The `myenv/` folder is created locally during installation and should not be committed to Git.

## Core Modules

### `app.py`
Main Streamlit application. Provides the user interface, analysis mode selection, recorded-media processing, realtime microphone analysis, transcript display, meeting insights, and question-answering interface.

### `core/transcriber.py`
Handles transcription for recorded audio. Supports Groq Whisper, Sarvam AI, and the configured local Whisper fallback.

### `core/summarizer.py`
Generates meeting titles and summaries from the transcript.

### `core/extractor.py`
Extracts:
- Action items
- Key decisions
- Important questions

### `core/rag_engine.py`
Builds the RAG chain, handles retrieval-related processing, and answers questions about recorded video or audio content.

### `core/vector_store.py`
Handles transcript vector-store and embedding operations used by the RAG implementation, as configured by the project.

### `core/realtime_analyzer.py`
Contains realtime audio-buffering, live audio transcription, live summary updates, and transcript-based question-answering logic.

### `utils/audio_processor.py`
Handles recorded-media input processing, including supported YouTube audio downloading, audio conversion, and audio chunking.

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/123a-star/AI-Meeting-And-Video-Assistant.git
cd AI-Meeting-And-Video-Assistant
```

If the repository is already downloaded, open the terminal inside the existing project folder instead of cloning it again.

### 2. Create a Virtual Environment

```powershell
python -m venv myenv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
& ".\myenv\Scripts\Activate.ps1"
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again:

```powershell
& ".\myenv\Scripts\Activate.ps1"
```

Confirm that the terminal prompt begins with `(myenv)`.

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies

Make sure `requirements.txt` contains the packages required by both analysis modes, including `streamlit-webrtc`.

Run:

```powershell
python -m pip install -r requirements.txt
```

Check for dependency conflicts:

```powershell
python -m pip check
```

### 6. Install FFmpeg

FFmpeg must be installed separately and available through the system `PATH`. It is used for supported audio/video processing and local Whisper workflows.

Verify that it is available:

```powershell
ffmpeg -version
```

If the command is not recognized, install FFmpeg and configure the Windows `PATH` before running the application.

## Environment Variables

Create a `.env` file in the project root directory.

Add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

Replace the placeholder values with your own API keys.

- `GROQ_API_KEY` is required for the configured Groq-powered transcription and language-model operations.
- `SARVAM_API_KEY` is required when the selected transcription workflow uses Sarvam AI.

Keep `.env` private. Do not upload real API keys to a public GitHub repository.

## Run the Application

First, activate the virtual environment and navigate to the project directory.

Run:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser.

## How to Use

### Video Analysis

1. Open the application.
2. Select `Video Analysis` in the analysis mode selector.
3. Enter a supported YouTube URL or local audio/video file path.
4. Select the desired transcription language, where applicable.
5. Start the analysis.
6. Review the generated transcript, title, summary, action items, key decisions, and questions.
7. Use the question-answering interface to ask questions about the recorded content.

### Realtime Meeting Analyzer

1. Open the application.
2. Select `Realtime Meeting Analyzer`.
3. Allow microphone access in the browser.
4. Start the microphone stream and begin speaking.
5. Review the transcript as audio chunks are processed.
6. Follow the periodically updated live summary.
7. Ask questions about the available meeting transcript through the supported Q&A interface.
8. Finish the meeting to run the available final-report processing.

The browser must be able to access the microphone. When accessing the app from another device, browser permissions and secure-context requirements may affect microphone access.

## Troubleshooting

### `ModuleNotFoundError: No module named 'streamlit_webrtc'`

Activate the project virtual environment and install the dependency:

```powershell
python -m pip install streamlit-webrtc
```

Add `streamlit-webrtc` to `requirements.txt` to keep the project setup reproducible.

### `ModuleNotFoundError` for Another Package

Check that the correct virtual environment is active:

```powershell
python -c "import sys; print(sys.executable)"
```

Then install the project's dependencies:

```powershell
python -m pip install -r requirements.txt
```

### FFmpeg Is Not Found

Check the installation:

```powershell
ffmpeg -version
```

Install FFmpeg and ensure its executable location is included in the system `PATH`.

### Microphone Is Not Working

- Allow microphone access in the browser.
- Verify that the correct microphone is selected in the operating system.
- Check that the WebRTC stream starts successfully.
- Review the Streamlit terminal for audio-processing errors.

### API Errors

Verify that the required API keys are present in `.env` and that the selected provider supports the requested operation. Also check API usage limits, network connectivity, and the relevant error message.

### Dependency Conflicts

Run the dependency check from the activated project environment:

```powershell
python -m pip check
```

Avoid modifying the global Anaconda `base` environment to fix a conflict that belongs to the project virtual environment.

## Security

- Never commit API keys, passwords, or other secrets.
- Keep `.env` out of version control.
- Do not commit the virtual environment or generated cache files.
- Avoid sharing private meeting transcripts or recordings without authorization.

## Future Improvements

- RAG-based retrieval for realtime meetings
- Persistent vector storage for meeting history
- Speaker identification and speaker-labelled transcripts
- More efficient incremental transcription and summarization
- Improved meeting search and historical meeting retrieval
- Exportable meeting reports
- Configurable transcription and summarization providers

## License

Add the applicable license and attribution details for this project and its dependencies before distributing the application.

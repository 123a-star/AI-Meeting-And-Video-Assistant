\# AI Meeting and Video Assistant



An AI-powered Meeting and Video Assistant that processes YouTube videos or local audio/video files and provides transcription, summarization, meeting insights, and RAG-based question answering.



\## Features



\- YouTube video processing

\- Local audio/video file processing

\- Audio conversion and chunking

\- English speech-to-text using Groq Whisper

\- Hinglish transcription using Sarvam AI

\- AI-generated meeting title

\- Meeting summary generation

\- Action item extraction

\- Key decision extraction

\- Important question extraction

\- Vector database creation

\- RAG-based question answering

\- Local Whisper fallback for English transcription



The user provides:

YouTube URL
Local audio file
Local video file
2️⃣ Audio Processing

The application downloads or converts the input into audio and divides it into manageable chunks.

3️⃣ Transcription

For English content:
Audio → Groq Whisper → Transcript
For Hinglish content:
Audio → Sarvam AI → Transcript
4️⃣ AI Analysis

The transcript is processed to generate:
Transcript
    │
    ├── 📝 Summary
    ├── 🏷️ Meeting Title
    ├── ✅ Action Items
    ├── 💡 Key Decisions
    └── ❓ Questions
5️⃣ Vector Database

The transcript is converted into embeddings and stored in ChromaDB.

6️⃣ RAG Question Answering

When a user asks a question:
User Question
      ↓
Vector Search
      ↓
Relevant Transcript Chunks
      ↓
Groq LLM
      ↓
Context-Aware Answer
User Question
      ↓
Vector Search
      ↓
Relevant Transcript Chunks
      ↓
Groq LLM
      ↓
Context-Aware Answer


🛠️ Technology Stack
Frontend
Streamlit
AI / LLM
Groq
Groq Whisper
Sarvam AI
OpenAI Whisper
RAG
LangChain
ChromaDB
Sentence Transformers
Audio Processing
yt-dlp
Pydub
FFmpeg
Programming Language
Python
📂 Project Structure
AI-Meeting-And-Video-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
│
└── utils/
    └── audio_processor.py
Core Modules
app.py

Main Streamlit application and user interface.

core/transcriber.py

Handles:

Groq Whisper transcription
Sarvam AI transcription
Local Whisper fallback
core/summarizer.py

Handles:

Meeting title generation
Meeting summarization
core/extractor.py

Extracts:

Action items
Key decisions
Questions
core/rag_engine.py

Handles:

RAG chain creation
Context retrieval
Question answering
core/vector_store.py

Handles transcript embeddings and vector database operations.

utils/audio_processor.py

Handles:

YouTube audio downloading
Audio conversion
Audio chunking



INSTALLATION
1. Clone the Repository
git clone https://github.com/123a-star/AI-Meeting-And-Video-Assistant.git
cd AI-Meeting-And-Video-Assistant

2. Create Virtual Environment
python -m venv myenv

3. Activate Virtual Environment
Windows PowerShell
& ".\myenv\Scripts\Activate.ps1"

If PowerShell blocks script execution:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

Then activate again:
& ".\myenv\Scripts\Activate.ps1"

4. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
🔑 Environment Variables

GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key

▶️ Run the Application


python -m streamlit run app.py
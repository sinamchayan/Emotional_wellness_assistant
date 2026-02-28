# 🧠 Hridaya — Emotional Wellness Assistant

> *"Hridaya" (हृदय) — Sanskrit for "Heart"*

Hridaya is an AI-powered emotional wellness assistant that conducts structured, clinically-inspired therapy sessions through multimodal emotion detection. It combines a fine-tuned RoBERTa model, an audio emotion classifier, and Google's Gemma 3 LLM to understand how you truly feel — not just what you say.

---

## ✨ Key Features

- 🎙️ **Multimodal Emotion Detection** — Fuses text analysis (RoBERTa + Gemma 3) with audio prosody analysis (MFCC/Chroma/Mel-spectrogram features) for accurate, conflict-aware emotion scoring
- 🧑‍⚕️ **Structured 11-Turn Therapy Sessions** — Guided conversations with a clinically-aware AI therapist powered by Gemma 3 27B
- 📊 **Daily & Weekly Analytics Dashboard** — Interactive Streamlit dashboard with Plotly charts showing emotional trends, stability scores, and session breakdowns
- 🤖 **AI-Generated Clinical Summaries** — Automatic extraction of stress triggers, happy moments, and wellness suggestions after each session
- 📄 **PDF Report Export** — Downloadable daily and weekly clinical PDF reports generated with ReportLab and a deep emotional profile from Gemma
- 🗄️ **ChromaDB Session Memory** — Completed sessions are persisted as semantic embeddings for historical RAG querying
- 🌐 **React + Vite Frontend** — A sleek dark-theme chat interface (Hridaya UI) with embedded Streamlit analytics dashboard via iframe

---

## 🏗️ Project Architecture

```
Emotional_wellness_assistant/
│
├── backend_api_app.py          # FastAPI backend — core chat, emotion fusion, PDF generation
├── emotional_dashboard.py      # Streamlit analytics dashboard (daily & weekly views)
│
├── production_emotion_model/   # Fine-tuned RoBERTa (text emotion classifier)
│   ├── model.safetensors       # Model weights (~476 MB)
│   ├── config.json             # Model config (7 emotion classes)
│   ├── tokenizer.json          # Tokenizer vocabulary
│   └── tokenizer_config.json
│
├── audio_emotion_model.pkl     # Sklearn-based audio emotion classifier (~31 MB)
│
├── neural-ninjas-ui/           # React + Vite frontend
│   └── src/
│       ├── App.jsx             # Main UI: chat, sidebar nav, embedded dashboard
│       ├── index.css           # Global styles
│       └── main.jsx            # Entry point
│
├── landing_page/               # Static HTML landing page with animations
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── patient_logs/               # Per-user session JSON logs (auto-created)
│   └── <username>/
│       └── Data_<user>_<date>_<timestamp>.json
│
├── rag_db/                     # ChromaDB vector store for session history
├── requirements.txt            # Python dependencies
└── config.yml                  # Streamlit config
```

---

## 🔄 How Hridaya Works — Data Flow

```
User (Browser)
     │
     │  Text + Optional Audio
     ▼
┌─────────────────────────────────────────────────────┐
│              React Frontend (Port 5173)              │
│  • 11-turn structured session UI                    │
│  • Voice recording (MediaRecorder + WebSpeechAPI)   │
│  • Embedded Streamlit dashboard (iframe, Port 8501) │
└─────────────────────┬───────────────────────────────┘
                      │  POST /chat  (FormData: text + audio)
                      ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)             │
│                                                     │
│  TURN 1: Collect daily schedule                     │
│                                                     │
│  TURNS 2–11: Multimodal Emotion Fusion Pipeline     │
│  ┌─────────────────────────────────────────────┐    │
│  │  1. Context Resolution (Gemma 3)            │    │
│  │     Resolve pronouns, extract emotion scores│    │
│  │                                             │    │
│  │  2. Text Emotion Analysis (RoBERTa)         │    │
│  │     Fine-tuned on 7-class emotion dataset   │    │
│  │                                             │    │
│  │  3. Conflict Detection & Weight Assignment  │    │
│  │     If RoBERTa < 70% confidence OR conflict │    │
│  │     → Trust Gemma more (80/20)              │    │
│  │     Else → Trust RoBERTa more (70/30)       │    │
│  │                                             │    │
│  │  4. Audio Emotion Analysis (if audio sent)  │    │
│  │     MFCC + Chroma + Mel features → sklearn  │    │
│  │     Final = 30% text fused + 70% audio      │    │
│  │                                             │    │
│  │  5. Detected Emotion → Gemma 3 Therapist    │    │
│  │     Generates empathetic, contextual reply  │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  TURN 11 (Final):                                   │
│  • Generate Clinical Summary (triggers, joy, tips)  │
│  • Save JSON log to ./patient_logs/<username>/      │
│  • Persist session to ChromaDB                      │
└─────────────┬───────────────────┬───────────────────┘
              │                   │
              ▼                   ▼
   patient_logs/<user>/       rag_db/
   Data_*.json               (ChromaDB)
              │
              ▼
┌─────────────────────────────────────────────────────┐
│         Streamlit Dashboard (Port 8501)             │
│  • Load JSON logs for selected user/date            │
│  • Daily View: Emotion pie, intensity line, flow    │
│  • Weekly View: Stacked bar, happiness pulse, AI    │
│    deep profile (Gemma), personalized weekly plan   │
│  • PDF export via /download-pdf and                 │
│    /download-weekly-pdf endpoints                   │
└─────────────────────────────────────────────────────┘
```

---

## 🤖 The Emotion Detection System (Deep Dive)

Hridaya uses a **3-layer multimodal fusion** approach:

### Layer 1 — Context Resolution (Gemma 3 27B)
Before any analysis, the raw user message is sent to Gemma 3, which:
- Resolves pronouns and vague references using conversation history
- Provides its own emotion probability distribution across 7 classes
- This "smart preprocessing" handles sarcasm and ambiguous language

### Layer 2 — Text Emotion Classification (RoBERTa)
A fine-tuned `RobertaForSequenceClassification` model (`production_emotion_model/`) classifies the resolved text into 7 emotions:

| ID | Emotion   |
|----|-----------|
| 0  | Neutral   |
| 1  | Anger     |
| 2  | Disgust   |
| 3  | Fear      |
| 4  | Happiness |
| 5  | Sadness   |
| 6  | Surprise  |

### Layer 3 — Conflict-Aware Fusion
The system detects **emotional conflict** (e.g., RoBERTa says "happy" but Gemma 3 says "sad") and adapts weights dynamically:

```
If conflict detected OR RoBERTa confidence < 70%:
    final_text_score = 0.20 × RoBERTa + 0.80 × Gemma
Else:
    final_text_score = 0.70 × RoBERTa + 0.30 × Gemma
```

### Layer 4 — Audio Prosody (when mic is used)
If voice input is provided, audio features are extracted:
- **MFCCs** (40 coefficients) — captures tonal quality and speech patterns
- **Chroma** (12 features) — captures harmonic content
- **Mel Spectrogram** — captures frequency-time patterns

These are fed into a pre-trained sklearn classifier (`audio_emotion_model.pkl`), and the result overrides the text score:
```
final_score = 0.30 × text_fused + 0.70 × audio
```

---

## 🗣️ The 11-Turn Session Structure

Each session follows a structured clinical protocol:

| Turn | Phase | Purpose |
|------|-------|---------|
| 1 | Schedule Collection | User shares their daily schedule (no emotion analysis) |
| 2–6 | Positive Exploration | AI asks open-ended questions about good moments, highlights |
| 7–10 | Challenge Exploration | AI gently probes difficult or stressful aspects of the day |
| 11 | Session Closure | AI provides a 2-sentence summary, a wellness exercise, and a goal for tomorrow |

After Turn 11:
- A **Clinical Summary** is generated (triggers, happy moments, suggestions)
- The full session is **saved as JSON** and **embedded in ChromaDB**
- The user can **Talk More** (up to 5 extra turns) or **View Analytics**

---

## 🖥️ Component Overview

### 1. `backend_api_app.py` — FastAPI Backend
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main session endpoint — processes text/audio, returns AI reply + emotions |
| `/history/{username}` | GET | Returns list of session dates for a given user |
| `/download-pdf/{username}/{date}` | GET | Generates and returns a daily session PDF |
| `/download-weekly-pdf/{username}` | GET | Generates a full weekly wellness PDF with AI deep profile |

### 2. `emotional_dashboard.py` — Streamlit Analytics
- **Daily View**: Emotion distribution pie chart, intensity line chart, turn-by-turn emotional flow, AI insights cards, PDF download button
- **Weekly View**: Stacked mood bar chart, happiness pulse, emotional stability area chart, day-by-day breakdown, weekly highlights, Gemma-generated deep emotional health profile, personalized next-week plan

### 3. `neural-ninjas-ui/` — React Frontend
- Built with **Vite + React + TailwindCSS**
- Sidebar navigation between Daily Session, Daily Insights, and Weekly Analysis
- Full voice recording support (MediaRecorder API + Web Speech API for live transcript)
- Analytics dashboard is embedded as an iframe pointing to Streamlit (port 8501)
- Analytics are **locked** until a session is completed (privacy-first design)
- Native browser PDF print for session transcript export

### 4. `landing_page/` — Static Landing Page
- Animated HTML/CSS/JS landing page
- Login and Anonymous Login buttons
- Stores username in `localStorage` before routing to the chat UI

---

## ⚙️ Setup Guide

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |
| Google Gemma API Key | Required (Google AI Studio) |

> **Note:** The models (`production_emotion_model/` and `audio_emotion_model.pkl`) must be present locally. They are tracked via Git LFS due to their size (~530 MB total).

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/sinamchayan/Emotional_wellness_assistant.git
cd Emotional_wellness_assistant
```

---

### Step 2 — Set Up Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
```

---

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Known Issue:** `openai-whisper` and `tensorflow`/`keras` may conflict. If you encounter dependency errors, install them in this order:
> ```bash
> pip install tensorflow
> pip install openai-whisper --no-deps
> ```

---

### Step 4 — Configure Your API Key

Open `backend_api_app.py` and `emotional_dashboard.py` and replace the API key:

```python
# In both files, find this line:
API_KEY = "your-google-ai-studio-key-here"
```

Get your free API key from [Google AI Studio](https://aistudio.google.com/).

---

### Step 5 — Install Frontend Dependencies

```bash
cd neural-ninjas-ui
npm install
cd ..
```

---

### Step 6 — Run All Services

You need **3 terminals** running simultaneously:

**Terminal 1 — FastAPI Backend:**
```bash
source .venv/bin/activate
cd /path/to/Emotional_wellness_assistant
uvicorn backend_api_app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Streamlit Dashboard:**
```bash
source .venv/bin/activate
cd /path/to/Emotional_wellness_assistant
streamlit run emotional_dashboard.py --server.port 8501
```

**Terminal 3 — React Frontend:**
```bash
cd neural-ninjas-ui
npm run dev
```

---

### Step 7 — Access the App

| Service | URL |
|---------|-----|
| Landing Page | Open `landing_page/index.html` in a browser |
| Chat UI (React) | http://localhost:5173 |
| Analytics Dashboard (Streamlit) | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|------------|
| LLM Backend | Google Gemma 3 27B (via Google GenAI SDK) |
| Text Emotion Model | Fine-tuned RoBERTa (`roberta-base`) |
| Audio Emotion Model | Sklearn classifier (MFCC/Chroma/Mel features via Librosa) |
| API Server | FastAPI + Uvicorn |
| Analytics Dashboard | Streamlit + Plotly |
| Session Storage | JSON files + ChromaDB (vector DB) |
| PDF Generation | ReportLab |
| Frontend | React 18 + Vite + TailwindCSS + Lucide Icons |

---

## 📁 Patient Data & Privacy

- All session logs are stored **locally** in `./patient_logs/<username>/` as JSON files
- No data is sent to any external server except the **Google GenAI API** (for Gemma 3 inference)
- Session data is also stored in a local **ChromaDB** instance (`./rag_db/`) for semantic search
- The landing page supports **Anonymous Login** — no account or personal data required

---

## 🧩 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Failed to fetch` on frontend | Ensure the FastAPI backend is running on port 8000 |
| ChromaDB SQLite FTS5 error | Run `python patch_db.py` to apply the SQLite patch |
| Streamlit analytics shows blank | Complete an 11-turn session first; analytics unlock after session ends |
| Weekly view locked | Requires **3 or more distinct session days** |
| Audio model not loading | Ensure `audio_emotion_model.pkl` is present in the root directory |
| RoBERTa model not loading | Ensure the `production_emotion_model/` folder contains `model.safetensors` |

---

## 🙏 Acknowledgements

- Model architecture based on **RoBERTa** by HuggingFace / Facebook AI
- Powered by **Google Gemma 3** via the Google GenAI SDK
- Audio feature extraction via **Librosa**
- Built with love by **Team Neural Ninjas** 🧠⚡

---

*"The mind is everything. What you think, you become."*

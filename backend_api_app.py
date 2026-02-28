import os
import json
import joblib
import torch
import librosa
import numpy as np
import tempfile
import glob
import re
import chromadb
import traceback
import warnings
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from google import genai
from google.genai import types
from typing import List, Dict, Optional
from datetime import datetime

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# 1. SETUP & CONFIGURATION

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "models/gemma-3-27b-it" 

EMOTIONS = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
POSITIVE_VALENCE = ["happiness", "surprise", "neutral"]
NEGATIVE_VALENCE = ["anger", "disgust", "fear", "sadness"]

try:
    chroma_client = chromadb.PersistentClient(path="./rag_db")
    collection = chroma_client.get_collection(name="wellness_interventions")
    print("✅ RAG Database Connected.")
except:
    collection = None

print("🧠 Initializing Neural Ninjas Intelligence...")
try:
    roberta = pipeline("text-classification", model="./production_emotion_model", top_k=None)
    audio_brain = joblib.load("./audio_emotion_model.pkl")
    print("✅ Local AI Models Loaded.")
except Exception as e:
    print(f"⚠️ Local AI Load Failed: {e}")
    roberta = None
    audio_brain = None

sessions = {}

# 2. CORE UTILITIES

# ... (existing imports)

async def generate_clinical_summary(client, history, schedule):
    """Generates structured extracted insights for the dashboard using strict Schema typing."""
    history_str = "\n".join([f"U: {h['u']} | B: {h['b']}" for h in history])
    
    prompt = f"""You are a Clinical Data Analyst. 
    Analyze this therapy session.
    
    USER SCHEDULE: {schedule}
    SESSION HISTORY: {history_str}
    
    TASK: Provide the requested JSON extraction.
    
        RULES:
    - Return ONLY raw JSON without markdown markers or backticks like ```json ... ```.
    - Do NOT use generic phrases.
    - Be specific to the user's words.
    - Triggers and happy_moments should be under 15 words.
    - Suggestions should be one actionable, unique wellness tip for tomorrow.

    EXPECTED JSON:
    {{
        "triggers": "1-2 specific events causing stress",
        "happy_moments": "Highlights mentioned by user that improved mood",
        "suggestions": "One actionable wellness tip"
    }}"""
    
    try:
        res = client.models.generate_content(
            model=MODEL_NAME, 
            contents=prompt,
        )
        import re
        match = re.search(r'\{.*\}', res.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(res.text)
    except Exception as e:
        print(f"❌ Summary Generation Error: {e}")
        return {
            "triggers": "Specific stress patterns were too complex to categorize this session.",
            "happy_moments": "User mentioned personal highlights that are being processed.",
            "suggestions": "Review your session notes and focus on maintaining your evening routine."
        }

def save_conversation_log(session_id, username, history, schedule, emo_scores, summary_obj):
    log_dir = f"./patient_logs/{username}"
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_data = {
        "timestamp": timestamp,
        "date": date_str,
        "username": username,
        "session_id": session_id,
        "schedule": schedule,
        "history": history,
        "emo_scores": emo_scores,
        "ai_insights": summary_obj,
        "summary_text": history[-1]['b'] if history else ""
    }
    
    # 1. Save locally for easy PDF generation and debugging
    file_path = f"{log_dir}/Data_{username}_{date_str}_{timestamp}.json"
    
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(session_data, f, indent=4)
        
    print(f"📄 Session log saved locally: {file_path}")
    
    # 2. Persist to ChromaDB for advanced historical RAG querying later
    if collection is not None:
        try:
            # Create a string representation of the conversation for semantic search
            history_text = "\n".join([f"User: {turn['u']}\nAI: {turn['b']}" for turn in history])
            document_content = f"Date: {date_str}\nSchedule: {schedule}\n\nTranscript:\n{history_text}\n\nInsights: Triggers: {summary_obj.get('triggers')}. Joy: {summary_obj.get('happy_moments')}."
            
            # Upsert into ChromaDB
            collection.add(
                documents=[document_content],
                metadatas=[{
                    "username": username,
                    "date": date_str,
                    "session_id": session_id,
                    "type": "completed_session"
                }],
                ids=[f"session_{username}_{timestamp}"]
            )
            print(f"🧠 Session {session_id} beautifully persisted to Chroma DB.")
        except Exception as e:
            print(f"⚠️ Chroma DB save failed: {e}")
            
    return file_path

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        return np.hstack((mfccs, chroma, mel))
    except: return None

# 3. MAIN CHAT ENDPOINT

@app.post("/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    is_extra_phase: bool = Form(False),
    username: str = Form("Guest")
):
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "turns": 1, "extra_turns": 0, "schedule": "", "emo_scores": []}
    
    sess = sessions[session_id]
    client = genai.Client(api_key=API_KEY)
    
    if (not is_extra_phase and sess["turns"] > 11) or (is_extra_phase and sess["extra_turns"] >= 5):
         return {"response": "Session is concluded.", "concluded": True}

    raw_text = text if text else ""
    history_str = "\n".join([f"U: {h['u']} | B: {h['b']}" for h in sess["history"][-2:]])

    # --- TURN 1: SCHEDULE COLLECTION ---
    if sess["turns"] == 1 and not is_extra_phase:
        sess["schedule"] = raw_text
        sess["turns"] += 1
        prompt = f"The user shared this schedule: {raw_text}. Briefly acknowledge and ask first about what was the best part of the day even if there was none, any moment that made the user happy (Turn 2/11). Max 40 words. DO NOT explain your instructions, JUST output the response."
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        ai_reply = response.text.strip()
        sess["history"].append({"u": raw_text, "b": ai_reply, "e": "neutral"})
        sess["emo_scores"].append({"turn": 1, "scores": {e: 0.0 for e in EMOTIONS}, "emotion": "neutral"})
        return {"response": ai_reply, "emotion": "neutral", "current_turn": 2, "is_final": False, "transcribed_text": raw_text}

    # --- TURNS 2-11: MULTIMODAL FUSION ---
    res_prompt = f"Resolve context for '{raw_text}' using HISTORY: {history_str} and SCHEDULE: {sess['schedule']}. Return JSON with 'resolved_text' and scores for {EMOTIONS} CONSTRAINT : Maintain the user's tone and emotions, do not exaggerate anything just resolve pronouns such as 'this', 'that', 'such'."
    try:
        res = client.models.generate_content(model=MODEL_NAME, contents=res_prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
        gemma_data = json.loads(res.text)
        processed_text = gemma_data.get("resolved_text", raw_text)
        gemma_scores = gemma_data.get("scores", {e: 0.1 for e in EMOTIONS})
    except:
        processed_text = raw_text
        gemma_scores = {e: 0.1 for e in EMOTIONS}

    roberta_scores = {e: 0.0 for e in EMOTIONS}
    if roberta:
        r_res = roberta(processed_text)[0]
        roberta_scores = {r['label']: r['score'] for r in r_res}
    
    top_rob = max(roberta_scores, key=roberta_scores.get)
    top_gem = max(gemma_scores, key=gemma_scores.get)
    is_conflict = (top_rob in POSITIVE_VALENCE and top_gem in NEGATIVE_VALENCE)

    tw_rob, tw_gem = (0.20, 0.80) if is_conflict or roberta_scores.get(top_rob, 0) < 0.70 else (0.70, 0.30)

    fused_text_scores = {e: (roberta_scores.get(e, 0.0) * tw_rob) + (gemma_scores.get(e, 0.0) * tw_gem) for e in EMOTIONS}

    final_scores = fused_text_scores
    if audio and audio_brain:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        features = extract_audio_features(tmp_path)
        if features is not None:
            probs = audio_brain.predict_proba(features.reshape(1, -1))[0]
            audio_raw_scores = {emo: float(p) for emo, p in zip(audio_brain.classes_, probs)}
            final_scores = {e: (fused_text_scores[e] * 0.3) + (audio_raw_scores[e] * 0.7) for e in EMOTIONS}
        os.remove(tmp_path)

    detected_emotion = max(final_scores, key=final_scores.get)
    sess["emo_scores"].append({"turn": sess["turns"], "scores": final_scores, "emotion": detected_emotion})
    
    if is_extra_phase: sess["extra_turns"] += 1
    else: sess["turns"] += 1
    
    current_count = sess["extra_turns"] if is_extra_phase else sess["turns"]
    is_final_turn = (not is_extra_phase and current_count == 11) or (is_extra_phase and current_count == 5)

    phase_instruction = "ask open-ended questions about what was good about the user's day" if current_count <= 6 else "ask something bad or challenging"
    
    final_prompt = f"""User Input: "{processed_text}"
    Detected Emotion: {detected_emotion.upper()}
    Schedule: {sess['schedule']}
    
    Task: Respond with professional clinical empathy. Address any hidden sarcasm or pain detected. Questions should only be answerable in a descriptive manner.
    One open-ended question only (not answerable in yes or no). You have 10 questions numbered from 2 i.e,(2-11). 
    Ask ONLY the question appropriate for current Turn {current_count}. 
    
    IF IS_FINAL_TURN=True:
    DO NOT ask a question. Provide a short 2-sentence emotional summary, one short wellness exercise, and a goal for tomorrow.

    CONSTRAINT: Do NOT ask repetetive questions. Do not start every response with "it sounds like" .Ask one question at a time. Respond ONLY with the dialogue. DO NOT explain your instructions. Do not include the number count in response. 
    IS_FINAL_TURN: {is_final_turn} | Current turn: {current_count}"""

    response = client.models.generate_content(model=MODEL_NAME, contents=final_prompt)
    ai_reply = response.text.strip()
    
    sess["history"].append({"u": processed_text, "b": ai_reply, "e": detected_emotion})
    
    if is_final_turn:
        # ASYNC extraction of summary fields for Streamlit
        summary_obj = await generate_clinical_summary(client, sess["history"], sess["schedule"])
        save_conversation_log(session_id, username, sess["history"], sess["schedule"], sess["emo_scores"], summary_obj)

    return {
        "response": ai_reply, 
        "emotion": detected_emotion, 
        "current_turn": current_count, 
        "is_final": is_final_turn,
        "analytics": sess["emo_scores"],
        "transcribed_text": processed_text
    }

@app.get("/history/{username}")
async def get_history(username: str):
    log_dir = f"./patient_logs/{username}"
    if not os.path.exists(log_dir):
        return {"dates": []}
    
    log_files = glob.glob(f"{log_dir}/Data_*.json")
    dates = set()
    for f in log_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 4:
            date_part = parts[-3]
            dates.add(date_part)
            
    return {"dates": sorted(list(dates), reverse=True)}

@app.get("/download-pdf/{username}/{date}")
async def download_pdf(username: str, date: str):
    log_dir = f"./patient_logs/{username}"
    log_files = glob.glob(f"{log_dir}/Data_*_{date}_*.json")
    
    if not log_files:
        raise HTTPException(status_code=404, detail="No logs found for this date")
        
    latest_file = max(log_files, key=os.path.getctime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import copy
    
    pdf_path = f"{log_dir}/report_{date}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    subtitle_style = styles['Heading2']
    subtitle_style.alignment = 1
    section_style = styles['Heading3']
    normal_style = styles['BodyText']
    user_style = copy.copy(styles['BodyText'])
    user_style.textColor = "#10b981" # Greenish tint for user
    ai_style = copy.copy(styles['BodyText'])
    ai_style.textColor = "#3b82f6" # Bluish tint for AI
    
    story = []
    
    # Header
    story.append(Paragraph("Wellness Session Report", title_style))
    display_date = "-".join(reversed(date.split("-")))  # yyyy-mm-dd -> dd-mm-yyyy
    story.append(Paragraph(f"User: {username} | Date: {display_date}", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Clinical Summary
    story.append(Paragraph("Clinical Summary", section_style))
    insights = data.get("ai_insights", {})
    story.append(Paragraph(f"<b>Triggers/Stressors:</b> {insights.get('triggers', 'N/A')}", normal_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Happy Moments:</b> {insights.get('happy_moments', 'N/A')}", normal_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Suggestions:</b> {insights.get('suggestions', 'N/A')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Session Details
    story.append(Paragraph("Session Details", section_style))
    story.append(Paragraph(f"<b>Reported Schedule:</b><br/>{data.get('schedule', 'N/A')}", normal_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>Closing Thoughts:</b><br/>{data.get('summary_text', '')}", normal_style))
    story.append(Spacer(1, 20))
    
    
    # Conversation Transcript
    story.append(Paragraph("Conversation Transcript", section_style))
    history = data.get("history", [])
    if not history:
        story.append(Paragraph("No transcript available.", normal_style))
    else:
        for idx, turn in enumerate(history):
            story.append(Paragraph(f"<b>User:</b> {turn.get('u', '')}", user_style))
            story.append(Spacer(1, 5))
            story.append(Paragraph(f"<b>AI:</b> {turn.get('b', '')}", ai_style))
            story.append(Spacer(1, 10))
    
    # Build PDF
    doc.build(story)
    
    return FileResponse(path=pdf_path, filename=f"Wellness_Report_{date}.pdf", media_type='application/pdf')

@app.get("/download-weekly-pdf/{username}")
async def download_weekly_pdf(username: str):
    log_dir = f"./patient_logs/{username}"
    log_files = glob.glob(f"{log_dir}/Data_*.json")

    if not log_files:
        raise HTTPException(status_code=404, detail="No session logs found for this user")

    log_files.sort(key=os.path.getctime)
    all_sessions = []
    for f in log_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                all_sessions.append(json.load(fp))
        except:
            continue

    if not all_sessions:
        raise HTTPException(status_code=404, detail="Could not load any session data")

    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import copy

    pdf_path = f"{log_dir}/weekly_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    
    title_style = copy.copy(styles['Heading1'])
    title_style.alignment = 1
    subtitle_style = copy.copy(styles['Heading2'])
    subtitle_style.alignment = 1
    section_style = copy.copy(styles['Heading3'])
    section_style.textColor = colors.HexColor('#10b981')
    normal_style = styles['BodyText']
    user_style = copy.copy(styles['BodyText'])
    user_style.textColor = colors.HexColor('#10b981')
    ai_style = copy.copy(styles['BodyText'])
    ai_style.textColor = colors.HexColor('#3b82f6')
    label_style = copy.copy(styles['BodyText'])
    label_style.textColor = colors.HexColor('#ef4444')

    story = []

    # ── COVER ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Weekly Wellness Report", title_style))
    from datetime import datetime
    week_label = datetime.now().strftime("%d-%m-%Y")
    story.append(Paragraph(f"User: {username}  |  Generated: {week_label}  |  Sessions: {len(all_sessions)}", subtitle_style))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#334155')))
    story.append(Spacer(1, 16))

    # ── PER-SESSION SUMMARIES ──────────────────────────────────────────────
    story.append(Paragraph("Day-by-Day Session Summaries", section_style))
    story.append(Spacer(1, 8))

    for sess in all_sessions:
        raw_date = sess.get('date', sess.get('timestamp', '')[:10])
        display_date = "-".join(reversed(raw_date.split("-")))
        insights = sess.get('ai_insights', {})
        emo_turns = [t for t in sess.get('emo_scores', []) if t.get('turn', 0) > 1]
        emotions = [t['emotion'] for t in emo_turns]
        dominant = max(set(emotions), key=emotions.count) if emotions else 'N/A'

        story.append(Paragraph(f"<b>Date: {display_date}  |  Dominant Emotion: {dominant.upper()}</b>", normal_style))
        story.append(Paragraph(f"<font color='#ef4444'><b>Triggers/Stressors:</b></font> {insights.get('triggers', 'N/A')}", normal_style))
        story.append(Paragraph(f"<font color='#10b981'><b>Happy Moments:</b></font> {insights.get('happy_moments', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Suggestion:</b> {insights.get('suggestions', 'N/A')}", normal_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#334155')))
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))

    # ── AGGREGATED HIGHLIGHTS ──────────────────────────────────────────────
    story.append(Paragraph("Week Highlights", section_style))
    story.append(Spacer(1, 8))
    all_triggers = [s.get('ai_insights', {}).get('triggers', '') for s in all_sessions if s.get('ai_insights', {}).get('triggers', '') not in ('', 'N/A')]
    all_happies  = [s.get('ai_insights', {}).get('happy_moments', '') for s in all_sessions if s.get('ai_insights', {}).get('happy_moments', '') not in ('', 'N/A')]

    story.append(Paragraph("<b>Repeated Stressors:</b>", normal_style))
    for t in all_triggers:
        story.append(Paragraph(f"• {t}", normal_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Moments of Joy:</b>", normal_style))
    for h in all_happies:
        story.append(Paragraph(f"• {h}", normal_style))
    story.append(Spacer(1, 16))

    # ── DEEP EMOTIONAL PROFILE (Gemini) ────────────────────────────────────
    story.append(Paragraph("Deep Emotional Health Profile", section_style))
    story.append(Spacer(1, 8))

    client = genai.Client(api_key=API_KEY)
    session_summaries_str = "\n".join([
        f"Date: {s.get('date','')[:10]} | Dominant: {max(set([t['emotion'] for t in s.get('emo_scores',[]) if t.get('turn',0)>1]), key=[t['emotion'] for t in s.get('emo_scores',[]) if t.get('turn',0)>1].count) if [t for t in s.get('emo_scores',[]) if t.get('turn',0)>1] else 'N/A'} | Triggers: {s.get('ai_insights',{}).get('triggers','N/A')} | Happy: {s.get('ai_insights',{}).get('happy_moments','N/A')}"
        for s in all_sessions
    ])

    profile_prompt = f"""You are a senior clinical psychologist writing a comprehensive weekly emotional health report for a patient.

Session data:
{session_summaries_str}

Write a detailed, empathetic, structured emotional health profile with EXACTLY these sections (use the section names as headers, one per line, then the body):

OVERALL EMOTIONAL STATE
DOMINANT EMOTIONS & PATTERNS
STRESS TRIGGERS & ROOT CAUSES
SOURCES OF JOY & POSITIVE ANCHORS
EMOTIONAL VULNERABILITIES
CLINICAL OBSERVATIONS
PERSONALIZED RECOMMENDATIONS

Rules: speak directly to the patient, be specific not generic, 3-5 sentences per section, no markdown symbols."""

    try:
        res = client.models.generate_content(model=MODEL_NAME, contents=profile_prompt)
        deep_profile = res.text.strip()
    except:
        deep_profile = "Deep profile generation unavailable. Please check your API key and try again."

    SECTION_HEADERS = {
        "OVERALL EMOTIONAL STATE", "DOMINANT EMOTIONS & PATTERNS",
        "STRESS TRIGGERS & ROOT CAUSES", "SOURCES OF JOY & POSITIVE ANCHORS",
        "EMOTIONAL VULNERABILITIES", "CLINICAL OBSERVATIONS",
        "PERSONALIZED RECOMMENDATIONS"
    }

    for line in deep_profile.split("\n"):
        stripped = line.strip()
        if stripped.upper() in SECTION_HEADERS:
            story.append(Spacer(1, 8))
            sec_style = copy.copy(styles['BodyText'])
            sec_style.textColor = colors.HexColor('#10b981')
            story.append(Paragraph(f"<b>{stripped.upper()}</b>", sec_style))
        elif stripped:
            story.append(Paragraph(stripped, normal_style))

    story.append(Spacer(1, 16))

    # ── NEXT-WEEK PLAN (Gemini) ─────────────────────────────────────────────
    story.append(Paragraph("Your Personalized Plan for Next Week", section_style))
    story.append(Spacer(1, 8))

    triggers_text = ". ".join(filter(None, all_triggers))
    happies_text  = ". ".join(filter(None, all_happies))
    plan_prompt = f"""You are a supportive wellness coach. Triggers this week: {triggers_text}. Joy sources: {happies_text}.
Write a warm, encouraging 1-paragraph plan for next week (under 100 words, speak directly to the user, no generic advice)."""

    try:
        plan_res = client.models.generate_content(model=MODEL_NAME, contents=plan_prompt)
        weekly_plan = plan_res.text.strip()
    except:
        weekly_plan = "Keep going—you've shown real resilience this week. Use what brought you joy as your fuel for next week."

    story.append(Paragraph(weekly_plan, normal_style))

    # ── BUILD ───────────────────────────────────────────────────────────────
    doc.build(story)
    return FileResponse(path=pdf_path, filename=f"Weekly_Wellness_Report_{username}.pdf", media_type='application/pdf')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

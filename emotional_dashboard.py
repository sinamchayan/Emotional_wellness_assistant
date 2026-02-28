import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import glob
import asyncio
from google import genai
from google.genai import types

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
API_KEY = "AIzaSyC_uGdJBiAw3V6M0hAzl8ymRLhCyehmWGk"
MODEL_NAME = "models/gemma-3-27b-it"

# Ordered for visual hierarchy: Sadness at bottom, Neutral centered, Happiness at top
EMOTIONS = ["sadness", "disgust", "anger", "neutral", "fear", "surprise", "happiness"]

st.set_page_config(page_title="Neural Ninjas Analytics", layout="wide")

st.markdown("""
    <style>
    /* ── Cosmic Nebula Theme — aligned with main chat UI ── */
    .main { background-color: transparent; color: #f3e8ff; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(14,5,45,0.72);
        padding: 22px;
        border-radius: 20px;
        border: 1px solid rgba(216,180,254,.18);
        backdrop-filter: blur(12px);
    }
    [data-testid="stMetricValue"] { color: #d8b4fe !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #c4b5fd !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; }

    /* Headings */
    h1, h2, h3 { color: #d8b4fe !important; font-weight: 800; letter-spacing: -0.5px; }
    h1 { font-size: 2rem !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(14,5,45,0.60) !important;
        border: 1px solid rgba(216,180,254,.14) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(12px);
    }

    /* Selectbox */
    [data-testid="stSelectbox"] label { color: #c4b5fd !important; }

    /* Divider */
    hr { border-color: rgba(216,180,254,.10) !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(147,51,234,.4); border-radius: 4px; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}

    /* Deep profile / plan box */
    .report-box {
        background: rgba(14,5,45,0.70);
        padding: 28px 32px;
        border-radius: 24px;
        border: 1px solid rgba(216,180,254,.16);
        line-height: 1.75;
        font-size: 1.05rem;
        color: #e9d5ff;
        backdrop-filter: blur(12px);
    }

    /* Trigger & happy moment cards in weekly view */
    .trigger-card {
        border-left: 5px solid #f87171;
        margin-bottom: 10px;
        padding: 14px 18px;
        background: rgba(239,68,68,.08);
        border-radius: 12px;
        color: #fecdd3;
    }
    .happy-card {
        border-left: 5px solid #a78bfa;
        margin-bottom: 10px;
        padding: 14px 18px;
        background: rgba(167,139,250,.08);
        border-radius: 12px;
        color: #ede9fe;
    }
    </style>
    """, unsafe_allow_html=True)

query_params = st.query_params
view_mode = query_params.get("view", "daily")
username = query_params.get("username", "Guest")
requested_date = query_params.get("date", None)

# ==========================================
# 2. DATA UTILITIES
# ==========================================
def load_latest_session():
    log_files = glob.glob(f"./patient_logs/{username}/Data_*.json")
    if not log_files: return None
    latest_file = max(log_files, key=os.path.getctime)
    with open(latest_file, 'r', encoding='utf-8') as f: return json.load(f)

def load_session_for_date(target_date):
    log_files = glob.glob(f"./patient_logs/{username}/Data_*_{target_date}_*.json")
    if not log_files: return None
    latest_file = max(log_files, key=os.path.getctime)
    with open(latest_file, 'r', encoding='utf-8') as f: return json.load(f)

def load_all_sessions():
    log_files = glob.glob(f"./patient_logs/{username}/Data_*.json")
    log_files.sort(key=os.path.getctime)
    data_list = []
    for f in log_files:
        try:
            with open(f, 'r', encoding='utf-8') as file: data_list.append(json.load(file))
        except: continue
    return data_list

async def get_dynamic_weekly_plan(triggers, happy_moments):
    """Calls Gemini to write a friendly, practical next-week plan based on triggers and happiness."""
    client = genai.Client(api_key=API_KEY)
    prompt = f"""You are a supportive friend and wellness coach. Look at these notes from a user's past week:
    
    THINGS THAT CAUSED STRESS OR SADNESS: {triggers}
    THINGS THAT MADE THEM HAPPY: {happy_moments}
    
    TASK: Write a warm, encouraging 1-paragraph plan for their next week. 
    1. Acknowledge the difficult moments kindly.
    2. Specifically suggest how they can use the things that made them happy to make next week better.
    3. DO NOT use technical words like 'anchors', 'valence', 'metrics', or 'surety'.
    4. Keep it under 100 words. Speak directly to the user.
    """
    try:
        res = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return res.text.strip()
    except:
        return "You've handled a lot this week. Looking at what made you happy, try to make more time for those specific activities next week—they clearly help you bounce back!"

async def get_deep_emotional_profile(all_sessions):
    """Generates a rich, structured emotional health profile across all sessions."""
    client = genai.Client(api_key=API_KEY)
    
    session_summaries = []
    for sess in all_sessions:
        date = sess.get('date', sess.get('timestamp', '')[:10])
        insights = sess.get('ai_insights', {})
        emotions = [t['emotion'] for t in sess.get('emo_scores', []) if t.get('turn', 0) > 1]
        dominant = max(set(emotions), key=emotions.count) if emotions else 'unknown'
        session_summaries.append(
            f"Date: {date} | Dominant Emotion: {dominant} | "
            f"Triggers: {insights.get('triggers', 'N/A')} | "
            f"Happy Moments: {insights.get('happy_moments', 'N/A')} | "
            f"Suggestion: {insights.get('suggestions', 'N/A')}"
        )
    
    sessions_str = "\n".join(session_summaries)
    
    prompt = f"""You are a senior clinical psychologist writing a comprehensive weekly emotional health report for a patient.

Here is the patient's session data across the week:
{sessions_str}

Write a detailed, empathetic, and structured emotional health profile with EXACTLY these sections. Use plain text, no markdown symbols:

OVERALL EMOTIONAL STATE
Describe the patient's general emotional baseline across the week. Was it stable, volatile, improving, declining? Be specific.

DOMINANT EMOTIONS & PATTERNS
Identify which emotions appeared most. Describe the emotional rhythm - when did they feel most anxious, calm, happy?

STRESS TRIGGERS & ROOT CAUSES
List and deeply analyze each recurring trigger. Why might this trigger cause stress? What underlying need might it reflect?

SOURCES OF JOY & POSITIVE ANCHORS
Describe in detail what brought the patient happiness. How can these be amplified?

EMOTIONAL VULNERABILITIES
What emotional patterns might lead to burnout or distress if not addressed? Be honest but compassionate.

CLINICAL OBSERVATIONS
Any notable behavioral or emotional patterns a therapist would flag. Keep this grounded and non-alarmist.

PERSONALIZED RECOMMENDATIONS
Give 3-5 specific, actionable recommendations tailored to this patient's exact situation.

RULES:
- Speak directly to the patient (use 'you')
- Be warm, non-judgmental, and specific - avoid generic advice
- Each section should be 3-5 sentences minimum
- Do NOT use terms like 'valence', 'anchors', 'metrics'
"""
    try:
        res = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return res.text.strip()
    except Exception as e:
        return f"Could not generate detailed profile at this time. Please try again. ({e})"

# ==========================================
# 3. DASHBOARD VIEWS
# ==========================================

if view_mode == "daily":
    st.title("Daily Mood Snapshot")
    
    # Extract all available dates
    log_files = glob.glob(f"./patient_logs/{username}/Data_*.json")
    available_dates = set()
    for f in log_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 4:
            available_dates.add(parts[-3])
            
    sorted_dates = sorted(list(available_dates), reverse=True)
    
    if not sorted_dates:
        st.info("Complete your first 11-turn session to unlock your daily insights.")
        st.stop()

    # Convert yyyy-mm-dd → dd-mm-yyyy for display
    def to_display(d):
        parts = d.split("-")
        return f"{parts[2]}-{parts[1]}-{parts[0]}" if len(parts) == 3 else d

    # Convert dd-mm-yyyy → yyyy-mm-dd for file lookups
    def to_storage(d):
        parts = d.split("-")
        return f"{parts[2]}-{parts[1]}-{parts[0]}" if len(parts) == 3 else d

    display_dates = [to_display(d) for d in sorted_dates]
        
    st.markdown("### 🗓️ View Historical Analytics")
    
    # Calculate default index for backend deep-linking
    default_ix = 0
    if requested_date and requested_date in sorted_dates:
        default_ix = sorted_dates.index(requested_date)
        
    cols = st.columns([3, 1])
    with cols[0]:
        selected_display_date = st.selectbox("Select Session Date", options=display_dates, index=default_ix)
    # Map display date back to storage format (yyyy-mm-dd) for file lookups & PDF URL
    selected_date = to_storage(selected_display_date)
    with cols[1]:
        st.write("")
        st.write("")
        pdf_url = f"https://training-independently-targeted-examining.trycloudflare.com/download-pdf/{username}/{selected_date}"
        st.link_button("📄 Download PDF Report", pdf_url, use_container_width=True)
        
    st.divider()
    
    data = load_session_for_date(selected_date)
    if not data:
        st.error("Error loading data for the selected date.")
        st.stop()
        
    diagnostic_turns = [item for item in data['emo_scores'] if item['turn'] > 1]
    emo_df = pd.DataFrame([item['scores'] for item in diagnostic_turns])
    labels = [item['emotion'] for item in diagnostic_turns]
    dominant_emo = max(set(labels), key=labels.count)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Vibe", dominant_emo.upper())
    col2.metric("Mood Clarity", f"{emo_df.max(axis=1).mean()*100:.1f}%")
    col3.metric("Analysis Scale", f"{len(labels)} Responses")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Emotion Mix")
        fig_pie = px.pie(names=labels, hole=0.5, color=labels, color_discrete_map={
            "happiness": "#a78bfa", "sadness": "#60a5fa", "anger": "#f87171", "fear": "#e879f9", "neutral": "#94a3b8"
        })
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f3e8ff")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("📈 Emotional Intensity")
        fig_line = px.line(x=range(2, len(labels)+2), y=emo_df.max(axis=1), markers=True)
        fig_line.update_traces(line_color='#d8b4fe', line_width=3, marker_color='#f0abfc', marker_size=10)
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,5,45,0.4)', font_color='#f3e8ff')
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("🌊 Turn-by-Turn Emotional Flow")
    fig_flow = go.Figure(go.Scatter(x=list(range(2, len(labels)+2)), y=labels, mode='lines+markers',
        line=dict(shape='hv', color='#d8b4fe', width=3), marker=dict(color='#f0abfc', size=10)))
    fig_flow.update_layout(
        yaxis=dict(categoryorder='array', categoryarray=EMOTIONS, gridcolor='rgba(216,180,254,.10)'),
        xaxis=dict(gridcolor='rgba(216,180,254,.08)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,5,45,0.40)',
        font_color='#f3e8ff', height=450)
    st.plotly_chart(fig_flow, use_container_width=True)

    st.divider()
    st.subheader("✨ Today's Key Findings")
    insights = data.get("ai_insights", {})
    sc1, sc2, sc3 = st.columns(3)
    sc1.info(f"**What triggered stress?**\n\n{insights.get('triggers', 'N/A')}")
    sc2.success(f"**What brought joy?**\n\n{insights.get('happy_moments', 'N/A')}")
    sc3.warning(f"**Focus for tomorrow**\n\n{insights.get('suggestions', 'N/A')}")

elif view_mode == "weekly":
    st.title("Your Weekly Wellness Journey")
    all_sessions = load_all_sessions()
    
    if len(all_sessions) < 1:
        st.warning("Talk to me for a few more days to see your weekly trends!")
        st.stop()

    unique_days = len(set([sess.get('date', sess.get('timestamp', '')[:8]) for sess in all_sessions if sess.get('date') or sess.get('timestamp')]))
    
    if unique_days < 3:
        remaining_days = 3 - unique_days
        percentage = (unique_days / 3.0) * 100
        
        st.markdown(f"""
        <div style="background:rgba(14,5,45,0.75);padding:40px;border-radius:28px;border:1px dashed rgba(216,180,254,.28);text-align:center;margin-top:50px;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:50vh;backdrop-filter:blur(12px);">
            <div style="background:rgba(7,1,32,.80);border-radius:50%;width:80px;height:80px;display:flex;align-items:center;justify-content:center;margin-bottom:25px;border:2px solid rgba(216,180,254,.25);">
                <span style="font-size:35px;">🔒</span>
            </div>
            <h2 style="color:#d8b4fe !important;margin-bottom:10px;font-weight:800;font-size:2rem;">Weekly Analysis Locked</h2>
            <p style="font-size:1.05rem;color:#c4b5fd;line-height:1.7;max-width:500px;margin:0 auto 30px auto;">
                We need a bit more data to spot your weekly trends accurately! You need 3 distinct days of session logs to unlock this dashboard.
            </p>
            <div style="background:rgba(7,1,32,.70);padding:20px 30px;border-radius:16px;border:1px solid rgba(216,180,254,.16);width:100%;max-width:400px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <span style="color:#e9d5ff;font-weight:600;">Unlock Progress</span>
                    <span style="color:#d8b4fe;font-weight:bold;">{unique_days}/3 Days</span>
                </div>
                <div style="width:100%;background:rgba(216,180,254,.15);border-radius:8px;height:10px;overflow:hidden;">
                    <div style="width:{percentage}%;background:linear-gradient(90deg,#9333ea,#d8b4fe);height:100%;border-radius:8px;"></div>
                </div>
                <p style="color:#c4b5fd;font-weight:600;margin-top:20px;font-size:1.05rem;">
                    🎯 Log <span style="color:#f3e8ff;font-size:1.3rem;">{remaining_days}</span> more {'day' if remaining_days == 1 else 'days'} to unlock!
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    plot_data = []
    weekly_triggers = []
    weekly_happies = []
    
    for sess in all_sessions:
        date_label = sess.get('date', sess.get('timestamp', '')[:10])
        dd_display_chart = "-".join(reversed(date_label.split("-")))
        d_turns = [t for t in sess['emo_scores'] if t['turn'] > 1]
        for t in d_turns:
            plot_data.append({"Date": dd_display_chart, "Emotion": t['emotion'], "Surety": max(t['scores'].values())})
        
        # Collect insights for aggregation
        weekly_triggers.append(sess['ai_insights'].get('triggers', ''))
        weekly_happies.append(sess['ai_insights'].get('happy_moments', ''))

    df = pd.DataFrame(plot_data)

    # 1. STACKED EMOTION TREND
    st.subheader("📅 Mood Patterns Across the Week")
    trend_counts = df.groupby(['Date', 'Emotion']).size().reset_index(name='Count')
    fig_trend = px.bar(trend_counts, x="Date", y="Count", color="Emotion",
                       color_discrete_map={"happiness": "#a78bfa", "sadness": "#60a5fa", "anger": "#f87171", "fear": "#e879f9", "neutral": "#94a3b8", "disgust": "#fb923c", "surprise": "#34d399"})
    fig_trend.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,5,45,0.40)',
                            font_color='#f3e8ff', xaxis=dict(gridcolor='rgba(216,180,254,.08)'), yaxis=dict(gridcolor='rgba(216,180,254,.08)'))
    st.plotly_chart(fig_trend, use_container_width=True)

    col_x, col_y = st.columns(2)
    with col_x:
        # 2. HAPPINESS PULSE
        st.subheader("😊 Happiness Levels")
        hap_pulse = df[df['Emotion'] == 'happiness'].groupby('Date').size() / df.groupby('Date').size() * 100
        fig_pulse = px.line(hap_pulse.reset_index(name='Index'), x='Date', y='Index', markers=True)
        fig_pulse.update_traces(line_color='#a78bfa', fill='tozeroy', fillcolor='rgba(167,139,250,.18)', line_width=3)
        fig_pulse.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,5,45,0.40)', font_color='#f3e8ff')
        st.plotly_chart(fig_pulse, use_container_width=True)
    with col_y:
        # 3. STABILITY PULSE
        st.subheader("🔒 Emotional Stability")
        avg_surety = df.groupby('Date')['Surety'].mean().reset_index()
        fig_area = px.area(avg_surety, x='Date', y='Surety')
        fig_area.update_traces(line_color='#c084fc', fillcolor='rgba(192,132,252,.18)')
        fig_area.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,5,45,0.40)', font_color='#f3e8ff')
        st.plotly_chart(fig_area, use_container_width=True)

    st.divider()
    
    # 4. PER-DAY DETAILED BREAKDOWN
    st.subheader("📋 Day-by-Day Emotional Breakdown")
    for sess in all_sessions:
        date_label = sess.get('date', sess.get('timestamp', '')[:10])
        dd_display = "-".join(reversed(date_label.split("-")))
        insights = sess.get('ai_insights', {})
        d_turns = [t for t in sess.get('emo_scores', []) if t.get('turn', 0) > 1]
        emotions_list = [t['emotion'] for t in d_turns]
        dominant = max(set(emotions_list), key=emotions_list.count) if emotions_list else 'N/A'
        emotion_colors = {"happiness": "#10b981", "sadness": "#3b82f6", "anger": "#ef4444",
                          "fear": "#a855f7", "neutral": "#64748b", "disgust": "#f59e0b", "surprise": "#38bdf8"}
        badge_color = emotion_colors.get(dominant, "#64748b")
        
        with st.expander(f"📅 {dd_display}  —  Dominant: {dominant.upper()}", expanded=False):
            ec1, ec2, ec3 = st.columns(3)
            ec1.markdown(f"""
            <div style='background:rgba(14,5,45,0.72);padding:16px;border-radius:14px;border-left:5px solid {badge_color};backdrop-filter:blur(8px);'>
                <div style='font-size:0.7rem;color:#a78bca;text-transform:uppercase;font-weight:700;letter-spacing:2px;'>Dominant Emotion</div>
                <div style='font-size:1.4rem;font-weight:800;color:{badge_color};margin-top:6px;'>{dominant.upper()}</div>
            </div>""", unsafe_allow_html=True)
            ec2.markdown(f"""
            <div style='background:rgba(14,5,45,0.72);padding:16px;border-radius:14px;border-left:5px solid #f87171;backdrop-filter:blur(8px);'>
                <div style='font-size:0.7rem;color:#a78bca;text-transform:uppercase;font-weight:700;letter-spacing:2px;'>Stress Triggers</div>
                <div style='font-size:0.95rem;color:#f3e8ff;margin-top:6px;line-height:1.5;'>{insights.get('triggers', 'N/A')}</div>
            </div>""", unsafe_allow_html=True)
            ec3.markdown(f"""
            <div style='background:rgba(14,5,45,0.72);padding:16px;border-radius:14px;border-left:5px solid #a78bfa;backdrop-filter:blur(8px);'>
                <div style='font-size:0.7rem;color:#a78bca;text-transform:uppercase;font-weight:700;letter-spacing:2px;'>Happy Moments</div>
                <div style='font-size:0.95rem;color:#f3e8ff;margin-top:6px;line-height:1.5;'>{insights.get('happy_moments', 'N/A')}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""
            <div style='background:rgba(7,1,32,0.65);padding:14px 20px;border-radius:12px;border:1px solid rgba(216,180,254,.14);margin-top:10px;backdrop-filter:blur(8px);'>
                <span style='color:#d8b4fe;font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;'>💡 Suggestion for next day</span><br/>
                <span style='color:#e9d5ff;font-size:0.95rem;line-height:1.6;'>{insights.get('suggestions', 'N/A')}</span>
            </div>""", unsafe_allow_html=True)
    
    st.divider()
    
    # 5. AGGREGATED HIGHLIGHTS
    st.subheader("🔍 Week Highlights")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### 🛑 Repeated Stressors")
        for t in weekly_triggers[-5:]:
            if t and t != "N/A": st.markdown(f"<div class='trigger-card'>{t}</div>", unsafe_allow_html=True)
    with r2:
        st.markdown("#### 🌟 Moments of Joy")
        for h in weekly_happies[-5:]:
            if h and h != "N/A": st.markdown(f"<div class='happy-card'>{h}</div>", unsafe_allow_html=True)

    st.divider()
    
    # 6. DEEP AI EMOTIONAL PROFILE
    st.subheader("🧠 Deep Emotional Health Profile")
    st.caption("AI-generated clinical analysis based on your full week of sessions.")
    with st.spinner("Generating your personalised emotional profile..."):
        deep_profile = asyncio.run(get_deep_emotional_profile(all_sessions))
    
    # Render each section with styled headings
    section_icons = {
        "OVERALL EMOTIONAL STATE": "🌡️",
        "DOMINANT EMOTIONS & PATTERNS": "📊",
        "STRESS TRIGGERS & ROOT CAUSES": "⚡",
        "SOURCES OF JOY & POSITIVE ANCHORS": "🌟",
        "EMOTIONAL VULNERABILITIES": "🛡️",
        "CLINICAL OBSERVATIONS": "🔬",
        "PERSONALIZED RECOMMENDATIONS": "🎯",
    }
    
    current_section = None
    current_lines = []
    
    def flush_section(section_name, lines):
        if not section_name or not lines: return
        icon = section_icons.get(section_name, "📌")
        border_colors = {"OVERALL EMOTIONAL STATE": "#93c5fd", "DOMINANT EMOTIONS & PATTERNS": "#c084fc",
                         "STRESS TRIGGERS & ROOT CAUSES": "#f87171", "SOURCES OF JOY & POSITIVE ANCHORS": "#a78bfa",
                         "EMOTIONAL VULNERABILITIES": "#fbbf24", "CLINICAL OBSERVATIONS": "#94a3b8",
                         "PERSONALIZED RECOMMENDATIONS": "#d8b4fe"}
        color = border_colors.get(section_name, "rgba(216,180,254,.3)")
        body = " ".join(lines).strip()
        st.markdown(f"""
        <div style='background:rgba(14,5,45,0.72);padding:22px 28px;border-radius:18px;border-left:6px solid {color};
                    margin-bottom:16px;line-height:1.75;backdrop-filter:blur(10px);'>
            <div style='font-size:0.75rem;color:{color};text-transform:uppercase;font-weight:800;
                        letter-spacing:2px;margin-bottom:10px;'>{icon} {section_name}</div>
            <div style='color:#e9d5ff;font-size:1rem;'>{body}</div>
        </div>""", unsafe_allow_html=True)
    
    for line in deep_profile.split("\n"):
        stripped = line.strip()
        if stripped.upper() in section_icons:
            flush_section(current_section, current_lines)
            current_section = stripped.upper()
            current_lines = []
        elif stripped:
            current_lines.append(stripped)
    flush_section(current_section, current_lines)
    
    # If model didn't use expected headings, fall back to plain text
    if not current_section:
        st.markdown(f"<div class='report-box'>{deep_profile}</div>", unsafe_allow_html=True)

    st.divider()

    # 7. DYNAMIC LLM PLAN
    st.subheader("🚀 Your Personalized Plan for Next Week")
    triggers_text = ". ".join(filter(None, weekly_triggers))
    happies_text = ". ".join(filter(None, weekly_happies))
    
    weekly_plan = asyncio.run(get_dynamic_weekly_plan(triggers_text, happies_text))
    
    st.markdown(f"""
    <div class="report-box">
        {weekly_plan}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 8. WEEKLY PDF DOWNLOAD
    st.subheader("📄 Export Weekly Report")
    st.caption("Download a comprehensive PDF summary of your full week's emotional journey.")
    weekly_pdf_url = f"https://training-independently-targeted-examining.trycloudflare.com/download-weekly-pdf/{username}"
    st.link_button("📥 Download Weekly PDF Report", weekly_pdf_url, use_container_width=False)

# Run with: streamlit run emotional_dashboard.py
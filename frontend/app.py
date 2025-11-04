import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import random
from pathlib import Path
from datetime import datetime

# Add backend to path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from backend.tools import predict_placement
# Optional: access trained model + scaler for simple explainability
try:
    from backend.tools import model as trained_model, scaler as trained_scaler
except Exception:
    trained_model, trained_scaler = None, None

# Page config
st.set_page_config(
    page_title="IQ Quiz & Placement Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Force dark theme (no toggle)
st.session_state["dark_mode"] = True

# Theme CSS (light/dark)
APP_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    :root {
        --bg: #0d1117; /* higher contrast */
        --bg-accent: radial-gradient(1000px circle at 0% 0%, #0f172a 10%, #0d1117 70%);
        --card: #111827;
        --text: #f1f5f9;
        --muted: #cbd5e1;
        --primary-grad: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        --success: #22c55e;
        --danger: #ef4444;
        --border: 1px solid rgba(148,163,184,0.25);
    }
    .stApp { background: var(--bg); color: var(--text); background-image: var(--bg-accent); font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif; }
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        color: var(--text);
        margin-bottom: 0.75rem;
        letter-spacing: .2px;
    }
    .quiz-question {
        background: var(--card);
        padding: 20px;
        border-radius: 14px;
        margin: 10px 0;
        border: var(--border);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .section-header {
        background: var(--primary-grad);
        color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        font-weight: 700;
        font-size: 1.1em;
    }
    .metric-card { background: var(--card); padding: 16px; border-radius: 14px; border: var(--border); backdrop-filter: blur(8px); }
    /* Buttons */
    .stButton>button { 
        background: #2563eb !important; 
        color: #ffffff !important; 
        border: 0 !important; 
        padding: 0.6rem 1rem !important; 
        border-radius: 12px !important; 
        box-shadow: 0 8px 20px rgba(37,99,235,0.35);
    }
    .stButton>button:hover { filter: brightness(1.05); }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background: rgba(17,24,39,0.95); 
        border-radius: 12px; 
        padding: 10px 14px; 
        border: var(--border);
    }
    /* Sidebar */
    section[data-testid="stSidebar"]>div { background: rgba(13,18,32,0.85); backdrop-filter: blur(8px); }
    /* Tables */
    .stDataFrame, .stTable { border: var(--border); border-radius: 12px; color: var(--text); }
    /* Ensure common text elements inherit readable color */
    p, li, label, span, h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
    small, .muted { color: var(--muted) !important; }
    /* Metrics */
    div[data-testid="stMetricValue"] { color: var(--text); }
    /* Improve visibility of radios, selects, multiselects */
    div[role="radiogroup"] label { color: var(--text) !important; background: transparent !important; }
    div[role="radiogroup"] label:hover { filter: brightness(1.1); }
    .stRadio { background: transparent; }
    [data-baseweb="select"] * { color: var(--text) !important; }
    [data-baseweb="select"] div { background-color: var(--card) !important; border-color: rgba(148,163,184,0.25) !important; }
    [data-baseweb="select"] svg { fill: var(--text) !important; }
    /* Select dropdown menu (global override for portaled BaseWeb menu) */
    div[role="listbox"][data-baseweb="menu"] { 
        background-color: #000000 !important; 
        border: 1px solid #475569 !important; 
        box-shadow: 0 12px 28px rgba(0,0,0,0.6) !important; 
        z-index: 10000 !important; 
    }
    div[role="listbox"][data-baseweb="menu"] [role="option"] { 
        color: #ffffff !important; 
        opacity: 1 !important; 
        background: transparent !important; 
    }
    div[role="listbox"][data-baseweb="menu"] [role="option"] * { 
        color: inherit !important; opacity: 1 !important; 
    }
    div[role="listbox"][data-baseweb="menu"] [role="option"][aria-selected="true"],
    div[role="listbox"][data-baseweb="menu"] [role="option"]:hover { 
        background-color: #111111 !important; 
        color: #ffffff !important; 
    }
    /* Fallback: any listbox options (even if attributes differ) */
    div[role="listbox"] [role="option"] { color: #ffffff !important; opacity: 1 !important; }
    div[role="listbox"] { background-color: #000000 !important; }

    /* Ultimate override: force white text on black for all BaseWeb menus */
    .stApp div[data-baseweb="menu"],
    .stApp div[role="listbox"][data-baseweb="menu"],
    .stApp div[role="listbox"] {
        background-color: #000000 !important;
        border: 1px solid #475569 !important;
        box-shadow: 0 12px 28px rgba(0,0,0,0.6) !important;
        z-index: 10000 !important;
    }
    .stApp div[data-baseweb="menu"] [role="option"],
    .stApp div[role="listbox"] [role="option"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        background: transparent !important;
        mix-blend-mode: normal !important;
    }
    .stApp div[data-baseweb="menu"] [role="option"] *,
    .stApp div[role="listbox"] [role="option"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
    }
    /* EXTRA universal fallbacks (handles future/emotion class changes) */
    .stApp [role="listbox"],
    .stApp [role="listbox"] * {
        background-color: #000000 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }
    .stApp [role="option"],
    .stApp [role="option"] * {
        background-color: transparent !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* GLOBAL OVERRIDES (no .stApp scope) to catch portaled menus */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="listbox"] * {
        background-color: #000000 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
    }
    [role="option"],
    [role="option"] * {
        background-color: transparent !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [role="option"][aria-selected="true"],
    [role="option"]:hover {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    /* Hover/selected */
    .stApp div[data-baseweb="menu"] [role="option"][aria-selected="true"],
    .stApp div[role="listbox"] [role="option"][aria-selected="true"],
    .stApp div[data-baseweb="menu"] [role="option"]:hover,
    .stApp div[role="listbox"] [role="option"]:hover {
        background-color: #111111 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* ================================ */
    /* Enter Your Profile: black/white  */
    /* ================================ */
    #profile-form * { color: #ffffff !important; }
    #profile-form label { color: #ffffff !important; }
    /* Number input */
    #profile-form [data-testid="stNumberInput"] input { 
        background-color: #000000 !important; 
        color: #ffffff !important; 
        border: 1px solid #334155 !important; 
    }
    /* Select/Multiselect inputs */
    #profile-form [data-baseweb="select"] div { 
        background-color: #000000 !important; 
        color: #ffffff !important; 
        border-color: #334155 !important; 
    }
    #profile-form [data-baseweb="select"] input { 
        background: transparent !important; 
        color: #ffffff !important; 
    }
    #profile-form [data-baseweb="select"] input::placeholder { color: #cbd5e1 !important; }
    /* Dropdown options in this section */
    #profile-form div[role="listbox"],
    #profile-form div[role="listbox"][data-baseweb="menu"] { background-color: #000000 !important; }
    #profile-form div[role="listbox"] [role="option"] { color: #ffffff !important; opacity: 1 !important; }
    #profile-form div[role="listbox"] [role="option"]:hover,
    #profile-form div[role="listbox"] [role="option"][aria-selected="true"] { background-color: #111111 !important; color: #ffffff !important; }

    /* Dataframe readability */
    [data-testid="stDataFrame"] * { color: #e2e8f0 !important; }
    [data-testid="stDataFrame"] thead th { background-color: #0f172a !important; color: #f8fafc !important; }
    [data-testid="stDataFrame"] tbody tr { background-color: #0b1220 !important; }
    [data-testid="stDataFrame"] tbody tr:nth-child(odd) { background-color: #0e1526 !important; }
    
    /* Buttons: download + file upload */
    [data-testid="stDownloadButton"]>button { 
        background: #3b82f6 !important; color: #ffffff !important; border: 0 !important; 
        border-radius: 10px !important; padding: 0.6rem 1rem !important; 
        box-shadow: 0 6px 16px rgba(59,130,246,0.35) !important; 
    }
    [data-testid="stDownloadButton"]>button:hover { filter: brightness(1.08) !important; }
    [data-testid="stFileUploader"] section div { background: #0b1220 !important; color: #e2e8f0 !important; border: var(--border) !important; }
    [data-testid="stFileUploader"] svg { fill: #e2e8f0 !important; }
    </style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# Simple router state
if "route" not in st.session_state:
    st.session_state.route = "landing"  # landing -> auth -> main

# ---------- Landing Page ----------
if st.session_state.route == "landing":
    st.markdown("""
    <div style='padding: 6rem 1rem; text-align:center;'>
        <h1 class='main-header' style='font-size:3rem;'>Placement Predictor</h1>
        <p style='font-size:1.15rem; color:#94a3b8; max-width:720px; margin:0 auto;'>
            Ace your placements with a focused IQ + CGPA assessment, live insights, and a personalized roadmap.
        </p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.route = "auth"
            st.rerun()
    st.markdown("""
    <div style='text-align:center; margin-top:1rem; color:#94a3b8;'>
        No signup required — you can continue as Guest on the next screen.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------- Auth Page ----------
if st.session_state.route == "auth":
    st.markdown("<p class='main-header'>Welcome</p>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .auth-card { background:#0b1220; border:1px solid rgba(148,163,184,0.25); border-radius:14px; padding:24px; }
    .muted { color:#94a3b8; }
    </style>
    """, unsafe_allow_html=True)
    left, right = st.columns([1,1])
    with left:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.subheader("Login")
        st.caption("Use any username and password for this demo")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login", type="primary"):
            if len(user) > 0:
                st.session_state["user_name"] = user
                st.session_state.route = "main"
                st.rerun()
            else:
                st.warning("Enter a username or use Guest.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        st.subheader("Or Continue as Guest")
        st.caption("Skip login and explore the app immediately")
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state["user_name"] = "Guest"
            st.session_state.route = "main"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# IQ Test Question Bank
IQ_QUESTIONS = {
    "Logical Reasoning": [
        {
            "question": "If all roses are flowers and some flowers fade quickly, then:",
            "options": ["All roses fade quickly", "Some roses might fade quickly", "No roses fade quickly", "All flowers are roses"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "Complete the sequence: 2, 6, 12, 20, 30, ?",
            "options": ["40", "42", "38", "36"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "If A = 1, B = 2, C = 3, what is the sum of CAT?",
            "options": ["24", "23", "22", "25"],
            "correct": 0,
            "difficulty": "easy"
        },
        {
            "question": "If 5 workers take 5 hours to complete 5 tasks, how long for 100 workers to complete 100 tasks?",
            "options": ["100 hours", "20 hours", "5 hours", "1 hour"],
            "correct": 2,
            "difficulty": "hard"
        },
        {
            "question": "What comes next: J, F, M, A, M, ?",
            "options": ["J", "S", "N", "D"],
            "correct": 0,
            "difficulty": "medium"
        }
    ],
    "Pattern Recognition": [
        {
            "question": "Find the odd one out: 3, 5, 7, 9, 12, 13",
            "options": ["3", "9", "12", "13"],
            "correct": 2,
            "difficulty": "easy"
        },
        {
            "question": "Complete: 1, 4, 9, 16, 25, ?",
            "options": ["30", "35", "36", "49"],
            "correct": 2,
            "difficulty": "easy"
        },
        {
            "question": "What's the pattern: AB, CD, EF, GH, ?",
            "options": ["IJ", "HI", "JK", "IK"],
            "correct": 0,
            "difficulty": "easy"
        },
        {
            "question": "Find the next: 2, 5, 11, 23, 47, ?",
            "options": ["94", "95", "96", "97"],
            "correct": 1,
            "difficulty": "hard"
        },
        {
            "question": "Complete: Z, X, V, T, R, ?",
            "options": ["Q", "P", "O", "N"],
            "correct": 1,
            "difficulty": "medium"
        }
    ],
    "Mathematical Ability": [
        {
            "question": "If x + 5 = 12, what is x?",
            "options": ["5", "6", "7", "8"],
            "correct": 2,
            "difficulty": "easy"
        },
        {
            "question": "What is 15% of 200?",
            "options": ["25", "30", "35", "40"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "If a shirt costs $80 after a 20% discount, what was the original price?",
            "options": ["$96", "$100", "$104", "$110"],
            "correct": 1,
            "difficulty": "hard"
        },
        {
            "question": "Simplify: (8 + 2) × 5 - 10",
            "options": ["30", "40", "50", "60"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "If 3x = 27, what is x²?",
            "options": ["9", "27", "81", "243"],
            "correct": 2,
            "difficulty": "medium"
        }
    ],
    "Verbal Reasoning": [
        {
            "question": "Choose the word most similar to 'HAPPY':",
            "options": ["Sad", "Joyful", "Angry", "Tired"],
            "correct": 1,
            "difficulty": "easy"
        },
        {
            "question": "Complete: Book is to Reading as Fork is to ?",
            "options": ["Eating", "Cooking", "Kitchen", "Food"],
            "correct": 0,
            "difficulty": "easy"
        },
        {
            "question": "Find the antonym of 'ABUNDANT':",
            "options": ["Plentiful", "Scarce", "Many", "Rich"],
            "correct": 1,
            "difficulty": "medium"
        },
        {
            "question": "Doctor : Patient :: Teacher : ?",
            "options": ["School", "Student", "Book", "Class"],
            "correct": 1,
            "difficulty": "easy"
        },
        {
            "question": "Which word doesn't belong: Apple, Banana, Carrot, Orange",
            "options": ["Apple", "Banana", "Carrot", "Orange"],
            "correct": 2,
            "difficulty": "easy"
        }
    ],
    "Spatial Reasoning": [
        {
            "question": "How many faces does a cube have?",
            "options": ["4", "6", "8", "12"],
            "correct": 1,
            "difficulty": "easy"
        },
        {
            "question": "If you fold a paper in half 3 times and make one cut, how many pieces will you have?",
            "options": ["4", "6", "8", "9"],
            "correct": 3,
            "difficulty": "hard"
        },
        {
            "question": "A clock shows 3:15. What is the angle between hour and minute hands?",
            "options": ["0°", "7.5°", "15°", "22.5°"],
            "correct": 1,
            "difficulty": "hard"
        },
        {
            "question": "How many edges does a triangular pyramid have?",
            "options": ["4", "5", "6", "7"],
            "correct": 2,
            "difficulty": "medium"
        },
        {
            "question": "If a square is rotated 90° clockwise, it will look:",
            "options": ["Different", "The same", "Larger", "Smaller"],
            "correct": 1,
            "difficulty": "easy"
        }
    ]
}

# Initialize session state
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False
if "iq_score" not in st.session_state:
    st.session_state.iq_score = None
if "test_data" not in st.session_state:
    st.session_state.test_data = []

def generate_quiz():
    """Generate 10 random questions (2 from each section)"""
    quiz = []
    for section, questions in IQ_QUESTIONS.items():
        selected = random.sample(questions, 2)  # 2 questions per section
        for q in selected:
            quiz.append({
                "section": section,
                "question": q["question"],
                "options": q["options"],
                "correct": q["correct"],
                "difficulty": q["difficulty"]
            })
    random.shuffle(quiz)
    return quiz

def calculate_iq(answers, questions):
    """Calculate IQ score based on answers"""
    correct_count = 0
    difficulty_bonus = 0
    
    for i, answer in enumerate(answers):
        if answer == questions[i]["correct"]:
            correct_count += 1
            # Bonus points for harder questions
            if questions[i]["difficulty"] == "hard":
                difficulty_bonus += 2
            elif questions[i]["difficulty"] == "medium":
                difficulty_bonus += 1
    
    # Base IQ calculation
    base_score = (correct_count / len(questions)) * 100
    
    # Convert to IQ scale (70-160)
    # 50% correct = 100 IQ (average)
    # 100% correct = 140 IQ (high)
    # 0% correct = 70 IQ (low)
    
    iq = 70 + (base_score / 100) * 70 + difficulty_bonus
    iq = min(max(iq, 70), 160)  # Clamp between 70-160
    
    return round(iq, 1), correct_count

# Main title
st.markdown('<p class="main-header">🧠 IQ Assessment & Placement Prediction</p>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">Build your employability DNA: take the IQ test, enter your details, and get a tailored roadmap.</div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Test Status")
    
    if st.session_state.quiz_completed:
        st.success("✅ IQ Test Completed")
        st.metric("Your IQ Score", st.session_state.iq_score)
    else:
        st.info("📝 Take the IQ test first")
    
    st.markdown("---")
    st.header("📈 Statistics")
    st.metric("Tests Completed", len(st.session_state.test_data))
    
    if len(st.session_state.test_data) > 0:
        df = pd.DataFrame(st.session_state.test_data)
        avg_iq = df['iq'].mean()
        st.metric("Average IQ", f"{avg_iq:.1f}")
        
        if st.button("💾 Download Test Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "iq_test_data.csv",
                "text/csv"
            )

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🧠 IQ Test", "🎓 Student View", "🏢 TPO Dashboard", "🧩 Explainability"])

# ===================================
# TAB 1: IQ Test
# ===================================
with tab1:
    st.header("🧠 IQ Assessment Test")
    st.info("This test consists of 10 questions across 5 cognitive areas. Answer carefully!")
    
    if not st.session_state.quiz_started:
        st.markdown("""
        ### Test Sections:
        - 🧩 Logical Reasoning
        - 🔍 Pattern Recognition  
        - 🔢 Mathematical Ability
        - 📝 Verbal Reasoning
        - 🎲 Spatial Reasoning
        
        **Instructions:**
        - You will get 2 questions from each section (10 total)
        - Questions are randomized each time
        - Choose the best answer for each question
        - Your IQ score will be calculated automatically
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start IQ Test", type="primary"):
                st.session_state.quiz_started = True
                st.session_state.quiz_questions = generate_quiz()
                st.session_state.current_question = 0
                st.session_state.user_answers = []
                st.session_state.quiz_completed = False
                st.rerun()
    
    elif st.session_state.quiz_started and not st.session_state.quiz_completed:
        # Show progress
        progress = st.session_state.current_question / len(st.session_state.quiz_questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}")
        
        # Get current question
        q = st.session_state.quiz_questions[st.session_state.current_question]
        
        # Section header
        st.markdown(f'<div class="section-header">{q["section"]}</div>', unsafe_allow_html=True)
        
        # Question
        st.markdown(f'<div class="quiz-question"><h3>{q["question"]}</h3></div>', unsafe_allow_html=True)
        
        # Options
        st.write("")
        answer = st.radio(
            "Select your answer:",
            options=range(len(q["options"])),
            format_func=lambda x: f"{chr(65+x)}. {q['options'][x]}",
            key=f"q_{st.session_state.current_question}"
        )
        
        st.write("")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("Next Question ➡️", type="primary"):
                st.session_state.user_answers.append(answer)
                
                if st.session_state.current_question < len(st.session_state.quiz_questions) - 1:
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    # Quiz completed
                    iq_score, correct_count = calculate_iq(
                        st.session_state.user_answers,
                        st.session_state.quiz_questions
                    )
                    st.session_state.iq_score = iq_score
                    st.session_state.quiz_completed = True
                    st.rerun()
    
    elif st.session_state.quiz_completed:
        st.success("🎉 IQ Test Completed!")
        
        col1, col2, col3 = st.columns(3)
        
        iq_score, correct_count = calculate_iq(
            st.session_state.user_answers,
            st.session_state.quiz_questions
        )
        
        with col1:
            st.metric("Your IQ Score", f"{iq_score}")
        with col2:
            st.metric("Correct Answers", f"{correct_count}/10")
        with col3:
            accuracy = (correct_count / 10) * 100
            st.metric("Accuracy", f"{accuracy:.0f}%")
        
        # IQ interpretation
        st.markdown("---")
        st.subheader("📊 Score Interpretation")
        
        if iq_score >= 130:
            interpretation = "🌟 **Exceptional** - Very superior intelligence"
            color = "#10b981"
        elif iq_score >= 120:
            interpretation = "⭐ **Superior** - Above average intelligence"
            color = "#3b82f6"
        elif iq_score >= 110:
            interpretation = "✨ **High Average** - Above average"
            color = "#8b5cf6"
        elif iq_score >= 90:
            interpretation = "👍 **Average** - Normal intelligence"
            color = "#f59e0b"
        elif iq_score >= 80:
            interpretation = "📌 **Low Average** - Below average"
            color = "#ef4444"
        else:
            interpretation = "📍 **Below Average** - Needs improvement"
            color = "#dc2626"
        
        st.markdown(f"<div style='background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 1.2em;'>{interpretation}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Test"):
                st.session_state.quiz_started = False
                st.session_state.quiz_completed = False
                st.session_state.user_answers = []
                st.session_state.current_question = 0
                st.rerun()
        
        with col2:
            if st.button("➡️ Continue to Placement Prediction", type="primary"):
                st.info("Go to the 'Student View' tab to continue")

# ===================================
# TAB 2: Placement Prediction
# ===================================
with tab2:
    st.header("🎓 Student View")
    
    if not st.session_state.quiz_completed:
        st.warning("⚠️ Please complete the IQ test first!")
        st.info("Go to the 'IQ Test' tab to take the assessment")
    else:
        st.success(f"✅ IQ Score Recorded: {st.session_state.iq_score}")
        
        st.markdown("---")
        st.subheader("Enter Your Profile")
        # Start styled container for profile form
        st.markdown("<div id='profile-form'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            cgpa = st.number_input(
                "CGPA (0-10 scale)",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
                help="Enter your cumulative GPA"
            )
            
            branch = st.selectbox(
                "Branch/Department",
                ["Computer Science", "IT", "ECE", "EEE", "Mechanical", "Civil", "Other"]
            )
            
            certifications = st.multiselect(
                "Certifications",
                ["Python", "SQL", "Excel", "Power BI", "ML Basics", "Deep Learning", "Cloud (AWS/Azure/GCP)", "DSA"],
                default=[]
            )
            
            projects = st.number_input(
                "Number of completed projects",
                min_value=0,
                max_value=50,
                value=1,
                step=1
            )
        
        with col2:
            year = st.selectbox(
                "Current Year",
                ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduated"]
            )
            
            internship = st.selectbox(
                "Internship Experience",
                ["None", "< 3 months", "3-6 months", "> 6 months"]
            )
            
            soft_skills = st.selectbox(
                "Soft Skills",
                ["Needs improvement", "Average", "Good", "Excellent"]
            )
            
            extracurricular = st.selectbox(
                "Extracurricular Involvement",
                ["Low", "Moderate", "High"]
            )
        # End styled container
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.button("🔮 Predict My Placement", type="primary"):
            with st.spinner("Analyzing your profile..."):
                try:
                    if st.session_state.iq_score is None:
                        st.warning("Please complete the IQ test first.")
                        st.stop()
                    pred, prob, influence = predict_placement(cgpa, st.session_state.iq_score)
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        status = "✅ PLACED" if pred == 1 else "❌ NOT PLACED"
                        color = "#10b981" if pred == 1 else "#ef4444"
                        st.markdown(f"<div style='background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;'><h2>{status}</h2></div>", unsafe_allow_html=True)
                    with result_col2:
                        st.metric("Placement Probability", f"{prob:.0%}")
                    with result_col3:
                        st.metric("Key Factor", influence)
                    with result_col4:
                        target_role = "Data Analyst" if ("Python" in certifications or "Excel" in certifications) else ("Software Engineer" if "DSA" in certifications else "IT Support")
                        st.metric("Recommended Track", target_role)
                    # Simple confidence for pathway (heuristic)
                    role_bonus = 0.1 if (target_role == "Data Analyst" and ("Python" in certifications or "Excel" in certifications)) else (0.1 if target_role == "Software Engineer" and "DSA" in certifications else 0.0)
                    pathway_conf = min(0.95, max(0.5, prob + role_bonus))
                    st.info(f"Pathway Confidence: {pathway_conf:.0%}")
                    
                    # Recommendations (rule-based augmentation)
                    st.markdown("---")
                    st.subheader("🧭 Personalized Recommendations")
                    recs = []
                    if cgpa < 7.5:
                        recs.append("Improve CGPA to 7.5+; focus on core subjects")
                    if st.session_state.iq_score < 110:
                        recs.append("Practice aptitude/logic daily for 2–3 weeks")
                    if internship == "None":
                        recs.append("Pursue a 6–8 week internship or live project")
                    if projects < 2:
                        recs.append("Complete 2 portfolio projects aligned to your track")
                    if soft_skills in ["Needs improvement", "Average"]:
                        recs.append("Take a communication + interview prep module")
                    if "Python" not in certifications and target_role in ["Data Analyst", "Software Engineer"]:
                        recs.append("Take a Python + SQL course and build one end‑to‑end project")
                    if target_role == "Data Analyst" and "Power BI" not in certifications:
                        recs.append("Learn Power BI and publish a dashboard case study")
                    if len(recs) == 0:
                        recs.append("Keep strengthening DSA, system design, and mock interviews")
                    
                    for r in recs:
                        st.write(f"• {r}")
                    
                    # Save to test data (for TPO dashboard)
                    test_record = {
                        "cgpa": cgpa,
                        "iq": st.session_state.iq_score,
                        "branch": branch,
                        "year": year,
                        "predicted_placement": pred,
                        "confidence": prob,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.test_data.append(test_record)
                    # Cache last inputs for explainability
                    st.session_state["last_inputs"] = {"cgpa": cgpa, "iq": st.session_state.iq_score}
                    st.session_state["last_prediction"] = {"pred": pred, "prob": prob, "role": target_role}
                    st.success(f"✅ Prediction saved! Total records: {len(st.session_state.test_data)}")
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
                    st.info("Make sure the model is trained. Check if backend/placement_model.pkl exists.")

# ===================================
# TAB 3: Results & Analytics
# ===================================
with tab3:
    st.header("🏢 TPO Dashboard")
    
    if len(st.session_state.test_data) == 0:
        st.info("No records yet. Ask students to complete IQ test and prediction.")
    else:
        df = pd.DataFrame(st.session_state.test_data)
        
        # Summary metrics
        st.subheader("📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Predicted Placed", int(df['predicted_placement'].sum()))
        with col3:
            st.metric("Avg Confidence", f"{(df['confidence'].mean() if 'confidence' in df else 0)*100:.0f}%")
        with col4:
            st.metric("Avg CGPA", f"{df['cgpa'].mean():.2f}")
        
        st.markdown("---")
        
        # Department readiness
        st.subheader("🏫 Department Readiness")
        try:
            dept = df.groupby('branch').agg(
                total=('branch', 'count'),
                placed_rate=('predicted_placement', 'mean')
            ).reset_index()
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.bar(dept['branch'], (dept['placed_rate']*100).round(1), color="#1f77b4")
            ax3.set_ylabel('Predicted Placement %')
            ax3.set_xlabel('Department')
            ax3.set_title('Readiness by Department')
            ax3.grid(True, axis='y', alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=20, ha='right')
            st.pyplot(fig3)
        except Exception as e:
            st.info(f"Not enough data to compute department view: {e}")
        
        st.markdown("---")
        st.subheader("📋 Records")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Records (CSV)",
            csv,
            "placement_records.csv",
            "text/csv",
            use_container_width=True
        )

with tab4:
    st.header("🧩 Explainability")
    if "last_inputs" not in st.session_state:
        st.info("Run a prediction in the Student View to see explanations.")
    else:
        cgpa_val = st.session_state["last_inputs"]["cgpa"]
        iq_val = st.session_state["last_inputs"]["iq"]
        st.write(f"Using last inputs → CGPA: {cgpa_val}, IQ: {iq_val}")

        try:
            if trained_model is not None and hasattr(trained_model, "coef_"):
                import numpy as np
                import matplotlib.pyplot as plt

                X = [[cgpa_val, iq_val]]
                if trained_scaler is not None:
                    Xs = trained_scaler.transform(X)
                else:
                    Xs = np.array(X)

                coefs = trained_model.coef_[0]
                contrib = coefs * Xs[0]
                labels = ["CGPA", "IQ"]

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#1f77b4" if v >= 0 else "#ef4444" for v in contrib]
                ax.bar(labels, contrib, color=colors)
                ax.set_title("Feature contribution (log-odds)")
                ax.axhline(0, color="#94a3b8", linewidth=1)
                st.pyplot(fig)

                st.caption("Approximate contribution computed from linear coefficients; not SHAP.")
            else:
                st.info("Detailed explainability not available for this model type. Consider adding SHAP.")
        except Exception as e:
            st.error(f"Explainability error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🧠 IQ Assessment & Placement Prediction System</p>
    <p style='font-size: 0.9rem;'>Built with Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
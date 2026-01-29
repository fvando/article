import streamlit as st
from src.core.i18n import TRANSLATIONS

def apply_corporate_style():
    # Language Selector in Sidebar
    lang_options = {"🇺🇸 English": "en", "🇧🇷 Português": "pt"}
    
    # Initialize session state if needed
    if "language" not in st.session_state:
        st.session_state["language"] = "en"
        
    # Reverse lookup for display
    current_key = [k for k, v in lang_options.items() if v == st.session_state["language"]]
    default_index = list(lang_options.keys()).index(current_key[0]) if current_key else 0
    
    selected_lang = st.sidebar.radio(
        "Language / Idioma",
        options=list(lang_options.keys()),
        index=default_index,
        key="lang_selector",
        horizontal=True
    )
    
    # Update state
    st.session_state["language"] = lang_options[selected_lang]

    st.markdown("""
    <style>
        /* IMPORT MODERN FONT */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* PROFESSIONAL SOBER THEME VARIABLES */
        :root {
            --primary-slate: #0F172A;    /* Slate-900 */
            --secondary-slate: #334155;  /* Slate-700 */
            --accent-blue: #2563EB;      /* Blue-600 */
            --bg-slate: #F8FAFC;         /* Slate-50 */
            --border-slate: #E2E8F0;     /* Slate-200 */
            --text-main: #1E293B;        /* Slate-800 */
            --text-muted: #64748B;       /* Slate-500 */
            --card-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        }

        /* GLOBAL RESET */
        .stApp {
            background-color: var(--bg-slate);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* SIDEBAR - DEEP SOBER LOOK */
        [data-testid="stSidebar"] {
            background-color: var(--primary-slate);
            border-right: 1px solid var(--border-slate);
        }
        [data-testid="stSidebar"] * {
            color: #F1F5F9 !important; /* Slate-100 */
        }
        [data-testid="stSidebar"] .stSelectbox label {
            color: #94A3B8 !important; /* Slate-400 */
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* METRICS - PREMIUM FLAT LOOK */
        [data-testid="stMetric"] {
            background-color: white;
            padding: 1.25rem;
            border-radius: 0.75rem;
            box-shadow: var(--card-shadow);
            border: 1px solid var(--border-slate);
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.85rem;
            font-weight: 500;
        }
        [data-testid="stMetricValue"] {
            color: var(--primary-slate) !important;
            font-size: 1.75rem;
            font-weight: 700;
        }

        /* BUTTONS - SOBER & ACCESSIBLE */
        .stButton button {
            background-color: var(--primary-slate);
            color: white;
            border-radius: 0.5rem;
            border: 1px solid var(--primary-slate);
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            background-color: var(--secondary-slate);
            border-color: var(--secondary-slate);
            color: white;
        }

        /* EXPANDERS */
        .streamlit-expanderHeader {
            background-color: white;
            border: 1px solid var(--border-slate);
            border-radius: 0.5rem;
            font-weight: 600;
            color: var(--text-main);
        }

        /* TITLES & HEADINGS */
        h1, h2, h3 {
            color: var(--primary-slate);
            font-weight: 700 !important;
        }

        /* CUSTOM UTILS */
        .stAlert {
            border-radius: 0.5rem;
            border: none;
            border-left: 4px solid #3B82F6;
        }

        /* HIDE DEFAULT FOOTER */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* header {visibility: hidden;}  <-- UNHIDDEN TO SHOW HAMBURGER MENU */
    </style>
    """, unsafe_allow_html=True)

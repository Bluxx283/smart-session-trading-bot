from datetime import datetime
import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from tradingview_ta import Interval, TA_Handler
import yfinance as yf

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Session Anomaly Detector | Institutional Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL BACKGROUND + HEADER STYLING ---
st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(6,8,14,0.92), rgba(6,8,14,0.95)),
            repeating-linear-gradient(135deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 2px, transparent 2px, transparent 28px),
            radial-gradient(circle at 15% 10%, rgba(80,140,255,0.10), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(255,80,140,0.08), transparent 40%),
            #05070c;
        background-attachment: fixed;
    }
    .ssad-page-header {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        padding: 2px 0 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 14px;
    }
    .ssad-page-header span { color: rgba(255,255,255,0.55); font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(subtitle):
    st.markdown(
        f'<div class="ssad-page-header">⚡ Smart Session Anomaly Detector '
        f'<span>| {subtitle}</span></div>',
        unsafe_allow_html=True,
    )


# Auto-refresh session feed every 20 seconds
st_autorefresh(interval=20000, key="ssad_feed_sync")

# --- SESSION STATE INITIALIZATION ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "broker_connected" not in st.session_state:
    st.session_state.broker_connected = False

# --- FLOATING AI BOT: SESSION STATE ---
if "bot_open" not in st.session_state:
    st.session_state.bot_open = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hey, I'm your trading co-pilot. Ask me about the market, "
                "chart signals, or build a strategy in the Strategies tab."
            ),
        }
    ]
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""
if "claude_model" not in st.session_state:
    st.session_state.claude_model = "claude-sonnet-4-5"
if "my_strategies" not in st.session_state:
    st.session_state.my_strategies = []
if "last_spoken_index" not in st.session_state:
    st.session_state.last_spoken_index = -1
if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = ""

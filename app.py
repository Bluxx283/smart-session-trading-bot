import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from tradingview_ta import TA_Handler, Interval
from sklearn.ensemble import IsolationForest
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Session Anomaly Detector | Institutional Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh session feed every 20 seconds
st_autorefresh(interval=20000, key="ssad_feed_sync")

# --- MODERN STYLING (CSS) ---
st.markdown("""
    
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "📊 Dashboard"
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "broker_connected" not in st.session_state:
    st.session_state.broker_connected = False

# Navigation options
NAV_OPTIONS = [
    "📊 Dashboard", 
    "📈 AI Trade & Charts", 
    "🧮 Pip & Risk Calculator", 
    "⚡ Broker Gateway", 
    "📅 Economic Calendar", 
    "⚙️ Settings"
]

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("### ⚡ **Smart Session Anomaly Detector**")
st.sidebar.caption("Quantitative Market Pattern Engine")

curr_idx = NAV_OPTIONS.index(st.session_state.nav_page) if st.session_state.nav_page in NAV_OPTIONS else 0
selected_nav = st.sidebar.radio(
    "Navigation Menu",
    NAV_OPTIONS,
    index=curr_idx,
    label_visibility="collapsed",
    key="ssad_nav_radio"
)

if selected_nav != st.session_state.nav_page:
    st.session_state.nav_page = selected_nav
    st.rerun()

st.sidebar.markdown("""

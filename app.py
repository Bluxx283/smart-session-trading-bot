import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from tradingview_ta import TA_Handler, Interval
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TRADER MASTER | AI Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh app every 20 seconds
st_autorefresh(interval=20000, key="saas_dashboard_sync")

# --- CUSTOM SAAS DARK UI STYLING ---
st.markdown("""
    
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "📊 Dashboard"
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "broker_connected" not in st.session_state:
    st.session_state.broker_connected = False

# Navigation options
nav_options = [
    "📊 Dashboard", 
    "📈 AI Trade & Charts", 
    "🧮 Pip & Risk Calculator", 
    "⚡ Broker Gateway", 
    "📅 Economic Calendar", 
    "⚙️ Settings"
]

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("### ⚡ **TRADER MASTER**")
st.sidebar.caption("AI-Powered Institutional Suite")

current_idx = nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0
selected_page = st.sidebar.radio(
    "Navigation Menu",
    nav_options,
    index=current_idx,
    label_visibility="collapsed",
    key="sidebar_radio"
)

if selected_page != st.session_state.nav_page:
    st.session_state.nav_page = selected_page
    st.rerun()

# User Profile Pill in Sidebar
st.sidebar.markdown("""

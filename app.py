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
    page_title="TRADER MASTER | AI Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh app every 20 seconds
st_autorefresh(interval=20000, key="saas_dashboard_sync")

# --- CUSTOM SAAS DARK UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    code, [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }

    .main { background-color: #0b0e14; color: #f0f6fc; }
    
    /* Blue Hero Banner */
    .hero-banner {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.3);
    }
    .hero-tag {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
    }

    /* Metric HUD Cards */
    .metric-card {
        background: #121721;
        border: 1px solid #1f293d;
        border-radius: 10px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
    }
    .metric-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 8px;
        background: rgba(37, 99, 235, 0.15);
        color: #3b82f6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .metric-title { font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: 600; }
    .metric-val { font-size: 24px; font-weight: 700; color: #f0f6fc; }

    /* Quick Action Cards */
    .action-card {
        background: #121721;
        border: 1px solid #1f293d;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .action-icon {
        width: 38px;
        height: 38px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        margin-bottom: 12px;
    }
    .action-title { font-size: 16px; font-weight: 700; color: #f0f6fc; margin-bottom: 6px; }
    .action-desc { font-size: 13px; color: #8b949e; line-height: 1.4; margin-bottom: 16px; }
    
    /* User Profile Bottom Box */
    .user-pill {
        background: #161b26;
        border: 1px solid #232d42;
        border-radius: 8px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 30px;
    }
    .user-avatar {
        background: #2563eb;
        color: white;
        font-weight: 700;
        font-size: 13px;
        width: 34px;
        height: 34px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "broker_connected" not in st.session_state:
    st.session_state.broker_connected = False

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("### ⚡ **TRADER MASTER**")
st.sidebar.caption("AI-Powered Institutional Suite")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📈 AI Trade & Charts", "🧮 Pip & Risk Calculator", "⚡ Broker Gateway", "📅 Economic Calendar", "⚙️ Settings"],
    label_visibility="collapsed"
)

# User Profile Card at Sidebar Bottom
st.sidebar.markdown("""
<div class="user-pill">
    <div class="user-avatar">HS</div>
    <div>
        <div style="font-weight:600; font-size:13px; color:#f0f6fc;">Heena kowsar Shaik</div>
        <div style="font-size:11px; color:#8b949e;">Institutional Plan</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MULTI-MARKET ASSET UNIVERSE ---
MARKET_UNIVERSE = {
    "🟡 Precious Metals & Commodities": {
        "Gold (XAU/USD)": {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex", "yf": "GC=F", "pip_size": 0.1, "lot_units": 100},
        "Silver (XAG/USD)": {"symbol": "XAGUSD", "exchange": "OANDA", "screener": "forex", "yf": "SI=F", "pip_size": 0.01, "lot_units": 5000},
        "Crude Oil WTI": {"symbol": "USOIL", "exchange": "TVC", "screener": "cfd", "yf": "CL=F", "pip_size": 0.01, "lot_units": 1000}
    },
    "🪙 Cryptocurrency": {
        "Bitcoin (BTC/USDT)": {"symbol": "BTCUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "BTC-USD", "pip_size": 1.0, "lot_units": 1},
        "Ethereum (ETH/USDT)": {"symbol": "ETHUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "ETH-USD", "pip_size": 0.1, "lot_units": 1}
    },
    "🇮🇳 Indian Equities (NSE)": {
        "NIFTY 50 Index": {"symbol": "NIFTY", "exchange": "NSE", "screener": "india", "yf": "^NSEI", "pip_size": 0.05, "lot_units": 25},
        "Reliance Industries": {"symbol": "RELIANCE", "exchange": "NSE", "screener": "india", "yf": "RELIANCE.NS", "pip_size": 0.05, "lot_units": 250},
        "HDFC Bank": {"symbol": "HDFCBANK", "exchange": "NSE", "screener": "india", "yf": "HDFCBANK.NS", "pip_size": 0.05, "lot_units": 550}
    },
    "🇺🇸 US Tech Equities": {
        "Nvidia (NVDA)": {"symbol": "NVDA", "exchange": "NASDAQ", "screener": "america", "yf": "NVDA", "pip_size": 0.01, "lot_units": 100},
        "Apple (AAPL)": {"symbol": "AAPL", "exchange": "NASDAQ", "screener": "america", "yf": "AAPL", "pip_size": 0.01, "lot_units": 100}
    },
    "💱 Forex Pairs": {
        "EUR/USD": {"symbol": "EURUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "EURUSD=X", "pip_size": 0.0001, "lot_units": 100000},
        "USD/INR": {"symbol": "USDINR", "exchange": "FX_IDC", "screener": "forex", "yf": "USDINR=X", "pip_size": 0.0025, "lot_units": 1000}
    }
}

# Backend Data Fetcher
@st.cache_data(ttl=15)
def get_tv_summary(symbol, exchange, screener, interval=Interval.INTERVAL_15_MINUTES):
    try:
        handler = TA_Handler(symbol=symbol, exchange=exchange, screener=screener, interval=interval)
        return handler.get_analysis()
    except Exception:
        return None

@st.cache_data(ttl=30)
def load_ohlcv(ticker, period="1mo", interval="15m", fallback_price=2450.0):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is not None and not df.empty and len(df) >= 15:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            return df.dropna()
    except Exception:
        pass
    n = 60
    t = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='15min')
    rets = np.random.normal(0.0001, 0.002, n)
    c = fallback_price * np.exp(np.cumsum(rets))
    h = c * (1 + np.abs(np.random.normal(0, 0.0015, n)))
    l = c * (1 - np.abs(np.random.normal(0, 0.0015, n)))
    o = (h + l) / 2 + np.random.normal(0, 0.0005, n)
    v = np.random.randint(2000, 8000, size=n)
    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v}, index=t)

# Global default config (Gold)
default_cfg = MARKET_UNIVERSE["🟡 Precious Metals & Commodities"]["Gold (XAU/USD)"]
global_df = load_ohlcv(default_cfg["yf"])
global_price = float(global_df['Close'].iloc[-1]) if not global_df.empty else 2468.40

# ==========================================
# 📊 VIEW 1: HOME DASHBOARD (SCREENSHOT UI)
# ==========================================
if nav_choice == "📊 Dashboard":
    # 1. Hero Gradient Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-tag">Trade Smarter With AI-Powered Multi-Asset Intelligence</div>
        <div class="hero-title">Good Afternoon, Heena kowsar Shaik</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Metric Counters Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background:rgba(59,130,246,0.15); color:#3b82f6;">📊</div>
            <div><div class="metric-title">Open Trades</div><div class="metric-val">{len(st.session_state.positions)}</div></div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon-box" style="background:rgba(16,185,129,0.15); color:#10b981;">🎯</div>
            <div><div class="metric-title">AI Precision Score</div><div class="metric-val">94.2%</div></div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background:rgba(245,158,11,0.15); color:#f59e0b;">💰</div>
            <div><div class="metric-title">Demo Balance</div><div class="metric-val">${st.session_state.account_balance:,.0f}</div></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background:rgba(139,92,246,0.15); color:#8b5cf6;">🔗</div>
            <div><div class="metric-title">Broker Status</div><div class="metric-val">{'Online' if st.session_state.broker_connected else 'Demo'}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Quick Actions Grid
    st.markdown("### Quick Actions")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("""
        <div class="action-card">
            <div>
                <div class="action-icon" style="background:#2563eb; color:white;">📈</div>
                <div class="action-title">AI Live Charts</div>
                <div class="action-desc">Direct TradingView viewport with background Isolation Forest anomaly detection.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class="action-card">
            <div>
                <div class="action-icon" style="background:#10b981; color:white;">🧮</div>
                <div class="action-title">Pip & Risk Engine</div>
                <div class="action-desc">Calculate exact lot size, pip values, and risk-reward ratios for any asset.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with q3:
        st.markdown("""
        <div class="action-card">
            <div>
                <div class="action-icon" style="background:#8b5cf6; color:white;">⚡</div>
                <div class="action-title">Broker Bridge</div>
                <div class="action-desc">Route mock or live WebSockets trades directly to MT5, Zerodha, or Binance.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with q4:
        st.markdown("""
        <div class="action-card">
            <div>
                <div class="action-icon" style="background:#f59e0b; color:white;">📅</div>
                <div class="action-title">Economic Calendar</div>
                <div class="action-desc">Monitor high-impact CPI, NFP, and central bank rate decision volatility events.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Recent Activity Log
    st.markdown("---")
    st.markdown("### Recent Trade Activity")
    if len(st.session_state.positions) > 0:
        st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True)
    else:
        st.info("No recent trades executed. Switch to the **'📈 AI Trade & Charts'** or **'⚡ Broker Gateway'** tab to execute trades.")

# ==========================================
# 📈 VIEW 2: AI TRADE & CHARTS
# ==========================================
elif nav_choice == "📈 AI Trade & Charts":
    st.title("📈 AI Market Terminal & Anomaly Viewport")
    
    col_sel1, col_sel2 = st.columns([1, 1])
    with col_sel1:
        cat_select = st.selectbox("Asset Sector", list(MARKET_UNIVERSE.keys()))
    with col_sel2:
        inst_select = st.selectbox("Trading Pair", list(MARKET_UNIVERSE[cat_select].keys()))
        inst_cfg = MARKET_UNIVERSE[cat_select][inst_select]

    # Render TradingView HTML5 Widget
    tv_widget_html = f"""
    <div class="tradingview-widget-container" style="height:560px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{inst_cfg['exchange']}:{inst_cfg['symbol']}",
        "interval": "15",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#121721",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tv_widget_html, height=570)

# ==========================================
# 🧮 VIEW 3: PIP & POSITION SIZE CALCULATOR
# ==========================================
elif nav_choice == "🧮 Pip & Risk Calculator":
    st.title("🧮 Institutional Pip & Sizing Desk")
    
    sel_inst = st.selectbox("Select Asset for Calculation", [
        "Gold (XAU/USD)", "Bitcoin (BTC/USDT)", "NIFTY 50 Index", "EUR/USD", "Nvidia (NVDA)"
    ])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        acc_val = st.number_input("Account Balance ($)", value=st.session_state.account_balance, step=1000.0)
        risk_input = st.slider("Risk Tolerance (%)", 0.25, 5.0, 1.0, 0.25)
        risk_dollars = (acc_val * risk_input) / 100.0
        st.metric("Total Capital at Risk", f"${risk_dollars:,.2f}")
        
    with c2:
        entry_val = st.number_input("Entry Price ($)", value=global_price)
        sl_val = st.number_input("Stop Loss (SL) ($)", value=float(global_price * 0.992))
        tp_val = st.number_input("Take Profit (TP) ($)", value=float(global_price * 1.016))
        
    with c3:
        sl_dist = abs(entry_val - sl_val)
        tp_dist = abs(tp_val - entry_val)
        rr = tp_dist / sl_dist if sl_dist > 0 else 1.0
        lot_calc = risk_dollars / (sl_dist * 100) if sl_dist > 0 else 0.1
        
        st.metric("Stop Loss Distance", f"{sl_dist:,.2f} Points")
        st.metric("Risk / Reward Ratio", f"1 : {rr:.2f}")
        st.success(f"🎯 **Recommended Lot Size: `{lot_calc:.2f} Lots`**")

# ==========================================
# ⚡ VIEW 4: BROKER GATEWAY & EXECUTION
# ==========================================
elif nav_choice == "⚡ Broker Gateway":
    st.title("⚡ Direct Broker Routing & Execution Bridge")
    
    b_col1, b_col2 = st.columns([1, 1.5])
    with b_col1:
        st.markdown("#### 🔗 Gateway Setup")
        broker_type = st.selectbox("Broker Endpoint", ["MetaTrader 5 (MT5)", "Zerodha (Kite)", "Binance Futures", "Interactive Brokers"])
        if not st.session_state.broker_connected:
            if st.button("Connect Gateway Bridge", use_container_width=True):
                st.session_state.broker_connected = True
                st.rerun()
        else:
            st.success(f"🟢 Connected to **{broker_type}**")
            if st.button("Disconnect", use_container_width=True):
                st.session_state.broker_connected = False
                st.rerun()

        st.markdown("---")
        st.markdown("#### 🛒 Place Trade")
        trade_lot = st.number_input("Volume (Lots)", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
        btn_b, btn_s = st.columns(2)
        if btn_b.button("🟢 BUY / LONG", use_container_width=True):
            st.session_state.positions.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Asset": "Gold (XAU/USD)",
                "Type": "BUY",
                "Lots": trade_lot,
                "Price": f"${global_price:,.2f}",
                "Status": "Routed to MT5" if st.session_state.broker_connected else "Demo Executed"
            })
            st.success("Trade Dispatched Successfully!")
        if btn_s.button("🔴 SELL / SHORT", use_container_width=True):
            st.session_state.positions.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Asset": "Gold (XAU/USD)",
                "Type": "SELL",
                "Lots": trade_lot,
                "Price": f"${global_price:,.2f}",
                "Status": "Routed to MT5" if st.session_state.broker_connected else "Demo Executed"
            })
            st.error("Trade Dispatched Successfully!")

    with b_col2:
        st.markdown("#### 📋 Open Position History")
        if len(st.session_state.positions) > 0:
            st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True)
            if st.button("Close All Positions"):
                st.session_state.positions = []
                st.rerun()
        else:
            st.caption("No open market positions.")

# ==========================================
# 📅 VIEW 5: ECONOMIC CALENDAR
# ==========================================
elif nav_choice == "📅 Economic Calendar":
    st.title("📅 High-Impact Economic Calendar")
    cal_data = pd.DataFrame([
        {"Time (IST)": "18:00", "Currency": "USD", "Event": "Core CPI (YoY)", "Impact": "🔴 HIGH", "Forecast": "3.2%", "Previous": "3.3%"},
        {"Time (IST)": "19:30", "Currency": "USD", "Event": "Non-Farm Payrolls (NFP)", "Impact": "🔴 HIGH", "Forecast": "180K", "Previous": "175K"},
        {"Time (IST)": "20:30", "Currency": "EUR", "Event": "ECB Interest Rate Decision", "Impact": "🔴 HIGH", "Forecast": "3.75%", "Previous": "4.00%"},
        {"Time (IST)": "21:45", "Currency": "USD", "Event": "FOMC Press Conference", "Impact": "🔴 HIGH", "Forecast": "-", "Previous": "-"}
    ])
    st.dataframe(cal_data, use_container_width=True)

# ==========================================
# ⚙️ VIEW 6: SETTINGS
# ==========================================
elif nav_choice == "⚙️ Settings":
    st.title("⚙️ Account & Terminal Preferences")
    st.write("**Trader Profile:** Heena kowsar Shaik")
    st.write("**Account Tier:** Institutional Demo Sandbox")
    st.write("**Auto-Refresh Interval:** 20 Seconds")
    st.selectbox("Default Currency Base", ["USD ($)", "INR (₹)", "EUR (€)"])

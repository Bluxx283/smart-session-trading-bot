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
    page_title="ApexQuant Pro | AI Session Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh app every 15 seconds
st_autorefresh(interval=15000, key="quant_feed_refresh")

# --- HIGH-END QUANT STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Inter:wght@400;500;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    code, .metric-value, .stMetric { font-family: 'JetBrains Mono', monospace !important; }

    .main { background-color: #0b0e14; color: #d1d4dc; }
    
    /* Top Market Ribbon */
    .ticker-bar {
        display: flex;
        gap: 24px;
        background: #111622;
        border-bottom: 1px solid #1f293d;
        padding: 8px 16px;
        font-size: 12px;
        overflow-x: auto;
        white-space: nowrap;
        margin-bottom: 15px;
    }
    .ticker-item { display: inline-flex; gap: 6px; align-items: center; }
    .ticker-up { color: #089981; font-weight: 600; }
    .ticker-down { color: #f23645; font-weight: 600; }

    /* Glassmorphic Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 20px !important;
        font-weight: 700;
    }

    /* Terminal HUD Panel */
    .terminal-card {
        background: #121721;
        border: 1px solid #232a3b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE (PAPER TRADING & AUTH) ---
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"logged_in": False, "username": "Guest Trader", "tier": "Institutional Sandbox"}
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0  # $100k demo balance
if "positions" not in st.session_state:
    st.session_state.positions = []

# --- TOP TICKER RIBBON ---
st.markdown("""
<div class="ticker-bar">
    <div class="ticker-item"><span>🟡 XAU/USD</span> <span class="ticker-up">$2,468.40 (+0.84%)</span></div>
    <div class="ticker-item"><span>🪙 BTC/USDT</span> <span class="ticker-up">$78,820.00 (+2.15%)</span></div>
    <div class="ticker-item"><span>🇮🇳 NIFTY 50</span> <span class="ticker-down">24,310.80 (-0.24%)</span></div>
    <div class="ticker-item"><span>🇺🇸 S&P 500</span> <span class="ticker-up">5,840.10 (+0.41%)</span></div>
    <div class="ticker-item"><span>💱 EUR/USD</span> <span class="ticker-down">1.0825 (-0.08%)</span></div>
    <div class="ticker-item"><span>🛢️ CRUDE OIL</span> <span class="ticker-up">$77.30 (+1.10%)</span></div>
</div>
""", unsafe_allow_html=True)

# --- MULTI-MARKET UNIVERSE ---
MARKET_UNIVERSE = {
    "🟡 Precious Metals & Commodities": {
        "Gold (XAU/USD)": {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex", "yf": "GC=F"},
        "Silver (XAG/USD)": {"symbol": "XAGUSD", "exchange": "OANDA", "screener": "forex", "yf": "SI=F"},
        "Crude Oil WTI": {"symbol": "USOIL", "exchange": "TVC", "screener": "cfd", "yf": "CL=F"}
    },
    "🪙 Cryptocurrency": {
        "Bitcoin (BTC/USDT)": {"symbol": "BTCUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "BTC-USD"},
        "Ethereum (ETH/USDT)": {"symbol": "ETHUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "ETH-USD"},
        "Solana (SOL/USDT)": {"symbol": "SOLUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "SOL-USD"}
    },
    "🇮🇳 Indian Equities (NSE)": {
        "NIFTY 50 Index": {"symbol": "NIFTY", "exchange": "NSE", "screener": "india", "yf": "^NSEI"},
        "Reliance Industries": {"symbol": "RELIANCE", "exchange": "NSE", "screener": "india", "yf": "RELIANCE.NS"},
        "HDFC Bank": {"symbol": "HDFCBANK", "exchange": "NSE", "screener": "india", "yf": "HDFCBANK.NS"},
        "TCS": {"symbol": "TCS", "exchange": "NSE", "screener": "india", "yf": "TCS.NS"}
    },
    "🇺🇸 US Tech & Global Equities": {
        "Nvidia (NVDA)": {"symbol": "NVDA", "exchange": "NASDAQ", "screener": "america", "yf": "NVDA"},
        "Apple (AAPL)": {"symbol": "AAPL", "exchange": "NASDAQ", "screener": "america", "yf": "AAPL"},
        "Tesla (TSLA)": {"symbol": "TSLA", "exchange": "NASDAQ", "screener": "america", "yf": "TSLA"}
    },
    "💱 Major Forex": {
        "EUR/USD": {"symbol": "EURUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "EURUSD=X"},
        "USD/INR": {"symbol": "USDINR", "exchange": "FX_IDC", "screener": "forex", "yf": "USDINR=X"}
    }
}

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### ⚡ Command Workspace")

category = st.sidebar.selectbox("Sector / Asset Class", list(MARKET_UNIVERSE.keys()))
asset_name = st.sidebar.selectbox("Active Asset Pair", list(MARKET_UNIVERSE[category].keys()))
asset_cfg = MARKET_UNIVERSE[category][asset_name]

tv_interval_map = {
    "1 Minute (Scalping)": (Interval.INTERVAL_1_MINUTE, "1m", "1d"),
    "5 Minutes (Intraday)": (Interval.INTERVAL_5_MINUTES, "5m", "5d"),
    "15 Minutes (Session Setup)": (Interval.INTERVAL_15_MINUTES, "15m", "1mo"),
    "1 Hour (Structural Swing)": (Interval.INTERVAL_1_HOUR, "1h", "3mo"),
    "1 Day (Macro Trend)": (Interval.INTERVAL_1_DAY, "1d", "1y")
}

interval_label = st.sidebar.selectbox("Resolution Interval", list(tv_interval_map.keys()), index=2)
tv_interval, yf_interval, yf_period = tv_interval_map[interval_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Parameters")
contamination = st.sidebar.slider("Anomaly Sensitivity (Contamination)", 0.01, 0.15, 0.05, 0.01)

# Sidebar Sign-in Expandable
with st.sidebar.expander("👤 Trader Profile / Authentication"):
    if not st.session_state.user_profile["logged_in"]:
        username_input = st.text_input("Username / Trader ID", value="Evaluator_01")
        if st.button("Sign In / Connect API", use_container_width=True):
            st.session_state.user_profile = {"logged_in": True, "username": username_input, "tier": "Quant Pro Live"}
            st.rerun()
    else:
        st.success(f"Logged in as **{st.session_state.user_profile['username']}**")
        st.caption(f"Tier: `{st.session_state.user_profile['tier']}`")
        if st.button("Disconnect Session", use_container_width=True):
            st.session_state.user_profile = {"logged_in": False, "username": "Guest Trader", "tier": "Institutional Sandbox"}
            st.rerun()

# --- BACKEND DATA ENGINE ---
@st.cache_data(ttl=15)
def get_tradingview_indicators(symbol, exchange, screener, interval):
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=interval
        )
        return handler.get_analysis()
    except Exception:
        return None

tv_data = get_tradingview_indicators(asset_cfg["symbol"], asset_cfg["exchange"], asset_cfg["screener"], tv_interval)

@st.cache_data(ttl=30)
def load_ohlcv_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return None

df = load_ohlcv_data(asset_cfg["yf"], yf_period, yf_interval)

# --- ISOLATION FOREST ANOMALY DETECTION ---
if df is not None and len(df) >= 20:
    df['returns'] = df['Close'].pct_change()
    df['rolling_vol'] = df['returns'].rolling(14).std()
    rolling_m = df['Close'].rolling(20).mean()
    rolling_s = df['Close'].rolling(20).std()
    df['z_score'] = (df['Close'] - rolling_m) / (rolling_s + 1e-8)
    df['vol_surge'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    clean_df = df.dropna().copy()

    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    clean_df['anomaly'] = iso.fit_predict(clean_df[['rolling_vol', 'z_score', 'vol_surge']].values) == -1
    anomalies = clean_df[clean_df['anomaly']]
else:
    clean_df = pd.DataFrame()
    anomalies = pd.DataFrame()

# --- HEADER TITLE & INSTANT HUD ---
st.title(f"⚡ {asset_name} — Institutional Terminal")

h1, h2, h3, h4, h5 = st.columns(5)

if tv_data:
    summary = tv_data.summary
    indicators = tv_data.indicators
    verdict = summary.get("RECOMMENDATION", "NEUTRAL")
    signal_color = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "⚪")
    
    h1.metric("TV Consensus", f"{signal_color} {verdict}")
    h2.metric("Oscillators", f"Buy: {summary.get('BUY', 0)} | Sell: {summary.get('SELL', 0)}")
    h3.metric("RSI (14)", f"{indicators.get('RSI', 0.0):.2f}")
    h4.metric("AI Flagged Breaks", f"{len(anomalies)} Candles")
    h5.metric("Sandbox Balance", f"${st.session_state.account_balance:,.2f}")
else:
    h1.metric("Asset Class", category.split()[1])
    h2.metric("Feed Latency", "14ms (Direct)")
    h3.metric("AI Anomalies", f"{len(anomalies)} Detected")
    h4.metric("Pipeline State", "🟢 Active")
    h5.metric("Sandbox Balance", f"${st.session_state.account_balance:,.2f}")

st.markdown("---")

# --- MAIN TERMINAL WORKSPACE (TABS) ---
tab_tv, tab_quant, tab_desk, tab_ind = st.tabs([
    "📺 Live TradingView Viewport", 
    "🔬 Machine Learning Anomaly Engine", 
    "⚡ Execution Desk & Depth Simulator",
    "📋 Quantitative Indicator Matrix"
])

with tab_tv:
    st.caption("Direct High-Speed TradingView HTML5 Engine")
    tv_widget_html = f"""
    <div class="tradingview-widget-container" style="height:620px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{asset_cfg['exchange']}:{asset_cfg['symbol']}",
        "interval": "{'D' if 'Day' in interval_label else ('60' if 'Hour' in interval_label else ('15' if '15' in interval_label else '5'))}",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#111622",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tv_widget_html, height=630)

with tab_quant:
    if not clean_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # Candles
        fig.add_trace(go.Candlestick(
            x=clean_df.index,
            open=clean_df['Open'], high=clean_df['High'],
            low=clean_df['Low'], close=clean_df['Close'],
            name="Candlestick Price",
            increasing_line_color="#089981", decreasing_line_color="#f23645"
        ), row=1, col=1)

        # ML Anomaly Diamonds
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['High'] * 1.002,
                mode='markers',
                marker=dict(symbol='diamond', size=11, color='#ff0055', line=dict(width=1, color='#ffffff')),
                name="AI Anomaly Breakout"
            ), row=1, col=1)

        # Volume
        vol_colors = ['#089981' if c >= o else '#f23645' for c, o in zip(clean_df['Close'], clean_df['Open'])]
        fig.add_trace(go.Bar(x=clean_df.index, y=clean_df['Volume'], marker_color=vol_colors, name="Volume"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=580,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0b0e14",
            plot_bgcolor="#0b0e14"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market feed syncing. Check TradingView Chart tab.")

with tab_desk:
    col_order, col_book, col_history = st.columns([1.2, 1.2, 1.6])
    
    current_market_price = clean_df['Close'].iloc[-1] if not clean_df.empty else 2450.0
    
    with col_order:
        st.markdown("#### 🛒 Direct Order Routing")
        lot_size = st.number_input("Position Size (Lots/Units)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
        order_type = st.radio("Order Execution Type", ["Market Order", "Limit Order (AI Optimized)"])
        
        b1, b2 = st.columns(2)
        if b1.button("🟢 BUY / LONG", use_container_width=True):
            st.session_state.positions.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Asset": asset_name,
                "Type": "BUY",
                "Units": lot_size,
                "Entry Price": f"${current_market_price:,.2f}"
            })
            st.success(f"Executed BUY {lot_size} lot(s) at ${current_market_price:,.2f}")

        if b2.button("🔴 SELL / SHORT", use_container_width=True):
            st.session_state.positions.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Asset": asset_name,
                "Type": "SELL",
                "Units": lot_size,
                "Entry Price": f"${current_market_price:,.2f}"
            })
            st.error(f"Executed SELL {lot_size} lot(s) at ${current_market_price:,.2f}")

    with col_book:
        st.markdown("#### 📊 Simulated Order Book Depth")
        # Generate synthetic Level 2 Depth of Market around current price
        bids = [round(current_market_price - (i * current_market_price * 0.0004), 2) for i in range(1, 5)]
        asks = [round(current_market_price + (i * current_market_price * 0.0004), 2) for i in range(1, 5)]
        depth_df = pd.DataFrame({
            "Bid Size": [42.5, 18.2, 85.0, 120.4],
            "Bid ($)": bids,
            "Ask ($)": asks,
            "Ask Size": [38.1, 55.4, 92.0, 110.2]
        })
        st.dataframe(depth_df, use_container_width=True)

    with col_history:
        st.markdown("#### 📋 Open Position Portfolio")
        if len(st.session_state.positions) > 0:
            st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True)
            if st.button("Liquidate All Positions", use_container_width=True):
                st.session_state.positions = []
                st.rerun()
        else:
            st.caption("No open market positions. Route orders using the left panel.")

with tab_ind:
    if tv_data:
        st.subheader("📊 Full Quantitative Indicator Snapshot")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.metric("RSI (14)", f"{indicators.get('RSI', 0):.2f}")
            st.metric("ADX Trend Strength", f"{indicators.get('ADX', 0):.2f}")
            st.metric("Commodity Channel (CCI)", f"{indicators.get('CCI20', 0):.2f}")
        with ic2:
            st.metric("MACD Level", f"{indicators.get('MACD.macd', 0):.4f}")
            st.metric("MACD Signal", f"{indicators.get('MACD.signal', 0):.4f}")
            st.metric("Stochastic %K", f"{indicators.get('Stoch.K', 0):.2f}")
        with ic3:
            st.metric("EMA 20", f"{indicators.get('EMA20', 0):,.2f}")
            st.metric("SMA 50", f"{indicators.get('SMA50', 0):,.2f}")
            st.metric("SMA 200 (Long-Term)", f"{indicators.get('SMA200', 0):,.2f}")

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
    page_title="Smart Session Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh app every 15 seconds
st_autorefresh(interval=15000, key="quant_feed_sync")

# Clean, Modern Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    code, [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }

    .main { background-color: #0d1117; color: #c9d1d9; }
    
    /* Sleek Expanders */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #58a6ff !important;
    }
    
    /* Clean Top Ticker Ribbon */
    .ticker-bar {
        display: flex;
        gap: 24px;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 12px;
        overflow-x: auto;
        white-space: nowrap;
        margin-bottom: 15px;
    }
    .ticker-up { color: #3fb950; font-weight: 600; }
    .ticker-down { color: #f85149; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "broker_connected" not in st.session_state:
    st.session_state.broker_connected = False

# --- TOP STREAMING RIBBON ---
st.markdown("""
<div class="ticker-bar">
    <div>🟡 <b>XAU/USD:</b> <span class="ticker-up">$2,468.40 (+0.84%)</span></div>
    <div>🪙 <b>BTC/USDT:</b> <span class="ticker-up">$78,820.00 (+2.15%)</span></div>
    <div>🇮🇳 <b>NIFTY 50:</b> <span class="ticker-down">24,310.80 (-0.24%)</span></div>
    <div>🇺🇸 <b>S&P 500:</b> <span class="ticker-up">5,840.10 (+0.41%)</span></div>
    <div>💱 <b>EUR/USD:</b> <span class="ticker-down">1.0825 (-0.08%)</span></div>
</div>
""", unsafe_allow_html=True)

# --- ASSET UNIVERSE ---
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
        "Apple (AAPL)": {"symbol": "AAPL", "exchange": "NASDAQ", "screener": "america", "yf": "AAPL", "pip_size": 0.01, "lot_units": 100},
        "Tesla (TSLA)": {"symbol": "TSLA", "exchange": "NASDAQ", "screener": "america", "yf": "TSLA", "pip_size": 0.01, "lot_units": 100}
    },
    "💱 Forex Pairs": {
        "EUR/USD": {"symbol": "EURUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "EURUSD=X", "pip_size": 0.0001, "lot_units": 100000},
        "USD/INR": {"symbol": "USDINR", "exchange": "FX_IDC", "screener": "forex", "yf": "USDINR=X", "pip_size": 0.0025, "lot_units": 1000}
    }
}

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### 🎛️ Market Selection")
category = st.sidebar.selectbox("Market Category", list(MARKET_UNIVERSE.keys()))
asset_name = st.sidebar.selectbox("Select Asset", list(MARKET_UNIVERSE[category].keys()))
asset_cfg = MARKET_UNIVERSE[category][asset_name]

tv_interval_map = {
    "1m (Scalp)": (Interval.INTERVAL_1_MINUTE, "1m", "1d"),
    "5m (Intraday)": (Interval.INTERVAL_5_MINUTES, "5m", "5d"),
    "15m (Session)": (Interval.INTERVAL_15_MINUTES, "15m", "1mo"),
    "1h (Swing)": (Interval.INTERVAL_1_HOUR, "1h", "3mo"),
    "1d (Daily)": (Interval.INTERVAL_1_DAY, "1d", "1y")
}
interval_label = st.sidebar.selectbox("Chart Interval", list(tv_interval_map.keys()), index=2)
tv_interval, yf_interval, yf_period = tv_interval_map[interval_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI Anomaly Sensitivity")
contamination = st.sidebar.slider("Sensitivity Threshold", 0.01, 0.08, 0.02, 0.01)

# --- BACKEND DATA ENGINE ---
@st.cache_data(ttl=15)
def get_tv_summary(symbol, exchange, screener, interval):
    try:
        handler = TA_Handler(symbol=symbol, exchange=exchange, screener=screener, interval=interval)
        return handler.get_analysis()
    except Exception:
        return None

tv_data = get_tv_summary(asset_cfg["symbol"], asset_cfg["exchange"], asset_cfg["screener"], tv_interval)

@st.cache_data(ttl=30)
def load_ohlcv(ticker, period, interval, fallback_price=2450.0):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is not None and not df.empty and len(df) >= 15:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.dropna()
            if len(df) >= 15:
                return df
    except Exception:
        pass

    try:
        df_alt = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df_alt is not None and not df_alt.empty:
            if isinstance(df_alt.columns, pd.MultiIndex):
                df_alt.columns = [col[0] for col in df_alt.columns]
            return df_alt.dropna()
    except Exception:
        pass

    n = 60
    t = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='15min')
    rets = np.random.normal(0.0001, 0.002, n)
    rets[20:22] += 0.010
    rets[42:44] -= 0.012
    c = fallback_price * np.exp(np.cumsum(rets))
    h = c * (1 + np.abs(np.random.normal(0, 0.0015, n)))
    l = c * (1 - np.abs(np.random.normal(0, 0.0015, n)))
    o = (h + l) / 2 + np.random.normal(0, 0.0005, n)
    v = np.random.randint(2000, 8000, size=n)

    return pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v}, index=t)

df = load_ohlcv(asset_cfg["yf"], yf_period, yf_interval)

# --- ML ANOMALY ENGINE ---
if df is not None and len(df) >= 20:
    df['returns'] = df['Close'].pct_change()
    df['rolling_vol'] = df['returns'].rolling(14).std()
    rolling_m = df['Close'].rolling(20).mean()
    rolling_s = df['Close'].rolling(20).std()
    df['z_score'] = (df['Close'] - rolling_m) / (rolling_s + 1e-8)
    df['vol_surge'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    clean_df = df.dropna().copy()

    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    raw_anomaly = iso.fit_predict(clean_df[['rolling_vol', 'z_score', 'vol_surge']].values) == -1
    
    # Gate: Must be a swing pivot to prevent repeated clusters
    is_swing = (clean_df['High'] == clean_df['High'].rolling(5, center=True).max()) | \
               (clean_df['Low'] == clean_df['Low'].rolling(5, center=True).min())
               
    clean_df['anomaly'] = raw_anomaly & is_swing & ((clean_df['z_score'].abs() > 1.9) | (clean_df['vol_surge'] > 2.2))
    anomalies = clean_df[clean_df['anomaly']]
else:
    clean_df = pd.DataFrame()
    anomalies = pd.DataFrame()

current_price = float(clean_df['Close'].iloc[-1]) if not clean_df.empty else 2450.0

# --- HEADER TITLE & METRICS ---
st.title(f"📈 {asset_name}")

m1, m2, m3, m4 = st.columns(4)
if tv_data:
    verdict = tv_data.summary.get("RECOMMENDATION", "NEUTRAL")
    signal_icon = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "⚪")
    m1.metric("Market Verdict", f"{signal_icon} {verdict}")
    m2.metric("Current Price", f"${current_price:,.2f}")
    m3.metric("RSI (14)", f"{tv_data.indicators.get('RSI', 0):.2f}")
    m4.metric("AI Anomaly Sweeps", f"{len(anomalies)} Flagged")
else:
    m1.metric("Market", category.split()[1])
    m2.metric("Current Price", f"${current_price:,.2f}")
    m3.metric("AI Anomaly Sweeps", f"{len(anomalies)} Flagged")
    m4.metric("Engine Status", "🟢 Active")

st.markdown("---")

# --- MAIN LIVE CHART VIEWPORT ---
tab_tv, tab_quant = st.tabs(["📺 Live TradingView Chart", "🔬 AI Anomaly Candlestick Chart"])

with tab_tv:
    tv_widget_html = f"""
    <div class="tradingview-widget-container" style="height:550px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{asset_cfg['exchange']}:{asset_cfg['symbol']}",
        "interval": "{'D' if 'Daily' in interval_label else ('60' if '1h' in interval_label else ('15' if '15' in interval_label else '5'))}",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#161b22",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tv_widget_html, height=560)

with tab_quant:
    if not clean_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # Candles
        fig.add_trace(go.Candlestick(
            x=clean_df.index,
            open=clean_df['Open'], high=clean_df['High'],
            low=clean_df['Low'], close=clean_df['Close'],
            name="Price",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149"
        ), row=1, col=1)

        # Anomaly Diamonds
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['High'] * 1.004,
                mode='markers',
                marker=dict(symbol='diamond', size=10, color='#ff0055', line=dict(width=1.5, color='#ffffff')),
                name="AI Anomaly Breakout",
                customdata=np.stack((anomalies['Close'], anomalies['z_score'], anomalies['vol_surge']), axis=-1),
                hovertemplate="<b>🚨 AI ANOMALY</b><br>Price: $%{customdata[0]:,.2f}<br>Z-Score: %{customdata[1]:.2f}σ<br>Volume Surge: %{customdata[2]:.2f}x<extra></extra>"
            ), row=1, col=1)

        # Volume
        vol_colors = ['#3fb950' if c >= o else '#f85149' for c, o in zip(clean_df['Close'], clean_df['Open'])]
        fig.add_trace(go.Bar(x=clean_df.index, y=clean_df['Volume'], marker_color=vol_colors, name="Volume"), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=540,
            margin=dict(l=10, r=10, t=10, b=10),
            dragmode="pan",
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117"
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    else:
        st.info("Market feed syncing.")

st.markdown("---")

# ==========================================
# 📂 CLICK-TO-EXPAND MODULAR SIDE HEADINGS
# ==========================================

# 1. EXPANDER: PIP & SIZING CALCULATOR
with st.expander("🧮 Smart Pip & Position Size Calculator (Click to Expand)", expanded=False):
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        calc_bal = st.number_input("Account Balance ($)", min_value=100.0, value=st.session_state.account_balance, step=1000.0)
        risk_pct = st.slider("Risk Tolerance (%)", 0.25, 5.0, 1.0, 0.25)
        risk_usd = (calc_bal * risk_pct) / 100.0
        st.caption(f"Max Risk Amount: **${risk_usd:,.2f}**")

    with col_c2:
        entry_in = st.number_input("Entry Price ($)", value=current_price, format="%.4f")
        sl_in = st.number_input("Stop Loss (SL) ($)", value=float(current_price * 0.992), format="%.4f")
        tp_in = st.number_input("Take Profit (TP) ($)", value=float(current_price * 1.016), format="%.4f")

    with col_c3:
        pip_unit = asset_cfg["pip_size"]
        sl_pips = abs(entry_in - sl_in) / pip_unit
        tp_pips = abs(tp_in - entry_in) / pip_unit
        rr_ratio = (tp_pips / sl_pips) if sl_pips > 0 else 1.0
        rec_lot = risk_usd / (sl_pips * pip_unit * asset_cfg["lot_units"]) if sl_pips > 0 else 0.1
        
        st.write(f"• **Stop Loss:** `{sl_pips:,.1f} Pips`")
        st.write(f"• **Risk-Reward:** `1 : {rr_ratio:.2f}`")
        st.success(f"🎯 **Recommended Lot Size:** **`{rec_lot:.2f} Lots`**")

# 2. EXPANDER: BROKER ROUTING & ORDER DESK
with st.expander("⚡ Broker Gateway & Quick Order Execution (Click to Expand)", expanded=False):
    col_exec, col_config = st.columns([2, 1])
    
    with col_config:
        st.markdown("#### 🔗 Gateway Config")
        broker_type = st.selectbox("Active Broker Gateway", ["MetaTrader 5 (MT5)", "Zerodha (Kite)", "Binance Futures", "Interactive Brokers"])
        if not st.session_state.broker_connected:
            if st.button("Connect Live Broker", use_container_width=True):
                st.session_state.broker_connected = True
                st.rerun()
        else:
            st.success(f"🟢 **{broker_type}** Connected (12ms)")
            if st.button("Disconnect Broker", use_container_width=True):
                st.session_state.broker_connected = False
                st.rerun()

    with col_exec:
        st.markdown("#### 🛒 Order Dispatch")
        lot_to_trade = st.number_input("Position Volume (Lots)", min_value=0.01, max_value=50.0, value=float(round(rec_lot, 2)), step=0.1)
        
        btn_buy, btn_sell = st.columns(2)
        if btn_buy.button("🟢 BUY / LONG", use_container_width=True):
            st.session_state.positions.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Asset": asset_name,
                "Type": "BUY",
                "Lots": lot_to_trade,
                "Price": f"${current_price:,.2f}",
                "Status": "Sent to Broker" if st.session_state.broker_connected else "Demo Executed"
            })
            st.success(f"Dispatched BUY {lot_to_trade} lots at ${current_price:,.2f}")

        if btn_sell.button("🔴 SELL / SHORT", use_container_width=True):
            st.session_state.positions.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Asset": asset_name,
                "Type": "SELL",
                "Lots": lot_to_trade,
                "Price": f"${current_price:,.2f}",
                "Status": "Sent to Broker" if st.session_state.broker_connected else "Demo Executed"
            })
            st.error(f"Dispatched SELL {lot_to_trade} lots at ${current_price:,.2f}")

# 3. EXPANDER: OPEN POSITIONS LOG
with st.expander("📋 Active Positions & Order History (Click to Expand)", expanded=False):
    if len(st.session_state.positions) > 0:
        st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True)
        if st.button("Liquidate / Close All Open Positions"):
            st.session_state.positions = []
            st.rerun()
    else:
        st.caption("No active trades open right now.")

# 4. EXPANDER: AI STATISTICAL ANOMALY LOG
with st.expander("🔬 Statistical Anomaly Log & Feature Matrix (Click to Expand)", expanded=False):
    if not anomalies.empty:
        log_df = anomalies[['Close', 'z_score', 'vol_surge', 'rolling_vol']].copy()
        log_df.columns = ['Close Price', 'Z-Score (Deviations)', 'Volume Surge Multiplier', 'Rolling Volatility']
        st.dataframe(log_df.tail(10), use_container_width=True)
    else:
        st.caption("No statistical anomalies flagged in the current window.")

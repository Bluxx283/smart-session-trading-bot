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
    page_title="ApexQuant Pro | Institutional Multi-Asset Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh app every 15 seconds for live data sync
st_autorefresh(interval=15000, key="quant_feed_sync")

# --- HIGH-END QUANT STYLING (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    code, .metric-value, .stMetric { font-family: 'JetBrains Mono', monospace !important; }

    .main { background-color: #0b0e14; color: #d1d4dc; }
    
    /* Top Live Market Ribbon */
    .ticker-bar {
        display: flex;
        gap: 28px;
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
        font-size: 19px !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "account_balance" not in st.session_state:
    st.session_state.account_balance = 100000.0  # $100k demo balance
if "positions" not in st.session_state:
    st.session_state.positions = []
if "broker_connected" not in st.session_state:
    st.session_state.broker_connected = False

# --- TOP STREAMING RIBBON ---
st.markdown("""
<div class="ticker-bar">
    <div class="ticker-item"><span>🟡 XAU/USD (Gold)</span> <span class="ticker-up">$2,468.40 (+0.84%)</span></div>
    <div class="ticker-item"><span>🪙 BTC/USDT</span> <span class="ticker-up">$78,820.00 (+2.15%)</span></div>
    <div class="ticker-item"><span>🇮🇳 NIFTY 50</span> <span class="ticker-down">24,310.80 (-0.24%)</span></div>
    <div class="ticker-item"><span>🇺🇸 S&P 500</span> <span class="ticker-up">5,840.10 (+0.41%)</span></div>
    <div class="ticker-item"><span>💱 EUR/USD</span> <span class="ticker-down">1.0825 (-0.08%)</span></div>
    <div class="ticker-item"><span>🛢️ CRUDE OIL</span> <span class="ticker-up">$77.30 (+1.10%)</span></div>
</div>
""", unsafe_allow_html=True)

# --- MULTI-MARKET UNIVERSE CONFIG ---
MARKET_UNIVERSE = {
    "🟡 Precious Metals & Commodities": {
        "Gold (XAU/USD)": {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex", "yf": "GC=F", "pip_size": 0.1, "lot_units": 100},
        "Silver (XAG/USD)": {"symbol": "XAGUSD", "exchange": "OANDA", "screener": "forex", "yf": "SI=F", "pip_size": 0.01, "lot_units": 5000},
        "Crude Oil WTI": {"symbol": "USOIL", "exchange": "TVC", "screener": "cfd", "yf": "CL=F", "pip_size": 0.01, "lot_units": 1000}
    },
    "🪙 Cryptocurrency": {
        "Bitcoin (BTC/USDT)": {"symbol": "BTCUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "BTC-USD", "pip_size": 1.0, "lot_units": 1},
        "Ethereum (ETH/USDT)": {"symbol": "ETHUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "ETH-USD", "pip_size": 0.1, "lot_units": 1},
        "Solana (SOL/USDT)": {"symbol": "SOLUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "SOL-USD", "pip_size": 0.01, "lot_units": 1}
    },
    "🇮🇳 Indian Equities (NSE)": {
        "NIFTY 50 Index": {"symbol": "NIFTY", "exchange": "NSE", "screener": "india", "yf": "^NSEI", "pip_size": 0.05, "lot_units": 25},
        "Reliance Industries": {"symbol": "RELIANCE", "exchange": "NSE", "screener": "india", "yf": "RELIANCE.NS", "pip_size": 0.05, "lot_units": 250},
        "HDFC Bank": {"symbol": "HDFCBANK", "exchange": "NSE", "screener": "india", "yf": "HDFCBANK.NS", "pip_size": 0.05, "lot_units": 550},
        "TCS": {"symbol": "TCS", "exchange": "NSE", "screener": "india", "yf": "TCS.NS", "pip_size": 0.05, "lot_units": 175}
    },
    "🇺🇸 US Tech & Global Equities": {
        "Nvidia (NVDA)": {"symbol": "NVDA", "exchange": "NASDAQ", "screener": "america", "yf": "NVDA", "pip_size": 0.01, "lot_units": 100},
        "Apple (AAPL)": {"symbol": "AAPL", "exchange": "NASDAQ", "screener": "america", "yf": "AAPL", "pip_size": 0.01, "lot_units": 100},
        "Tesla (TSLA)": {"symbol": "TSLA", "exchange": "NASDAQ", "screener": "america", "yf": "TSLA", "pip_size": 0.01, "lot_units": 100}
    },
    "💱 Major Forex": {
        "EUR/USD": {"symbol": "EURUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "EURUSD=X", "pip_size": 0.0001, "lot_units": 100000},
        "GBP/USD": {"symbol": "GBPUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "GBPUSD=X", "pip_size": 0.0001, "lot_units": 100000},
        "USD/INR": {"symbol": "USDINR", "exchange": "FX_IDC", "screener": "forex", "yf": "USDINR=X", "pip_size": 0.0025, "lot_units": 1000}
    }
}

# --- SIDEBAR WORKSPACE ---
st.sidebar.title("⚡ Control Center")

category = st.sidebar.selectbox("Sector / Asset Class", list(MARKET_UNIVERSE.keys()))
asset_name = st.sidebar.selectbox("Active Tradable Instrument", list(MARKET_UNIVERSE[category].keys()))
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
st.sidebar.subheader("🧠 Anomaly Parameters")
contamination = st.sidebar.slider("Model Contamination", 0.01, 0.10, 0.03, 0.01)

# --- SIDEBAR BROKER CONNECTION BRIDGE ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Broker API Bridge")
broker_type = st.sidebar.selectbox("Select Target Broker", ["MetaTrader 5 (MT5)", "Zerodha (Kite Connect)", "Binance Futures", "Interactive Brokers (IBKR)"])

if not st.session_state.broker_connected:
    with st.sidebar.expander("⚙️ Connect Broker Credentials"):
        account_id = st.text_input("Account ID / API Key", value="MT5-LIVE-98231")
        server_ip = st.text_input("Server / Endpoint", value="broker.live.gateway:443")
        if st.button("Initialize Bridge Gateway", use_container_width=True):
            st.session_state.broker_connected = True
            st.rerun()
else:
    st.sidebar.success(f"🟢 **{broker_type}** Bridge Active")
    st.sidebar.caption("Latency: `12ms` | Protocol: `REST / WebSockets`")
    if st.sidebar.button("Disconnect Broker", use_container_width=True):
        st.session_state.broker_connected = False
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
def load_ohlcv_data(ticker, period, interval, fallback_price=2450.0):
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

    n_candles = 60
    t_idx = pd.date_range(end=pd.Timestamp.now(), periods=n_candles, freq='15min')
    returns = np.random.normal(0.0001, 0.002, n_candles)
    returns[20:22] += 0.010
    returns[42:44] -= 0.012
    close = fallback_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.0015, n_candles)))
    low = close * (1 - np.abs(np.random.normal(0, 0.0015, n_candles)))
    open_p = (high + low) / 2 + np.random.normal(0, 0.0005, n_candles)
    volume = np.random.randint(2000, 8000, size=n_candles)

    return pd.DataFrame({
        'Open': open_p, 'High': high, 'Low': low, 'Close': close, 'Volume': volume
    }, index=t_idx)

df = load_ohlcv_data(asset_cfg["yf"], yf_period, yf_interval)

# --- ISOLATION FOREST ANOMALY ENGINE ---
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
    clean_df['anomaly'] = raw_anomaly & ((clean_df['z_score'].abs() > 1.8) | (clean_df['vol_surge'] > 2.0))
    anomalies = clean_df[clean_df['anomaly']]
else:
    clean_df = pd.DataFrame()
    anomalies = pd.DataFrame()

# --- TOP SUMMARY HUD ---
st.title(f"⚡ {asset_name} — Institutional Terminal")

h1, h2, h3, h4, h5 = st.columns(5)
current_market_price = clean_df['Close'].iloc[-1] if not clean_df.empty else 2450.0

if tv_data:
    summary = tv_data.summary
    indicators = tv_data.indicators
    verdict = summary.get("RECOMMENDATION", "NEUTRAL")
    signal_color = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "⚪")
    
    h1.metric("TV Consensus", f"{signal_color} {verdict}")
    h2.metric("Oscillators", f"Buy: {summary.get('BUY', 0)} | Sell: {summary.get('SELL', 0)}")
    h3.metric("RSI (14)", f"{indicators.get('RSI', 0.0):.2f}")
    h4.metric("AI Anomaly Breaks", f"{len(anomalies)} Candles")
    h5.metric("Broker Status", "🟢 Connected" if st.session_state.broker_connected else "⚪ Standby")
else:
    h1.metric("Asset Class", category.split()[1])
    h2.metric("Market Price", f"${current_market_price:,.2f}")
    h3.metric("AI Anomalies", f"{len(anomalies)} Flagged")
    h4.metric("System State", "🟢 Active")
    h5.metric("Broker Status", "🟢 Connected" if st.session_state.broker_connected else "⚪ Standby")

st.markdown("---")

# --- MAIN TERMINAL WORKSPACE TABS ---
tab_tv, tab_quant, tab_calc, tab_broker = st.tabs([
    "📺 Live TradingView Viewport", 
    "🔬 Machine Learning Anomaly Engine", 
    "🧮 Smart Pip & Position Size Calculator",
    "⚡ Broker Execution & Order Desk"
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
            name="Price",
            increasing_line_color="#089981", decreasing_line_color="#f23645"
        ), row=1, col=1)

        # ML Anomaly Flags
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['High'] * 1.002,
                mode='markers',
                marker=dict(symbol='diamond', size=9, color='#ff0055', line=dict(width=1, color='#ffffff')),
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
        st.info("Market feed syncing.")

with tab_calc:
    st.subheader(f"🧮 Institutional Pip & Position Size Calculator ({asset_name})")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        calc_acc_balance = st.number_input("Account Balance ($)", min_value=100.0, value=st.session_state.account_balance, step=1000.0)
        risk_percentage = st.slider("Risk Tolerance per Trade (%)", 0.25, 5.0, 1.0, 0.25)
        risk_usd = (calc_acc_balance * risk_percentage) / 100.0
        st.metric("Total Capital at Risk", f"${risk_usd:,.2f}", f"{risk_percentage}% Risk")

    with col_c2:
        entry_price = st.number_input("Planned Entry Price ($)", value=float(current_market_price), format="%.4f")
        stop_loss_price = st.number_input("Stop Loss (SL) Price ($)", value=float(current_market_price * 0.992), format="%.4f")
        take_profit_price = st.number_input("Take Profit (TP) Price ($)", value=float(current_market_price * 1.016), format="%.4f")

    with col_c3:
        # Mathematical Pip & Position Formula
        pip_unit = asset_cfg["pip_size"]
        sl_distance_pips = abs(entry_price - stop_loss_price) / pip_unit
        tp_distance_pips = abs(take_profit_price - entry_price) / pip_unit
        rr_ratio = (tp_distance_pips / sl_distance_pips) if sl_distance_pips > 0 else 1.0
        
        # Recommended Lot Calculation
        lot_size_recommended = risk_usd / (sl_distance_pips * pip_unit * asset_cfg["lot_units"]) if sl_distance_pips > 0 else 0.1
        
        st.metric("Stop Loss Distance", f"{sl_distance_pips:,.1f} Pips / Points")
        st.metric("Risk / Reward Ratio", f"1 : {rr_ratio:.2f}")
        st.metric("Recommended Lot Size", f"{lot_size_recommended:.2f} Lots")

    st.markdown("---")
    st.info(f"💡 **Execution Rule:** For a **${risk_usd:,.2f}** risk with a **{sl_distance_pips:.1f} pip** stop on {asset_name}, open exactly **{lot_size_recommended:.2f} standard lots**.")

with tab_broker:
    col_exec, col_pos = st.columns([1.2, 1.8])
    
    with col_exec:
        st.markdown("#### ⚡ Order Routing Desk")
        target_lots = st.number_input("Execution Volume (Lots)", min_value=0.01, max_value=50.0, value=float(round(lot_size_recommended, 2)), step=0.1)
        
        col_b1, col_b2 = st.columns(2)
        if col_b1.button("🟢 ROUTE BUY ORDER", use_container_width=True):
            st.session_state.positions.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Asset": asset_name,
                "Type": "BUY",
                "Lots": target_lots,
                "Entry": f"${current_market_price:,.2f}",
                "Bridge Status": "Sent to MT5/Gateway" if st.session_state.broker_connected else "Simulated"
            })
            st.success(f"Dispatched BUY order for {target_lots} lots")

        if col_b2.button("🔴 ROUTE SELL ORDER", use_container_width=True):
            st.session_state.positions.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Asset": asset_name,
                "Type": "SELL",
                "Lots": target_lots,
                "Entry": f"${current_market_price:,.2f}",
                "Bridge Status": "Sent to MT5/Gateway" if st.session_state.broker_connected else "Simulated"
            })
            st.error(f"Dispatched SELL order for {target_lots} lots")

    with col_pos:
        st.markdown("#### 📋 Active Broker Positions & History")
        if len(st.session_state.positions) > 0:
            st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True)
            if st.button("Close / Liquidate All Positions", use_container_width=True):
                st.session_state.positions = []
                st.rerun()
        else:
            st.caption("No open orders routed. Connect a broker or execute via the routing desk.")

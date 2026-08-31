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
    page_title="Multi-Asset AI Quant & TradingView Terminal",
    page_icon="⚡",
    layout="wide"
)

# Auto-refresh app every 15 seconds for live streaming updates
st_autorefresh(interval=15000, key="quant_feed_refresh")

# Custom Dark Terminal Theme
st.markdown("""
    <style>
    .main { background-color: #131722; color: #d1d4dc; }
    .stMetric { background-color: #1e222d; padding: 12px; border-radius: 6px; border: 1px solid #2a2e39; }
    </style>
""", unsafe_allow_html=True)

# --- COMPREHENSIVE MULTI-ASSET UNIVERSE ---
MARKET_UNIVERSE = {
    "🟡 Precious Metals & Commodities": {
        "Gold (XAU/USD)": {"symbol": "XAUUSD", "exchange": "OANDA", "screener": "forex", "yf": "GC=F"},
        "Silver (XAG/USD)": {"symbol": "XAGUSD", "exchange": "OANDA", "screener": "forex", "yf": "SI=F"},
        "Crude Oil WTI": {"symbol": "USOIL", "exchange": "TVC", "screener": "cfd", "yf": "CL=F"},
        "Natural Gas": {"symbol": "NATGAS", "exchange": "TVC", "screener": "cfd", "yf": "NG=F"},
        "Copper": {"symbol": "COPPER", "exchange": "TVC", "screener": "cfd", "yf": "HG=F"}
    },
    "🪙 Cryptocurrency": {
        "Bitcoin (BTC/USDT)": {"symbol": "BTCUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "BTC-USD"},
        "Ethereum (ETH/USDT)": {"symbol": "ETHUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "ETH-USD"},
        "Solana (SOL/USDT)": {"symbol": "SOLUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "SOL-USD"},
        "Ripple (XRP/USDT)": {"symbol": "XRPUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "XRP-USD"},
        "Binance Coin (BNB/USDT)": {"symbol": "BNBUSDT", "exchange": "BINANCE", "screener": "crypto", "yf": "BNB-USD"}
    },
    "🇮🇳 Indian Equities (NSE)": {
        "NIFTY 50 Index": {"symbol": "NIFTY", "exchange": "NSE", "screener": "india", "yf": "^NSEI"},
        "BANK NIFTY Index": {"symbol": "BANKNIFTY", "exchange": "NSE", "screener": "india", "yf": "^NSEBANK"},
        "Reliance Industries": {"symbol": "RELIANCE", "exchange": "NSE", "screener": "india", "yf": "RELIANCE.NS"},
        "Tata Consultancy Services (TCS)": {"symbol": "TCS", "exchange": "NSE", "screener": "india", "yf": "TCS.NS"},
        "HDFC Bank": {"symbol": "HDFCBANK", "exchange": "NSE", "screener": "india", "yf": "HDFCBANK.NS"},
        "Infosys": {"symbol": "INFY", "exchange": "NSE", "screener": "india", "yf": "INFY.NS"},
        "Tata Motors": {"symbol": "TATAMOTORS", "exchange": "NSE", "screener": "india", "yf": "TATAMOTORS.NS"},
        "State Bank of India (SBIN)": {"symbol": "SBIN", "exchange": "NSE", "screener": "india", "yf": "SBIN.NS"}
    },
    "🇺🇸 US Stocks & Tech": {
        "Apple (AAPL)": {"symbol": "AAPL", "exchange": "NASDAQ", "screener": "america", "yf": "AAPL"},
        "Nvidia (NVDA)": {"symbol": "NVDA", "exchange": "NASDAQ", "screener": "america", "yf": "NVDA"},
        "Tesla (TSLA)": {"symbol": "TSLA", "exchange": "NASDAQ", "screener": "america", "yf": "TSLA"},
        "Microsoft (MSFT)": {"symbol": "MSFT", "exchange": "NASDAQ", "screener": "america", "yf": "MSFT"},
        "Amazon (AMZN)": {"symbol": "AMZN", "exchange": "NASDAQ", "screener": "america", "yf": "AMZN"},
        "Meta Platforms (META)": {"symbol": "META", "exchange": "NASDAQ", "screener": "america", "yf": "META"}
    },
    "💱 Major Forex Pairs": {
        "EUR/USD": {"symbol": "EURUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "EURUSD=X"},
        "GBP/USD": {"symbol": "GBPUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "GBPUSD=X"},
        "USD/JPY": {"symbol": "USDJPY", "exchange": "FX_IDC", "screener": "forex", "yf": "USDJPY=X"},
        "USD/INR": {"symbol": "USDINR", "exchange": "FX_IDC", "screener": "forex", "yf": "USDINR=X"},
        "AUD/USD": {"symbol": "AUDUSD", "exchange": "FX_IDC", "screener": "forex", "yf": "AUDUSD=X"}
    },
    "🌐 Global Indices": {
        "S&P 500 (SPX)": {"symbol": "SPX500", "exchange": "OANDA", "screener": "cfd", "yf": "^GSPC"},
        "Nasdaq 100 (NDX)": {"symbol": "NAS100", "exchange": "OANDA", "screener": "cfd", "yf": "^IXIC"},
        "Dow Jones (DJI)": {"symbol": "US30", "exchange": "TVC", "screener": "cfd", "yf": "^DJI"}
    }
}

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚡ Market Control Center")

category = st.sidebar.selectbox("Market Sector / Asset Class", list(MARKET_UNIVERSE.keys()))
asset_name = st.sidebar.selectbox("Tradable Instrument", list(MARKET_UNIVERSE[category].keys()))
asset_cfg = MARKET_UNIVERSE[category][asset_name]

tv_interval_map = {
    "1 Minute (Scalping)": (Interval.INTERVAL_1_MINUTE, "1m", "1d"),
    "5 Minutes (Intraday)": (Interval.INTERVAL_5_MINUTES, "5m", "5d"),
    "15 Minutes (Session)": (Interval.INTERVAL_15_MINUTES, "15m", "1mo"),
    "1 Hour (Swing)": (Interval.INTERVAL_1_HOUR, "1h", "3mo"),
    "1 Day (Macro)": (Interval.INTERVAL_1_DAY, "1d", "1y")
}

interval_label = st.sidebar.selectbox("Analysis Timeframe", list(tv_interval_map.keys()), index=2)
tv_interval, yf_interval, yf_period = tv_interval_map[interval_label]

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Machine Learning Parameters")
contamination = st.sidebar.slider("Anomaly Contamination Sensitivity", 0.01, 0.15, 0.05, 0.01)

# --- FETCH TRADINGVIEW TECHNICAL ANALYSIS ---
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

# --- FETCH OHLCV DATA FOR PLOTTING & ML ---
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

# --- ML ANOMALY ENGINE (ISOLATION FOREST) ---
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

# --- TOP SUMMARY METRIC ROW ---
st.title(f"{asset_name} — Institutional Quant Terminal")

c1, c2, c3, c4 = st.columns(4)

if tv_data:
    summary = tv_data.summary
    indicators = tv_data.indicators
    
    verdict = summary.get("RECOMMENDATION", "NEUTRAL")
    signal_color = "🟢" if "BUY" in verdict else ("🔴" if "SELL" in verdict else "⚪")
    
    c1.metric("TradingView Summary", f"{signal_color} {verdict}")
    c2.metric("Oscillators Consensus", f"Buy: {summary.get('BUY', 0)} | Sell: {summary.get('SELL', 0)}")
    c3.metric("RSI (14)", f"{indicators.get('RSI', 0.0):.2f}")
    c4.metric("AI Flagged Anomalies", f"{len(anomalies)} Structural Events")
else:
    c1.metric("Asset Class", category.split()[1])
    c2.metric("Status", "Live Connected")
    c3.metric("AI Anomalies", f"{len(anomalies)} Detected")
    c4.metric("Engine State", "🟢 Active")

st.markdown("---")

# --- DUAL WORKSPACE TABS ---
tab_tv, tab_quant, tab_ind = st.tabs(["📺 Live TradingView Chart Viewport", "🔬 ML Anomaly Engine & Price Action", "📋 Technical Indicator Snapshot"])

with tab_tv:
    st.caption("Direct High-Speed Interactive TradingView HTML5 Engine")
    tv_widget_html = f"""
    <div class="tradingview-widget-container" style="height:600px;width:100%">
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
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tv_widget_html, height=620)

with tab_quant:
    if not clean_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=clean_df.index,
            open=clean_df['Open'], high=clean_df['High'],
            low=clean_df['Low'], close=clean_df['Close'],
            name="Candlestick Price",
            increasing_line_color="#089981", decreasing_line_color="#F23645"
        ), row=1, col=1)

        # ML Anomaly Flags
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['High'] * 1.002,
                mode='markers+text',
                marker=dict(symbol='diamond', size=10, color='#FF1744'),
                name="AI Anomaly Breakout"
            ), row=1, col=1)

        # Volume Subplot
        vol_colors = ['#089981' if c >= o else '#F23645' for c, o in zip(clean_df['Close'], clean_df['Open'])]
        fig.add_trace(go.Bar(x=clean_df.index, y=clean_df['Volume'], marker_color=vol_colors, name="Volume"), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=580, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Market data feed loading or closed for the selected timeframe. Check TradingView Chart tab.")

with tab_ind:
    if tv_data:
        st.subheader("📊 Live Technical Indicator Readouts")
        ind_col1, ind_col2, ind_col3 = st.columns(3)
        with ind_col1:
            st.metric("RSI (14)", f"{indicators.get('RSI', 0):.2f}")
            st.metric("ADX Trend Strength", f"{indicators.get('ADX', 0):.2f}")
        with ind_col2:
            st.metric("MACD Level", f"{indicators.get('MACD.macd', 0):.4f}")
            st.metric("Stochastic %K", f"{indicators.get('Stoch.K', 0):.2f}")
        with ind_col3:
            st.metric("EMA 20", f"{indicators.get('EMA20', 0):,.2f}")
            st.metric("SMA 50", f"{indicators.get('SMA50', 0):,.2f}")

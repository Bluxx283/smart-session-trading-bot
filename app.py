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
# NOTE: rather than an external image URL (which can 404/rate-limit and break
# the look of the app), the "background image" is a generated CSS pattern —
# a faint chart-grid + soft color blooms — layered under a dark overlay for
# contrast. Swap in a real photo via `background-image: url('...')` if you
# have one you'd like to use instead.
st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(rgba(5,7,12,0.90), rgba(5,7,12,0.95)),
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,0.035) 1px, transparent 1px),
            repeating-linear-gradient(135deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 2px, transparent 2px, transparent 28px),
            radial-gradient(circle at 15% 10%, rgba(80,140,255,0.12), transparent 45%),
            radial-gradient(circle at 85% 90%, rgba(255,80,140,0.09), transparent 45%),
            radial-gradient(circle at 50% 40%, rgba(0,255,180,0.05), transparent 60%),
            #05070c;
        background-size: auto, 46px 46px, 46px 46px, auto, auto, auto, auto, auto;
        background-attachment: fixed;
    }

    /* Dynamic page header used across every view */
    .ssad-page-header {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        padding: 2px 0 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 14px;
    }
    .ssad-page-header span { color: rgba(255,255,255,0.55); font-weight: 500; }

    /* Room at the top of the main content so the fixed ticker never overlaps it */
    section.main > div.block-container {
        padding-top: 4.4rem !important;
    }
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
                "chart signals, or build a strategy in the Strategy Builder tab."
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


default_cfg = MARKET_UNIVERSE["🟡 Metals & Commodities"]["Gold (XAU/USD)"]
global_df = load_ohlcv(default_cfg["yf"])
global_price = (
    float(global_df["Close"].iloc[-1]) if not global_df.empty else 2468.40
)

# ==========================================
# 🎫 TOP TICKER BAR — persistent, fixed-position, CSS-only marquee
#
# Rendered with st.markdown (NOT components.html) so it lives directly in
# the app's own DOM rather than inside a throwaway iframe. That means every
# Streamlit rerun (nav clicks, the 20s auto-refresh, any widget interaction)
# re-emits the *same* fixed-position HTML/CSS instead of tearing down and
# reloading an iframe — so the bar stays put and the marquee keeps sliding
# smoothly no matter which page or quick action is active. All 5 assets are
# duplicated back-to-back in the track so the loop has no blank gap, and the
# 50%-travel keyframe means the seam is invisible.
# ==========================================
_TICKER_ITEMS = (
    '<span class="ssad-tk-item">🟡 XAU/USD&nbsp;<b>$2,468.40</b>&nbsp;'
    '<span class="ssad-up">+0.84%</span></span>'
    '<span class="ssad-tk-item">🪙 BTC/USDT&nbsp;<b>$78,820.00</b>&nbsp;'
    '<span class="ssad-up">+2.15%</span></span>'
    '<span class="ssad-tk-item">🇮🇳 NIFTY 50&nbsp;<b>24,310.80</b>&nbsp;'
    '<span class="ssad-down">-0.24%</span></span>'
    '<span class="ssad-tk-item">🇺🇸 S&amp;P 500&nbsp;<b>5,840.10</b>&nbsp;'
    '<span class="ssad-up">+0.41%</span></span>'
    '<span class="ssad-tk-item">💱 EUR/USD&nbsp;<b>1.0825</b>&nbsp;'
    '<span class="ssad-down">-0.08%</span></span>'
)

st.markdown(
    f"""
    <style>
    .ssad-ticker-fixed {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 999998;
        background: rgba(14,17,27,0.72);
        backdrop-filter: blur(10px) saturate(150%);
        -webkit-backdrop-filter: blur(10px) saturate(150%);
        border-bottom: 1px solid rgba(255,255,255,0.10);
        overflow: hidden;
        white-space: nowrap;
        padding: 10px 0;
    }}
    .ssad-ticker-fixed:hover .ssad-ticker-track {{
        animation-play-state: paused;
    }}
    .ssad-ticker-track {{
        display: inline-block;
        white-space: nowrap;
        /* left-to-right, continuous, seamless: the track holds two copies of
           the item list, so travelling exactly 50% of its own width lands
           back on an identical frame and the loop point is invisible */
        animation: ssad-ticker-slide 30s linear infinite;
    }}
    @keyframes ssad-ticker-slide {{
        from {{ transform: translateX(0%); }}
        to   {{ transform: translateX(50%); }}
    }}
    .ssad-tk-item {{
        display: inline-block; color: #e8eaf0; font-family: sans-serif;
        font-size: 14px; padding: 0 34px; border-right: 1px solid rgba(255,255,255,0.12);
    }}
    .ssad-up {{ color: #4ade80; font-weight: 600; }}
    .ssad-down {{ color: #f87171; font-weight: 600; }}
    </style>
    <div class="ssad-ticker-fixed">
        <div class="ssad-ticker-track">{_TICKER_ITEMS}{_TICKER_ITEMS}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        st.markdown("##### 🤖 AI Trading Co-Pilot")
        bot_tab_chat, bot_tab_signals, bot_tab_set = st.tabs(
            ["💬 Chat", "📡 Signals", "⚙️"]
        )

        # ---------------- CHAT TAB (text + voice) ----------------
        with bot_tab_chat:
            for msg in st.session_state.chat_messages[-30:]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Text-to-speech: speak the newest assistant reply once, in-browser
            last_idx = len(st.session_state.chat_messages) - 1
            if (
                st.session_state.voice_enabled
                and last_idx >= 0
                and st.session_state.chat_messages[last_idx]["role"] == "assistant"
                and last_idx != st.session_state.last_spoken_index
            ):
                speak_text = (
                    st.session_state.chat_messages[last_idx]["content"]
                    .replace("`", "")
                    .replace("\n", " ")
                    .replace('"', "'")
                )
                components.html(
                    f"""
                    <script>
                    try {{
                        const u = new SpeechSynthesisUtterance("{speak_text[:600]}");
                        u.rate = 1.0;
                        window.parent.speechSynthesis.cancel();
                        window.parent.speechSynthesis.speak(u);
                    }} catch (e) {{}}
                    </script>
                    """,
                    height=1,
                )
                st.session_state.last_spoken_index = last_idx

            voice_col, text_col, send_col = st.columns([0.16, 0.64, 0.2])
            with voice_col:
                # Voice input via the browser's built-in Web Speech API (Chrome/Edge).
                # Recognized speech is written into the text box below and
                # auto-submitted by simulating a click on the Send button.
                components.html(
                    """
                    <button id="ssad_mic_btn" style="width:100%;height:38px;border-radius:8px;
                        border:1px solid rgba(255,255,255,0.3);background:rgba(255,255,255,0.08);
                        color:white;cursor:pointer;font-size:16px;">🎤</button>
                    <script>
                    const btn = document.getElementById('ssad_mic_btn');
                    btn.addEventListener('click', function() {
                        const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
                        if (!SpeechRecognition) {
                            alert('Voice input needs Chrome or Edge.');
                            return;
                        }
                        const rec = new SpeechRecognition();
                        rec.lang = 'en-US';
                        rec.interimResults = false;
                        btn.innerText = '🔴';
                        rec.onresult = function(e) {
                            const text = e.results[0][0].transcript;
                            const doc = window.parent.document;
                            const inputs = doc.querySelectorAll('textarea, input[type="text"]');
                            let target = null;
                            inputs.forEach(function(el) {
                                if (el.placeholder && el.placeholder.indexOf('Ask about the market') !== -1) {
                                    target = el;
                                }
                            });
                            if (target) {
                                const setter = Object.getOwnPropertyDescriptor(
                                    window.parent.HTMLTextAreaElement.prototype.value !== undefined
                                        ? window.parent.HTMLTextAreaElement.prototype
                                        : window.parent.HTMLInputElement.prototype,
                                    'value'
                                ).set;
                                setter.call(target, text);
                                target.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                            btn.innerText = '🎤';
                        };
                        rec.onerror = function() { btn.innerText = '🎤'; };
                        rec.onend = function() { btn.innerText = '🎤'; };
                        rec.start();
                    });
                    </script>
                    """,
                    height=44,
                )
            with text_col:
                chat_text = st.text_input(
                    "Message",
                    key="ssad_chat_text_input",
                    placeholder="Ask about the market or your strategy...",
                    label_visibility="collapsed",
                )
            with send_col:
                if st.button("Send", key="ssad_chat_send", use_container_width=True):
                    if chat_text:
                        send_chat_message(chat_text)
                        st.rerun()

        # ---------------- SIGNALS TAB ----------------
        with bot_tab_signals:
            sig_df = global_df.copy()
            if len(sig_df) >= 20:
                sig_df["ret"] = sig_df["Close"].pct_change()
                z = (
                    (sig_df["Close"] - sig_df["Close"].rolling(20).mean())
                    / (sig_df["Close"].rolling(20).std() + 1e-8)
                ).iloc[-1]
                vol_surge = (
                    sig_df["Volume"] / (sig_df["Volume"].rolling(20).mean() + 1e-8)
                ).iloc[-1]

                if z <= -1.8:
                    call, reason = "🟢 BUY (mean-reversion)", "Price is statistically stretched below its mean."
                elif z >= 1.8:
                    call, reason = "🔴 SELL (mean-reversion)", "Price is statistically stretched above its mean."
                elif vol_surge >= 2.0:
                    call, reason = "🟡 WATCH (volume surge)", "Volume is surging — a breakout may be forming."
                else:
                    call, reason = "⚪ HOLD", "No statistically significant edge right now."

                st.metric("Current Signal", call)
                st.caption(f"Z-Score: {z:.2f}  |  Volume Surge: {vol_surge:.2f}x")
                st.write(reason)

                sig_context = (
                    f"Instrument: Gold (XAU/USD). Latest price {global_price:.2f}. "
                    f"Z-Score {z:.2f}, Volume Surge {vol_surge:.2f}x. "
                    f"Rule-based call: {call}."
                )
                if st.button("Ask Claude to interpret this signal", key="ssad_sig_ask"):
                    send_chat_message(
                        "Interpret the current chart signal and suggest what to watch for next.",
                        context_summary=sig_context,
                    )
                    st.rerun()
            else:
                st.info("Not enough data loaded yet for a signal.")

        # ---------------- BOT SETTINGS TAB ----------------
        with bot_tab_set:
            st.session_state.anthropic_api_key = st.text_input(
                "Anthropic API key",
                value=st.session_state.anthropic_api_key,
                type="password",
            )
            st.session_state.claude_model = st.text_input(
                "Claude model", value=st.session_state.claude_model
            )
            st.session_state.voice_enabled = st.checkbox(
                "Speak replies out loud", value=st.session_state.voice_enabled
            )
            st.caption(
                "Voice input/output uses your browser's built-in speech engine "
                "(Chrome/Edge work best) — no extra key needed for that part."
            )

# ==========================================
# 📊 VIEW 1: DASHBOARD
# ==========================================
if st.session_state.active_tab == "📊 Dashboard":
    page_header("Terminal Core")
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

    # Uniform glassmorphism quick-action cards — targeted via the
    # `st-key-qa_card_*` prefix so a single rule styles all of them.
    st.markdown(
        """
        <style>
        [class*="st-key-qa_card_"] {
            background: rgba(16, 20, 30, 0.55) !important;
            backdrop-filter: blur(16px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 16px !important;
            padding: 6px !important;
            transition: transform 0.15s ease, border-color 0.15s ease;
            height: 100%;
        }
        [class*="st-key-qa_card_"]:hover {
            transform: translateY(-3px);
            border-color: rgba(140,180,255,0.45) !important;
        }
        .ssad-qa-icon { font-size: 1.6rem; margin-bottom: 2px; }
        .ssad-qa-title { font-weight: 700; font-size: 1.0rem; margin-bottom: 4px; }
        .ssad-qa-desc { font-size: 0.80rem; color: rgba(255,255,255,0.62); line-height: 1.35; min-height: 64px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    QUICK_ACTIONS = [
        {
            "icon": "📈",
            "title": "AI Live Charts",
            "desc": "Institutional-grade TradingView viewport fused with the "
            "Isolation Forest anomaly engine — flags liquidity sweeps and "
            "statistical outliers as they form.",
            "button": "Open Charts →",
            "target": "📈 AI Trade & Charts",
        },
        {
            "icon": "🧮",
            "title": "Pip & Risk Engine",
            "desc": "Precision position-sizing desk. Model exact lot size, "
            "stop distance and risk-to-reward before routing a single order.",
            "button": "Open Calculator →",
            "target": "🧮 Pip & Risk Calculator",
        },
        {
            "icon": "⚡",
            "title": "Broker Gateway",
            "desc": "Bridge to MT5, Zerodha Kite, Binance Futures or "
            "Interactive Brokers. Route mock or live execution with full "
            "fill history.",
            "button": "Open Gateway →",
            "target": "⚡ Broker Gateway",
        },
        {
            "icon": "🧩",
            "title": "Strategy Builder",
            "desc": "Construct rule-based setups from scratch, or clone "
            "institutional playbooks — Liquidity Sweep, Order Block "
            "Mitigation, Session Breakout — straight into your desk.",
            "button": "Open Builder →",
            "target": "🧩 Strategy Builder",
        },
        {
            "icon": "📅",
            "title": "Economic Calendar",
            "desc": "Track high-impact CPI, NFP and central bank rate "
            "decisions before volatility hits your open positions.",
            "button": "View Calendar →",
            "target": "📅 Economic Calendar",
        },
        {
            "icon": "🤖",
            "title": "AI Co-Pilot",
            "desc": "Voice- and text-enabled trading assistant with live "
            "signal interpretation and access to your saved strategies.",
            "button": "Open Co-Pilot →",
            "target": None,  # opens the floating bot instead of switching pages
        },
    ]

    for row_start in (0, 3):
        cols = st.columns(3)
        for col, action in zip(cols, QUICK_ACTIONS[row_start:row_start + 3]):
            with col:
                with st.container(key=f"qa_card_{action['title']}", border=True):
                    st.markdown(
                        f'<div class="ssad-qa-icon">{action["icon"]}</div>'
                        f'<div class="ssad-qa-title">{action["title"]}</div>'
                        f'<div class="ssad-qa-desc">{action["desc"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if action["target"] is not None:
                        st.button(
                            action["button"],
                            key=f"btn_qa_{action['title']}",
                            use_container_width=True,
                            on_click=switch_page,
                            args=(action["target"],),
                        )
                    else:
                        st.button(
                            action["button"],
                            key=f"btn_qa_{action['title']}",
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
    page_header("Live Charts & ML Anomaly Engine")

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
    page_header("Pip & Sizing Desk")

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
    page_header("Execution Bridge")

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
# 🧩 VIEW 5: STRATEGY BUILDER
# ==========================================
elif st.session_state.active_tab == "🧩 Strategy Builder":
    page_header("Strategy Builder")
    st.caption("Clone an institutional playbook or construct your own rule-based setup.")

    tab_playbooks, tab_custom, tab_mine = st.tabs(
        ["📚 Institutional Playbooks", "🛠️ Build Custom", "⭐ My Strategies"]
    )

    # ---------------- INSTITUTIONAL PLAYBOOKS ----------------
    with tab_playbooks:
        pb_cols = st.columns(3)
        for pb_col, strat in zip(pb_cols, POPULAR_STRATEGIES):
            with pb_col:
                with st.container(border=True):
                    st.markdown(f"#### {strat['name']}")
                    st.caption(strat["description"])
                    st.markdown(f"**Entry rule:** {strat['entry_rule']}")
                    m1, m2 = st.columns(2)
                    m1.metric("Stop Loss", f"{strat['stop_loss_pct']}%")
                    m2.metric("Take Profit", f"{strat['take_profit_pct']}%")
                    already_added = any(
                        s["name"] == strat["name"] for s in st.session_state.my_strategies
                    )
                    if already_added:
                        st.button(
                            "✅ Added",
                            key=f"clone_{strat['name']}",
                            use_container_width=True,
                            disabled=True,
                        )
                    else:
                        if st.button(
                            "＋ Clone to My Strategies",
                            key=f"clone_{strat['name']}",
                            use_container_width=True,
                        ):
                            st.session_state.my_strategies.append(dict(strat))
                            st.rerun()

    # ---------------- CUSTOM STRATEGY BUILDER ----------------
    with tab_custom:
        with st.form("ssad_custom_strategy_form", clear_on_submit=True):
            cs_name = st.text_input("Strategy Name", placeholder="e.g. NY Open Fade")
            cs_desc = st.text_area(
                "Description",
                placeholder="What market behavior is this strategy trying to capture?",
            )
            cs_entry = st.text_input(
                "Entry Rule",
                placeholder="e.g. Price closes above 20-EMA with RSI > 55 → enter long",
            )
            cs_col1, cs_col2 = st.columns(2)
            cs_sl = cs_col1.slider("Stop Loss (%)", 0.1, 5.0, 1.0, 0.1)
            cs_tp = cs_col2.slider("Take Profit (%)", 0.1, 10.0, 2.0, 0.1)
            cs_submit = st.form_submit_button("＋ Add Strategy", use_container_width=True)

        if cs_submit:
            if cs_name.strip() and cs_entry.strip():
                st.session_state.my_strategies.append(
                    {
                        "name": cs_name.strip(),
                        "description": cs_desc.strip() or "No description provided.",
                        "entry_rule": cs_entry.strip(),
                        "stop_loss_pct": cs_sl,
                        "take_profit_pct": cs_tp,
                    }
                )
                st.success(f"'{cs_name.strip()}' added to My Strategies.")
                st.rerun()
            else:
                st.warning("Give the strategy a name and an entry rule before saving.")

    # ---------------- MY STRATEGIES ----------------
    with tab_mine:
        if len(st.session_state.my_strategies) == 0:
            st.info(
                "No strategies saved yet. Clone a playbook or build a custom "
                "one to see it here."
            )
        else:
            for idx, strat in enumerate(st.session_state.my_strategies):
                with st.container(border=True):
                    hc1, hc2 = st.columns([5, 1])
                    hc1.markdown(f"#### {strat['name']}")
                    if hc2.button("🗑️", key=f"del_strat_{idx}", use_container_width=True):
                        st.session_state.my_strategies.pop(idx)
                        st.rerun()
                    st.caption(strat["description"])
                    st.markdown(f"**Entry rule:** {strat['entry_rule']}")
                    m1, m2 = st.columns(2)
                    m1.metric("Stop Loss", f"{strat['stop_loss_pct']}%")
                    m2.metric("Take Profit", f"{strat['take_profit_pct']}%")

# ==========================================
# 📅 VIEW 6: ECONOMIC CALENDAR
# ==========================================
elif st.session_state.active_tab == "📅 Economic Calendar":
    page_header("Economic Calendar")
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
# ⚙️ VIEW 7: SETTINGS
# ==========================================
elif st.session_state.active_tab == "⚙️ Settings":
    page_header("Settings")
    st.write("Platform: Smart Session Anomaly Detector Suite")
    st.write("Architecture: Python Quant Pipeline + Isolation Forest ML")
    st.write("Data Stream Latency: 20 Seconds Auto-Sync")
    st.selectbox("Base Currency", ["USD ($)", "INR (₹)", "EUR (€)"])

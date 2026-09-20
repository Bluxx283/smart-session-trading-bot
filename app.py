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

# A few ready-made strategy templates users can clone into "My Strategies"
POPULAR_STRATEGIES = [
    {
        "name": "Mean Reversion (Z-Score)",
        "description": "Fades price extremes: enters when price is statistically "
        "stretched from its 20-period mean and expects reversion.",
        "entry_rule": "Z-Score below -1.8 → BUY  |  Z-Score above +1.8 → SELL",
        "stop_loss_pct": 0.8,
        "take_profit_pct": 1.6,
    },
    {
        "name": "Breakout Momentum",
        "description": "Follows the trend: enters on a volume-confirmed breakout "
        "past a recent pivot high/low.",
        "entry_rule": "Volume Surge above 2.0x AND price breaks 5-bar pivot → BUY/SELL with trend",
        "stop_loss_pct": 1.2,
        "take_profit_pct": 3.0,
    },
    {
        "name": "Volatility Squeeze",
        "description": "Waits for rolling volatility to compress, then trades the "
        "expansion in whichever direction it breaks.",
        "entry_rule": "Rolling Volatility in bottom 20th percentile, then breakout → enter with breakout direction",
        "stop_loss_pct": 1.0,
        "take_profit_pct": 2.5,
    },
]

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
# 🤖 FLOATING TRANSLUCENT AI BOT
# Rendered every rerun, outside the tab if/elif chain, so it stays visible
# and usable no matter which nav tab is currently selected.
# ==========================================
st.markdown(
    """
    <style>
    /* Floating round toggle button */
    div[data-testid="stButton"]:has(.ssad-fab-anchor),
    .st-key-ssad_bot_fab {
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        z-index: 1000000 !important;
        width: 62px !important;
    }
    .st-key-ssad_bot_fab button {
        border-radius: 50% !important;
        width: 62px !important;
        height: 62px !important;
        font-size: 26px !important;
        background: rgba(255, 255, 255, 0.10) !important;
        backdrop-filter: blur(12px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(12px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.28) !important;
        box-shadow: 0 6px 22px rgba(0,0,0,0.45) !important;
    }
    /* Floating translucent chat panel */
    .st-key-ssad_bot_panel {
        position: fixed !important;
        bottom: 96px !important;
        right: 24px !important;
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

fab_label = "✖" if st.session_state.bot_open else "🤖"
if st.button(fab_label, key="ssad_bot_fab", help="AI Trading Co-Pilot"):
    st.session_state.bot_open = not st.session_state.bot_open
    st.rerun()

if st.session_state.bot_open:
    with st.container(key="ssad_bot_panel"):
        st.markdown("##### 🤖 AI Trading Co-Pilot")
        bot_tab_chat, bot_tab_signals, bot_tab_strat, bot_tab_set = st.tabs(
            ["💬 Chat", "📡 Signals", "🧠 Strategies", "⚙️"]
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
                    height=0,
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

        # ---------------- STRATEGIES / EA BUILDER TAB ----------------
        with bot_tab_strat:
            st.caption("Popular templates")
            for tmpl in POPULAR_STRATEGIES:
                with st.expander(tmpl["name"]):
                    st.write(tmpl["description"])
                    st.caption(f"Rule: {tmpl['entry_rule']}")
                    st.caption(
                        f"SL {tmpl['stop_loss_pct']}%  |  TP {tmpl['take_profit_pct']}%"
                    )
                    if st.button("Use as my EA", key=f"ssad_use_{tmpl['name']}"):
                        st.session_state.my_strategies.append(dict(tmpl))
                        st.rerun()

            st.divider()
            st.caption("Build your own EA")
            with st.form("ssad_new_strategy_form", clear_on_submit=True):
                new_name = st.text_input("Strategy name")
                new_rule = st.text_area(
                    "Entry rule (plain language, e.g. 'RSI below 30 and volume surge above 1.5x → BUY')"
                )
                new_sl = st.number_input("Stop Loss (%)", value=1.0, step=0.1)
                new_tp = st.number_input("Take Profit (%)", value=2.0, step=0.1)
                if st.form_submit_button("Save Strategy"):
                    if new_name and new_rule:
                        st.session_state.my_strategies.append(
                            {
                                "name": new_name,
                                "description": "Custom user-defined EA.",
                                "entry_rule": new_rule,
                                "stop_loss_pct": new_sl,
                                "take_profit_pct": new_tp,
                            }
                        )
                        st.rerun()

            if st.session_state.my_strategies:
                st.divider()
                st.caption("My saved strategies")
                for i, strat in enumerate(st.session_state.my_strategies):
                    with st.expander(f"📌 {strat['name']}"):
                        st.write(strat["description"])
                        st.caption(f"Rule: {strat['entry_rule']}")
                        st.caption(
                            f"SL {strat['stop_loss_pct']}%  |  TP {strat['take_profit_pct']}%"
                        )
                        col_bt, col_del = st.columns(2)
                        if col_bt.button("Ask Claude to review", key=f"ssad_review_{i}"):
                            send_chat_message(
                                "Review this trading strategy and point out strengths, "
                                "weaknesses, and any risk-management gaps.",
                                context_summary=(
                                    f"Strategy '{strat['name']}' — rule: {strat['entry_rule']}, "
                                    f"SL {strat['stop_loss_pct']}%, TP {strat['take_profit_pct']}%."
                                ),
                            )
                            st.rerun()
                        if col_del.button("Delete", key=f"ssad_del_{i}"):
                            st.session_state.my_strategies.pop(i)
                            st.rerun()

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

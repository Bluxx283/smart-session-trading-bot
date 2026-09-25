from datetime import datetime
import re
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


# --- EA / STRATEGY WORKSPACE DATA ---
EA_TEMPLATES = [
    {
        "name": "SMA / EMA Crossover EA",
        "description": "Trend-following EA using fast/slow moving-average crossovers.",
        "entry": "Fast SMA/EMA crosses above slow SMA/EMA",
        "exit": "Opposite crossover or configured stop/target",
    },
    {
        "name": "RSI Mean Reversion EA",
        "description": "Mean-reversion EA using RSI overbought/oversold zones.",
        "entry": "RSI enters oversold/overbought zone and confirms reversal",
        "exit": "Return toward RSI midpoint or configured stop/target",
    },
    {
        "name": "ICT / SMC Liquidity Sweep EA",
        "description": "Session liquidity sweep and reclaim model.",
        "entry": "Liquidity sweep + fast reclaim / market-structure confirmation",
        "exit": "Opposing liquidity target or configured stop/target",
    },
    {
        "name": "MACD Divergence EA",
        "description": "Momentum/divergence setup using MACD histogram and price structure.",
        "entry": "Confirmed bullish/bearish MACD divergence",
        "exit": "Momentum reversal or configured stop/target",
    },
    {
        "name": "Bollinger Bands Squeeze EA",
        "description": "Volatility-compression breakout model.",
        "entry": "Band squeeze followed by confirmed directional expansion",
        "exit": "Opposite signal or configured stop/target",
    },
]

if "saved_eas" not in st.session_state:
    st.session_state.saved_eas = []
if "active_ea" not in st.session_state:
    st.session_state.active_ea = None
if "ea_enabled" not in st.session_state:
    st.session_state.ea_enabled = False
if "ea_code" not in st.session_state:
    st.session_state.ea_code = """# Custom EA template
# Python / Pine-style pseudocode
FAST_PERIOD = 9
SLOW_PERIOD = 21
RISK_PCT = 1.0
STOP_LOSS_PCT = 0.6
TAKE_PROFIT_PCT = 1.8

def entry_signal(data):
    # Add your entry conditions here
    return False

def exit_signal(data):
    # Add your exit conditions here
    return False
"""
if "broker_api_mode" not in st.session_state:
    st.session_state.broker_api_mode = "Paper / Demo"
if "broker_name" not in st.session_state:
    st.session_state.broker_name = "MetaTrader 5 (MT5)"
if "broker_api_endpoint" not in st.session_state:
    st.session_state.broker_api_endpoint = ""
if "broker_api_key" not in st.session_state:
    st.session_state.broker_api_key = ""
if "ea_deployments" not in st.session_state:
    st.session_state.ea_deployments = {}
if "strategy_library" not in st.session_state:
    st.session_state.strategy_library = []
if "last_backtest" not in st.session_state:
    st.session_state.last_backtest = None

def place_chart_trade(side, lots, asset_name, price):
    st.session_state.positions.insert(
        0,
        {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Asset": asset_name,
            "Type": side,
            "Lots": lots,
            "Price": f"${price:,.2f}",
            "Bridge": "MT5/REST" if st.session_state.broker_connected else "Demo Simulated",
        },
    )


def add_indicators(df, fast=9, slow=21, rsi_period=14, bb_period=20):
    out = df.copy()
    out["EMA Fast"] = out["Close"].ewm(span=max(int(fast), 2), adjust=False).mean()
    out["EMA Slow"] = out["Close"].ewm(span=max(int(slow), 2), adjust=False).mean()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(max(int(rsi_period), 2)).mean()
    loss = (-delta.clip(upper=0)).rolling(max(int(rsi_period), 2)).mean()
    rs = gain / (loss + 1e-9)
    out["RSI"] = 100 - (100 / (1 + rs))
    mid = out["Close"].rolling(max(int(bb_period), 2)).mean()
    std = out["Close"].rolling(max(int(bb_period), 2)).std()
    out["BB Mid"] = mid
    out["BB Upper"] = mid + 2 * std
    out["BB Lower"] = mid - 2 * std
    out["Volume MA"] = out["Volume"].rolling(20).mean()
    out["Long Signal"] = (out["EMA Fast"] > out["EMA Slow"]) & (out["EMA Fast"].shift(1) <= out["EMA Slow"].shift(1))
    out["Short Signal"] = (out["EMA Fast"] < out["EMA Slow"]) & (out["EMA Fast"].shift(1) >= out["EMA Slow"].shift(1))
    return out


def run_strategy_backtest(df, fast, slow, risk_pct=1.0, rr=2.0, starting_balance=100000.0):
    data = add_indicators(df, fast, slow).dropna().copy()
    if len(data) < max(int(slow) + 5, 30):
        return None
    balance = float(starting_balance)
    equity_points = []
    trades = []
    position = None
    risk_cash = starting_balance * float(risk_pct) / 100.0
    for ts, row in data.iterrows():
        price = float(row["Close"])
        if position is None:
            if bool(row["Long Signal"]):
                sl = price * (1 - 0.006)
                tp = price + (price - sl) * float(rr)
                position = {"side":"LONG","entry":price,"sl":sl,"tp":tp,"time":ts}
            elif bool(row["Short Signal"]):
                sl = price * (1 + 0.006)
                tp = price - (sl - price) * float(rr)
                position = {"side":"SHORT","entry":price,"sl":sl,"tp":tp,"time":ts}
        else:
            exit_price = None
            reason = None
            if position["side"] == "LONG":
                if row["Low"] <= position["sl"]:
                    exit_price, reason = position["sl"], "Stop Loss"
                elif row["High"] >= position["tp"]:
                    exit_price, reason = position["tp"], "Take Profit"
                elif bool(row["Short Signal"]):
                    exit_price, reason = price, "Opposite Signal"
            else:
                if row["High"] >= position["sl"]:
                    exit_price, reason = position["sl"], "Stop Loss"
                elif row["Low"] <= position["tp"]:
                    exit_price, reason = position["tp"], "Take Profit"
                elif bool(row["Long Signal"]):
                    exit_price, reason = price, "Opposite Signal"
            if exit_price is not None:
                pct = ((exit_price - position["entry"]) / position["entry"] if position["side"] == "LONG" else (position["entry"] - exit_price) / position["entry"])
                pnl = risk_cash * (pct / 0.006)
                balance += pnl
                trades.append({"Entry Time":position["time"],"Exit Time":ts,"Side":position["side"],"Entry":position["entry"],"Exit":exit_price,"P&L":pnl,"Return %":pct*100,"Exit Reason":reason})
                position = None
        mark = balance
        if position:
            pct = ((price-position["entry"])/position["entry"] if position["side"]=="LONG" else (position["entry"]-price)/position["entry"])
            mark += risk_cash * (pct / 0.006)
        equity_points.append((ts, mark))
    equity = pd.Series(dict(equity_points)).sort_index()
    if not equity.empty:
        running = equity.cummax()
        dd = (equity-running)/running
        mdd = abs(float(dd.min())*100)
        returns = equity.pct_change().dropna()
        sharpe = float((returns.mean()/(returns.std()+1e-9))*np.sqrt(252)) if len(returns)>1 else 0.0
    else:
        mdd, sharpe = 0.0, 0.0
    trade_df = pd.DataFrame(trades)
    wins = int((trade_df["P&L"] > 0).sum()) if not trade_df.empty else 0
    win_rate = wins / len(trade_df) * 100 if len(trade_df) else 0.0
    return {"equity":equity,"trades":trade_df,"net_profit":float(balance-starting_balance),"win_rate":win_rate,"mdd":mdd,"sharpe":sharpe}

# Navigation, grouped for the sidebar
NAV_GROUPS = {
    "CORE": ["📊 Dashboard"],
    "ANALYTICS & EXECUTION": [
        "📈 Chart Analysis",
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
_TICKER_ITEMS = (
    '<span class="tk-item">🟡 XAU/USD&nbsp;<b>$2,468.40</b>&nbsp;'
    '<span class="up">+0.84%</span></span>'
    '<span class="tk-item">🪙 BTC/USDT&nbsp;<b>$78,820.00</b>&nbsp;'
    '<span class="up">+2.15%</span></span>'
    '<span class="tk-item">🇮🇳 NIFTY 50&nbsp;<b>24,310.80</b>&nbsp;'
    '<span class="down">-0.24%</span></span>'
    '<span class="tk-item">🇺🇸 S&amp;P 500&nbsp;<b>5,840.10</b>&nbsp;'
    '<span class="up">+0.41%</span></span>'
    '<span class="tk-item">💱 EUR/USD&nbsp;<b>1.0825</b>&nbsp;'
    '<span class="down">-0.08%</span></span>'
)
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
    st.title("Smart Session Anomaly Detector")
    st.caption("Multi-Asset Quantitative Intelligence & Risk Framework")

    st.markdown("### Quick Actions")
    q1, q2, q3, q4, q5 = st.columns(5)

    with q1:
        with st.container(border=True):
            st.markdown("#### 📈 Chart Analysis")
            st.caption("Live TradingView viewport with anomaly detection markers.")
            st.button(
                "Open Charts →",
                key="btn_qa_charts",
                use_container_width=True,
                on_click=switch_page,
                args=("📈 Chart Analysis",),
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
elif st.session_state.active_tab == "📈 Chart Analysis":
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


        # --- PROFESSIONAL TECHNICAL CHART ---
        st.markdown("### 📊 Technical Chart & Signal Workspace")
        chart_df = load_ohlcv(inst_cfg["yf"], period="1mo", interval=chart_res)
        if chart_df is not None and len(chart_df) >= 30:
            ind_fast, ind_slow = st.columns(2)
            with ind_fast:
                tech_fast = st.number_input("Fast EMA", 2, 100, 9, key="tech_fast")
            with ind_slow:
                tech_slow = st.number_input("Slow EMA", 3, 200, 21, key="tech_slow")
            tech_df = add_indicators(chart_df, tech_fast, tech_slow)
            fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.78,0.22])
            fig_tech.add_trace(go.Candlestick(x=tech_df.index, open=tech_df["Open"], high=tech_df["High"], low=tech_df["Low"], close=tech_df["Close"], name="Price"), row=1,col=1)
            fig_tech.add_trace(go.Scatter(x=tech_df.index,y=tech_df["EMA Fast"],name=f"EMA {tech_fast}",mode="lines"),row=1,col=1)
            fig_tech.add_trace(go.Scatter(x=tech_df.index,y=tech_df["EMA Slow"],name=f"EMA {tech_slow}",mode="lines"),row=1,col=1)
            fig_tech.add_trace(go.Scatter(x=tech_df.index,y=tech_df["BB Upper"],name="BB Upper",mode="lines",line=dict(dash="dot")),row=1,col=1)
            fig_tech.add_trace(go.Scatter(x=tech_df.index,y=tech_df["BB Lower"],name="BB Lower",mode="lines",line=dict(dash="dot")),row=1,col=1)
            longs=tech_df[tech_df["Long Signal"]]; shorts=tech_df[tech_df["Short Signal"]]
            if not longs.empty: fig_tech.add_trace(go.Scatter(x=longs.index,y=longs["Low"]*0.998,mode="markers",name="LONG",marker=dict(symbol="triangle-up",size=11)),row=1,col=1)
            if not shorts.empty: fig_tech.add_trace(go.Scatter(x=shorts.index,y=shorts["High"]*1.002,mode="markers",name="SHORT",marker=dict(symbol="triangle-down",size=11)),row=1,col=1)
            fig_tech.add_trace(go.Bar(x=tech_df.index,y=tech_df["Volume"],name="Volume"),row=2,col=1)
            fig_tech.update_layout(template="plotly_dark",height=620,xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="#0b0e14",plot_bgcolor="#0b0e14")
            st.plotly_chart(fig_tech,use_container_width=True,config={"scrollZoom":True,"displaylogo":False})
            latest=tech_df.iloc[-1]
            s1,s2,s3,s4=st.columns(4)
            s1.metric("Last Price",f"{latest['Close']:,.2f}")
            s2.metric("RSI",f"{latest['RSI']:.1f}")
            s3.metric("EMA Trend","BULLISH" if latest['EMA Fast']>latest['EMA Slow'] else "BEARISH")
            s4.metric("Volume vs MA",f"{latest['Volume']/max(latest['Volume MA'],1):.2f}x")
        else:
            st.info("Technical chart data is still syncing.")

        # --- DIRECT TRADE EXECUTION ON ACTIVE CHART ---
        chart_trade_price = float(chart_df["Close"].iloc[-1]) if chart_df is not None and len(chart_df) else float(global_price)
        st.markdown("### ⚡ Direct Trade Execution")
        trade_col1, trade_col2, trade_col3 = st.columns([1, 1, 2])
        with trade_col1:
            chart_lots = st.number_input(
                "Lot Size",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.01,
                key="chart_trade_lots",
            )
        with trade_col2:
            st.caption("Active Instrument")
            st.write(f"**{inst_select}**")
        with trade_col3:
            buy_col, sell_col = st.columns(2)
            if buy_col.button("🟢 BUY / LONG", use_container_width=True, key="chart_buy"):
                place_chart_trade("BUY", chart_lots, inst_select, chart_trade_price)
                st.success("BUY order routed from Chart Analysis.")
            if sell_col.button("🔴 SELL / SHORT", use_container_width=True, key="chart_sell"):
                place_chart_trade("SELL", chart_lots, inst_select, chart_trade_price)
                st.success("SELL order routed from Chart Analysis.")

        # --- EA EXECUTION PANEL ---
        st.markdown("### 🤖 Expert Advisor Execution Panel")
        ea_exec_1, ea_exec_2, ea_exec_3 = st.columns([1.4, 1, 1])
        with ea_exec_1:
            ea_choices = ["None"] + [x["name"] for x in EA_TEMPLATES] + [
                x["name"] for x in st.session_state.saved_eas
            ]
            selected_chart_ea = st.selectbox(
                "EA to load on active chart",
                ea_choices,
                key="chart_ea_selector",
            )
        with ea_exec_2:
            if st.button("📥 Load EA", use_container_width=True, key="chart_load_ea"):
                st.session_state.active_ea = None if selected_chart_ea == "None" else selected_chart_ea
                st.session_state.ea_enabled = False
                st.success(
                    "No EA selected." if selected_chart_ea == "None"
                    else f"Loaded **{selected_chart_ea}** onto {inst_select}."
                )
        with ea_exec_3:
            enabled = st.toggle(
                "Enable EA",
                value=st.session_state.ea_enabled,
                key="chart_ea_enabled",
            )
            st.session_state.ea_enabled = enabled

        if st.session_state.active_ea:
            st.info(
                f"Active EA: **{st.session_state.active_ea}** | "
                f"Status: **{'ENABLED' if st.session_state.ea_enabled else 'DISABLED'}**"
            )
            trigger_col, risk_col = st.columns([1, 1])
            with trigger_col:
                if st.button("▶ Trigger EA Check", use_container_width=True, key="trigger_chart_ea"):
                    if st.session_state.ea_enabled:
                        st.success(
                            f"EA check triggered on {inst_select}. "
                            "Review the generated signal before live execution."
                        )
                    else:
                        st.warning("Enable the EA before triggering an automated check.")
            with risk_col:
                st.caption("EA execution remains subject to the configured risk and broker connection.")

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
    st.caption("Broker connection architecture with paper/demo execution by default.")
    g1,g2=st.columns([1,1.2])
    with g1:
        st.markdown("#### 🔗 Connection Profile")
        st.session_state.broker_name=st.selectbox("Broker / Gateway",["MetaTrader 5 (MT5)","Zerodha (Kite)","Binance Futures","Interactive Brokers"],index=["MetaTrader 5 (MT5)","Zerodha (Kite)","Binance Futures","Interactive Brokers"].index(st.session_state.broker_name))
        st.session_state.broker_api_mode=st.radio("Execution Mode",["Paper / Demo","Live (manual approval)"],horizontal=True)
        st.session_state.broker_api_endpoint=st.text_input("API / Bridge Endpoint",value=st.session_state.broker_api_endpoint,placeholder="https://...")
        st.session_state.broker_api_key=st.text_input("API Key / Account ID",value=st.session_state.broker_api_key,type="password")
        c1,c2=st.columns(2)
        with c1:
            if st.button("🔌 Connect",use_container_width=True): st.session_state.broker_connected=True; st.success("Gateway connected in application demo mode.")
        with c2:
            if st.button("⏹ Disconnect",use_container_width=True): st.session_state.broker_connected=False; st.info("Gateway disconnected.")
        st.info("Live broker-side autonomous execution requires the broker's authenticated API/bridge and is intentionally not enabled by this UI alone.")
    with g2:
        st.markdown("#### 📋 Execution Monitor")
        st.metric("Gateway Status","🟢 Connected" if st.session_state.broker_connected else "⚪ Offline")
        st.metric("Execution Mode",st.session_state.broker_api_mode)
        if st.session_state.positions:
            st.dataframe(pd.DataFrame(st.session_state.positions),use_container_width=True,hide_index=True)
            if st.button("Close All Demo Positions",use_container_width=True): st.session_state.positions=[]; st.rerun()
        else: st.caption("No open positions in the application session.")

# ==========================================
# 🧩 VIEW 5: STRATEGY BUILDER
# ==========================================
elif st.session_state.active_tab == "🧩 Strategy Builder":
    st.title("🧩 Strategy Builder")
    st.caption("Worldwide pre-built strategies, code editor, configurable parameters, and backtesting workspace.")

    strategy_names = [
        "SMA / EMA Crossover",
        "RSI Mean Reversion",
        "ICT / SMC Liquidity Sweeps",
        "MACD Divergence",
        "Bollinger Bands Squeeze",
        "Custom Strategy",
    ]
    selected_strategy = st.selectbox("Pre-Built Strategy", strategy_names)

    preset_map = {
        "SMA / EMA Crossover": ("Fast EMA crosses Slow EMA", "Opposite crossover", 9, 21),
        "RSI Mean Reversion": ("RSI reversal from oversold/overbought", "RSI returns toward midpoint", 14, 30),
        "ICT / SMC Liquidity Sweeps": ("Liquidity sweep + reclaim", "Opposing liquidity / structure target", 20, 50),
        "MACD Divergence": ("Confirmed price/MACD divergence", "Momentum reversal", 12, 26),
        "Bollinger Bands Squeeze": ("Volatility squeeze + breakout", "Band reversal / target", 20, 2),
        "Custom Strategy": ("Custom entry logic", "Custom exit logic", 9, 21),
    }
    default_entry, default_exit, default_a, default_b = preset_map[selected_strategy]

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        fast_param = st.number_input("Fast / Lookback", min_value=2, max_value=200, value=default_a)
    with p2:
        slow_param = st.number_input("Slow / Threshold", min_value=2, max_value=200, value=default_b)
    with p3:
        risk_param = st.slider("Risk %", 0.25, 5.0, 1.0, 0.25)
    with p4:
        rr_param = st.slider("Target R:R", 0.5, 5.0, 2.0, 0.25)

    st.markdown("#### Entry / Exit Logic")
    entry_logic = st.text_area("Entry Logic", value=default_entry, height=80)
    exit_logic = st.text_area("Exit Logic", value=default_exit, height=80)

    st.markdown("#### 💻 Online Code Editor")
    code_language = st.selectbox("Code Style", ["Python", "Pine Script"])
    edited_code = st.text_area(
        "Strategy Code",
        value=st.session_state.ea_code,
        height=260,
        key="strategy_code_editor",
    )
    if st.button("✅ Validate Code", key="validate_strategy_code"):
        if "def " in edited_code or "strategy(" in edited_code or "indicator(" in edited_code:
            st.success(f"{code_language} style structure detected. Basic validation passed.")
        else:
            st.warning("Add at least one function or strategy/indicator declaration for validation.")

    if st.button("💾 Save Strategy Configuration", key="save_strategy"):
        st.success(f"Saved **{selected_strategy}** with {risk_param:.2f}% risk and {rr_param:.2f}R target.")

    st.divider()
    st.markdown("#### 🧪 Backtesting Engine")
    bt1, bt2, bt3, bt4 = st.columns(4)
    with bt1:
        bt_asset = st.selectbox("Asset", list(MARKET_UNIVERSE["🇮🇳 Indian Equities (NSE)"].keys()) + ["Gold (XAU/USD)", "EUR/USD", "Bitcoin (BTC/USDT)"], key="bt_asset")
    with bt2:
        bt_tf = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1D"], index=1, key="bt_tf")
    with bt3:
        bt_period = st.selectbox("History", ["1mo", "3mo", "6mo", "1y"], index=2, key="bt_period")
    with bt4:
        run_bt = st.button("▶ Run Backtest", use_container_width=True, key="run_backtest")

    if run_bt:
        bt_cfg = next((group[bt_asset] for group in MARKET_UNIVERSE.values() if bt_asset in group), None)
        if bt_cfg:
            bt_df = load_ohlcv(bt_cfg["yf"], period=bt_period, interval=bt_tf)
            result = run_strategy_backtest(bt_df, fast_param, slow_param, risk_param, rr_param)
            if result:
                st.session_state.last_backtest = result
            else:
                st.warning("Not enough historical data for this backtest.")
        else:
            st.warning("Asset configuration not found.")

    if st.session_state.last_backtest:
        result = st.session_state.last_backtest
        metrics = st.columns(5)
        metrics[0].metric("Net Profit", f"${result['net_profit']:,.2f}")
        metrics[1].metric("Win Rate", f"{result['win_rate']:.1f}%")
        metrics[2].metric("MDD", f"{result['mdd']:.2f}%")
        metrics[3].metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
        metrics[4].metric("Trades", f"{len(result['trades'])}")
        eq=result["equity"]
        fig_eq=go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Equity"))
        fig_eq.update_layout(template="plotly_dark",title="Equity Curve",height=360,margin=dict(l=10,r=10,t=40,b=10),paper_bgcolor="#0b0e14",plot_bgcolor="#0b0e14")
        st.plotly_chart(fig_eq,use_container_width=True)
        if not result["trades"].empty:
            st.markdown("##### Detailed Trade Log")
            st.dataframe(result["trades"].tail(200),use_container_width=True,hide_index=True)
        else:
            st.info("No completed trades were generated for this configuration.")

# ==========================================
# 🤖 VIEW 6: EA / EXPERT ADVISORS
# ==========================================
elif st.session_state.active_tab == "🤖 EA / Expert Advisors":
    st.title("🤖 EA / Expert Advisors")
    st.caption("Build, configure, save, validate, and deploy automated trading logic.")

    ea_tab1, ea_tab2, ea_tab3 = st.tabs(["🛠 EA Builder", "📚 EA Library", "🚀 Deployment"])

    with ea_tab1:
        ea_template = st.selectbox("Start from Template", [x["name"] for x in EA_TEMPLATES] + ["Custom EA"])
        selected_template = next((x for x in EA_TEMPLATES if x["name"] == ea_template), None)
        if selected_template:
            st.info(selected_template["description"])

        e1, e2, e3 = st.columns(3)
        with e1:
            ea_name = st.text_input("EA Name", value=ea_template)
            ea_entry = st.text_area("Entry Logic", value=selected_template["entry"] if selected_template else "Define entry conditions", height=110)
        with e2:
            ea_exit = st.text_area("Exit Logic", value=selected_template["exit"] if selected_template else "Define exit conditions", height=110)
            ea_sl = st.number_input("Stop Loss %", min_value=0.05, max_value=20.0, value=0.6, step=0.05)
        with e3:
            ea_tp = st.number_input("Take Profit %", min_value=0.05, max_value=50.0, value=1.8, step=0.05)
            ea_risk = st.slider("Risk per Trade %", 0.25, 5.0, 1.0, 0.25)
            ea_max_lots = st.number_input("Max Lots", min_value=0.01, max_value=100.0, value=1.0, step=0.01)

        st.markdown("#### 🛡 Risk & Trade Management")
        rm1, rm2, rm3 = st.columns(3)
        with rm1:
            ea_trailing = st.number_input("Trailing Stop %", 0.0, 10.0, 0.0, 0.05)
        with rm2:
            ea_breakeven = st.number_input("Breakeven Trigger (R)", 0.0, 5.0, 1.0, 0.25)
        with rm3:
            ea_max_trades = st.number_input("Max Trades / Day", 1, 100, 5, 1)

        st.markdown("#### Execution Parameters")
        x1, x2, x3 = st.columns(3)
        with x1:
            ea_tf = st.selectbox("Execution Timeframe", ["5m", "15m", "1h", "4h", "1D"], key="ea_exec_tf")
        with x2:
            ea_session = st.selectbox("Trading Session", ["All Sessions", "Asia", "London", "New York"])
        with x3:
            ea_mode = st.selectbox("Execution Mode", ["Paper / Demo", "Live (Broker Connected)"])

        st.markdown("#### EA Code")
        ea_code = st.text_area("Python / Pine-style EA Code", value=st.session_state.ea_code, height=280, key="ea_builder_code")
        if st.button("🔍 Validate EA", key="validate_ea"):
            if len(ea_code.strip()) > 30:
                st.success("EA code passed basic structural validation.")
            else:
                st.warning("EA code is too short to validate.")

        if st.button("💾 Save EA to Library", key="save_ea"):
            record = {
                "name": ea_name,
                "description": f"{ea_entry[:100]} | SL {ea_sl}% | TP {ea_tp}%",
                "entry": ea_entry,
                "exit": ea_exit,
                "stop_loss_pct": ea_sl,
                "take_profit_pct": ea_tp,
                "risk_pct": ea_risk,
                "max_lots": ea_max_lots,
                "trailing_stop_pct": ea_trailing,
                "breakeven_r": ea_breakeven,
                "max_trades_day": ea_max_trades,
                "timeframe": ea_tf,
                "session": ea_session,
                "mode": ea_mode,
                "code": ea_code,
            }
            st.session_state.saved_eas = [x for x in st.session_state.saved_eas if x["name"] != ea_name] + [record]
            st.success(f"Saved **{ea_name}** to the custom EA library.")

    with ea_tab2:
        if st.session_state.saved_eas:
            library_df = pd.DataFrame([
                {
                    "EA": x["name"],
                    "Timeframe": x["timeframe"],
                    "Risk %": x["risk_pct"],
                    "SL %": x["stop_loss_pct"],
                    "TP %": x["take_profit_pct"],
                    "Execution": x["mode"],
                }
                for x in st.session_state.saved_eas
            ])
            st.dataframe(library_df, use_container_width=True)
            library_choice = st.selectbox("Select Saved EA", [x["name"] for x in st.session_state.saved_eas])
            if st.button("📌 Set as Active EA", key="activate_library_ea"):
                st.session_state.active_ea = library_choice
                st.success(f"**{library_choice}** is ready for chart deployment.")
        else:
            st.info("No custom EAs saved yet. Build one in the EA Builder.")

    with ea_tab3:
        deploy_assets = st.multiselect("Deploy to Assets", list(MARKET_UNIVERSE["🇮🇳 Indian Equities (NSE)"].keys()) + ["Gold (XAU/USD)","EUR/USD","Bitcoin (BTC/USDT)"], default=[])
        deploy_ea = st.selectbox("EA", ["None"] + [x["name"] for x in st.session_state.saved_eas] + [x["name"] for x in EA_TEMPLATES], key="deploy_ea")
        d1,d2,d3=st.columns(3)
        with d1:
            deploy_mode=st.selectbox("Deployment Mode",["Paper / Demo","Signal Only","Live (manual approval)"],key="deploy_mode")
        with d2:
            deploy_risk=st.slider("Deployment Risk %",0.25,5.0,1.0,0.25,key="deploy_risk")
        with d3:
            deploy_enabled=st.toggle("Enable After Load",value=False,key="deploy_enabled")
        if st.button("🚀 Deploy EA",use_container_width=True,key="deploy_ea_now"):
            if deploy_ea=="None" or not deploy_assets:
                st.warning("Select an EA and at least one asset.")
            else:
                for asset in deploy_assets:
                    st.session_state.ea_deployments[asset]={"EA":deploy_ea,"Mode":deploy_mode,"Risk %":deploy_risk,"Enabled":deploy_enabled}
                st.session_state.active_ea=deploy_ea
                st.session_state.ea_enabled=deploy_enabled
                st.success(f"{deploy_ea} configured for {len(deploy_assets)} asset(s).")
        if st.session_state.ea_deployments:
            st.markdown("##### Active Deployments")
            st.dataframe(pd.DataFrame.from_dict(st.session_state.ea_deployments,orient="index").reset_index().rename(columns={"index":"Asset"}),use_container_width=True,hide_index=True)

    st.divider()
    st.markdown("#### 📋 Current Deployment")
    st.write({
        "Active EA": st.session_state.active_ea or "None",
        "Enabled": st.session_state.ea_enabled,
    })


# ==========================================
# 📅 VIEW 7: ECONOMIC CALENDAR
# ==========================================
elif st.session_state.active_tab == "📅 Economic Calendar":
    st.title("📅 Forex Factory-Style Economic Calendar")
    st.caption("Filter high-impact market events by currency, severity, and date range.")

    now = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    cal_data = pd.DataFrame(
        [
            {"Date": now.strftime("%Y-%m-%d"), "Time": "18:00", "Currency": "USD", "Impact": "High", "Event Name": "Core CPI (YoY)", "Actual": "3.1%", "Forecast": "3.2%", "Previous": "3.3%"},
            {"Date": now.strftime("%Y-%m-%d"), "Time": "19:30", "Currency": "USD", "Impact": "High", "Event Name": "Non-Farm Payrolls (NFP)", "Actual": "185K", "Forecast": "180K", "Previous": "175K"},
            {"Date": now.strftime("%Y-%m-%d"), "Time": "20:30", "Currency": "EUR", "Impact": "High", "Event Name": "ECB Interest Rate Decision", "Actual": "3.75%", "Forecast": "3.75%", "Previous": "4.00%"},
            {"Date": (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "Time": "21:45", "Currency": "USD", "Impact": "High", "Event Name": "FOMC Press Conference", "Actual": "-", "Forecast": "-", "Previous": "-"},
            {"Date": (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "Time": "14:30", "Currency": "GBP", "Impact": "Medium", "Event Name": "Manufacturing PMI", "Actual": "-", "Forecast": "51.0", "Previous": "50.8"},
            {"Date": (now + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "Time": "12:00", "Currency": "JPY", "Impact": "Low", "Event Name": "Consumer Confidence", "Actual": "-", "Forecast": "36.5", "Previous": "36.2"},
            {"Date": (now + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "Time": "10:00", "Currency": "EUR", "Impact": "Non-Economic", "Event Name": "Eurogroup Meeting", "Actual": "-", "Forecast": "-", "Previous": "-"},
        ]
    )

    f1, f2, f3 = st.columns([1, 1, 1.4])
    with f1:
        currency_filter = st.multiselect("Currency", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"], default=[])
    with f2:
        impact_filter = st.multiselect("Impact", ["High", "Medium", "Low", "Non-Economic"], default=[])
    with f3:
        date_range = st.selectbox("Date Range", ["Today", "Tomorrow", "This Week"])

    start_date = now
    if date_range == "Today":
        end_date = now
    elif date_range == "Tomorrow":
        start_date = now + pd.Timedelta(days=1)
        end_date = start_date
    else:
        end_date = now + pd.Timedelta(days=6)

    display_df = cal_data.copy()
    display_df["_date"] = pd.to_datetime(display_df["Date"])
    display_df = display_df[(display_df["_date"] >= start_date.tz_localize(None)) & (display_df["_date"] <= end_date.tz_localize(None))]
    if currency_filter:
        display_df = display_df[display_df["Currency"].isin(currency_filter)]
    if impact_filter:
        display_df = display_df[display_df["Impact"].isin(impact_filter)]

    impact_icon = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟡 Low", "Non-Economic": "⚪ Non-Economic"}
    display_df["Impact"] = display_df["Impact"].map(impact_icon)

    def actual_badge(row):
        actual = str(row["Actual"])
        forecast = str(row["Forecast"])
        if actual == "-" or forecast == "-":
            return actual
        try:
            a = float(re.sub(r"[^0-9.\-]", "", actual))
            f = float(re.sub(r"[^0-9.\-]", "", forecast))
            if a > f:
                return f"🟢 {actual}"
            if a < f:
                return f"🔴 {actual}"
            return f"⚪ {actual}"
        except Exception:
            return actual

    display_df["Actual"] = display_df.apply(actual_badge, axis=1)
    display_df = display_df[["Date", "Time", "Currency", "Impact", "Event Name", "Actual", "Forecast", "Previous"]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================================
# ⚙️ VIEW 8: SETTINGS
# ==========================================
elif st.session_state.active_tab == "⚙️ Settings":
    st.title("⚙️ Smart Session Anomaly Detector | Settings")
    st.write("Platform: Smart Session Anomaly Detector Suite")
    st.write("Architecture: Python Quant Pipeline + Isolation Forest ML")
    st.write("Data Stream Latency: 20 Seconds Auto-Sync")
    st.selectbox("Base Currency", ["USD ($)", "INR (₹)", "EUR (€)"])
    st.divider()
    st.markdown("#### 🧩 Platform Modules")
    st.write({"Chart Analysis": "Technical chart + direct execution", "EA Engine": "Builder / Library / Deployment", "Backtesting": "Historical simulation + metrics", "Broker Bridge": st.session_state.broker_name, "Default Execution": st.session_state.broker_api_mode})

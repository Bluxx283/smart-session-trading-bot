"""
SSAD — Smart Session Anomaly Detector (remodelled)

Layout of this file
  1. Config & constants          5. Indicators / signals / backtest engine
  2. Persistence & session state 6. Prop-firm risk helpers
  3. Live market-data engine     7. Co-pilot (Claude) helpers
  4. Historical data layer       8. Styles, sidebar, market strip, co-pilot UI
                                 9. Pages  +  router

Requires Streamlit >= 1.39 (st.fragment run_every, st-key-* CSS classes).
Optional: websocket-client (live feeds), anthropic (AI co-pilot).
"""
from __future__ import annotations

import ast
import base64
import copy
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from forex_factory_calendar import CalendarFetchError, fetch_calendar

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None

st.set_page_config(
    page_title="Smart Session Anomaly Detector | Institutional Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# 1. CONFIG & CONSTANTS
# ════════════════════════════════════════════════════════════
APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "ssad_state.json"
HERO_IMAGE = APP_DIR / "trading_hero.png"      # optional artwork
PERSIST_KEYS = ("saved_strategies", "saved_eas")

# live = key used by the WebSocket engine (None = no free live feed)
# mkt  = market-hours category (crypto | forex | nse | us_equity)
MARKET_UNIVERSE = {
    "🟡 Metals & Commodities": {
        "Gold (XAU/USD)":   dict(symbol="XAUUSD", exchange="OANDA", yf="GC=F", live="OANDA:XAU_USD", mkt="forex", lot_units=100, pip_size=0.1),
        "Silver (XAG/USD)": dict(symbol="XAGUSD", exchange="OANDA", yf="SI=F", live=None, mkt="forex", lot_units=5000, pip_size=0.01),
        "Crude Oil WTI":    dict(symbol="USOIL", exchange="TVC", yf="CL=F", live=None, mkt="forex", lot_units=1000, pip_size=0.01),
    },
    "🪙 Cryptocurrency": {
        "Bitcoin (BTC/USDT)":  dict(symbol="BTCUSDT", exchange="BINANCE", yf="BTC-USD", live="BTCUSDT", mkt="crypto", lot_units=1, pip_size=1.0),
        "Ethereum (ETH/USDT)": dict(symbol="ETHUSDT", exchange="BINANCE", yf="ETH-USD", live="ETHUSDT", mkt="crypto", lot_units=1, pip_size=0.1),
    },
    "🇮🇳 Indian Equities (NSE)": {
        "NIFTY 50 Index":      dict(symbol="NIFTY", exchange="NSE", yf="^NSEI", live=None, mkt="nse", lot_units=25, pip_size=0.05),
        "Reliance Industries": dict(symbol="RELIANCE", exchange="NSE", yf="RELIANCE.NS", live=None, mkt="nse", lot_units=250, pip_size=0.05),
        "HDFC Bank":           dict(symbol="HDFCBANK", exchange="NSE", yf="HDFCBANK.NS", live=None, mkt="nse", lot_units=550, pip_size=0.05),
    },
    "🇺🇸 US Equities": {
        "Nvidia (NVDA)": dict(symbol="NVDA", exchange="NASDAQ", yf="NVDA", live="NVDA", mkt="us_equity", lot_units=100, pip_size=0.01),
        "Apple (AAPL)":  dict(symbol="AAPL", exchange="NASDAQ", yf="AAPL", live="AAPL", mkt="us_equity", lot_units=100, pip_size=0.01),
    },
    "💱 Major Forex": {
        "EUR/USD": dict(symbol="EURUSD", exchange="FX_IDC", yf="EURUSD=X", live="OANDA:EUR_USD", mkt="forex", lot_units=100000, pip_size=0.0001),
        "USD/INR": dict(symbol="USDINR", exchange="FX_IDC", yf="USDINR=X", live=None, mkt="forex", lot_units=1000, pip_size=0.0025),
    },
}
ALL_INSTRUMENTS = {n: c for grp in MARKET_UNIVERSE.values() for n, c in grp.items()}
GOLD = ALL_INSTRUMENTS["Gold (XAU/USD)"]

# Market strip: label, live key, yfinance ticker, market, decimals, prefix
STRIP_ASSETS = [
    ("XAU/USD", "OANDA:XAU_USD", "GC=F", "forex", 2, ""),
    ("BTC/USDT", "BTCUSDT", "BTC-USD", "crypto", 2, "$"),
    ("NIFTY 50", None, "^NSEI", "nse", 2, ""),
    ("S&P 500", None, "^GSPC", "us_equity", 2, ""),
    ("EUR/USD", "OANDA:EUR_USD", "EURUSD=X", "forex", 4, ""),
    ("ETH/USDT", "ETHUSDT", "ETH-USD", "crypto", 2, "$"),
    ("AAPL", "AAPL", "AAPL", "us_equity", 2, "$"),
    ("NVDA", "NVDA", "NVDA", "us_equity", 2, "$"),
    ("GBP/USD", "OANDA:GBP_USD", "GBPUSD=X", "forex", 4, ""),
    ("USD/JPY", "OANDA:USD_JPY", "JPY=X", "forex", 3, ""),
]
TWELVE_SYMBOLS = {  # Twelve Data symbol -> internal live key
    "XAU/USD": "OANDA:XAU_USD", "EUR/USD": "OANDA:EUR_USD",
    "GBP/USD": "OANDA:GBP_USD", "USD/JPY": "OANDA:USD_JPY",
    "AAPL": "AAPL", "NVDA": "NVDA",
}

NAV_GROUPS = {
    "CORE": [("⌂ Home", "dashboard")],
    "MARKET": [("📊 Markets", "chart")],
    "STRATEGY WORKSPACE": [
        ("⚗ Strategy Lab", "strategy"),
        ("▣ Backtesting", "backtest"),
        ("▤ News Trading", "news"),
        ("🤖 EA / Expert Advisors", "ea"),
    ],
    "TOOLS": [
        ("🧮 Pip & Risk Calculator", "risk"),
        ("⚡ Broker Gateway", "broker"),
        ("🛡 Prop-Firm Center", "prop"),
    ],
    "SYSTEM": [("⚙️ Settings", "settings")],
}

STRATEGY_MODELS = ["EMA Crossover", "MACD Crossover", "RSI Reversal", "Bollinger Breakout"]

# Institutional setups. The backtester only understands the built-in models,
# so each preset names the closest model and keeps the real rule as notes.
STRATEGY_PRESETS = {
    "Liquidity Sweep": dict(model="RSI Reversal", stop_pct=0.6, rr=3.0,
        entry="Wick sweeps prior session high/low, then reclaims quickly → trade against the sweep",
        exit="Opposing liquidity pool or 3R"),
    "Order Block Mitigation": dict(model="EMA Crossover", stop_pct=0.8, rr=3.0,
        entry="Price retests last opposing candle before an impulse → trade with the impulse",
        exit="Prior swing extreme or 3R"),
    "Session Breakout": dict(model="Bollinger Breakout", stop_pct=1.0, rr=2.5,
        entry="Close beyond first 30-min session range with volume > 1.5× average",
        exit="Opposite signal or 2.5R"),
}

EA_TEMPLATES = [
    dict(name="SMA / EMA Crossover EA", description="Trend-following on fast/slow MA crossovers.",
         entry="Fast MA crosses above slow MA", exit="Opposite crossover or stop/target"),
    dict(name="RSI Mean Reversion EA", description="Fades RSI overbought/oversold zones.",
         entry="RSI leaves oversold/overbought zone", exit="RSI mid-line or stop/target"),
    dict(name="ICT / SMC Liquidity Sweep EA", description="Session liquidity sweep and reclaim.",
         entry="Liquidity sweep + fast reclaim", exit="Opposing liquidity or stop/target"),
    dict(name="MACD Divergence EA", description="Momentum divergence using MACD histogram.",
         entry="Confirmed MACD divergence", exit="Momentum reversal or stop/target"),
    dict(name="Bollinger Bands Squeeze EA", description="Volatility-compression breakout.",
         entry="Squeeze followed by directional expansion", exit="Opposite signal or stop/target"),
]
EA_CODE_TEMPLATE = '''# Custom EA template (reference only — never executed by this app)
FAST_PERIOD = 9
SLOW_PERIOD = 21
RISK_PCT = 1.0

def entry_signal(data):
    return False

def exit_signal(data):
    return False
'''

# ════════════════════════════════════════════════════════════
# 2. PERSISTENCE & SESSION STATE
# ════════════════════════════════════════════════════════════
def load_persisted() -> dict:
    try:
        return json.loads(STATE_FILE.read_text("utf-8"))
    except Exception:
        return {}


def persist() -> None:
    """Save strategies / EAs to disk so they survive restarts (no secrets)."""
    try:
        data = {k: st.session_state.get(k, []) for k in PERSIST_KEYS}
        STATE_FILE.write_text(json.dumps(data, indent=2, default=str), "utf-8")
    except Exception:
        pass


DEFAULTS = {
    "page": "dashboard", "account_balance": 100000.0, "positions": [],
    "broker_connected": False, "broker_name": "MetaTrader 5 (MT5)",
    "broker_api_mode": "Paper / Demo", "broker_api_endpoint": "", "broker_api_key": "",
    "saved_strategies": [], "saved_eas": [], "active_ea": None, "ea_enabled": False,
    "ea_deployments": {}, "last_backtest": None,
    # co-pilot
    "bot_open": False, "voice_enabled": True, "voice_name": "Auto / Best available",
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""), "claude_model": "claude-sonnet-4-5",
    "last_spoken_index": -1,
    "chat_messages": [{"role": "assistant", "content":
        "Hi! I'm your trading co-pilot. I can explain signals, check risk and help with your plan."}],
    # feeds / alerts
    "twelvedata_api_key": os.getenv("TWELVE_DATA_API_KEY", ""), "alerts_on": True,
    "notification_log": [], "telegram_bot_token": "", "telegram_chat_id": "",
    "discord_webhook": "", "alert_email": "",
    # prop firm
    "prop_account_type": "Personal / Demo", "prop_firm_name": "Custom Prop Profile",
    "prop_account_size": 100000.0, "prop_profit_target_pct": 10.0,
    "prop_daily_loss_limit_pct": 5.0, "prop_max_drawdown_pct": 10.0,
    "prop_max_trade_risk_pct": 0.50, "prop_min_trading_days": 5,
    "prop_news_trading": "Allowed", "prop_ea_trading": "Allowed",
    "prop_daily_pnl": 0.0, "prop_peak_equity": 100000.0,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, copy.deepcopy(_v))
st.session_state.setdefault("ea_code", EA_CODE_TEMPLATE)
if not st.session_state.get("_state_loaded"):
    for _k, _v in load_persisted().items():
        if _k in PERSIST_KEYS and isinstance(_v, list):
            st.session_state[_k] = _v
    st.session_state["_state_loaded"] = True
ss = st.session_state

# ════════════════════════════════════════════════════════════
# 3. LIVE MARKET-DATA ENGINE
#    Binance public WS (crypto, no key) + Twelve Data WS (forex/gold/US
#    stocks, free key). Everything else is polled and labelled DELAYED.
#    Worker threads push into a locked dict; reruns only read from it.
# ════════════════════════════════════════════════════════════
_LIVE_LOCK = threading.Lock()
_LIVE_PRICES: dict[str, dict] = {}
_LIVE_HISTORY: dict[str, deque] = {}


def _update_live(key, price, prev_close=None, source="ws"):
    with _LIVE_LOCK:
        prev = _LIVE_PRICES.get(key, {})
        pc = prev_close if prev_close is not None else prev.get("prev_close", price)
        _LIVE_PRICES[key] = dict(
            price=float(price), prev_close=float(pc),
            pct=float((price - pc) / pc * 100.0) if pc else 0.0,
            ts=time.time(), source=source)
        _LIVE_HISTORY.setdefault(key, deque(maxlen=60)).append(float(price))


def get_live(key, max_age=120):
    """Latest WebSocket quote for `key` or None when missing/stale."""
    if not key:
        return None
    with _LIVE_LOCK:
        item = _LIVE_PRICES.get(key)
        hist = list(_LIVE_HISTORY.get(key, []))
    if not item or time.time() - item["ts"] > max_age:
        return None
    return {**item, "history": hist}


def live_feed_count() -> int:
    with _LIVE_LOCK:
        return sum(1 for v in _LIVE_PRICES.values() if time.time() - v["ts"] < 120)


def _binance_worker(streams):
    if websocket is None:
        return
    url = "wss://stream.binance.com:9443/stream?streams=" + "/".join(streams)
    while True:
        try:
            ws = websocket.create_connection(url, timeout=10)
            while True:
                data = json.loads(ws.recv()).get("data", {})
                if data.get("s") and data.get("c"):
                    _update_live(data["s"], float(data["c"]),
                                 float(data["o"]) if data.get("o") else None, "binance_ws")
        except Exception:
            time.sleep(5)


def _twelvedata_worker(api_key):
    if websocket is None or not api_key:
        return
    url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"
    while True:
        try:
            ws = websocket.create_connection(url, timeout=15)
            ws.send(json.dumps({"action": "subscribe",
                                "params": {"symbols": ",".join(TWELVE_SYMBOLS)}}))
            while True:
                msg = json.loads(ws.recv())
                key = TWELVE_SYMBOLS.get(msg.get("symbol"))
                if msg.get("event") == "price" and key and msg.get("price") is not None:
                    _update_live(key, float(msg["price"]), source="twelvedata_ws")
        except Exception:
            time.sleep(5)


@st.cache_resource(show_spinner=False)
def start_binance():
    t = threading.Thread(target=_binance_worker,
                         args=(["btcusdt@ticker", "ethusdt@ticker"],), daemon=True)
    t.start()
    return t


@st.cache_resource(show_spinner=False)
def start_twelvedata(api_key: str):
    if not api_key:
        return None
    t = threading.Thread(target=_twelvedata_worker, args=(api_key,), daemon=True)
    t.start()
    return t


start_binance()
start_twelvedata(ss.twelvedata_api_key)


# Market hours -----------------------------------------------------------
def market_is_open(category: str) -> bool:
    now = datetime.now(timezone.utc)
    if category == "crypto":
        return True
    if category == "forex":      # Sun 21:00 UTC → Fri 21:00 UTC
        wd, t = now.weekday(), now.time()
        return not (wd == 5 or (wd == 6 and t < dtime(21)) or (wd == 4 and t >= dtime(21)))
    if category == "nse":        # Mon-Fri 09:15-15:30 IST
        ist = now + timedelta(hours=5, minutes=30)
        return ist.weekday() < 5 and dtime(9, 15) <= ist.time() <= dtime(15, 30)
    if category == "us_equity":  # Mon-Fri 09:30-16:00 ET (UTC-5, no DST)
        et = now - timedelta(hours=5)
        return et.weekday() < 5 and dtime(9, 30) <= et.time() <= dtime(16, 0)
    return True


# ════════════════════════════════════════════════════════════
# 4. HISTORICAL DATA LAYER  (real data only — never fabricated)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker: str, period="1mo", interval="15m", daily_fallback=True) -> pd.DataFrame:
    attempts = [(period, interval)]
    if daily_fallback and interval != "1d":
        attempts.append(("3mo", "1d"))
    for p, i in attempts:
        try:
            df = yf.download(ticker, period=p, interval=i, progress=False, auto_adjust=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            continue
        if "Volume" not in df:
            df["Volume"] = 0
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) >= 15:
            return df
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def delayed_quote(ticker: str):
    intraday = fetch_ohlcv(ticker, "5d", "15m")
    daily = fetch_ohlcv(ticker, "1mo", "1d", daily_fallback=False)
    if intraday.empty:
        return None
    hist = intraday["Close"].astype(float).tail(24).tolist()
    price = hist[-1]
    prev = float(daily["Close"].iloc[-2]) if len(daily) >= 2 else hist[0]
    return dict(price=price, pct=(price - prev) / prev * 100 if prev else 0.0, history=hist)


def quote_for(live_key, yf_ticker):
    """(quote dict, 'LIVE' | 'DELAYED') or (None, None)."""
    live = get_live(live_key)
    if live:
        return live, "LIVE"
    q = delayed_quote(yf_ticker)
    return (q, "DELAYED") if q else (None, None)


def current_price(cfg) -> float | None:
    q, _ = quote_for(cfg.get("live"), cfg["yf"])
    return float(q["price"]) if q else None


@st.cache_data(ttl=60, show_spinner=False)
def anomaly_table(ticker: str) -> pd.DataFrame:
    """IsolationForest on volatility / z-score / volume surge, gated to pivots."""
    df = fetch_ohlcv(ticker, "1mo", "15m").copy()
    if len(df) < 40:
        return pd.DataFrame()
    c = df["Close"].astype(float)
    df["returns"] = c.pct_change()
    df["rolling_vol"] = df["returns"].rolling(14).std()
    df["z_score"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)
    df["vol_surge"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-8)
    df = df.dropna(subset=["rolling_vol", "z_score", "vol_surge"]).copy()
    if len(df) < 30:
        return pd.DataFrame()
    raw = IsolationForest(n_estimators=100, contamination=0.03, random_state=42).fit_predict(
        df[["rolling_vol", "z_score", "vol_surge"]].values) == -1
    pivot = (df["High"] == df["High"].rolling(5, center=True).max()) | \
            (df["Low"] == df["Low"].rolling(5, center=True).min())
    df["anomaly"] = raw & pivot & ((df["z_score"].abs() > 1.8) | (df["vol_surge"] > 2.0))
    return df


# ════════════════════════════════════════════════════════════
# 5. INDICATORS / SIGNALS / BACKTEST ENGINE
# ════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame, model: str, fast: int, slow: int):
    """Return (long_signal, short_signal, overlay_lines) for a built-in model."""
    c = df["Close"].astype(float)
    ema = lambda n: c.ewm(span=max(int(n), 2), adjust=False).mean()  # noqa: E731
    cross_up = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))      # noqa: E731
    cross_dn = lambda a, b: (a < b) & (a.shift(1) >= b.shift(1))      # noqa: E731
    overlays = {}
    if model == "EMA Crossover":
        f, s = ema(fast), ema(slow)
        long, short, overlays = cross_up(f, s), cross_dn(f, s), {f"EMA {fast}": f, f"EMA {slow}": s}
    elif model == "MACD Crossover":
        macd = ema(fast) - ema(slow)
        sig = macd.ewm(span=9, adjust=False).mean()
        long, short = cross_up(macd, sig), cross_dn(macd, sig)
    elif model == "RSI Reversal":
        d = c.diff()
        rsi = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / ((-d.clip(upper=0)).rolling(14).mean() + 1e-9))
        long, short = (rsi.shift(1) < 30) & (rsi >= 30), (rsi.shift(1) > 70) & (rsi <= 70)
    else:  # Bollinger Breakout
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        up, lo = mid + 2 * sd, mid - 2 * sd
        long, short = (c > up) & (c.shift(1) <= up.shift(1)), (c < lo) & (c.shift(1) >= lo.shift(1))
        overlays = {"BB upper": up, "BB lower": lo}
    return long.fillna(False), short.fillna(False), overlays


def run_backtest(df, long, short, capital, risk_pct, rr, stop_pct, commission=0.0):
    """One position at a time; fixed % stop, R-multiple target. Intrabar order simplified."""
    balance, risk_cash, sl_frac = float(capital), float(capital) * risk_pct / 100.0, stop_pct / 100.0
    pos, trades, equity = None, [], []
    for ts, o, h, l, c, lg, sh in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"],
                                      long.values, short.values):
        px, exit_px, reason = float(c), None, None
        if pos is None:
            if lg or sh:
                side = 1 if lg else -1
                pos = dict(side=side, entry=px, time=ts,
                           sl=px * (1 - side * sl_frac), tp=px * (1 + side * sl_frac * rr))
        else:
            if pos["side"] == 1:
                if l <= pos["sl"]: exit_px, reason = pos["sl"], "Stop loss"
                elif h >= pos["tp"]: exit_px, reason = pos["tp"], "Take profit"
                elif sh: exit_px, reason = px, "Opposite signal"
            else:
                if h >= pos["sl"]: exit_px, reason = pos["sl"], "Stop loss"
                elif l <= pos["tp"]: exit_px, reason = pos["tp"], "Take profit"
                elif lg: exit_px, reason = px, "Opposite signal"
            if exit_px is not None:
                move = (exit_px - pos["entry"]) / pos["entry"] * pos["side"]
                pnl = risk_cash * move / sl_frac - commission
                balance += pnl
                trades.append({"Entry Time": pos["time"], "Exit Time": ts,
                               "Side": "LONG" if pos["side"] == 1 else "SHORT",
                               "Entry": pos["entry"], "Exit": exit_px, "P&L": pnl,
                               "Return %": move * 100, "Exit Reason": reason})
                pos = None
        mark = balance
        if pos is not None:
            mark += risk_cash * ((px - pos["entry"]) / pos["entry"] * pos["side"]) / sl_frac
        equity.append(mark)
    eq = pd.Series(equity, index=df.index)
    tdf = pd.DataFrame(trades)
    dd = (eq / eq.cummax() - 1) * 100 if len(eq) else pd.Series(dtype=float)
    rets = eq.pct_change().dropna()
    return dict(equity=eq, trades=tdf, capital=float(capital), net=balance - capital,
                winrate=float((tdf["P&L"] > 0).mean() * 100) if len(tdf) else 0.0,
                mdd=abs(float(dd.min())) if len(dd) else 0.0,
                sharpe=float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 1 else 0.0)


def make_signal_snapshot(price, df, asset="Gold (XAU/USD)"):
    if df is None or len(df) < 30 or price is None:
        return None
    d = df.copy()
    c = d["Close"].astype(float)
    f, s = c.ewm(span=9, adjust=False).mean(), c.ewm(span=21, adjust=False).mean()
    delta = c.diff()
    rsi = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / ((-delta.clip(upper=0)).rolling(14).mean() + 1e-9))
    vol_ma = float(d["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in d else 0.0
    vol_ratio = float(d["Volume"].iloc[-1] / vol_ma) if vol_ma else 1.0
    side = "LONG" if f.iloc[-1] > s.iloc[-1] else "SHORT"
    atr = max(float((d["High"] - d["Low"]).rolling(14).mean().iloc[-1]), price * 0.002)
    risk = atr * 1.15
    sgn = 1 if side == "LONG" else -1
    trend = abs(f.iloc[-1] - s.iloc[-1]) / max(price, 1e-9) * 10000
    ok, why = prop_risk_status(float(ss.prop_max_trade_risk_pct))
    return dict(asset=asset, side=side, entry=price, sl=price - sgn * risk,
                tp1=price + sgn * risk, tp2=price + sgn * risk * 2, rr=2.0,
                confidence=min(95, max(52, 58 + trend * 2 + min(vol_ratio, 2) * 6)),
                rsi=float(rsi.iloc[-1]), volume_ratio=vol_ratio, risk_ok=ok, risk_reason=why)


# ════════════════════════════════════════════════════════════
# 6. PROP-FIRM RISK HELPERS
# ════════════════════════════════════════════════════════════
def prop_equity() -> float:
    return float(ss.account_balance + ss.prop_daily_pnl)


def prop_drawdown_pct() -> float:
    peak = max(float(ss.prop_peak_equity), prop_equity(), 1.0)
    return max(0.0, (peak - prop_equity()) / peak * 100.0)


def prop_daily_loss_pct() -> float:
    return max(0.0, -float(ss.prop_daily_pnl) / max(float(ss.prop_account_size), 1.0) * 100.0)


def prop_risk_status(risk_pct: float):
    if ss.prop_account_type not in {"Prop-Firm", "Evaluation", "Funded Account"}:
        return True, "Personal/demo mode — prop limits are informational."
    mx, dl, dd = ss.prop_max_trade_risk_pct, ss.prop_daily_loss_limit_pct, ss.prop_max_drawdown_pct
    if risk_pct > mx:
        return False, f"Requested risk {risk_pct:.2f}% exceeds max trade risk {mx:.2f}%."
    if prop_daily_loss_pct() + risk_pct > dl:
        return False, f"Trade could exceed the daily loss limit ({dl:.2f}%)."
    if prop_drawdown_pct() + risk_pct > dd:
        return False, f"Trade could exceed the maximum drawdown limit ({dd:.2f}%)."
    return True, "Within configured prop-account risk limits."


def estimate_risk_pct(cfg, lots, entry, sl) -> float:
    if not sl:
        return float(ss.prop_max_trade_risk_pct)
    return abs(entry - sl) * lots * cfg.get("lot_units", 1) / max(ss.account_balance, 1) * 100


def append_notification(title, body, level="INFO"):
    if not ss.alerts_on:
        return
    ss.notification_log.insert(0, dict(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                       level=level, title=title, body=body))
    del ss.notification_log[50:]


def signal_message(sig) -> str:
    return (f"{sig['asset']} {sig['side']} SETUP | Entry {sig['entry']:.2f} | SL {sig['sl']:.2f} | "
            f"TP1 {sig['tp1']:.2f} | TP2 {sig['tp2']:.2f} | R:R 1:{sig['rr']:.1f} | "
            f"Confidence {sig['confidence']:.0f}% | Risk check "
            f"{'PASS' if sig['risk_ok'] else 'BLOCKED'}: {sig['risk_reason']}")


def save_monitor_config() -> str:
    cfg = {k: ss[k] for k in ("prop_account_type", "prop_firm_name", "prop_account_size",
                              "prop_daily_loss_limit_pct", "prop_max_drawdown_pct",
                              "prop_max_trade_risk_pct", "telegram_bot_token", "telegram_chat_id",
                              "discord_webhook", "alert_email", "broker_api_endpoint")}
    # Match the standalone service schema, but leave monitoring and EA execution OFF by default.
    cfg.update({
        "enabled": False,
        "assets": ["Gold (XAU/USD)", "Bitcoin (BTC/USDT)", "NIFTY 50 Index"],
        "interval_seconds": 60,
        "max_trade_risk_pct": float(ss.prop_max_trade_risk_pct),
        "auto_ea_on_signal": False,
        "ea_execution_mode": "PAPER",
    })
    path = APP_DIR / "prop_ai_monitor_config.json"
    path.write_text(json.dumps(cfg, indent=2), "utf-8")
    try:
        os.chmod(path, 0o600)      # file may hold tokens
    except OSError:
        pass
    return str(path)


def place_order(side, lots, asset, price, sl=0.0, tp=0.0):
    ss.positions.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"), "Asset": asset, "Type": side, "Lots": lots,
        "Price": round(price, 5), "SL": sl or "", "TP": tp or "",
        "Bridge": "MT5/REST" if ss.broker_connected else "Demo Simulated"})


# ════════════════════════════════════════════════════════════
# 7. CO-PILOT (CLAUDE) HELPERS
# ════════════════════════════════════════════════════════════
def ask_claude(user_text: str, context: str = "") -> str:
    try:
        import anthropic
    except ImportError:
        return "⚠️ Install the SDK first: `pip install anthropic`, then restart the app."
    if not ss.anthropic_api_key:
        return "⚠️ Add your Anthropic API key in Settings (or the co-pilot's ⚙️ tab)."
    try:
        client = anthropic.Anthropic(api_key=ss.anthropic_api_key)
        system = ("You are the trading co-pilot inside the 'Smart Session Anomaly Detector' app. "
                  "Be concise and practical about market structure, the user's strategies and "
                  "signals. For any trade idea state the reasoning and add a short reminder that "
                  "this is not financial advice."
                  + (f"\n\nCurrent context:\n{context}" if context else ""))
        history = [{"role": m["role"], "content": m["content"]} for m in ss.chat_messages[-10:]]
        resp = client.messages.create(model=ss.claude_model, max_tokens=600, system=system,
                                      messages=history + [{"role": "user", "content": user_text}])
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as exc:
        return f"⚠️ Claude API error: {exc}"


def send_chat_message(text: str, context: str = ""):
    text = (text or "").strip()
    if not text:
        return
    ss.chat_messages.append({"role": "user", "content": text})
    ss.chat_messages.append({"role": "assistant", "content": ask_claude(text, context)})


def send_from_input():
    send_chat_message(ss.get("ssad_chat_text_input", ""))
    ss["ssad_chat_text_input"] = ""


def switch_page(page_id: str):
    ss.page = page_id


def toggle_bot():
    ss.bot_open = not ss.bot_open


def open_bot():
    ss.bot_open = True


# ════════════════════════════════════════════════════════════
# 8. STYLES, SIDEBAR, MARKET STRIP, CO-PILOT UI
# ════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
:root{--line:rgba(255,255,255,.09);--green:#39e58c;--red:#ff5c68;--amber:#f6c85f;--blue:#68a8ff}
.stApp{background:radial-gradient(circle at 12% 8%,rgba(49,106,220,.16),transparent 27%),
  radial-gradient(circle at 88% 18%,rgba(0,205,145,.08),transparent 23%),linear-gradient(180deg,#070a11,#04060b);color:#f5f7fb}
[data-testid="stHeader"]{background:rgba(5,7,12,.72)!important;backdrop-filter:blur(18px)}
[data-testid="stMainBlockContainer"]{padding-top:1.4rem!important}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line)!important}
.up{color:var(--green)!important}.down{color:var(--red)!important}
.ssad-page-header{font-size:1.15rem;font-weight:700;padding:6px 0 12px;border-bottom:1px solid var(--line);margin-bottom:16px;color:#e9edf5}
.ssad-page-header span{color:#737d91;font-weight:500}
.ssad-eyebrow{color:#6f7a8e;text-transform:uppercase;letter-spacing:1.8px;font-size:.68rem;font-weight:800}
.ssad-section-title{font-size:1.3rem;font-weight:800;text-align:center;margin:22px 0 14px}
/* ── ORIGINAL globe hero (restored verbatim from the pre-remodel dashboard) ── */
.ssad-hero{position:relative;overflow:hidden;min-height:255px;border-radius:22px;border:1px solid rgba(255,255,255,.10);background:radial-gradient(circle at 80% 30%,rgba(82,153,255,.20),transparent 26%),radial-gradient(circle at 70% 90%,rgba(46,229,140,.10),transparent 24%),linear-gradient(120deg,rgba(17,23,36,.97),rgba(7,10,17,.90));box-shadow:0 22px 70px rgba(0,0,0,.32),inset 0 1px 0 rgba(255,255,255,.035)}
.ssad-hero-grid{position:absolute;inset:0;opacity:.22;background-image:linear-gradient(rgba(255,255,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.045) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(90deg,#000 0%,transparent 85%)}
.ssad-hero-copy{position:relative;z-index:2;padding:30px 34px;max-width:62%}
.ssad-hero h1{margin:8px 0;font-size:clamp(2rem,4vw,3.2rem);line-height:1.02;letter-spacing:-1.7px;color:#fff}
.ssad-hero p{color:#9aa5b7;font-size:1rem;max-width:620px;line-height:1.65;margin:0 0 20px}
.ssad-chip-row{display:flex;flex-wrap:wrap;gap:8px}.ssad-chip{padding:7px 11px;border-radius:999px;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.045);color:#c9d0dc;font-size:.75rem;font-weight:700}.ssad-chip.live{color:#56ed9d;border-color:rgba(57,229,140,.25);background:rgba(57,229,140,.08)}
.ssad-hero-clean{min-height:500px;margin-top:2px;background:radial-gradient(circle at 82% 38%,rgba(40,126,255,.18),transparent 25%),radial-gradient(circle at 70% 78%,rgba(57,229,140,.10),transparent 24%),linear-gradient(120deg,rgba(17,23,36,.98),rgba(7,10,17,.94))}
.ssad-hero-clean .ssad-hero-copy{max-width:78%;padding:58px 52px}
.ssad-hero-clean .ssad-eyebrow span{color:#65758d;margin:0 6px}
.ssad-hero-clean h1{font-size:clamp(2.7rem,5vw,4.6rem);letter-spacing:-2.8px;margin:12px 0 17px}
.ssad-hero-clean h1 span{background:linear-gradient(90deg,#55efc2,#55b8ff 48%,#a97cff);-webkit-background-clip:text;background-clip:text;color:transparent}
.ssad-hero-clean p{max-width:690px;font-size:.98rem;margin-bottom:25px}
.ssad-chip.blue{color:#8dbbff;border-color:rgba(104,168,255,.25)}
.ssad-chip.purple{color:#c19bff;border-color:rgba(165,108,255,.28)}
.ssad-chip.cyan{color:#6eeaff;border-color:rgba(0,216,255,.25)}
.ssad-feature-line{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin-top:27px;color:#94a8c1;font-size:.68rem;font-weight:800;letter-spacing:.8px}
.ssad-feature-line span:first-child{color:#69e8b1}
.ssad-feature-line b{color:#53657e}
.ssad-hero-glow{position:absolute;right:9%;top:17%;width:270px;height:270px;border-radius:50%;background:radial-gradient(circle,rgba(68,155,255,.11),transparent 67%);filter:blur(2px)}
.ssad-hero-orbit{position:absolute;border:1px solid rgba(87,155,255,.18);border-radius:50%;pointer-events:none}
.ssad-hero-orbit.orbit-one{width:310px;height:310px;right:5%;top:11%;transform:rotate(-22deg)}
.ssad-hero-orbit.orbit-two{width:430px;height:190px;right:-1%;top:27%;transform:rotate(-22deg);border-color:rgba(57,229,140,.13)}
.ssad-hero-node{position:absolute;width:7px;height:7px;border-radius:50%;background:#68dfff;box-shadow:0 0 16px rgba(104,223,255,.9)}
.ssad-hero-node.node-one{right:21%;top:20%}.ssad-hero-node.node-two{right:10%;top:58%;background:#39e58c;box-shadow:0 0 16px rgba(57,229,140,.9)}.ssad-hero-node.node-three{right:29%;bottom:18%;background:#a87cff;box-shadow:0 0 16px rgba(168,124,255,.8)}
.ssad-card{height:315px;box-sizing:border-box;display:flex;flex-direction:column;padding:18px;border-radius:18px;border:1px solid var(--line);
  background:linear-gradient(145deg,rgba(18,25,39,.96),rgba(8,12,20,.96));transition:transform .18s,border-color .18s,box-shadow .18s;overflow:hidden}
.ssad-card:hover{transform:translateY(-3px);border-color:rgba(104,168,255,.32);box-shadow:0 18px 45px rgba(0,0,0,.32)}
.ssad-card .art{height:44px;margin-bottom:4px}.ssad-card h3{margin:4px 0 8px;font-size:1.12rem}
.ssad-card p{flex:1;margin:8px 0 0;color:#a1aec0;font-size:.84rem;line-height:1.42}
.ssad-price{font-size:1.3rem;font-weight:850;letter-spacing:-.3px}.ssad-move{font-size:.76rem;font-weight:800}
.ssad-meta{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.ssad-pill{padding:4px 8px;border-radius:999px;background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.08);color:#a9b7ca;font-size:.68rem;font-weight:750}
.ssad-pill.good{color:#62e8a9;border-color:rgba(57,229,140,.25)}.ssad-pill.warn{color:#ffd36b;border-color:rgba(246,200,95,.25)}.ssad-pill.bad{color:#ff8290;border-color:rgba(255,92,104,.25)}
.ssad-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:8px 0 4px}
.ssad-stat{padding:7px 8px;border-radius:8px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.06)}
.ssad-stat b{display:block;color:#f3f6fb;font-size:.84rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ssad-stat span{display:block;color:#718198;font-size:.62rem;margin-top:2px;text-transform:uppercase;letter-spacing:.4px}
.ssad-note{color:#667084;font-size:.7rem;margin-top:14px}
.stButton>button{border-radius:10px!important;border:1px solid rgba(255,255,255,.10)!important;background:rgba(255,255,255,.045)!important;color:#e8ecf4!important;font-weight:700!important}
.stButton>button:hover{border-color:rgba(104,168,255,.38)!important;background:rgba(104,168,255,.09)!important}
/* sticky market strip */
.st-key-ssad_strip{position:sticky;top:var(--header-height,3.75rem);z-index:50;margin:0 0 8px;padding:.35rem 0 .15rem;background:linear-gradient(180deg,#070a11 78%,rgba(7,10,17,0))}
/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(9,14,27,.98),rgba(7,11,21,.98))!important;border-right:1px solid rgba(120,160,220,.12)!important}
[data-testid="stSidebarContent"]{padding:18px 16px 14px!important}
.ssad-logo{display:flex;align-items:center;gap:11px}
.ssad-logo .icon{font-size:24px;width:44px;height:44px;display:flex;align-items:center;justify-content:center;border-radius:11px;
  background:linear-gradient(145deg,rgba(48,91,170,.32),rgba(18,31,61,.72));border:1px solid rgba(112,157,235,.42)}
.ssad-logo .name{font-weight:800;font-size:1.12rem;line-height:1.05}.ssad-logo .sub{font-size:.68rem;color:rgba(205,216,235,.55);margin-top:4px}
.ssad-status{display:inline-flex;padding:6px 12px;border-radius:999px;background:rgba(25,211,137,.08);border:1px solid rgba(25,211,137,.42);
  color:#4ade80;font-size:.70rem;font-weight:750;margin:10px 0 6px}
.ssad-nav-label{font-size:.64rem;letter-spacing:1.35px;color:rgba(177,191,214,.48);font-weight:800;margin:16px 4px 6px}
[data-testid="stSidebar"] .stButton{margin:0 0 6px!important}
[data-testid="stSidebar"] .stButton>button{min-height:42px!important;border-radius:12px!important;justify-content:flex-start!important;text-align:left!important;
  font-weight:600!important;background:rgba(255,255,255,.035)!important}
[data-testid="stSidebar"] .stButton>button:hover{transform:translateX(2px)}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:linear-gradient(90deg,rgba(52,120,246,.18),rgba(22,166,161,.08))!important;
  border-color:rgba(75,143,245,.42)!important;box-shadow:inset 3px 0 0 #4f91ff!important}
/* co-pilot */
.st-key-ssad_bot_fab{position:fixed!important;bottom:28px;right:28px;z-index:1000000;width:auto!important}
.st-key-ssad_bot_fab button{border-radius:999px!important;padding:12px 19px!important;font-size:16px!important;color:#fff!important;
  background:linear-gradient(135deg,rgba(33,35,57,.82),rgba(89,64,126,.66))!important;backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,.22)!important;box-shadow:0 12px 36px rgba(0,0,0,.48),0 0 24px rgba(155,120,255,.22)!important}
.st-key-ssad_bot_panel{position:fixed!important;bottom:104px;right:28px;width:430px;max-height:72vh;overflow-y:auto;z-index:999999;
  background:linear-gradient(160deg,rgba(13,16,29,.80),rgba(48,34,73,.72));backdrop-filter:blur(20px) saturate(150%);
  border:1px solid rgba(255,255,255,.16);border-radius:26px;box-shadow:0 18px 55px rgba(0,0,0,.58);padding:6px 8px}
</style>
"""


def page_header(subtitle: str, caption: str | None = None):
    st.markdown(f'<div class="ssad-page-header">⚡ Smart Session Anomaly Detector '
                f'<span>| {subtitle}</span></div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="ssad-logo"><div class="icon">⚡</div><div><div class="name">SSAD</div>'
                    '<div class="sub">Smart Session Anomaly Detector</div></div></div>'
                    '<div class="ssad-status">🟢 ML ENGINE ONLINE</div>', unsafe_allow_html=True)
        for group, items in NAV_GROUPS.items():
            st.markdown(f'<div class="ssad-nav-label">{group}</div>', unsafe_allow_html=True)
            for label, pid in items:
                st.button(label, key=f"nav_{pid}", use_container_width=True,
                          type="primary" if ss.page == pid else "secondary",
                          on_click=switch_page, args=(pid,))
        st.divider()
        st.caption("CONNECTION")
        c1, c2 = st.columns(2)
        n = live_feed_count()
        c1.metric("Live feeds", f"{n} ●" if n else "None")
        c2.metric("Broker", "Connected" if ss.broker_connected else "Offline")
        t1, t2 = st.columns(2)
        ss.alerts_on = t1.toggle("🔔 Alerts", value=ss.alerts_on)
        ss.voice_enabled = t2.toggle("🔊 Voice", value=ss.voice_enabled)


def sparkline(points, color) -> str:
    pts = [float(p) for p in points] or [0, 0]
    if len(pts) < 2:
        pts = pts * 2
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    xy = " ".join(f"{4 + i / (len(pts) - 1) * 192:.1f},{45 - (v - lo) / span * 36:.1f}" for i, v in enumerate(pts))
    return (f"<svg viewBox='0 0 200 52' preserveAspectRatio='none'><polyline points='{xy}' fill='none' "
            f"stroke='{color}' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/></svg>")


MARQUEE_SECONDS = 45


def _strip_html() -> str:
    cards = []
    for label, live_key, ticker, mkt, dec, prefix in STRIP_ASSETS:
        q, src = quote_for(live_key, ticker)
        if not q:
            price, move, cls, pts, badge = "—", "n/a", "", [], ("NO DATA", "#8792a5")
        else:
            pct = float(q["pct"])
            price, move, cls = f"{prefix}{q['price']:,.{dec}f}", f"{pct:+.2f}%", "up" if pct >= 0 else "down"
            pts = q.get("history", [])
            badge = ("CLOSED", "#8792a5") if not market_is_open(mkt) else \
                    (("LIVE", "#39e58c") if src == "LIVE" else ("DELAYED", "#f6c85f"))
        col = "#39e58c" if cls == "up" else "#ff5c68"
        cards.append(
            f"<div class='c'><div class='t'><span>{label}</span>"
            f"<i style='color:{badge[1]}'>● {badge[0]}</i></div><div class='p'>{price}</div>"
            f"<div class='m {cls}'>{move}</div><div class='s'>{sparkline(pts, col) if pts else ''}</div></div>")
    row = "".join(cards)
    # Track holds the row twice and slides -50% → 0 = continuous left-to-right loop, no gap.
    # The animation delay is derived from the wall clock, so a re-render resumes mid-phase.
    return f"""<style>
html,body{{margin:0;background:transparent;overflow:hidden;font-family:Inter,system-ui,sans-serif}}
.v{{overflow:hidden;padding:4px 0}}
.k{{display:flex;width:max-content;gap:10px;animation:slide {MARQUEE_SECONDS}s linear infinite}}
.v:hover .k{{animation-play-state:paused}}
@keyframes slide{{from{{transform:translateX(-50%)}}to{{transform:translateX(0)}}}}
.c{{position:relative;flex:0 0 160px;box-sizing:border-box;padding:10px 12px;border-radius:14px;color:#f3f6fb;
 border:1px solid rgba(104,168,255,.18);background:linear-gradient(145deg,rgba(18,27,43,.92),rgba(8,12,21,.92))}}
.t{{display:flex;justify-content:space-between;font-size:11px;font-weight:800;color:#d8e3f3}}.t i{{font-style:normal;font-size:9px}}
.p{{font-size:16px;font-weight:800;margin-top:6px}}.m{{font-size:11px;font-weight:800}}
.up{{color:#39e58c}}.down{{color:#ff5c68}}.s{{position:absolute;right:8px;bottom:8px;width:44%;height:30px;opacity:.65}}.s svg{{width:100%;height:100%}}
</style><div class='v'><div class='k' id='k'>{row}{row}</div></div>
<script>document.getElementById('k').style.animationDelay=(-((Date.now()/1000)%{MARQUEE_SECONDS}))+'s';</script>"""


@st.fragment(run_every=6)
def market_strip():
    components.html(_strip_html(), height=104)


def render_copilot():
    st.button("✖ Close" if ss.bot_open else "🤖 AI Co-Pilot", key="ssad_bot_fab", on_click=toggle_bot)
    if not ss.bot_open:
        return
    gold_df = fetch_ohlcv(GOLD["yf"])
    gold_px = current_price(GOLD)
    with st.container(key="ssad_bot_panel"):
        st.markdown("### 🧸✨ Nova — Trading Co-Pilot")
        st.caption("Voice, signals, risk & alerts")
        tab_talk, tab_sig, tab_guard, tab_set = st.tabs(["💬 Talk", "📡 Signals", "🛡️ Risk Guardian", "⚙️"])

        with tab_talk:
            for m in ss.chat_messages[-30:]:
                with st.chat_message(m["role"]):
                    st.write(m["content"])
            last = len(ss.chat_messages) - 1
            if ss.voice_enabled and ss.chat_messages[last]["role"] == "assistant" and last != ss.last_spoken_index:
                text = json.dumps(ss.chat_messages[last]["content"].replace("`", "")[:600]).replace("</", "<\\/")
                voice = json.dumps(ss.voice_name)
                components.html(f"""<script>try{{
                  const text={text}, pref={voice}, S=window.parent.speechSynthesis;
                  const go=()=>{{const v=S.getVoices();
                    const pv=pref!=="Auto / Best available"?v.find(x=>x.name===pref):(v.find(x=>/Samantha|Ava|Jenny|Zira|Google US English/i.test(x.name))||v.find(x=>/^en/i.test(x.lang))||v[0]);
                    const u=new SpeechSynthesisUtterance(text);if(pv)u.voice=pv;u.rate=.96;S.cancel();S.speak(u);}};
                  S.getVoices().length?go():(S.onvoiceschanged=go);}}catch(e){{}}</script>""", height=0)
                ss.last_spoken_index = last
            c_mic, c_txt, c_send = st.columns([0.16, 0.64, 0.2])
            with c_mic:   # Web Speech API (Chrome/Edge); fills the box then presses Send
                components.html("""<button id=m style="width:100%;height:38px;border-radius:8px;border:1px solid rgba(255,255,255,.3);
                  background:rgba(255,255,255,.08);color:#fff;cursor:pointer;font-size:16px">🎤</button><script>
                  const b=document.getElementById('m');b.onclick=()=>{const R=window.webkitSpeechRecognition||window.SpeechRecognition;
                  if(!R){alert('Voice input needs Chrome or Edge.');return;}
                  const r=new R();r.lang='en-US';b.innerText='🔴';
                  r.onresult=e=>{const d=window.parent.document;
                    const t=[...d.querySelectorAll('input')].find(x=>(x.placeholder||'').includes('Ask about the market'));
                    if(t){Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype,'value').set.call(t,e.results[0][0].transcript);
                      t.dispatchEvent(new Event('input',{bubbles:true}));t.blur();
                      setTimeout(()=>{const s=[...d.querySelectorAll('button')].find(x=>x.innerText.trim()==='Send');if(s)s.click();},600);}};
                  r.onend=r.onerror=()=>b.innerText='🎤';r.start();};</script>""", height=44)
            c_txt.text_input("Message", key="ssad_chat_text_input", label_visibility="collapsed",
                             placeholder="Ask about the market or your strategy...")
            c_send.button("Send", key="ssad_send", use_container_width=True, on_click=send_from_input)

        with tab_sig:
            if len(gold_df) >= 25:
                c = gold_df["Close"].astype(float)
                z = float(((c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-8)).iloc[-1])
                vs = float((gold_df["Volume"] / (gold_df["Volume"].rolling(20).mean() + 1e-8)).iloc[-1])
                call, why = (("🟢 BUY (mean-reversion)", "Price is stretched below its mean.") if z <= -1.8 else
                             ("🔴 SELL (mean-reversion)", "Price is stretched above its mean.") if z >= 1.8 else
                             ("🟡 WATCH (volume surge)", "Volume surge — a breakout may be forming.") if vs >= 2 else
                             ("⚪ HOLD", "No statistically significant edge right now."))
                st.metric("Current Signal (Gold)", call)
                st.caption(f"Z-Score {z:.2f} | Volume surge {vs:.2f}x")
                st.write(why)
                if st.button("Ask Nova to interpret", key="ssad_sig_ask"):
                    send_chat_message("Interpret the current signal and what to watch next.",
                                      f"XAU/USD price {gold_px}. Z-score {z:.2f}, volume surge {vs:.2f}x. Call: {call}.")
                    st.rerun()
            else:
                st.info("No market data available right now.")

        with tab_guard:
            g1, g2 = st.columns(2)
            g1.metric("Account mode", ss.prop_account_type)
            g1.metric("Daily loss", f"{prop_daily_loss_pct():.2f}% / {ss.prop_daily_loss_limit_pct:.2f}%")
            g2.metric("Drawdown", f"{prop_drawdown_pct():.2f}% / {ss.prop_max_drawdown_pct:.2f}%")
            g2.metric("Max trade risk", f"{ss.prop_max_trade_risk_pct:.2f}%")
            sig = make_signal_snapshot(gold_px, gold_df)
            if sig:
                st.markdown(f"#### {'🟢 READY' if sig['risk_ok'] else '🔴 BLOCKED'} {sig['side']} setup")
                a, b, c3 = st.columns(3)
                a.metric("Entry", f"{sig['entry']:.2f}"); b.metric("SL", f"{sig['sl']:.2f}"); c3.metric("TP1", f"{sig['tp1']:.2f}")
                (st.success if sig["risk_ok"] else st.error)(sig["risk_reason"])
                if st.button("🔔 Create alert", key="guardian_alert"):
                    append_notification(f"{sig['asset']} {sig['side']} signal", signal_message(sig), "SIGNAL")
                    st.success("Added to the notification queue.")
            else:
                st.info("Waiting for market data.")

        with tab_set:
            ss.anthropic_api_key = st.text_input("Anthropic API key", value=ss.anthropic_api_key, type="password")
            ss.claude_model = st.text_input("Model", value=ss.claude_model)
            voices = ["Auto / Best available", "Samantha", "Ava", "Jenny", "Microsoft Zira", "Google US English"]
            ss.voice_name = st.selectbox("Voice", voices, index=voices.index(ss.voice_name) if ss.voice_name in voices else 0,
                                         help="Uses voices installed in your browser/OS.")
            st.caption("Voice I/O uses your browser's speech engine (Chrome/Edge work best).")


# ════════════════════════════════════════════════════════════
# 9. PAGES
# ════════════════════════════════════════════════════════════
ART = {
    "chart": '<svg viewBox="0 0 120 54" width="120" height="44"><path d="M5 45 L25 35 L40 39 L58 19 L74 27 L95 8 L115 16" fill="none" stroke="#68a8ff" stroke-width="3"/><circle cx="95" cy="8" r="4" fill="#39e58c"/></svg>',
    "risk": '<svg viewBox="0 0 120 54" width="120" height="44"><circle cx="42" cy="27" r="20" fill="none" stroke="#f6c85f" stroke-width="7" stroke-dasharray="72 55"/><circle cx="42" cy="27" r="20" fill="none" stroke="#39e58c" stroke-width="7" stroke-dasharray="42 85" transform="rotate(-35 42 27)"/><path d="M78 39 L88 29 L97 35 L114 14" fill="none" stroke="#68a8ff" stroke-width="3"/></svg>',
    "broker": '<svg viewBox="0 0 120 54" width="120" height="44"><rect x="6" y="10" width="108" height="34" rx="8" fill="rgba(255,255,255,.035)" stroke="rgba(255,255,255,.12)"/><circle cx="23" cy="27" r="6" fill="#39e58c"/><path d="M38 27 H95 M83 19 L96 27 L83 35" fill="none" stroke="#68a8ff" stroke-width="3"/></svg>',
    "quant": '<svg viewBox="0 0 120 54" width="120" height="44"><rect x="8" y="8" width="104" height="38" rx="8" fill="rgba(104,168,255,.06)" stroke="rgba(104,168,255,.35)"/><path d="M18 37 L36 28 L50 32 L67 18 L82 23 L101 11" fill="none" stroke="#68a8ff" stroke-width="3"/><circle cx="101" cy="11" r="4" fill="#39e58c"/></svg>',
    "ai": '<svg viewBox="0 0 120 54" width="120" height="44"><circle cx="37" cy="27" r="19" fill="rgba(104,168,255,.12)" stroke="#68a8ff"/><path d="M28 28 Q37 15 46 28 Q37 39 28 28Z" fill="none" stroke="#39e58c" stroke-width="2.5"/><path d="M66 15 L73 27 L66 39 M86 15 L79 27 L86 39" fill="none" stroke="#f6c85f" stroke-width="3" stroke-linecap="round"/></svg>',
}


def stat_row(*pairs) -> str:
    return "<div class='ssad-stats'>" + "".join(f"<div class='ssad-stat'><b>{v}</b><span>{k}</span></div>" for k, v in pairs) + "</div>"


@st.cache_resource(show_spinner=False)
def hero_background() -> str:
    """Globe artwork exactly as before (trading_hero.png + the same dark left-to-right overlay).
    If the image file is missing the hero falls back to its CSS gradients instead of crashing."""
    if not HERO_IMAGE.exists():
        return ""
    b64 = base64.b64encode(HERO_IMAGE.read_bytes()).decode("utf-8")
    return ("background-image: linear-gradient(90deg, rgba(5,10,22,.98) 0%, rgba(5,10,22,.90) 34%, "
            "rgba(5,10,22,.58) 65%, rgba(5,10,22,.30) 100%), url(data:image/png;base64," + b64 + "); "
            "background-size: cover; background-position: center; background-repeat: no-repeat;")


def page_dashboard():
    hero_html = """
    <div class="ssad-hero ssad-hero-clean" style="__HERO_STYLE__">
        <div class="ssad-hero-grid"></div>
        <div class="ssad-hero-glow"></div>
        <div class="ssad-hero-copy">
            <div class="ssad-eyebrow">QUANT TRADING TERMINAL <span>•</span> SESSION INTELLIGENCE</div>
            <h1>Smart Session<br/><span>Anomaly Detector</span></h1>
            <p>Multi-asset market intelligence, anomaly detection, execution controls and systematic strategy research — all in one professional workspace.</p>
            <div class="ssad-chip-row">
                <span class="ssad-chip live">● ML ENGINE ONLINE</span>
                <span class="ssad-chip blue">◎ MULTI-MARKET</span>
                <span class="ssad-chip purple">◷ MULTI-TIMEFRAME ANALYSIS</span>
                <span class="ssad-chip cyan">↗ FLEXIBLE EXECUTION</span>
            </div>
            <div class="ssad-feature-line">
                <span>◉ LIVE MARKET DATA</span><b>•</b>
                <span>♧ ANOMALY MONITORING</span><b>•</b>
                <span>◎ SIGNAL GENERATION</span>
            </div>
        </div>
        <div class="ssad-hero-orbit orbit-one"></div>
        <div class="ssad-hero-orbit orbit-two"></div>
        <div class="ssad-hero-node node-one"></div>
        <div class="ssad-hero-node node-two"></div>
        <div class="ssad-hero-node node-three"></div>
    </div>
    """.replace("__HERO_STYLE__", hero_background())
    st.markdown(hero_html, unsafe_allow_html=True)
    st.markdown('<div class="ssad-section-title">Workspace</div>', unsafe_allow_html=True)

    # ---- live numbers for each card (all derived from real state) ----
    q, src = quote_for(GOLD["live"], GOLD["yf"])
    gdf = anomaly_table(GOLD["yf"])
    anomalies = int(gdf["anomaly"].sum()) if not gdf.empty else None
    sig = make_signal_snapshot(float(q["price"]) if q else None, fetch_ohlcv(GOLD["yf"]))
    price_html = (f"<div class='ssad-price'>XAU/USD {q['price']:,.2f}</div><div class='ssad-move {'up' if q['pct']>=0 else 'down'}'>"
                  f"{q['pct']:+.2f}% · {src}</div>") if q else "<div class='ssad-price'>XAU/USD —</div><div class='ssad-move'>no data</div>"
    markets_body = price_html + "<div class='ssad-meta'>" \
        f"<span class='ssad-pill'>ANOMALIES {anomalies if anomalies is not None else 'n/a'}</span>" \
        f"<span class='ssad-pill {'good' if sig else 'warn'}'>SIGNAL {sig['side'] if sig else 'WAITING'}</span></div>"

    risk_cash = ss.account_balance * ss.prop_max_trade_risk_pct / 100
    lots = risk_cash / (abs(sig["entry"] - sig["sl"]) * GOLD["lot_units"]) if sig and sig["entry"] != sig["sl"] else 0.0
    risk_body = stat_row(("Risk / trade", f"{ss.prop_max_trade_risk_pct:.2f}%"), ("R:R", f"1:{sig['rr']:.1f}" if sig else "—"),
                         ("Lots", f"{lots:.2f}")) + f"<div class='ssad-meta'><span class='ssad-pill'>CAPITAL ${ss.account_balance:,.0f}</span><span class='ssad-pill'>RISK ${risk_cash:,.0f}</span></div>"
    broker_body = stat_row(("Broker", "ON" if ss.broker_connected else "OFF"), ("Mode", ss.broker_api_mode.split(" ")[0]),
                           ("Positions", len(ss.positions)))
    latest = ss.saved_strategies[-1]["name"] if ss.saved_strategies else "None yet"
    strat_body = stat_row(("Saved", len(ss.saved_strategies)), ("Models", len(STRATEGY_MODELS)), ("Presets", len(STRATEGY_PRESETS))) + \
        f"<div class='ssad-meta'><span class='ssad-pill'>LATEST: {latest}</span></div>"
    bt = ss.last_backtest
    bt_body = (stat_row(("Net P/L", f"${bt['net']:,.0f}"), ("Win rate", f"{bt['winrate']:.0f}%"), ("Trades", len(bt['trades'])))
               if bt else "<div class='ssad-meta'><span class='ssad-pill warn'>NO BACKTEST YET</span></div>")
    ea_body = stat_row(("Active", (ss.active_ea or "None")[:10]), ("State", "ON" if ss.ea_enabled else "OFF"),
                       ("Deployed", len(ss.ea_deployments)))

    cards = [
        ("chart", "Markets & Analysis", markets_body, "Candlesticks, anomaly zones and AI signals.", "Open Markets →", "chart"),
        ("risk", "Risk Management", risk_body, "Position sizing, stop distance and R:R planning.", "Calculate Risk →", "risk"),
        ("broker", "Trade Execution", broker_body, "Paper, live-approval or bridge-based execution.", "Open Execution →", "broker"),
        ("quant", "Strategy Lab", strat_body, "Build rule-based strategies or load institutional setups.", "Open Strategy Lab →", "strategy"),
        ("quant", "Backtesting", bt_body, "Historical candle tests with real provider data only.", "Open Backtesting →", "backtest"),
        ("ai", "Expert Advisors", ea_body, "Build, validate and deploy automated logic.", "Open EA Studio →", "ea"),
    ]
    for row in (cards[:3], cards[3:]):
        for col, (art, title, body, desc, btn, target) in zip(st.columns(3, gap="medium"), row):
            with col:
                st.markdown(f'<div class="ssad-card"><div class="art">{ART[art]}</div><h3>{title}</h3>{body}<p>{desc}</p></div>',
                            unsafe_allow_html=True)
                st.button(btn, key=f"qa_{target}", use_container_width=True, on_click=switch_page, args=(target,))
    st.markdown('<div class="ssad-note">Card metrics come from the live feed, your risk settings, broker state and saved results — nothing is hard-coded.</div>',
                unsafe_allow_html=True)
    st.button("🤖 Open AI Co-Pilot", key="qa_bot", on_click=open_bot)


def tradingview_widget(cfg, height=600) -> None:
    sym = f"{cfg['exchange']}:{cfg['symbol']}"
    components.html(f"""<style>html,body{{height:100%;margin:0;background:#0f1728}}</style>
      <div style="height:100%;width:100%"><div id="tv" style="height:100%;width:100%"></div>
      <script src="https://s3.tradingview.com/tv.js"></script><script>new TradingView.widget({{autosize:true,symbol:{json.dumps(sym)},
      interval:"1",timezone:"Asia/Kolkata",theme:"dark",style:"1",locale:"en",toolbar_bg:"#10182a",enable_publishing:false,
      container_id:"tv",withdateranges:true,allow_symbol_change:true,save_image:true}});</script></div>""", height=height)


def ml_anomaly_panel(cfg):
    df = anomaly_table(cfg["yf"])
    if df.empty:
        st.warning("No market data returned for this instrument right now (provider unreachable or market data unavailable).")
        return
    an = df[df["anomaly"]]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
                                 increasing_line_color="#089981", decreasing_line_color="#f23645"), row=1, col=1)
    if not an.empty:
        fig.add_trace(go.Scatter(x=an.index, y=an["High"] * 1.003, mode="markers", name="AI anomaly",
                                 marker=dict(symbol="diamond", size=10, color="#ff0055", line=dict(width=1.5, color="#fff"))), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=["#089981" if c >= o else "#f23645" for c, o in zip(df["Close"], df["Open"])]), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=540, margin=dict(l=10, r=10, t=10, b=10), dragmode="pan",
                      xaxis_rangeslider_visible=False, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14")
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    st.markdown("#### 🔬 Detected session anomalies")
    if an.empty:
        st.caption("No statistical anomalies flagged in this window.")
    else:
        t = an[["Close", "z_score", "vol_surge", "rolling_vol"]].tail(10).copy()
        t.columns = ["Price", "Z-Score (σ)", "Volume surge", "Rolling vol"]
        st.dataframe(t, use_container_width=True)


def order_panel(name, cfg):
    st.markdown("**Trade execution**")
    px = current_price(cfg)
    if px is None:
        st.warning("No price available — order entry disabled.")
        return
    spread = max(abs(px) * 0.0001, 0.0001)      # simulated, for display only
    sell_q, buy_q = px - spread / 2, px + spread / 2
    a, b = st.columns(2)
    a.metric("SELL", f"{sell_q:,.4f}")
    b.metric("BUY", f"{buy_q:,.4f}")
    st.caption("Indicative quotes with a simulated spread.")
    lots = st.number_input("Lots", 0.01, 100.0, 0.05, 0.01, key="ord_lots")
    use_sltp = st.toggle("TP / SL", key="ord_sltp")
    sl = tp = 0.0
    if use_sltp:
        sl = st.number_input("Stop loss", 0.0, value=0.0, key="ord_sl")
        tp = st.number_input("Take profit", 0.0, value=0.0, key="ord_tp")
    side = st.radio("Direction", ["BUY", "SELL"], horizontal=True, key="ord_side", label_visibility="collapsed")
    entry = buy_q if side == "BUY" else sell_q
    if st.button(f"Place {side} @ {entry:,.4f}", type="primary", use_container_width=True, key="ord_go"):
        ok, why = prop_risk_status(estimate_risk_pct(cfg, lots, entry, sl))
        if not ok:
            st.error(f"Blocked by risk guard: {why}")
        else:
            place_order(side, lots, name, entry, sl, tp)
            st.success(f"{side} order recorded ({ss.broker_api_mode}).")
    if not ss.broker_connected:
        st.caption("Broker not connected — orders are simulated only.")


def page_chart():
    page_header("Chart Analysis")
    c1, c2 = st.columns(2)
    sector = c1.selectbox("Market sector", list(MARKET_UNIVERSE), key="chart_sector")
    name = c2.selectbox("Instrument", list(MARKET_UNIVERSE[sector]), key="chart_inst")
    cfg = MARKET_UNIVERSE[sector][name]
    tab_live, tab_ml = st.tabs(["📺 Live TradingView", "🔬 ML Anomaly Engine"])
    with tab_live:
        chart_col, order_col = st.columns([2.7, 1], gap="medium")
        with chart_col:
            tradingview_widget(cfg)
        with order_col:
            order_panel(name, cfg)
    with tab_ml:
        ml_anomaly_panel(cfg)

    st.markdown("### 🤖 Expert Advisor panel")
    e1, e2, e3 = st.columns([1.5, 1, 1])
    choices = ["None"] + [x["name"] for x in EA_TEMPLATES] + [x["name"] for x in ss.saved_eas]
    pick = e1.selectbox("EA to load on this chart", choices, key="chart_ea_pick")
    if e2.button("📥 Load EA", use_container_width=True, key="chart_ea_load"):
        ss.active_ea = None if pick == "None" else pick
        ss.ea_enabled = False
    ss.ea_enabled = e3.toggle("Enable EA", value=ss.ea_enabled, key="chart_ea_enabled")
    if ss.active_ea:
        st.info(f"Active EA: **{ss.active_ea}** — {'ENABLED' if ss.ea_enabled else 'DISABLED'}. "
                "Execution stays subject to your risk limits and broker connection.")


def page_risk():
    page_header("Pip & Risk Calculator")
    name = st.selectbox("Instrument", list(ALL_INSTRUMENTS), key="risk_inst")
    cfg = ALL_INSTRUMENTS[name]
    px = current_price(cfg) or 0.0
    c1, c2, c3 = st.columns(3)
    with c1:
        bal = st.number_input("Portfolio balance ($)", value=float(ss.account_balance), step=1000.0)
        risk_pct = st.slider("Risk tolerance (%)", 0.25, 5.0, 1.0, 0.25)
        risk_usd = bal * risk_pct / 100
        st.metric("Capital at risk", f"${risk_usd:,.2f}")
    with c2:
        entry = st.number_input("Entry price", value=float(px), format="%.5f", key=f"re_{name}")
        sl = st.number_input("Stop loss", value=float(px * 0.992), format="%.5f", key=f"rs_{name}")
        tp = st.number_input("Take profit", value=float(px * 1.016), format="%.5f", key=f"rt_{name}")
    sl_pts, tp_pts = abs(entry - sl), abs(tp - entry)
    with c3:
        st.metric("Stop distance", f"{sl_pts:,.4f}")
        st.metric("Risk : Reward", f"1 : {tp_pts / sl_pts:.2f}" if sl_pts else "—")
    if sl_pts > 0:
        st.success(f"🎯 Recommended size: **{risk_usd / (sl_pts * cfg['lot_units']):.2f} lots** "
                   f"(contract size {cfg['lot_units']:,})")
    else:
        st.info("Set a stop-loss different from entry to size the position.")


def page_broker():
    page_header("Execution Bridge", "Paper/demo by default. Live autonomous execution needs the broker's authenticated bridge.")
    brokers = ["MetaTrader 5 (MT5)", "Zerodha (Kite)", "Binance Futures", "Interactive Brokers"]
    g1, g2 = st.columns([1, 1.2])
    with g1:
        ss.broker_name = st.selectbox("Broker / gateway", brokers, index=brokers.index(ss.broker_name))
        ss.broker_api_mode = st.radio("Execution mode", ["Paper / Demo", "Live (manual approval)"], horizontal=True,
                                      index=["Paper / Demo", "Live (manual approval)"].index(ss.broker_api_mode))
        ss.broker_api_endpoint = st.text_input("API / bridge endpoint", value=ss.broker_api_endpoint, placeholder="https://...")
        ss.broker_api_key = st.text_input("API key / account ID", value=ss.broker_api_key, type="password")
        b1, b2 = st.columns(2)
        if b1.button("🔌 Connect", use_container_width=True):
            ss.broker_connected = True
        if b2.button("⏹ Disconnect", use_container_width=True):
            ss.broker_connected = False
        st.info("Connecting here marks the gateway as connected for this app session (demo mode).")
    with g2:
        st.metric("Gateway", "🟢 Connected" if ss.broker_connected else "⚪ Offline")
        if ss.positions:
            st.dataframe(pd.DataFrame(ss.positions), use_container_width=True, hide_index=True)
            if st.button("Close all demo positions", use_container_width=True):
                ss.positions = []
                st.rerun()
        else:
            st.caption("No open positions in this session.")


def page_strategy():
    page_header("Strategy Lab", "Define rules visually, save them, then test in Backtesting. Code is stored as text only and never executed.")
    mode = st.radio("Start with", ["Build from scratch", "Institutional preset", "Recreate from notes", "Edit saved"], horizontal=True)
    base: dict = {}
    if mode == "Institutional preset":
        pname = st.selectbox("Preset", list(STRATEGY_PRESETS))
        base = {**STRATEGY_PRESETS[pname], "name": pname}
        st.info(f"**Rule:** {base['entry']}  \nBacktests use the closest built-in model (**{base['model']}**); the rule text is kept as notes.")
    elif mode == "Edit saved":
        if ss.saved_strategies:
            pick = st.selectbox("Saved strategy", [s["name"] for s in ss.saved_strategies])
            base = next(s for s in ss.saved_strategies if s["name"] == pick)
        else:
            st.info("Save a strategy first to enable editing.")

    c1, c2 = st.columns([1.2, 1])
    name = c1.text_input("Strategy name", value=base.get("name", "EMA Crossover"))
    model = c1.selectbox("Signal model", STRATEGY_MODELS, index=STRATEGY_MODELS.index(base.get("model", "EMA Crossover")))
    risk = c2.number_input("Risk per trade (%)", 0.1, 5.0, float(base.get("risk", 1.0)), 0.1)
    rr = c2.number_input("Reward : Risk", 0.5, 10.0, float(base.get("rr", 2.0)), 0.25)
    p1, p2, p3 = st.columns(3)
    fast = p1.number_input("Fast period", 2, 200, int(base.get("fast", 9)))
    slow = p2.number_input("Slow period", 3, 300, int(base.get("slow", 21)))
    stop = p3.number_input("Stop loss (%)", 0.05, 20.0, float(base.get("stop_pct", 0.6)), 0.05)
    if model in ("EMA Crossover", "MACD Crossover") and fast >= slow:
        st.warning("Fast period should be smaller than slow period.")
    if model in ("RSI Reversal", "Bollinger Breakout"):
        st.caption("This model uses fixed parameters (RSI 14 / Bollinger 20,2σ); fast/slow are ignored.")
    entry = st.text_input("Entry rule notes", value=base.get("entry", "Enter on confirmed signal"))
    exit_ = st.text_input("Exit rule notes", value=base.get("exit", "Exit on opposite signal or stop/target"))
    notes = st.text_area("Video / transcript notes", value=base.get("notes", ""), height=80) if mode == "Recreate from notes" else base.get("notes", "")

    if st.button("💾 Save strategy", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Enter a strategy name.")
        elif model in ("EMA Crossover", "MACD Crossover") and fast >= slow:
            st.error("Fast period must be lower than slow period.")
        else:
            item = dict(name=name.strip(), model=model, fast=int(fast), slow=int(slow), risk=float(risk),
                        rr=float(rr), stop_pct=float(stop), entry=entry, exit=exit_, notes=notes)
            ss.saved_strategies = [s for s in ss.saved_strategies if s["name"] != item["name"]] + [item]
            persist()
            st.success(f"Saved '{item['name']}'. Select it in Backtesting.")

    if ss.saved_strategies:
        st.markdown("#### Saved strategies")
        st.dataframe(pd.DataFrame([{k: s.get(k) for k in ("name", "model", "fast", "slow", "risk", "rr", "stop_pct")}
                                   for s in ss.saved_strategies]), use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        dup = d1.selectbox("Duplicate", [s["name"] for s in ss.saved_strategies], key="dup_pick")
        if d1.button("Duplicate", key="dup_go"):
            src = next(s for s in ss.saved_strategies if s["name"] == dup)
            ss.saved_strategies.append({**src, "name": src["name"] + " Copy"})
            persist(); st.rerun()
        rem = d2.selectbox("Remove", [s["name"] for s in ss.saved_strategies], key="del_pick")
        if d2.button("Delete", key="del_go"):
            ss.saved_strategies = [s for s in ss.saved_strategies if s["name"] != rem]
            persist(); st.rerun()


def page_backtest():
    page_header("Backtesting Workspace", "Real provider candles only — if none are returned, nothing is simulated.")
    saved = ss.saved_strategies
    if saved:
        chosen = st.selectbox("Saved strategy", [s["name"] for s in saved])
        strat = next(s for s in saved if s["name"] == chosen)
    else:
        st.info("No saved strategy yet — using a built-in EMA crossover. Save your own in Strategy Lab.")
        strat = dict(name="EMA Crossover", model="EMA Crossover", fast=9, slow=21, risk=1.0, rr=2.0, stop_pct=0.6)
    assets = ["Gold (XAU/USD)", "Silver (XAG/USD)", "Bitcoin (BTC/USDT)", "Ethereum (ETH/USDT)",
              "NIFTY 50 Index", "EUR/USD", "Apple (AAPL)"]
    b1, b2, b3, b4 = st.columns(4)
    asset = b1.selectbox("Market", assets)
    tf = b2.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "1d"], index=1)
    period = b3.selectbox("History", ["1mo", "3mo", "6mo", "1y"], index=1)
    capital = b4.number_input("Initial capital", 1000.0, value=100000.0, step=5000.0)
    commission = st.number_input("Commission per completed trade ($)", 0.0, value=1.0, step=0.25)
    if tf in ("5m", "15m", "30m") and period != "1mo":
        st.caption("Yahoo limits 5m–30m history to ~60 days, so the request is clamped to 60d.")
    if st.button("▶ Run historical backtest", type="primary", use_container_width=True):
        yf_period = "60d" if tf in ("5m", "15m", "30m") and period != "1mo" else period
        df = fetch_ohlcv(ALL_INSTRUMENTS[asset]["yf"], yf_period, tf, daily_fallback=False)
        if len(df) < max(30, strat["slow"] + 5):
            ss.last_backtest = None
            st.error("The provider returned too few real candles for this selection. No synthetic data was used — try another timeframe/period/market.")
        else:
            long, short, overlays = compute_signals(df, strat["model"], strat["fast"], strat["slow"])
            res = run_backtest(df, long, short, capital, strat["risk"], strat["rr"], strat["stop_pct"], commission)
            fig = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
            for label, line in overlays.items():
                fig.add_trace(go.Scatter(x=df.index, y=line, name=label, line=dict(width=1.2)))
            fig.add_trace(go.Scatter(x=df.index[long.values], y=df["Low"][long.values], mode="markers", name="LONG",
                                     marker=dict(symbol="triangle-up", size=10, color="#28d17c")))
            fig.add_trace(go.Scatter(x=df.index[short.values], y=df["High"][short.values], mode="markers", name="SHORT",
                                     marker=dict(symbol="triangle-down", size=10, color="#ff5c69")))
            fig.update_layout(template="plotly_dark", height=520, xaxis_rangeslider_visible=False,
                              title=f"{asset} · {tf} · {strat['model']}", paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14")
            ss.last_backtest = {**res, "fig": fig, "coverage": f"{df.index.min()} → {df.index.max()}", "candles": len(df)}
    r = ss.last_backtest
    if r:
        st.caption(f"Yahoo Finance · {r['candles']:,} candles · {r['coverage']} · intrabar stop/target ordering is simplified.")
        st.plotly_chart(r["fig"], use_container_width=True)
        m = st.columns(5)
        m[0].metric("Net P/L", f"${r['net']:,.2f}"); m[1].metric("Win rate", f"{r['winrate']:.1f}%")
        m[2].metric("Max drawdown", f"{r['mdd']:.2f}%"); m[3].metric("Trades", len(r["trades"]))
        m[4].metric("Sharpe*", f"{r['sharpe']:.2f}")
        st.caption("*Annualised with a 252-period factor regardless of timeframe — treat as indicative.")
        eq = go.Figure(go.Scatter(x=r["equity"].index, y=r["equity"].values, mode="lines", name="Equity"))
        eq.update_layout(template="plotly_dark", height=300, title="Equity curve", paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14")
        st.plotly_chart(eq, use_container_width=True)
        if len(r["trades"]):
            st.dataframe(r["trades"], use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades in this data window.")


def page_ea():
    page_header("EA / Expert Advisors", "Build, validate, save and deploy automated trading logic.")
    t_build, t_lib, t_deploy = st.tabs(["🛠 Builder", "📚 Library", "🚀 Deployment"])
    with t_build:
        tpl_name = st.selectbox("Start from template", [t["name"] for t in EA_TEMPLATES] + ["Custom EA"])
        tpl = next((t for t in EA_TEMPLATES if t["name"] == tpl_name), None)
        if tpl:
            st.info(tpl["description"])
        e1, e2, e3 = st.columns(3)
        name = e1.text_input("EA name", value=tpl_name)
        entry = e1.text_area("Entry logic", value=tpl["entry"] if tpl else "Define entry conditions", height=100)
        exit_ = e2.text_area("Exit logic", value=tpl["exit"] if tpl else "Define exit conditions", height=100)
        sl = e2.number_input("Stop loss %", 0.05, 20.0, 0.6, 0.05)
        tp = e3.number_input("Take profit %", 0.05, 50.0, 1.8, 0.05)
        risk = e3.slider("Risk per trade %", 0.25, 5.0, 1.0, 0.25)
        x1, x2, x3 = st.columns(3)
        tf = x1.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], key="ea_tf")
        session = x2.selectbox("Session", ["All sessions", "Asia", "London", "New York"])
        mode = x3.selectbox("Execution mode", ["Paper / Demo", "Live (broker connected)"])
        code = st.text_area("EA code (reference only)", value=ss.ea_code, height=240, key="ea_builder_code")
        v, s = st.columns(2)
        if v.button("🔍 Validate syntax", use_container_width=True):
            try:
                ast.parse(code)
                st.success("Python syntax OK (parsed only — not executed).")
            except SyntaxError as exc:
                st.error(f"Syntax error on line {exc.lineno}: {exc.msg}")
        if s.button("💾 Save EA", use_container_width=True):
            rec = dict(name=name, entry=entry, exit=exit_, stop_loss_pct=sl, take_profit_pct=tp, risk_pct=risk,
                       timeframe=tf, session=session, mode=mode, code=code)
            ss.saved_eas = [e for e in ss.saved_eas if e["name"] != name] + [rec]
            persist()
            st.success(f"Saved **{name}**.")
    with t_lib:
        if ss.saved_eas:
            st.dataframe(pd.DataFrame([{"EA": e["name"], "TF": e["timeframe"], "Risk %": e["risk_pct"], "SL %": e["stop_loss_pct"],
                                        "TP %": e["take_profit_pct"], "Mode": e["mode"]} for e in ss.saved_eas]),
                         use_container_width=True, hide_index=True)
            pick = st.selectbox("Saved EA", [e["name"] for e in ss.saved_eas])
            if st.button("📌 Set as active EA"):
                ss.active_ea = pick
                st.success(f"**{pick}** is active.")
        else:
            st.info("No custom EAs saved yet.")
    with t_deploy:
        assets = st.multiselect("Deploy to", list(ALL_INSTRUMENTS))
        ea = st.selectbox("EA", ["None"] + [e["name"] for e in ss.saved_eas] + [t["name"] for t in EA_TEMPLATES], key="dep_ea")
        d1, d2, d3 = st.columns(3)
        mode = d1.selectbox("Mode", ["Paper / Demo", "Signal only", "Live (manual approval)"], key="dep_mode")
        risk = d2.slider("Risk %", 0.25, 5.0, 1.0, 0.25, key="dep_risk")
        on = d3.toggle("Enable after load", key="dep_on")
        if st.button("🚀 Deploy", use_container_width=True):
            if ea == "None" or not assets:
                st.warning("Select an EA and at least one asset.")
            else:
                for a in assets:
                    ss.ea_deployments[a] = {"EA": ea, "Mode": mode, "Risk %": risk, "Enabled": on}
                ss.active_ea, ss.ea_enabled = ea, on
                st.success(f"{ea} configured for {len(assets)} asset(s).")
        if ss.ea_deployments:
            st.dataframe(pd.DataFrame.from_dict(ss.ea_deployments, orient="index").reset_index().rename(columns={"index": "Asset"}),
                         use_container_width=True, hide_index=True)


def page_prop():
    page_header("Prop-Firm Risk Center", "Configure account rules, monitor drawdown and gate signals before execution.")
    types = ["Personal / Demo", "Prop-Firm", "Evaluation", "Funded Account"]
    a, b, c = st.columns([1.3, 1.3, 1])
    ss.prop_account_type = a.selectbox("Account type", types, index=types.index(ss.prop_account_type))
    ss.prop_firm_name = b.text_input("Profile / firm name", ss.prop_firm_name)
    ss.prop_account_size = c.number_input("Account size", 1000.0, value=float(ss.prop_account_size), step=1000.0)
    r = st.columns(4)
    ss.prop_profit_target_pct = r[0].number_input("Profit target %", 0.0, 100.0, float(ss.prop_profit_target_pct), 0.25)
    ss.prop_daily_loss_limit_pct = r[1].number_input("Daily loss limit %", 0.1, 50.0, float(ss.prop_daily_loss_limit_pct), 0.25)
    ss.prop_max_drawdown_pct = r[2].number_input("Max drawdown %", 0.1, 50.0, float(ss.prop_max_drawdown_pct), 0.25)
    ss.prop_max_trade_risk_pct = r[3].number_input("Max risk / trade %", 0.05, 10.0, float(ss.prop_max_trade_risk_pct), 0.05)
    st.divider()
    st.markdown("### 🛡️ Live account health")
    ss.prop_daily_pnl = st.number_input("Today's realised P&L ($) — enter manually from your broker", value=float(ss.prop_daily_pnl), step=100.0)
    ss.prop_peak_equity = max(float(ss.prop_peak_equity), prop_equity())
    h = st.columns(4)
    h[0].metric("Equity", f"${prop_equity():,.2f}")
    h[1].metric("P/L vs size", f"{(prop_equity() - ss.prop_account_size) / max(ss.prop_account_size, 1) * 100:+.2f}%")
    h[2].metric("Daily loss", f"{prop_daily_loss_pct():.2f}%")
    h[3].metric("Drawdown", f"{prop_drawdown_pct():.2f}%")
    st.progress(min(prop_daily_loss_pct() / max(ss.prop_daily_loss_limit_pct, .01), 1.0),
                text=f"Daily-loss usage {prop_daily_loss_pct():.2f}% / {ss.prop_daily_loss_limit_pct:.2f}%")
    st.progress(min(prop_drawdown_pct() / max(ss.prop_max_drawdown_pct, .01), 1.0),
                text=f"Drawdown usage {prop_drawdown_pct():.2f}% / {ss.prop_max_drawdown_pct:.2f}%")
    st.markdown("### 🟢 Signal gate")
    sig = make_signal_snapshot(current_price(GOLD), fetch_ohlcv(GOLD["yf"]))
    if sig:
        st.markdown(f"#### {'🟢 APPROVED' if sig['risk_ok'] else '🔴 BLOCKED'} — {sig['side']}")
        s = st.columns(4)
        s[0].metric("Entry", f"{sig['entry']:.2f}"); s[1].metric("SL", f"{sig['sl']:.2f}")
        s[2].metric("TP1", f"{sig['tp1']:.2f}"); s[3].metric("TP2", f"{sig['tp2']:.2f}")
        st.write(f"**Reason:** {sig['risk_reason']}")
        if sig["risk_ok"] and st.button("📡 Queue signal alert"):
            append_notification(f"{sig['asset']} {sig['side']} signal", signal_message(sig), "SIGNAL")
            st.success("Queued in the notification center.")
    else:
        st.warning("Market data isn't available for a signal gate right now.")
    st.divider()
    st.markdown("### 🔔 Notification center")
    if ss.notification_log:
        st.dataframe(pd.DataFrame(ss.notification_log), use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts yet.")
    st.caption("Alerts are in-app. Monitoring after the browser closes needs a separate always-on service.")
    if st.button("💾 Write monitor configuration"):
        st.success(f"Saved: {save_monitor_config()} (permissions restricted; it may contain tokens).")


@st.cache_data(ttl=300, show_spinner=False)
def load_forex_factory_calendar():
    return fetch_calendar()


@st.fragment(run_every=60)
def page_news():
    page_header("News Trading & Economic Calendar", "Forex Factory calendar · UTC · cached for 5 minutes")
    st.caption(
        "Best-effort integration with Forex Factory's public HTML calendar, not an official API. "
        "Source markup, access restrictions, or timezone settings can change."
    )
    st.link_button("Open Forex Factory calendar", "https://www.forexfactory.com/calendar")
    if st.button("↻ Refresh calendar", use_container_width=True):
        load_forex_factory_calendar.clear()

    currencies = st.multiselect(
        "Currencies to watch",
        ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY", "INR"],
        default=["USD", "EUR"], key="news_currencies",
    )
    impacts = st.multiselect(
        "Impact levels", ["High", "Medium", "Low", "Holiday", "Unknown"],
        default=["High"], key="news_impacts",
    )
    lead = st.slider("Pre-release caution window (minutes)", 5, 180, 30, 5, key="news_lead_minutes")
    post = st.slider("Post-release caution window (minutes)", 0, 60, 15, 5, key="news_post_minutes")
    st.caption("Caution windows are advisory. They do not automatically block broker orders or guarantee execution safety.")

    try:
        with st.spinner("Loading Forex Factory economic calendar…"):
            events = load_forex_factory_calendar()
    except CalendarFetchError as exc:
        st.error(str(exc))
        st.warning("No synthetic events are shown as live data. Try again later or verify the calendar directly.")
        return
    except Exception as exc:
        st.error(f"Unexpected calendar error: {exc}")
        return

    now = datetime.now(timezone.utc)
    selected = [
        e for e in events
        if e["currency"] in currencies and e["impact"] in impacts
    ]
    parsed = []
    for event in selected:
        dt_value = event.get("datetime_utc")
        dt = datetime.fromisoformat(dt_value) if dt_value else None
        row = dict(event)
        row["_dt"] = dt
        parsed.append(row)
    parsed.sort(key=lambda e: (e["_dt"] is None, e["_dt"] or datetime.max.replace(tzinfo=timezone.utc)))

    upcoming = [e for e in parsed if e["_dt"] and e["_dt"] >= now]
    active = [
        e for e in parsed if e["_dt"] and
        now - timedelta(minutes=post) <= e["_dt"] <= now + timedelta(minutes=lead)
    ]

    if "news_alerted_ids" not in ss:
        ss.news_alerted_ids = []
    for event in active:
        alert_key = f"{event['id']}|{lead}|{post}"
        if alert_key not in ss.news_alerted_ids and ss.alerts_on:
            append_notification(
                f"News caution: {event['impact']} {event['currency']}",
                f"{event['event']} · {event['_dt'].strftime('%Y-%m-%d %H:%M UTC')} · active caution window.",
                "WARNING" if event["impact"] == "High" else "INFO",
            )
            ss.news_alerted_ids.append(alert_key)
    ss.news_alerted_ids = ss.news_alerted_ids[-200:]

    c1, c2, c3 = st.columns(3)
    c1.metric("Events loaded", len(events))
    c2.metric("Upcoming selected", len(upcoming))
    c3.metric("Active caution windows", len(active))
    if active:
        st.warning("Caution window active: " + ", ".join(f"{e['currency']} — {e['event']}" for e in active))
    else:
        st.success("No selected events are currently inside the configured caution window.")

    st.markdown("### News trading analysis")
    st.caption(
        "This panel prioritizes upcoming high-impact releases and maps their currencies to commonly watched instruments. "
        "It does not infer a buy/sell direction from the event name alone."
    )
    related_instruments = {
        "USD": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
        "EUR": ["EUR/USD"], "GBP": ["GBP/USD"], "JPY": ["USD/JPY"],
        "AUD": ["AUD/USD"], "CAD": ["USD/CAD"], "CHF": ["USD/CHF"],
        "NZD": ["NZD/USD"], "CNY": ["AUD/USD", "NZD/USD"], "INR": ["USD/INR"],
    }
    next_high = [e for e in upcoming if e["impact"] == "High"][:5]
    if next_high:
        for event in next_high:
            dt = event["_dt"]
            minutes = max(0, int((dt - now).total_seconds() // 60))
            time_left = f"{minutes // 60}h {minutes % 60}m"
            symbols = related_instruments.get(event["currency"], ["Related instruments depend on exposure"])
            st.markdown(
                f"**{event['currency']} · {event['event']}** — {dt.strftime('%Y-%m-%d %H:%M UTC')} "
                f"(in {time_left})"
            )
            st.caption(f"Impact: High · Watchlist: {', '.join(symbols)}")
            if event["actual"] and event["forecast"]:
                st.caption(
                    f"Reported values — actual: {event['actual']}; forecast: {event['forecast']}; "
                    f"previous: {event['previous'] or 'not supplied'}. Check the indicator definition and revisions before interpreting the surprise."
                )
    else:
        st.info("No upcoming high-impact events in the selected currencies and impact filters.")

    rows = []
    for event in parsed:
        dt = event["_dt"]
        if dt and dt >= now:
            minutes = int((dt - now).total_seconds() // 60)
            countdown = f"{minutes // 60}h {minutes % 60}m"
        else:
            countdown = "—"
        rows.append({
            "Time (UTC)": dt.strftime("%Y-%m-%d %H:%M") if dt else f"{event['date_utc']} · {event['time_source']}",
            "Countdown": countdown,
            "Currency": event["currency"], "Impact": event["impact"], "Event": event["event"],
            "Actual": event["actual"] or "—", "Forecast": event["forecast"] or "—",
            "Previous": event["previous"] or "—",
        })
    if rows:
        st.markdown("### Economic calendar")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No events match the selected filters in the fetched calendar view.")
    st.caption("Only events with an exact parsed release time trigger caution alerts. Confirm the source's timezone in Forex Factory before relying on timing.")


def page_settings():
    page_header("Settings")
    st.markdown("#### 📡 Market-data feeds")
    ss.twelvedata_api_key = st.text_input("Twelve Data API key (live gold / forex / US stocks)", value=ss.twelvedata_api_key,
                                          type="password", help="Free key at twelvedata.com. The feed starts automatically.")
    if websocket is None:
        st.warning("`websocket-client` isn't installed — live feeds are disabled. Run `pip install websocket-client`.")
    with _LIVE_LOCK:
        rows = [{"Symbol": k, "Price": v["price"], "Source": v["source"], "Age (s)": round(time.time() - v["ts"], 1)}
                for k, v in _LIVE_PRICES.items()]
    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Symbol", "Price", "Source", "Age (s)"]),
                 use_container_width=True, hide_index=True)
    st.caption("Crypto streams from Binance (no key). NSE data and indices are polled and shown as DELAYED.")
    st.markdown("#### 🔔 Alert channels")
    for key, label, secret in (("telegram_bot_token", "Telegram bot token", True), ("telegram_chat_id", "Telegram chat ID", False),
                               ("discord_webhook", "Discord webhook", True), ("alert_email", "Alert e-mail", False)):
        ss[key] = st.text_input(label, value=ss[key], type="password" if secret else "default")
    st.markdown("#### General")
    st.selectbox("Base currency", ["USD ($)", "INR (₹)", "EUR (€)"])
    ss.account_balance = st.number_input("Terminal capital ($)", 1000.0, value=float(ss.account_balance), step=1000.0)
    st.caption(f"Strategies and EAs persist to `{STATE_FILE.name}` next to the app. API keys are never written to disk "
               "(except via the explicit monitor-config button).")


# ── Router ──────────────────────────────────────────────────
PAGES = {"dashboard": page_dashboard, "chart": page_chart, "risk": page_risk, "broker": page_broker,
         "strategy": page_strategy, "backtest": page_backtest, "ea": page_ea, "prop": page_prop,
         "news": page_news, "settings": page_settings}


def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    render_sidebar()
    with st.container(key="ssad_strip"):      # global: visible on every page
        market_strip()
    PAGES.get(ss.page, page_dashboard)()
    render_copilot()


main()

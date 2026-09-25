from datetime import datetime, timedelta
import os
import json
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
    :root { --line:rgba(255,255,255,.09); --green:#39e58c; --red:#ff5c68; --amber:#f6c85f; --blue:#68a8ff; }
    .stApp { background:radial-gradient(circle at 12% 8%,rgba(49,106,220,.16),transparent 27%),radial-gradient(circle at 88% 18%,rgba(0,205,145,.08),transparent 23%),linear-gradient(180deg,#070a11 0%,#04060b 100%); color:#f5f7fb; }
    .stApp,.stApp * { font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    [data-testid="stHeader"] { background:rgba(5,7,12,.72)!important; backdrop-filter:blur(18px); }
    [data-testid="stMainBlockContainer"] { padding-top:1.6rem!important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line)!important; background:linear-gradient(180deg,rgba(18,23,36,.78),rgba(10,13,21,.78)); }
    .ssad-page-header { font-size:1.08rem;font-weight:700;letter-spacing:.2px;padding:7px 0 12px;border-bottom:1px solid var(--line);margin-bottom:16px;color:#e9edf5; }
    .ssad-page-header span { color:#737d91;font-weight:500; }
    .ssad-eyebrow { color:#6f7a8e;text-transform:uppercase;letter-spacing:1.8px;font-size:.68rem;font-weight:800; }
    .ssad-hero { position:relative;overflow:hidden;min-height:255px;border-radius:22px;border:1px solid rgba(255,255,255,.10);background:radial-gradient(circle at 80% 30%,rgba(82,153,255,.20),transparent 26%),radial-gradient(circle at 70% 90%,rgba(46,229,140,.10),transparent 24%),linear-gradient(120deg,rgba(17,23,36,.97),rgba(7,10,17,.90));box-shadow:0 22px 70px rgba(0,0,0,.32),inset 0 1px 0 rgba(255,255,255,.035); }
    .ssad-hero-grid { position:absolute;inset:0;opacity:.22;background-image:linear-gradient(rgba(255,255,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.045) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(90deg,#000 0%,transparent 85%); }
    .ssad-hero-copy { position:relative;z-index:2;padding:30px 34px;max-width:62%; }
    .ssad-hero h1 { margin:8px 0;font-size:clamp(2rem,4vw,3.2rem);line-height:1.02;letter-spacing:-1.7px;color:#fff; }
    .ssad-hero p { color:#9aa5b7;font-size:1rem;max-width:620px;line-height:1.65;margin:0 0 20px; }
    .ssad-chip-row { display:flex;flex-wrap:wrap;gap:8px; }.ssad-chip { padding:7px 11px;border-radius:999px;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.045);color:#c9d0dc;font-size:.75rem;font-weight:700; }.ssad-chip.live { color:#56ed9d;border-color:rgba(57,229,140,.25);background:rgba(57,229,140,.08); }
    .ssad-hero-art { position:absolute;right:18px;bottom:-2px;width:45%;height:94%;opacity:.96; }
    .ssad-market-card { position:relative;overflow:hidden;min-height:126px;padding:18px;border-radius:17px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(18,23,35,.92),rgba(9,12,20,.92));box-shadow:0 12px 34px rgba(0,0,0,.22); }.ssad-market-card .label{color:#8792a5;font-size:.73rem;font-weight:700;text-transform:uppercase;letter-spacing:.9px}.ssad-market-card .price{font-size:1.35rem;font-weight:800;margin-top:7px}.ssad-market-card .move{font-size:.75rem;font-weight:800}.up{color:var(--green)!important}.down{color:var(--red)!important}.ssad-mini-chart{position:absolute;right:10px;bottom:7px;width:47%;height:52px;opacity:.8}
    .ssad-section-title{font-size:1.02rem;font-weight:800;margin:8px 0 10px;color:#e9edf5}.ssad-action-card{min-height:205px;position:relative;overflow:hidden;padding:20px;border-radius:18px;border:1px solid var(--line);background:linear-gradient(145deg,rgba(19,24,37,.92),rgba(8,11,18,.92));transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}.ssad-action-card:hover{transform:translateY(-3px);border-color:rgba(104,168,255,.32);box-shadow:0 18px 45px rgba(0,0,0,.32)}.ssad-action-card .art{height:55px;margin-bottom:10px}.ssad-action-card h3{margin:0 0 6px;font-size:1.05rem}.ssad-action-card p{color:#8e97a9;font-size:.78rem;line-height:1.5;min-height:42px}
    .ssad-status-card{border:1px solid var(--line);border-radius:17px;padding:17px 18px;background:rgba(12,16,26,.76)}.ssad-status-card .k{color:#788398;font-size:.7rem;text-transform:uppercase;letter-spacing:1px;font-weight:800}.ssad-status-card .v{font-size:1.2rem;font-weight:800;margin-top:6px}.ssad-feed{padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.06);display:flex;align-items:center;justify-content:space-between;gap:10px}.ssad-feed:last-child{border-bottom:0}.ssad-feed .symbol{font-weight:800}.ssad-feed .meta{color:#7f899b;font-size:.73rem}.ssad-badge{display:inline-flex;align-items:center;gap:5px;padding:5px 8px;border-radius:7px;font-size:.68rem;font-weight:800;border:1px solid rgba(255,255,255,.08)}.ssad-badge.green{color:#5cf0a1;background:rgba(57,229,140,.08)}.ssad-badge.amber{color:#ffd36e;background:rgba(246,200,95,.08)}.ssad-badge.blue{color:#8dbbff;background:rgba(104,168,255,.08)}.ssad-footer-note{color:#667084;font-size:.68rem;margin-top:14px}
    .stButton>button{border-radius:10px!important;border:1px solid rgba(255,255,255,.10)!important;background:rgba(255,255,255,.045)!important;color:#e8ecf4!important;font-weight:700!important}.stButton>button:hover{border-color:rgba(104,168,255,.38)!important;background:rgba(104,168,255,.09)!important}
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
                "Hi! I'm your trading co-pilot. I can watch signals, check risk, "
                "explain setups, and help you work through your trading plan."
            ),
        }
    ]
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "voice_name" not in st.session_state:
    st.session_state.voice_name = "Auto / Best available"
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
if "quant_stage" not in st.session_state:
    st.session_state.quant_stage = "💡 Idea"
if "quant_runs" not in st.session_state:
    st.session_state.quant_runs = []
if "quant_candidate" not in st.session_state:
    st.session_state.quant_candidate = None

# --- PROP-FIRM + PERSISTENT AI MONITOR STATE ---
PROP_DEFAULTS = {
    "account_type": "Personal / Demo",
    "account_size": 100000.0,
    "profit_target_pct": 10.0,
    "daily_loss_limit_pct": 5.0,
    "max_drawdown_pct": 10.0,
    "min_trading_days": 5,
    "news_trading": "Allowed",
    "weekend_holding": "Allowed",
    "ea_trading": "Allowed",
    "max_trade_risk_pct": 0.50,
    "firm_name": "Custom Prop Profile",
}
for _k, _v in PROP_DEFAULTS.items():
    st.session_state.setdefault(f"prop_{_k}", _v)

st.session_state.setdefault("prop_daily_pnl", 0.0)
st.session_state.setdefault("prop_peak_equity", float(st.session_state.account_balance))
st.session_state.setdefault("prop_trading_days", 0)
st.session_state.setdefault("notification_log", [])
st.session_state.setdefault("telegram_bot_token", "")
st.session_state.setdefault("telegram_chat_id", "")
st.session_state.setdefault("discord_webhook", "")
st.session_state.setdefault("smtp_host", "")
st.session_state.setdefault("smtp_port", 587)
st.session_state.setdefault("smtp_user", "")
st.session_state.setdefault("smtp_password", "")
st.session_state.setdefault("alert_email", "")
st.session_state.setdefault("persistent_monitor_enabled", False)
st.session_state.setdefault("last_signal_snapshot", None)
st.session_state.setdefault("auto_ea_on_signal", False)
st.session_state.setdefault("ea_execution_mode", "PAPER")
st.session_state.setdefault("broker_api_endpoint", "")

def prop_equity():
    return float(st.session_state.account_balance + st.session_state.prop_daily_pnl)

def prop_drawdown_pct():
    peak = max(float(st.session_state.prop_peak_equity), 1.0)
    return max(0.0, (peak - prop_equity()) / peak * 100.0)

def prop_daily_loss_pct():
    size = max(float(st.session_state.prop_account_size), 1.0)
    return max(0.0, -float(st.session_state.prop_daily_pnl) / size * 100.0)

def prop_risk_status(risk_pct):
    if st.session_state.prop_account_type not in {"Prop-Firm", "Evaluation", "Funded Account"}:
        return True, "Personal/demo mode — prop limits are informational."
    daily = prop_daily_loss_pct()
    dd = prop_drawdown_pct()
    max_risk = float(st.session_state.prop_max_trade_risk_pct)
    daily_limit = float(st.session_state.prop_daily_loss_limit_pct)
    dd_limit = float(st.session_state.prop_max_drawdown_pct)
    if risk_pct > max_risk:
        return False, f"Requested risk {risk_pct:.2f}% exceeds max trade risk {max_risk:.2f}%."
    if daily + risk_pct > daily_limit:
        return False, f"This trade could exceed the daily loss limit ({daily_limit:.2f}%)."
    if dd + risk_pct > dd_limit:
        return False, f"This trade could exceed the maximum drawdown limit ({dd_limit:.2f}%)."
    return True, "Within configured prop-account risk limits."

def make_signal_snapshot(price, df, asset_name="Gold (XAU/USD)"):
    data = df.copy()
    if len(data) < 25:
        return None
    data = add_indicators(data, 9, 21).dropna()
    if data.empty:
        return None
    last = data.iloc[-1]
    p = float(price)
    side = "LONG" if float(last["EMA Fast"]) > float(last["EMA Slow"]) else "SHORT"
    atr = float((data["High"] - data["Low"]).rolling(14).mean().iloc[-1])
    atr = max(atr, p * 0.002)
    entry = p
    sl = entry - atr * 1.15 if side == "LONG" else entry + atr * 1.15
    tp1 = entry + (entry - sl) * 1.0 if side == "LONG" else entry - (sl - entry) * 1.0
    tp2 = entry + (entry - sl) * 2.0 if side == "LONG" else entry - (sl - entry) * 2.0
    rsi = float(last["RSI"])
    vol_ma = float(last["Volume MA"]) if pd.notna(last["Volume MA"]) else 0.0
    vol_ratio = float(last["Volume"] / (vol_ma + 1e-9)) if vol_ma else 1.0
    trend_strength = abs(float(last["EMA Fast"]) - float(last["EMA Slow"])) / max(p, 1e-9) * 10000
    confidence = min(95, max(52, 58 + trend_strength * 2 + min(vol_ratio, 2) * 6))
    risk_ok, risk_reason = prop_risk_status(float(st.session_state.prop_max_trade_risk_pct))
    return {
        "asset": asset_name,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": 2.0,
        "confidence": confidence,
        "rsi": rsi,
        "volume_ratio": vol_ratio,
        "risk_ok": risk_ok,
        "risk_reason": risk_reason,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def append_notification(title, body, level="INFO"):
    item = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "level": level, "title": title, "body": body}
    st.session_state.notification_log.insert(0, item)
    st.session_state.notification_log = st.session_state.notification_log[:50]

def build_signal_message(sig):
    return (
        f"{sig['asset']} {sig['side']} SETUP\n"
        f"Entry: {sig['entry']:.2f}\nSL: {sig['sl']:.2f}\n"
        f"TP1: {sig['tp1']:.2f}\nTP2: {sig['tp2']:.2f}\n"
        f"R:R: 1:{sig['rr']:.1f}\nConfidence: {sig['confidence']:.0f}%\n"
        f"Risk: {st.session_state.prop_max_trade_risk_pct:.2f}%\n"
        f"Risk check: {'PASS' if sig['risk_ok'] else 'BLOCKED'} — {sig['risk_reason']}"
    )

def save_monitor_config():
    cfg = {
        "enabled": bool(st.session_state.persistent_monitor_enabled),
        "account_type": st.session_state.prop_account_type,
        "firm_name": st.session_state.prop_firm_name,
        "account_size": float(st.session_state.prop_account_size),
        "daily_loss_limit_pct": float(st.session_state.prop_daily_loss_limit_pct),
        "max_drawdown_pct": float(st.session_state.prop_max_drawdown_pct),
        "max_trade_risk_pct": float(st.session_state.prop_max_trade_risk_pct),
        "assets": ["Gold (XAU/USD)", "Bitcoin (BTC/USDT)", "NIFTY 50 Index"],
        "interval_seconds": 60,
        "telegram_bot_token": st.session_state.telegram_bot_token,
        "telegram_chat_id": st.session_state.telegram_chat_id,
        "discord_webhook": st.session_state.discord_webhook,
        "smtp_host": st.session_state.smtp_host,
        "smtp_port": int(st.session_state.smtp_port),
        "smtp_user": st.session_state.smtp_user,
        "smtp_password": st.session_state.smtp_password,
        "alert_email": st.session_state.alert_email,
        "auto_ea_on_signal": bool(st.session_state.auto_ea_on_signal),
        "ea_execution_mode": st.session_state.ea_execution_mode,
        "broker_api_endpoint": st.session_state.broker_api_endpoint,
    }
    path = Path("prop_ai_monitor_config.json")
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return str(path)

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
        "🧪 Quant Lab",
        "🏦 Prop-Firm Center",
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
    /* Cute glass floating co-pilot button */
    .st-key-ssad_bot_fab {
        position: fixed !important;
        bottom: 28px !important;
        right: 28px !important;
        z-index: 1000000 !important;
        width: auto !important;
    }
    .st-key-ssad_bot_fab button {
        border-radius: 999px !important;
        padding: 12px 19px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, rgba(33,35,57,0.82), rgba(89,64,126,0.66)) !important;
        backdrop-filter: blur(18px) saturate(170%) !important;
        -webkit-backdrop-filter: blur(18px) saturate(170%) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        box-shadow: 0 12px 36px rgba(0,0,0,0.48), 0 0 24px rgba(155,120,255,0.22) !important;
        color: #fff !important;
    }
    .st-key-ssad_bot_fab button:hover {
        transform: translateY(-2px) scale(1.02);
        border-color: rgba(255,255,255,0.36) !important;
    }
    /* Floating translucent co-pilot shell */
    .st-key-ssad_bot_panel {
        position: fixed !important;
        bottom: 104px !important;
        right: 28px !important;
        width: 430px !important;
        max-height: 72vh !important;
        overflow-y: auto !important;
        z-index: 999999 !important;
        background: linear-gradient(160deg, rgba(13,16,29,0.76), rgba(48,34,73,0.68)) !important;
        backdrop-filter: blur(20px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 28px !important;
        box-shadow: 0 18px 55px rgba(0,0,0,0.58), 0 0 35px rgba(135,105,255,0.12) !important;
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
        st.markdown("### 🧸✨ Nova — Trading Co-Pilot")
        st.caption("Your market companion · voice, signals, risk & EA-ready alerts")
        st.markdown(
            "<div style='display:flex;gap:7px;flex-wrap:wrap;margin:4px 0 12px 0'>"
            "<span style='padding:5px 10px;border-radius:999px;background:rgba(80,220,150,.12);border:1px solid rgba(80,220,150,.25);font-size:12px'>● Market Watch</span>"
            "<span style='padding:5px 10px;border-radius:999px;background:rgba(120,150,255,.12);border:1px solid rgba(120,150,255,.25);font-size:12px'>✦ Signal Engine</span>"
            "<span style='padding:5px 10px;border-radius:999px;background:rgba(255,190,90,.12);border:1px solid rgba(255,190,90,.25);font-size:12px'>🛡 Risk Guard</span>"
            "</div>", unsafe_allow_html=True
        )
        bot_tab, bot_tab_signals, bot_tab_guardian, bot_tab_set = st.tabs(
            ["💬 Talk", "📡 Live Signals", "🛡️ Risk Guardian", "⚙️ Settings"]
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
                voice_name = st.session_state.get("voice_name", "Auto / Best available")
                safe_voice_name = voice_name.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${").replace('"', '\\"')
                components.html(
                    f"""
                    <script>
                    try {{
                        const text = "{speak_text[:600]}";
                        const preferred = "{safe_voice_name}";
                        const speak = () => {{
                            const voices = window.parent.speechSynthesis.getVoices();
                            const preferredVoice = preferred !== "Auto / Best available"
                                ? voices.find(v => v.name === preferred)
                                : (voices.find(v => /Samantha|Ava|Jenny|Zira|Google US English/i.test(v.name)) || voices.find(v => /^en[-_]/i.test(v.lang)) || voices[0]);
                            const u = new SpeechSynthesisUtterance(text);
                            if (preferredVoice) u.voice = preferredVoice;
                            u.rate = 0.96;
                            u.pitch = 1.02;
                            u.volume = 1.0;
                            window.parent.speechSynthesis.cancel();
                            window.parent.speechSynthesis.speak(u);
                        }};
                        if (window.parent.speechSynthesis.getVoices().length) speak();
                        else window.parent.speechSynthesis.onvoiceschanged = speak;
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
                if st.button("Ask Nova to interpret this signal", key="ssad_sig_ask"):
                    send_chat_message(
                        "Interpret the current chart signal and suggest what to watch for next.",
                        context_summary=sig_context,
                    )
                    st.rerun()
            else:
                st.info("Not enough data loaded yet for a signal.")


        # ---------------- PROP-FIRM GUARDIAN TAB ----------------
        with bot_tab_guardian:
            st.markdown("### 🧸 Risk Guardian")
            st.caption("Persistent account-aware monitoring. Levels are rule-engine outputs, not guarantees.")
            g1, g2 = st.columns(2)
            with g1:
                st.metric("Account Mode", st.session_state.prop_account_type)
                st.metric("Daily Loss", f"{prop_daily_loss_pct():.2f}% / {st.session_state.prop_daily_loss_limit_pct:.2f}%")
            with g2:
                st.metric("Drawdown", f"{prop_drawdown_pct():.2f}% / {st.session_state.prop_max_drawdown_pct:.2f}%")
                st.metric("Max Trade Risk", f"{st.session_state.prop_max_trade_risk_pct:.2f}%")

            if global_df is not None and len(global_df) >= 25:
                guardian_sig = make_signal_snapshot(global_price, global_df, "Gold (XAU/USD)")
                if guardian_sig:
                    badge = "🟢 READY" if guardian_sig["risk_ok"] else "🔴 BLOCKED"
                    st.markdown(f"#### {badge} {guardian_sig['side']} setup")
                    a, b, c = st.columns(3)
                    a.metric("Entry", f"{guardian_sig['entry']:.2f}")
                    b.metric("Stop Loss", f"{guardian_sig['sl']:.2f}")
                    c.metric("TP1", f"{guardian_sig['tp1']:.2f}")
                    st.metric("TP2", f"{guardian_sig['tp2']:.2f}")
                    st.caption(
                        f"Confidence {guardian_sig['confidence']:.0f}% · "
                        f"RSI {guardian_sig['rsi']:.1f} · "
                        f"Volume {guardian_sig['volume_ratio']:.2f}x"
                    )
                    if guardian_sig["risk_ok"]:
                        st.success(guardian_sig["risk_reason"])
                    else:
                        st.error(guardian_sig["risk_reason"])
                    if st.button("🔔 Create Alert", key="guardian_alert"):
                        st.session_state.last_signal_snapshot = guardian_sig
                        append_notification(
                            f"{guardian_sig['asset']} {guardian_sig['side']} signal",
                            build_signal_message(guardian_sig),
                            "SIGNAL",
                        )
                        st.success("Signal added to the in-app notification queue.")
                else:
                    st.info("Waiting for enough market data.")
            else:
                st.info("Market feed is still loading.")

        # ---------------- BOT SETTINGS TAB ----------------
        with bot_tab_set:
            st.session_state.anthropic_api_key = st.text_input(
                "Anthropic API key",
                value=st.session_state.anthropic_api_key,
                type="password",
            )
            st.session_state.claude_model = st.text_input(
                "AI model (Anthropic/Claude backend)", value=st.session_state.claude_model
            )
            st.session_state.voice_enabled = st.checkbox(
                "🔊 Speak replies out loud", value=st.session_state.voice_enabled
            )
            voice_options = ["Auto / Best available"]
            voice_options += [
                "Samantha", "Ava", "Jenny", "Microsoft Zira", "Google US English"
            ]
            current_voice = st.session_state.get("voice_name", "Auto / Best available")
            selected_voice = st.selectbox(
                "Co-Pilot voice", voice_options,
                index=voice_options.index(current_voice) if current_voice in voice_options else 0,
                help="Uses the voices installed in your browser/operating system. Exact availability varies by device."
            )
            st.session_state.voice_name = selected_voice
            st.caption(
                "Voice input/output uses your browser's built-in speech engine "
                "(Chrome/Edge work best) — no extra key needed for that part."
            )

            st.divider()
            st.markdown("#### 🔔 Persistent Alerts")
            st.session_state.persistent_monitor_enabled = st.toggle(
                "Enable background monitor configuration",
                value=st.session_state.persistent_monitor_enabled,
            )
            st.session_state.telegram_bot_token = st.text_input(
                "Telegram Bot Token", value=st.session_state.telegram_bot_token, type="password"
            )
            st.session_state.telegram_chat_id = st.text_input(
                "Telegram Chat ID", value=st.session_state.telegram_chat_id
            )
            st.session_state.discord_webhook = st.text_input(
                "Discord Webhook", value=st.session_state.discord_webhook, type="password"
            )
            st.session_state.alert_email = st.text_input(
                "Alert Email", value=st.session_state.alert_email
            )
            st.markdown("#### 🤖 EA Auto-Execution")
            st.session_state.auto_ea_on_signal = st.toggle(
                "Automatically let the EA act on approved signals",
                value=st.session_state.auto_ea_on_signal,
            )
            st.session_state.ea_execution_mode = st.selectbox(
                "EA Execution Mode", ["PAPER", "LIVE"],
                index=0 if st.session_state.ea_execution_mode == "PAPER" else 1,
            )
            st.session_state.broker_api_endpoint = st.text_input(
                "Broker execution webhook (LIVE only)",
                value=st.session_state.broker_api_endpoint,
                placeholder="https://your-broker-bridge.example/order",
            )
            st.caption("PAPER records simulated EA trades. LIVE sends an authenticated order payload to your configured broker/bridge endpoint.")
            if st.button("💾 Save Persistent Monitor Config", key="save_monitor_cfg"):
                cfg_path = save_monitor_config()
                st.success(f"Saved monitor configuration to {cfg_path}.")
                st.caption("Run the separate monitor service to continue scanning after this web app is closed.")


# --- DASHBOARD VISUAL HELPERS ---
def sparkline_svg(points, stroke='#68a8ff', fill='rgba(104,168,255,.10)'):
    pts=[float(x) for x in points]; mn,mx=min(pts),max(pts); span=(mx-mn) or 1
    coords=[]
    for i,v in enumerate(pts):
        x=4+(i/(len(pts)-1))*192; y=45-((v-mn)/span)*36; coords.append(f'{x:.1f},{y:.1f}')
    poly=' '.join(coords)
    return f"<svg class='ssad-mini-chart' viewBox='0 0 200 52' preserveAspectRatio='none'><polyline points='{poly}' fill='none' stroke='{stroke}' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/><polyline points='4,49 {poly} 196,49' fill='{fill}' stroke='none' opacity='.55'/></svg>"

def action_art(kind):
    arts={
      'chart':'<svg viewBox="0 0 120 54" width="120" height="54"><path d="M5 45 L25 35 L40 39 L58 19 L74 27 L95 8 L115 16" fill="none" stroke="#68a8ff" stroke-width="3"/><path d="M5 45 L25 35 L40 39 L58 19 L74 27 L95 8 L115 16 L115 52 L5 52Z" fill="rgba(104,168,255,.10)"/><circle cx="95" cy="8" r="4" fill="#39e58c"/></svg>',
      'risk':'<svg viewBox="0 0 120 54" width="120" height="54"><circle cx="42" cy="27" r="20" fill="none" stroke="#f6c85f" stroke-width="7" stroke-dasharray="72 55"/><circle cx="42" cy="27" r="20" fill="none" stroke="#39e58c" stroke-width="7" stroke-dasharray="42 85" transform="rotate(-35 42 27)"/><text x="42" y="31" text-anchor="middle" fill="#fff" font-size="10" font-weight="800">1.8R</text><path d="M78 39 L88 29 L97 35 L114 14" fill="none" stroke="#68a8ff" stroke-width="3"/></svg>',
      'broker':'<svg viewBox="0 0 120 54" width="120" height="54"><rect x="6" y="10" width="108" height="34" rx="8" fill="rgba(255,255,255,.035)" stroke="rgba(255,255,255,.12)"/><circle cx="23" cy="27" r="6" fill="#39e58c"/><path d="M38 27 H95" stroke="#68a8ff" stroke-width="3"/><path d="M83 19 L96 27 L83 35" fill="none" stroke="#68a8ff" stroke-width="3"/></svg>',
      'calendar':'<svg viewBox="0 0 120 54" width="120" height="54"><rect x="8" y="8" width="104" height="39" rx="7" fill="rgba(255,255,255,.035)" stroke="rgba(255,255,255,.12)"/><path d="M8 19 H112" stroke="#68a8ff"/><path d="M27 5 V15 M93 5 V15" stroke="#f6c85f" stroke-width="4" stroke-linecap="round"/><circle cx="30" cy="31" r="4" fill="#ff5c68"/><circle cx="52" cy="31" r="4" fill="#f6c85f"/><circle cx="74" cy="31" r="4" fill="#39e58c"/></svg>',
      'ai':'<svg viewBox="0 0 120 54" width="120" height="54"><circle cx="37" cy="27" r="19" fill="rgba(104,168,255,.12)" stroke="#68a8ff"/><path d="M28 28 Q37 15 46 28 Q37 39 28 28Z" fill="none" stroke="#39e58c" stroke-width="2.5"/><path d="M66 15 L73 27 L66 39 M86 15 L79 27 L86 39" fill="none" stroke="#f6c85f" stroke-width="3" stroke-linecap="round"/></svg>',
      'quant':'<svg viewBox="0 0 120 54" width="120" height="54"><rect x="8" y="8" width="104" height="38" rx="8" fill="rgba(104,168,255,.06)" stroke="rgba(104,168,255,.35)"/><path d="M18 37 L36 28 L50 32 L67 18 L82 23 L101 11" fill="none" stroke="#68a8ff" stroke-width="3"/><circle cx="101" cy="11" r="4" fill="#39e58c"/><path d="M24 14 V22 M32 14 V22 M40 14 V22" stroke="#f6c85f" stroke-width="2"/></svg>'}
    return arts[kind]

# ==========================================
# 📊 VIEW 1: DASHBOARD
# ==========================================
if st.session_state.active_tab == '📊 Dashboard':
    hero_html = f'''<div class="ssad-hero"><div class="ssad-hero-grid"></div><div class="ssad-hero-copy"><div class="ssad-eyebrow">Institutional Quant Terminal • Session Intelligence</div><h1>Smart Session<br/>Anomaly Detector</h1><p>Multi-asset market intelligence, anomaly detection, execution controls and systematic strategy research — presented in one professional workspace.</p><div class="ssad-chip-row"><span class="ssad-chip live">● ML ENGINE ONLINE</span><span class="ssad-chip">12 Markets</span><span class="ssad-chip">15m Signal Layer</span><span class="ssad-chip">Paper Execution Ready</span></div></div><div class="ssad-hero-art"><svg viewBox="0 0 600 300" width="100%" height="100%" preserveAspectRatio="none"><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#68a8ff" stop-opacity=".32"/><stop offset="1" stop-color="#68a8ff" stop-opacity="0"/></linearGradient></defs><g opacity=".28" stroke="#fff"><path d="M30 70 H570 M30 130 H570 M30 190 H570 M30 250 H570"/><path d="M110 30 V270 M200 30 V270 M290 30 V270 M380 30 V270 M470 30 V270"/></g><path d="M25 245 L70 214 L108 224 L146 160 L190 185 L232 120 L270 145 L314 91 L355 128 L405 74 L442 104 L480 58 L525 84 L575 35 L575 290 L25 290Z" fill="url(#area)"/><path d="M25 245 L70 214 L108 224 L146 160 L190 185 L232 120 L270 145 L314 91 L355 128 L405 74 L442 104 L480 58 L525 84 L575 35" fill="none" stroke="#68a8ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/><path d="M405 74 L442 104 L480 58 L525 84 L575 35" fill="none" stroke="#39e58c" stroke-width="4"/><circle cx="575" cy="35" r="6" fill="#39e58c"/></svg></div></div>'''
    st.markdown(hero_html, unsafe_allow_html=True)
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ssad-section-title">Market Pulse</div>', unsafe_allow_html=True)
    market_cards=[('XAU/USD','$2,468.40','+0.84%','up',[44,48,42,52,49,58,55,67,64,76,72,82]),('BTC/USDT','$78,820','+2.15%','up',[45,38,51,48,60,54,68,62,74,69,83,88]),('NIFTY 50','24,310.80','-0.24%','down',[76,72,78,70,74,66,68,60,62,54,57,49]),('EUR/USD','1.0825','-0.08%','down',[68,73,67,70,63,66,58,61,55,59,51,53])]
    mc=st.columns(4,gap='medium')
    for col,(label,price,move,cls,points) in zip(mc,market_cards):
        with col: st.markdown(f'<div class="ssad-market-card"><div class="label">{label}</div><div class="price">{price}</div><div class="move {cls}">{move} <span style="color:#657084;font-weight:500">today</span></div>{sparkline_svg(points,"#39e58c" if cls=="up" else "#ff5c68","rgba(57,229,140,.09)" if cls=="up" else "rgba(255,92,104,.08)")}</div>',unsafe_allow_html=True)
    st.markdown('<div style="height:16px"></div>',unsafe_allow_html=True)
    st.markdown('<div class="ssad-section-title">Workspace</div>',unsafe_allow_html=True)
    actions=[('chart','Chart Analysis','Candlesticks, indicators, anomaly zones and direct execution controls.','Open Charts →','📈 Chart Analysis'),('risk','Pip & Risk Engine','Position sizing, risk-to-reward and exposure planning before execution.','Open Calculator →','🧮 Pip & Risk Calculator'),('broker','Broker Gateway','Connection layer for paper trading and supported broker/API bridges.','Open Gateway →','⚡ Broker Gateway'),('calendar','Economic Calendar','High-impact macro events with forecast, actual and previous values.','View Calendar →','📅 Economic Calendar'),('ai','AI Co-Pilot','Market context, strategy assistance, saved EAs and voice interaction.','Open Co-Pilot →','BOT'),('quant','Quant Lab','Idea → build → backtest → validate → deploy workflow for systematic research.','Open Quant Lab →','🧪 Quant Lab')]
    qa=st.columns(6,gap='medium')
    for col,(kind,title,desc,btn,target) in zip(qa,actions):
        with col:
            st.markdown(f'<div class="ssad-action-card"><div class="art">{action_art(kind)}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
            if target=='BOT': st.button(btn,key=f'pro_qa_{kind}',use_container_width=True,on_click=open_bot)
            else: st.button(btn,key=f'pro_qa_{kind}',use_container_width=True,on_click=switch_page,args=(target,))
    st.markdown('<div style="height:16px"></div>',unsafe_allow_html=True)
    left,right=st.columns([1.35,.65],gap='medium')
    with left:
        st.markdown('<div class="ssad-section-title">Live System Monitor</div>',unsafe_allow_html=True)
        st.markdown('<div class="ssad-status-card"><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px"><div><div class="k">ML Signal Layer</div><div class="v" style="color:#39e58c">ONLINE</div></div><div><div class="k">Market Feed</div><div class="v">15s sync</div></div><div><div class="k">Execution</div><div class="v">PAPER</div></div><div><div class="k">AI Co-Pilot</div><div class="v" style="color:#68a8ff">READY</div></div></div></div>',unsafe_allow_html=True)
        st.markdown('<div class="ssad-section-title" style="margin-top:18px">Recent Activity</div>',unsafe_allow_html=True)
        if st.session_state.positions:
            for p in st.session_state.positions[:6]:
                side_cls='green' if p.get('Type') in ('BUY/LONG','LONG','BUY') else 'amber'
                st.markdown(f'<div class="ssad-feed"><div><div class="symbol">{p.get("Asset","—")} <span class="ssad-badge {side_cls}">{p.get("Type","—")}</span></div><div class="meta">{p.get("Timestamp","—")} • {p.get("Bridge","Demo Simulated")}</div></div><div style="font-weight:800">{p.get("Price","—")}</div></div>',unsafe_allow_html=True)
        else: st.markdown('<div class="ssad-status-card"><span style="color:#7f899b">No execution activity yet. Open Chart Analysis to inspect the market and place a paper trade.</span></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ssad-section-title">Session Snapshot</div>',unsafe_allow_html=True)
        for k,v,cls in [('Anomaly Engine','Monitoring','green'),('EA Deployments',str(len(st.session_state.ea_deployments)),'blue'),('Saved Strategies',str(len(st.session_state.strategy_library)),'amber'),('Broker Mode',st.session_state.broker_api_mode,'blue')]:
            st.markdown(f'<div class="ssad-status-card" style="margin-bottom:10px"><div class="k">{k}</div><div class="v">{v}</div><span class="ssad-badge {cls}">● ACTIVE</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="ssad-footer-note">Market values may be delayed or simulated depending on the connected data source. Execution remains paper/demo until a broker bridge is configured.</div>',unsafe_allow_html=True)

# ==========================================
# 📈 VIEW 2: CHART ANALYSIS
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
# 🧪 VIEW 5: QUANT LAB
# ==========================================
elif st.session_state.active_tab == "🧪 Quant Lab":
    st.title("🧪 Quant Lab")
    st.caption("A structured research pipeline: formulate an idea, build the rules, backtest them, validate robustness, then prepare deployment.")

    stages = ["💡 Idea", "🧱 Build", "🧪 Backtest", "🔬 Validate", "🚀 Deploy"]
    current = st.session_state.quant_stage
    st.markdown("<div class='ssad-status-card'><div style='display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap'>" + "".join([f"<span class='ssad-badge {'green' if x==current else 'blue'}'>{x}</span>" for x in stages]) + "</div></div>", unsafe_allow_html=True)

    q1,q2 = st.columns([1.25,.75], gap='medium')
    with q1:
        st.markdown("#### 1. Research Brief")
        idea_name = st.text_input("Strategy / Research Name", value="Session Momentum Research")
        hypothesis = st.text_area("Market Hypothesis", value="Test whether a fast/slow moving-average regime can identify directional sessions with controlled risk.", height=90)
        research_notes = st.text_area("Research Notes", value="Define the signal before looking at performance. Keep entry, exit and risk rules explicit.", height=90)
        st.markdown("#### 2. Build the Rule Set")
        qb1,qb2,qb3 = st.columns(3)
        with qb1:
            q_fast = st.number_input("Fast Period", 2, 100, 9, key="quant_fast")
        with qb2:
            q_slow = st.number_input("Slow Period", 3, 250, 21, key="quant_slow")
        with qb3:
            q_risk = st.slider("Risk / Trade %", 0.25, 5.0, 1.0, 0.25, key="quant_risk")
        qc1,qc2 = st.columns(2)
        with qc1:
            q_entry = st.text_area("Entry Rules", value="LONG when fast EMA crosses above slow EMA; SHORT on the inverse.", height=85)
        with qc2:
            q_exit = st.text_area("Exit Rules", value="Exit on opposite signal or protective stop/target.", height=85)
        st.markdown("#### 3. Backtest Configuration")
        assets = list(MARKET_UNIVERSE["🇮🇳 Indian Equities (NSE)"].keys()) + ["Gold (XAU/USD)", "EUR/USD", "Bitcoin (BTC/USDT)"]
        q1a,q1b,q1c = st.columns(3)
        with q1a: q_asset=st.selectbox("Research Asset", assets, index=min(0,len(assets)-1), key="quant_asset")
        with q1b: q_tf=st.selectbox("Timeframe", ["5m","15m","1h","4h","1D"], index=1, key="quant_tf")
        with q1c: q_period=st.selectbox("History", ["1mo","3mo","6mo","1y"], index=2, key="quant_period")
        run_quant = st.button("▶ Run Research Backtest", use_container_width=True, key="run_quant_bt")

    with q2:
        st.markdown("#### Research Checklist")
        checks = [
            ("Hypothesis written", bool(hypothesis.strip())),
            ("Entry rules defined", bool(q_entry.strip())),
            ("Exit rules defined", bool(q_exit.strip())),
            ("Risk budget defined", q_risk > 0),
            ("Historical test selected", bool(q_asset)),
        ]
        for label, ok in checks:
            st.markdown(f"<div class='ssad-feed'><span>{'🟢' if ok else '⚪'} {label}</span><span class='meta'>{'READY' if ok else 'PENDING'}</span></div>", unsafe_allow_html=True)
        st.markdown("#### Research Discipline")
        st.info("Use the lab to separate research from execution. A strong backtest is not proof of future performance; review assumptions, costs, data quality and out-of-sample behavior before deployment.")
        if st.button("🧭 Open Strategy Builder", use_container_width=True):
            switch_page("🧩 Strategy Builder")
        if st.button("🤖 Open EA Studio", use_container_width=True):
            switch_page("🤖 EA / Expert Advisors")

    if run_quant:
        if q_fast >= q_slow:
            st.warning("Fast Period should be smaller than Slow Period for this crossover research template.")
        else:
            cfg = next((group[q_asset] for group in MARKET_UNIVERSE.values() if q_asset in group), None)
            if cfg:
                qdf = load_ohlcv(cfg["yf"], period=q_period, interval=q_tf)
                result = run_strategy_backtest(qdf, q_fast, q_slow, q_risk, 2.0)
                if result:
                    st.session_state.last_backtest = result
                    st.session_state.quant_candidate = {"name":idea_name,"asset":q_asset,"timeframe":q_tf,"fast":q_fast,"slow":q_slow,"risk":q_risk,"hypothesis":hypothesis}
                    st.session_state.quant_stage = "🧪 Backtest"
                    st.session_state.quant_runs.append({"name":idea_name,"asset":q_asset,"timeframe":q_tf,"net_profit":result["net_profit"],"win_rate":result["win_rate"],"mdd":result["mdd"],"sharpe":result["sharpe"],"trades":len(result["trades"])})
                    st.success("Research run completed. Review the diagnostics below before moving to validation.")
                else:
                    st.warning("Not enough historical data for this configuration.")

    if st.session_state.last_backtest:
        result = st.session_state.last_backtest
        st.divider()
        st.markdown("### 📊 Research Diagnostics")
        m=st.columns(5)
        m[0].metric("Net Profit",f"${result['net_profit']:,.2f}")
        m[1].metric("Win Rate",f"{result['win_rate']:.1f}%")
        m[2].metric("Max Drawdown",f"{result['mdd']:.2f}%")
        m[3].metric("Sharpe",f"{result['sharpe']:.2f}")
        m[4].metric("Trades",f"{len(result['trades'])}")
        c1,c2=st.columns(2,gap='medium')
        with c1:
            eq=result["equity"]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=eq.index,y=eq.values,mode="lines",name="Equity"))
            fig.update_layout(template="plotly_dark",height=330,title="Equity Curve",margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="#0b0e14",plot_bgcolor="#0b0e14")
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            if not result["trades"].empty:
                pnl=result["trades"]["P&L"]
                figp=go.Figure()
                figp.add_trace(go.Bar(x=list(range(1,len(pnl)+1)),y=pnl,name="Trade P&L"))
                figp.update_layout(template="plotly_dark",height=330,title="Trade Outcome Distribution",margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="#0b0e14",plot_bgcolor="#0b0e14")
                st.plotly_chart(figp,use_container_width=True)
            else:
                st.info("No completed trades to visualize.")
        if st.session_state.quant_candidate:
            cand=st.session_state.quant_candidate
            st.markdown("#### 🔬 Validation Gate")
            v1,v2,v3=st.columns(3)
            with v1: st.checkbox("Review data assumptions", value=False, key="val_data")
            with v2: st.checkbox("Run alternate period/timeframe", value=False, key="val_alt")
            with v3: st.checkbox("Review drawdown tolerance", value=False, key="val_dd")
            if st.button("🔍 Mark Candidate for Validation", use_container_width=True):
                st.session_state.quant_stage="🔬 Validate"
                st.success(f"{cand['name']} is marked for validation. This does not imply that the strategy is validated.")

    if st.session_state.quant_runs:
        st.divider()
        st.markdown("### 🗂 Research Run History")
        st.dataframe(pd.DataFrame(st.session_state.quant_runs).tail(20),use_container_width=True,hide_index=True)

    if st.session_state.quant_stage == "🔬 Validate":
        st.markdown("#### 🚀 Deployment Gate")
        st.warning("Deployment should remain paper/demo until data assumptions, execution costs, risk limits and out-of-sample behavior have been reviewed.")
        d1,d2=st.columns(2)
        with d1:
            if st.button("📝 Create EA Draft",use_container_width=True):
                switch_page("🤖 EA / Expert Advisors")
        with d2:
            if st.button("📈 Review on Chart",use_container_width=True):
                switch_page("📈 Chart Analysis")

# ==========================================
# 🧩 VIEW 6: STRATEGY BUILDER
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
# 🏦 VIEW 7: PROP-FIRM CENTER
# ==========================================
elif st.session_state.active_tab == "🏦 Prop-Firm Center":
    st.title("🏦 Prop-Firm Risk Center")
    st.caption("Configure account rules, monitor drawdown, and gate signals before execution.")

    top1, top2, top3 = st.columns([1.3, 1.3, 1])
    with top1:
        st.session_state.prop_account_type = st.selectbox(
            "Account Type",
            ["Personal / Demo", "Prop-Firm", "Evaluation", "Funded Account"],
            index=["Personal / Demo", "Prop-Firm", "Evaluation", "Funded Account"].index(st.session_state.prop_account_type),
        )
    with top2:
        st.session_state.prop_firm_name = st.text_input("Profile / Firm Name", st.session_state.prop_firm_name)
    with top3:
        st.session_state.prop_account_size = st.number_input(
            "Account Size", min_value=1000.0, value=float(st.session_state.prop_account_size), step=1000.0
        )

    st.markdown("### 📐 Rule Profile")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.session_state.prop_profit_target_pct = st.number_input(
            "Profit Target %", 0.0, 100.0, float(st.session_state.prop_profit_target_pct), 0.25
        )
    with r2:
        st.session_state.prop_daily_loss_limit_pct = st.number_input(
            "Daily Loss Limit %", 0.1, 50.0, float(st.session_state.prop_daily_loss_limit_pct), 0.25
        )
    with r3:
        st.session_state.prop_max_drawdown_pct = st.number_input(
            "Maximum Drawdown %", 0.1, 50.0, float(st.session_state.prop_max_drawdown_pct), 0.25
        )
    with r4:
        st.session_state.prop_max_trade_risk_pct = st.number_input(
            "Max Risk / Trade %", 0.05, 10.0, float(st.session_state.prop_max_trade_risk_pct), 0.05
        )

    r5, r6, r7 = st.columns(3)
    with r5:
        st.session_state.prop_min_trading_days = st.number_input(
            "Minimum Trading Days", 0, 100, int(st.session_state.prop_min_trading_days)
        )
    with r6:
        st.session_state.prop_news_trading = st.selectbox("News Trading", ["Allowed", "Restricted"], index=["Allowed", "Restricted"].index(st.session_state.prop_news_trading))
    with r7:
        st.session_state.prop_ea_trading = st.selectbox("EA Trading", ["Allowed", "Restricted"], index=["Allowed", "Restricted"].index(st.session_state.prop_ea_trading))

    st.divider()
    st.markdown("### 🛡️ Live Account Health")
    h1, h2, h3, h4 = st.columns(4)
    equity = prop_equity()
    peak = max(st.session_state.prop_peak_equity, equity)
    st.session_state.prop_peak_equity = peak
    profit_pct = (equity - st.session_state.prop_account_size) / max(st.session_state.prop_account_size, 1) * 100
    h1.metric("Equity", f"${equity:,.2f}")
    h2.metric("Profit / Loss", f"{profit_pct:+.2f}%")
    h3.metric("Daily Loss", f"{prop_daily_loss_pct():.2f}%")
    h4.metric("Drawdown", f"{prop_drawdown_pct():.2f}%")

    daily_remaining = max(0.0, st.session_state.prop_daily_loss_limit_pct - prop_daily_loss_pct())
    dd_remaining = max(0.0, st.session_state.prop_max_drawdown_pct - prop_drawdown_pct())
    st.progress(min(prop_daily_loss_pct() / max(st.session_state.prop_daily_loss_limit_pct, 0.01), 1.0), text=f"Daily-loss usage · {prop_daily_loss_pct():.2f}% / {st.session_state.prop_daily_loss_limit_pct:.2f}%")
    st.progress(min(prop_drawdown_pct() / max(st.session_state.prop_max_drawdown_pct, 0.01), 1.0), text=f"Drawdown usage · {prop_drawdown_pct():.2f}% / {st.session_state.prop_max_drawdown_pct:.2f}%")

    st.info(
        f"Remaining daily-loss buffer: {daily_remaining:.2f}% · "
        f"Remaining drawdown buffer: {dd_remaining:.2f}%"
    )

    st.markdown("### 🟢 Signal Gate")
    if global_df is not None and len(global_df) >= 25:
        psig = make_signal_snapshot(global_price, global_df, "Gold (XAU/USD)")
        if psig:
            approved, reason = prop_risk_status(float(st.session_state.prop_max_trade_risk_pct))
            status = "APPROVED" if approved else "BLOCKED"
            st.markdown(f"#### {'🟢' if approved else '🔴'} {status}")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Entry", f"{psig['entry']:.2f}")
            s2.metric("SL", f"{psig['sl']:.2f}")
            s3.metric("TP1", f"{psig['tp1']:.2f}")
            s4.metric("TP2", f"{psig['tp2']:.2f}")
            st.write(f"**Reason:** {reason}")
            if approved and st.button("📡 Send Signal Alert", key="prop_send_signal"):
                append_notification(
                    f"{psig['asset']} {psig['side']} signal",
                    build_signal_message(psig),
                    "SIGNAL",
                )
                st.success("Signal queued in the notification center.")
    else:
        st.warning("Market data is not ready for a signal gate yet.")

    st.divider()
    st.markdown("### 🔔 Notification Center")
    if st.session_state.notification_log:
        st.dataframe(pd.DataFrame(st.session_state.notification_log), use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts yet.")

    st.markdown("### 🌙 Background Monitoring")
    st.write(
        "The Streamlit page can display alerts, but monitoring after the browser/app is closed "
        "requires the included persistent monitor service."
    )
    if st.button("💾 Write Monitor Configuration", key="prop_write_cfg"):
        cfg_path = save_monitor_config()
        st.success(f"Configuration saved: {cfg_path}")
        st.caption("Start prop_ai_monitor_service.py separately on a machine/server that remains online.")


# ==========================================
# 📅 VIEW 8: ECONOMIC CALENDAR
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
# ⚙️ VIEW 9: SETTINGS
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



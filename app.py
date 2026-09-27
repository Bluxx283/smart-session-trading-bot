from datetime import datetime, timedelta, time as dtime
from collections import deque
import os
import json
import re
import threading
import time
import base64
from pathlib import Path

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

try:
    import websocket
except ImportError:
    websocket = None


# ============================================================
# LIVE MARKET DATA ENGINE
# ============================================================

_LIVE_LOCK = threading.Lock()

_LIVE_PRICES = {}
_LIVE_HISTORY = {}


def _update_live(symbol, price, prev_close=None, source="ws"):
    with _LIVE_LOCK:

        previous = _LIVE_PRICES.get(symbol, {})

        pc = (
            prev_close
            if prev_close is not None
            else previous.get("prev_close", price)
        )

        pct = (
            ((price - pc) / pc) * 100.0
            if pc
            else 0.0
        )

        _LIVE_PRICES[symbol] = {
            "price": float(price),
            "prev_close": float(pc),
            "pct": float(pct),
            "ts": time.time(),
            "source": source,
        }

        history = _LIVE_HISTORY.setdefault(
            symbol,
            deque(maxlen=60)
        )

        history.append(float(price))


def get_live(symbol, max_age=120):

    with _LIVE_LOCK:

        item = _LIVE_PRICES.get(symbol)

        history = list(
            _LIVE_HISTORY.get(symbol, [])
        )

    if not item:
        return None

    if time.time() - item["ts"] > max_age:
        return None

    result = dict(item)

    result["history"] = history

    return result


# ============================================================
# BINANCE LIVE STREAM
# ============================================================

def _binance_ws_worker(streams):

    if websocket is None:
        return

    url = (
        "wss://stream.binance.com:9443/stream?streams="
        + "/".join(streams)
    )

    while True:

        try:

            ws = websocket.create_connection(
                url,
                timeout=20
            )

            while True:

                message = ws.recv()

                if not message:
                    break

                data = json.loads(message)

                payload = data.get("data", {})

                symbol = payload.get("s")
                price = payload.get("c")
                previous = payload.get("x")

                if symbol and price:

                    _update_live(
                        symbol,
                        float(price),
                        float(previous)
                        if previous
                        else None,
                        source="Binance"
                    )

        except Exception:
            time.sleep(3)


# ============================================================
# FINNHUB LIVE STREAM
# ============================================================

def _finnhub_ws_worker(api_key, symbols):

    if websocket is None:
        return

    url = (
        "wss://ws.finnhub.io?token="
        + api_key
    )

    while True:

        try:

            ws = websocket.create_connection(
                url,
                timeout=20
            )

            for symbol in symbols:

                ws.send(
                    json.dumps({
                        "type": "subscribe",
                        "symbol": symbol
                    })
                )

            while True:

                message = ws.recv()

                if not message:
                    break

                data = json.loads(message)

                for trade in data.get("data", []):

                    symbol = trade.get("s")
                    price = trade.get("p")

                    if symbol and price:

                        _update_live(
                            symbol,
                            float(price),
                            source="Finnhub"
                        )

        except Exception:
            time.sleep(5)


# ============================================================
# START LIVE FEEDS
# ============================================================

@st.cache_resource
def start_live_feeds(finnhub_key=""):

    threads = []

    crypto_thread = threading.Thread(
        target=_binance_ws_worker,
        args=(
            [
                "btcusdt@ticker",
                "ethusdt@ticker"
            ],
        ),
        daemon=True
    )

    crypto_thread.start()

    threads.append(
        crypto_thread
    )

    if finnhub_key:

        market_thread = threading.Thread(
            target=_finnhub_ws_worker,
            args=(
                finnhub_key,
                [
                    "OANDA:XAU_USD",
                    "OANDA:EUR_USD",
                    "AAPL",
                    "NVDA"
                ]
            ),
            daemon=True
        )

        market_thread.start()

        threads.append(
            market_thread
        )

    return threads


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=(
        "Smart Session Anomaly Detector "
        "| Quant Trading Terminal"
    ),
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL UI
# ============================================================

st.markdown(
    """
<style>

:root {

    --line:
        rgba(255,255,255,.09);

    --green:
        #39e58c;

    --red:
        #ff5c68;

    --amber:
        #f6c85f;

    --blue:
        #68a8ff;

    --cyan:
        #55efc2;

    --purple:
        #a97cff;
}


/* =========================================
   APPLICATION BACKGROUND
   ========================================= */

.stApp {

    background:
        radial-gradient(
            circle at 12% 8%,
            rgba(49,106,220,.16),
            transparent 27%
        ),

        radial-gradient(
            circle at 88% 18%,
            rgba(0,205,145,.08),
            transparent 23%
        ),

        linear-gradient(
            180deg,
            #070a11 0%,
            #04060b 100%
        );

    color:
        #f5f7fb;
}


.stApp,
.stApp * {

    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


[data-testid="stHeader"] {

    background:
        rgba(5,7,12,.72)
        !important;

    backdrop-filter:
        blur(18px);
}


[data-testid="stMainBlockContainer"] {

    padding-top:
        1.6rem !important;
}


/* =========================================
   BASIC BUTTONS
   ========================================= */

.stButton > button {

    border-radius:
        10px !important;

    border:
        1px solid
        rgba(255,255,255,.10)
        !important;

    background:
        rgba(255,255,255,.045)
        !important;

    color:
        #e8ecf4
        !important;

    font-weight:
        700
        !important;

    transition:
        all .18s ease;
}


.stButton > button:hover {

    border-color:
        rgba(104,168,255,.38)
        !important;

    background:
        rgba(104,168,255,.09)
        !important;
}


/* =========================================
   MOVING FLOATING MARKET CARDS
   ========================================= */

.ssad-floating-wrap {

    position:
        relative;

    margin:
        2px -4px 24px;

    overflow:
        hidden;

    padding:
        5px 4px 12px;

    mask-image:
        linear-gradient(
            90deg,
            transparent 0%,
            #000 4%,
            #000 96%,
            transparent 100%
        );
}


.ssad-floating-track {

    display:
        flex;

    align-items:
        stretch;

    gap:
        14px;

    width:
        max-content;

    animation:
        ssadMarketFloat
        30s
        linear
        infinite;

    will-change:
        transform;
}


/*
   Stop the moving market tape when
   the user hovers over it.
*/

.ssad-floating-wrap:hover
.ssad-floating-track {

    animation-play-state:
        paused;
}


/* =========================================
   MARKET CARD
   ========================================= */

.ssad-floating-market {

    position:
        relative;

    flex:
        0 0 194px;

    min-width:
        194px;

    height:
        104px;

    padding:
        13px;

    overflow:
        hidden;

    border:
        1px solid
        rgba(104,168,255,.22);

    border-radius:
        16px;

    background:
        linear-gradient(
            145deg,
            rgba(18,29,48,.94),
            rgba(7,12,22,.96)
        );

    box-shadow:
        0 12px 30px
        rgba(0,0,0,.24),

        inset 0 1px 0
        rgba(255,255,255,.04);

    backdrop-filter:
        blur(14px);

    transition:
        transform .2s ease,
        border-color .2s ease,
        box-shadow .2s ease;
}


.ssad-floating-market:hover {

    transform:
        translateY(-5px);

    border-color:
        rgba(104,168,255,.52);

    box-shadow:
        0 18px 42px
        rgba(0,0,0,.34),

        0 0 24px
        rgba(45,135,255,.08);
}


/* =========================================
   MARKET CARD CONTENT
   ========================================= */

.ssad-floating-top {

    display:
        flex;

    align-items:
        center;

    gap:
        7px;

    color:
        #d8e3f3;

    font-size:
        .70rem;

    font-weight:
        800;

    white-space:
        nowrap;
}


.ssad-floating-dot {

    width:
        7px;

    height:
        7px;

    flex:
        0 0 auto;

    border-radius:
        50%;

    background:
        #39e58c;

    box-shadow:
        0 0 10px
        rgba(57,229,140,.85);
}


.ssad-floating-price {

    margin-top:
        8px;

    color:
        #f3f6fb;

    font-size:
        1.08rem;

    font-weight:
        850;

    letter-spacing:
        -.3px;
}


.ssad-floating-move {

    margin-top:
        2px;

    font-size:
        .67rem;

    font-weight:
        800;
}


.ssad-floating-market
.ssad-mini-chart {

    position:
        absolute;

    right:
        7px;

    bottom:
        5px;

    width:
        43%;

    height:
        38px;

    opacity:
        .58;
}


/* =========================================
   CONTINUOUS MOVEMENT
   ========================================= */

@keyframes ssadMarketFloat {

    0% {

        transform:
            translateX(0);
    }

    100% {

        transform:
            translateX(
                calc(-50% - 7px)
            );
    }
}


/* =========================================
   SMALL VERTICAL VARIATION
   ========================================= */

.ssad-floating-track
.ssad-floating-market:nth-child(2n) {

    transform:
        translateY(4px);
}


.ssad-floating-track
.ssad-floating-market:nth-child(3n) {

    transform:
        translateY(-2px);
}


/* =========================================
   ADD MARKET BUTTON
   ========================================= */

.ssad-floating-add {

    position:
        absolute;

    right:
        8px;

    top:
        50%;

    transform:
        translateY(-50%);

    z-index:
        10;

    width:
        48px !important;

    height:
        48px !important;

    padding:
        0 !important;

    border-radius:
        50% !important;

    font-size:
        1.4rem !important;

    color:
        #b8c9df !important;

    background:
        rgba(5,10,18,.92)
        !important;

    border:
        1px solid
        rgba(104,168,255,.40)
        !important;

    box-shadow:
        0 10px 28px
        rgba(0,0,0,.30)
        !important;
}


/* =========================================
   HERO
   ========================================= */

.ssad-hero {

    position:
        relative;

    overflow:
        hidden;

    min-height:
        390px;

    margin-top:
        2px;

    border-radius:
        22px;

    border:
        1px solid
        rgba(255,255,255,.10);

    background:

        radial-gradient(
            circle at 82% 38%,
            rgba(40,126,255,.18),
            transparent 25%
        ),

        radial-gradient(
            circle at 70% 78%,
            rgba(57,229,140,.10),
            transparent 24%
        ),

        linear-gradient(
            120deg,
            rgba(17,23,36,.98),
            rgba(7,10,17,.94)
        );

    box-shadow:
        0 22px 70px
        rgba(0,0,0,.32),

        inset 0 1px 0
        rgba(255,255,255,.035);
}


/* =========================================
   HERO GRID
   ========================================= */

.ssad-hero-grid {

    position:
        absolute;

    inset:
        0;

    opacity:
        .22;

    background-image:

        linear-gradient(
            rgba(255,255,255,.045)
            1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(255,255,255,.045)
            1px,
            transparent 1px
        );

    background-size:
        42px 42px;

    mask-image:
        linear-gradient(
            90deg,
            #000 0%,
            transparent 85%
        );
}


/* =========================================
   HERO COPY
   ========================================= */

.ssad-hero-copy {

    position:
        relative;

    z-index:
        2;

    max-width:
        78%;

    padding:
        58px 52px;
}


.ssad-eyebrow {

    color:
        #6f7a8e;

    text-transform:
        uppercase;

    letter-spacing:
        1.8px;

    font-size:
        .68rem;

    font-weight:
        800;
}


.ssad-eyebrow span {

    color:
        #65758d;

    margin:
        0 6px;
}


.ssad-hero h1 {

    margin:
        12px 0 17px;

    color:
        #fff;

    font-size:
        clamp(
            2.7rem,
            5vw,
            4.6rem
        );

    line-height:
        1.02;

    letter-spacing:
        -2.8px;
}


.ssad-hero h1 span {

    background:
        linear-gradient(
            90deg,
            #55efc2,
            #55b8ff 48%,
            #a97cff
        );

    -webkit-background-clip:
        text;

    background-clip:
        text;

    color:
        transparent;
}


.ssad-hero p {

    max-width:
        690px;

    margin:
        0 0 25px;

    color:
        #9aa5b7;

    font-size:
        .98rem;

    line-height:
        1.65;
}


/* =========================================
   HERO CHIPS
   ========================================= */

.ssad-chip-row {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;
}


.ssad-chip {

    padding:
        7px 11px;

    border-radius:
        999px;

    border:
        1px solid
        rgba(255,255,255,.10);

    background:
        rgba(255,255,255,.045);

    color:
        #c9d0dc;

    font-size:
        .75rem;

    font-weight:
        700;
}


.ssad-chip.live {

    color:
        #56ed9d;

    border-color:
        rgba(57,229,140,.25);

    background:
        rgba(57,229,140,.08);
}


.ssad-chip.blue {

    color:
        #8dbbff;

    border-color:
        rgba(104,168,255,.25);
}


.ssad-chip.purple {

    color:
        #c19bff;

    border-color:
        rgba(165,108,255,.28);
}


.ssad-chip.cyan {

    color:
        #6eeaff;

    border-color:
        rgba(0,216,255,.25);
}


/* =========================================
   HERO FEATURE LINE
   ========================================= */

.ssad-feature-line {

    display:
        flex;

    align-items:
        center;

    flex-wrap:
        wrap;

    gap:
        12px;

    margin-top:
        27px;

    color:
        #94a8c1;

    font-size:
        .68rem;

    font-weight:
        800;

    letter-spacing:
        .8px;
}


.ssad-feature-line span:first-child {

    color:
        #69e8b1;
}


.ssad-feature-line b {

    color:
        #53657e;
}


/* =========================================
   GLOBAL MARKET GLOBE
   ========================================= */

.ssad-global-globe {

    position:
        absolute;

    right:
        -1.5%;

    top:
        2%;

    width:
        47%;

    height:
        96%;

    object-fit:
        contain;

    object-position:
        center;

    opacity:
        .90;

    mix-blend-mode:
        screen;

    filter:
        saturate(1.08)
        contrast(1.03);

    pointer-events:
        none;

    user-select:
        none;
}


/* =========================================
   GLOBE GLOW
   ========================================= */

.ssad-hero-glow {

    position:
        absolute;

    right:
        9%;

    top:
        17%;

    width:
        270px;

    height:
        270px;

    border-radius:
        50%;

    background:
        radial-gradient(
            circle,
            rgba(68,155,255,.11),
            transparent 67%
        );
}


/* =========================================
   ORBIT EFFECTS
   ========================================= */

.ssad-hero-orbit {

    position:
        absolute;

    border:
        1px solid
        rgba(87,155,255,.18);

    border-radius:
        50%;

    pointer-events:
        none;
}


.ssad-hero-orbit.orbit-one {

    width:
        310px;

    height:
        310px;

    right:
        5%;

    top:
        11%;

    transform:
        rotate(-22deg);
}


.ssad-hero-orbit.orbit-two {

    width:
        430px;

    height:
        190px;

    right:
        -1%;

    top:
        27%;

    transform:
        rotate(-22deg);

    border-color:
        rgba(57,229,140,.13);
}


/* =========================================
   GLOWING NODES
   ========================================= */

.ssad-hero-node {

    position:
        absolute;

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        #68dfff;

    box-shadow:
        0 0 16px
        rgba(104,223,255,.9);
}


.ssad-hero-node.node-one {

    right:
        21%;

    top:
        20%;
}


.ssad-hero-node.node-two {

    right:
        10%;

    top:
        58%;

    background:
        #39e58c;

    box-shadow:
        0 0 16px
        rgba(57,229,140,.9);
}


.ssad-hero-node.node-three {

    right:
        29%;

    bottom:
        18%;

    background:
        #a87cff;

    box-shadow:
        0 0 16px
        rgba(168,124,255,.8);
}


/* =========================================
   RESPONSIVE
   ========================================= */

@media (max-width: 900px) {

    .ssad-hero {

        min-height:
            430px;
    }

    .ssad-hero-copy {

        max-width:
            100%;

        padding:
            42px 30px;
    }

    .ssad-hero-orbit,
    .ssad-hero-glow {

        opacity:
            .35;

        right:
            -100px;
    }

    .ssad-global-globe {

        right:
            -20%;

        width:
            62%;

        opacity:
            .45;
    }

    .ssad-floating-market {

        min-width:
            165px;

        flex-basis:
            165px;
    }
}


@media (max-width: 650px) {

    .ssad-hero {

        min-height:
            470px;
    }

    .ssad-hero-copy {

        padding:
            34px 22px;
    }

    .ssad-hero h1 {

        font-size:
            2.5rem;
    }

    .ssad-feature-line {

        gap:
            8px;

        font-size:
            .62rem;
    }

    .ssad-global-globe {

        display:
            none;
    }

    .ssad-chip {

        font-size:
            .67rem;
    }
}

</style>
""",
    unsafe_allow_html=True
)

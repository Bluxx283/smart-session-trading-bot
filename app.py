<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quant Trading Terminal</title>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: "Inter", sans-serif;
    background:
        radial-gradient(circle at 75% 35%, rgba(0, 132, 255, 0.08), transparent 30%),
        linear-gradient(135deg, #020a17, #03101f 55%, #020916);
    color: #eaf4ff;
    min-height: 100vh;
    overflow: hidden;
}

/* =========================
   TOP HEADER
========================= */

.topbar {
    height: 64px;
    border-bottom: 1px solid rgba(80, 160, 255, 0.15);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    background: rgba(2, 10, 23, 0.75);
    backdrop-filter: blur(18px);
}

.brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo {
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #00f5d4;
    font-size: 28px;
    font-weight: 800;
}

.brand-name {
    font-size: 16px;
    letter-spacing: 2px;
    font-weight: 700;
}

.brand-name span {
    color: #8fa9c9;
    font-weight: 500;
}

.top-status {
    display: flex;
    align-items: center;
    gap: 28px;
    color: #b8c9df;
    font-size: 13px;
}

.live-status {
    display: flex;
    align-items: center;
    gap: 8px;
}

.live-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #00e6a8;
    box-shadow: 0 0 12px #00e6a8;
}

.user {
    display: flex;
    align-items: center;
    gap: 9px;
}

.avatar {
    width: 31px;
    height: 31px;
    border-radius: 50%;
    background: linear-gradient(135deg, #8db5ff, #c9dcff);
    color: #17263b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}

/* =========================
   SIDEBAR
========================= */

.sidebar {
    position: fixed;
    top: 64px;
    left: 0;
    bottom: 0;
    width: 228px;
    border-right: 1px solid rgba(80, 160, 255, 0.13);
    background: rgba(2, 10, 23, 0.72);
    padding: 22px 12px;
}

.nav-item {
    height: 48px;
    display: flex;
    align-items: center;
    gap: 17px;
    padding: 0 18px;
    margin-bottom: 5px;
    border-radius: 9px;
    color: #a9bdd7;
    font-size: 14px;
    transition: 0.25s;
}

.nav-icon {
    width: 22px;
    text-align: center;
    font-size: 19px;
}

.nav-item:hover {
    background: rgba(0, 200, 255, 0.08);
    color: #fff;
}

.nav-item.active {
    background: linear-gradient(
        90deg,
        rgba(0, 209, 226, 0.22),
        rgba(0, 105, 180, 0.08)
    );
    color: #00f4e4;
    border-left: 3px solid #00e8d0;
}

/* =========================
   MAIN
========================= */

.main {
    margin-left: 228px;
    min-height: calc(100vh - 64px);
    position: relative;
    padding: 50px 42px;
    overflow: hidden;
}

/* =========================
   FLOATING MARKET TICKERS
========================= */

.market-tickers {
    position: absolute;
    top: 42px;
    right: 30px;
    display: flex;
    gap: 14px;
    align-items: center;
}

.market-card {
    min-width: 128px;
    padding: 10px 14px;
    border-radius: 14px;
    border: 1px solid rgba(70, 150, 255, 0.25);
    background: rgba(9, 26, 49, 0.72);
    box-shadow:
        inset 0 0 20px rgba(30, 120, 255, 0.04),
        0 10px 35px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(15px);
}

.market-top {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #dbe8fa;
    font-weight: 600;
}

.market-icon {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.btc {
    background: rgba(247, 147, 26, 0.15);
    color: #ffae39;
}

.apple {
    background: rgba(255,255,255,0.1);
    color: white;
}

.fx {
    background: rgba(40, 130, 255, 0.15);
    color: #67aaff;
}

.gold {
    background: rgba(255, 180, 0, 0.14);
    color: #ffc43d;
}

.eth {
    background: rgba(140, 90, 255, 0.15);
    color: #a980ff;
}

.market-price {
    margin-left: 34px;
    margin-top: -2px;
    font-size: 12px;
    color: #c7d5e8;
}

.change {
    margin-left: 34px;
    font-size: 10px;
    margin-top: 3px;
}

.positive {
    color: #00e6a8;
}

.negative {
    color: #ff5575;
}

.add-market {
    width: 43px;
    height: 43px;
    border-radius: 50%;
    border: 1px solid rgba(80, 150, 255, 0.25);
    background: rgba(10, 28, 51, 0.7);
    color: #b8cce6;
    font-size: 22px;
}

/* =========================
   HERO
========================= */

.hero {
    position: relative;
    max-width: 900px;
    padding-top: 40px;
    z-index: 3;
}

.eyebrow {
    color: #00e9d0;
    font-size: 13px;
    letter-spacing: 3px;
    font-weight: 600;
    margin-bottom: 25px;
}

.eyebrow span {
    color: #7286a5;
    margin: 0 10px;
}

h1 {
    font-size: clamp(55px, 6vw, 82px);
    line-height: 0.98;
    letter-spacing: -4px;
    font-weight: 800;
    max-width: 720px;
}

h1 .gradient {
    background: linear-gradient(
        90deg,
        #00f0c8,
        #00c7ff 45%,
        #a66cff
    );
    -webkit-background-clip: text;
    color: transparent;
}

.description {
    max-width: 650px;
    margin-top: 27px;
    font-size: 17px;
    line-height: 1.65;
    color: #afc2dc;
}

/* =========================
   FEATURE BADGES
========================= */

.features {
    display: flex;
    gap: 13px;
    margin-top: 35px;
    flex-wrap: wrap;
}

.feature {
    height: 47px;
    padding: 0 18px;
    border-radius: 25px;
    border: 1px solid rgba(70, 170, 255, 0.5);
    display: flex;
    align-items: center;
    gap: 9px;
    background: rgba(4, 25, 47, 0.5);
    color: #d4e5fa;
    font-size: 12px;
    letter-spacing: .5px;
    backdrop-filter: blur(10px);
}

.feature.green {
    border-color: #00e6b5;
    color: #00e6b5;
}

.feature.blue {
    border-color: #238cff;
}

.feature.purple {
    border-color: #9a5cff;
    color: #c09bff;
}

.feature.cyan {
    border-color: #00d8ff;
    color: #6eeaff;
}

.feature-icon {
    font-size: 17px;
}

/* =========================
   LOWER FEATURE STRIP
========================= */

.feature-strip {
    display: flex;
    align-items: center;
    gap: 26px;
    margin-top: 38px;
    color: #9db2cf;
    font-size: 12px;
    letter-spacing: .8px;
}

.strip-item {
    display: flex;
    align-items: center;
    gap: 9px;
}

.strip-icon {
    color: #b9d7ff;
    font-size: 18px;
}

.separator {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #65809f;
}

/* =========================
   GLOBE
========================= */

.globe {
    position: absolute;
    right: 15px;
    top: 155px;
    width: 430px;
    height: 430px;
    border-radius: 50%;
    opacity: .82;
    background:
        radial-gradient(
            circle at 35% 35%,
            rgba(0, 226, 255, .2),
            transparent 45%
        ),
        radial-gradient(
            circle,
            rgba(0, 105, 255, .13),
            transparent 65%
        );
    border: 1px solid rgba(0, 172, 255, .4);
    box-shadow:
        inset 0 0 70px rgba(0, 162, 255, .13),
        0 0 60px rgba(0, 137, 255, .08);
}

/* longitude / latitude lines */

.globe::before,
.globe::after {
    content: "";
    position: absolute;
    inset: 7%;
    border-radius: 50%;
    border: 1px solid rgba(0, 210, 255, .25);
}

.globe::after {
    inset: 20%;
    transform: rotate(65deg);
}

/* orbit rings */

.orbit {
    position: absolute;
    inset: -30px;
    border: 1px solid rgba(93, 81, 255, .45);
    border-radius: 50%;
    transform: rotate(-25deg);
}

.orbit.two {
    inset: 5px;
    transform: rotate(42deg);
    border-color: rgba(0, 230, 210, .35);
}

.globe-dot {
    position: absolute;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00e8ff;
    box-shadow: 0 0 15px #00e8ff;
}

.dot1 { top: 28%; left: 22%; }
.dot2 { top: 46%; right: 22%; }
.dot3 { bottom: 27%; left: 47%; }
.dot4 { top: 20%; right: 40%; }
.dot5 { bottom: 35%; right: 18%; }

/* =========================
   BOTTOM WAVE
========================= */

.wave {
    position: absolute;
    bottom: -100px;
    left: -100px;
    width: 700px;
    height: 220px;
    opacity: .3;
    border-top: 1px solid rgba(24, 126, 255, .4);
    border-radius: 50%;
    transform: rotate(8deg);
}

.wave::before,
.wave::after {
    content: "";
    position: absolute;
    width: 100%;
    height: 100%;
    border-top: 1px solid rgba(24, 126, 255, .3);
    border-radius: 50%;
    top: 20px;
}

.wave::after {
    top: 42px;
}

/* =========================
   FOOTER MESSAGE
========================= */

.footer-message {
    position: absolute;
    bottom: 45px;
    left: 42px;
    color: #8fa5c1;
    font-size: 13px;
}

.footer-message span {
    color: #00e5ca;
    font-size: 20px;
    vertical-align: middle;
    margin-right: 10px;
}

/* =========================
   RESPONSIVE
========================= */

@media (max-width: 1200px) {

    .market-tickers {
        right: 20px;
        transform: scale(.85);
        transform-origin: right top;
    }

    .globe {
        right: -80px;
        opacity: .45;
    }
}

@media (max-width: 900px) {

    body {
        overflow: auto;
    }

    .sidebar {
        display: none;
    }

    .main {
        margin-left: 0;
        padding: 30px 25px;
    }

    .market-tickers {
        position: relative;
        top: auto;
        right: auto;
        margin-bottom: 35px;
        transform: none;
        flex-wrap: wrap;
    }

    .hero {
        padding-top: 0;
    }

    .globe {
        opacity: .18;
        right: -120px;
    }

    h1 {
        font-size: 55px;
    }
}

@media (max-width: 600px) {

    .top-status {
        display: none;
    }

    h1 {
        font-size: 45px;
        letter-spacing: -2px;
    }

    .description {
        font-size: 15px;
    }

    .feature-strip {
        flex-wrap: wrap;
    }

    .globe {
        display: none;
    }

    .market-card {
        min-width: 115px;
    }
}
</style>
</head>

<body>

<!-- =========================
     TOP BAR
========================= -->

<header class="topbar">

    <div class="brand">

        <div class="logo">▮▮▮</div>

        <div class="brand-name">
            QUANT TRADING TERMINAL
            <span>•</span>
            <span>SESSION INTELLIGENCE</span>
        </div>

    </div>

    <div class="top-status">

        <div class="live-status">
            <div class="live-dot"></div>
            Live Market Data
        </div>

        <div id="clock">10:24:17 (UTC)</div>

        <div>♧</div>

        <div class="user">
            <div class="avatar">T</div>
            Trader
            <span>⌄</span>
        </div>

    </div>

</header>


<!-- =========================
     SIDEBAR
========================= -->

<aside class="sidebar">

    <div class="nav-item active">
        <div class="nav-icon">⌂</div>
        Home
    </div>

    <div class="nav-item">
        <div class="nav-icon">◎</div>
        Markets
    </div>

    <div class="nav-item">
        <div class="nav-icon">☆</div>
        Watchlist
    </div>

    <div class="nav-item">
        <div class="nav-icon">⌁</div>
        Analysis
    </div>

    <div class="nav-item">
        <div class="nav-icon">♧</div>
        Anomaly Detector
    </div>

    <div class="nav-item">
        <div class="nav-icon">ϟ</div>
        Signals
    </div>

    <div class="nav-item">
        <div class="nav-icon">▣</div>
        Backtesting
    </div>

    <div class="nav-item">
        <div class="nav-icon">△</div>
        Strategy Lab
    </div>

    <div class="nav-item">
        <div class="nav-icon">▤</div>
        Journal
    </div>

    <div class="nav-item">
        <div class="nav-icon">⚙</div>
        Settings
    </div>

</aside>


<!-- =========================
     MAIN DASHBOARD
========================= -->

<main class="main">

    <!-- Floating market tickers -->

    <div class="market-tickers">

        <div class="market-card">
            <div class="market-top">
                <div class="market-icon btc">₿</div>
                BTC
            </div>
            <div class="market-price">$67,432.18</div>
            <div class="change positive">+1.82%</div>
        </div>

        <div class="market-card">
            <div class="market-top">
                <div class="market-icon apple">●</div>
                AAPL
            </div>
            <div class="market-price">$175.32</div>
            <div class="change positive">+1.21%</div>
        </div>

        <div class="market-card">
            <div class="market-top">
                <div class="market-icon fx">$</div>
                EURUSD
            </div>
            <div class="market-price">1.0724</div>
            <div class="change negative">-0.21%</div>
        </div>

        <div class="market-card">
            <div class="market-top">
                <div class="market-icon gold">◆</div>
                XAUUSD
            </div>
            <div class="market-price">2,336.45</div>
            <div class="change positive">+0.72%</div>
        </div>

        <div class="market-card">
            <div class="market-top">
                <div class="market-icon eth">♦</div>
                ETH
            </div>
            <div class="market-price">$2,345.76</div>
            <div class="change positive">+2.31%</div>
        </div>

        <button class="add-market">+</button>

    </div>


    <!-- Globe -->

    <div class="globe">

        <div class="orbit"></div>
        <div class="orbit two"></div>

        <div class="globe-dot dot1"></div>
        <div class="globe-dot dot2"></div>
        <div class="globe-dot dot3"></div>
        <div class="globe-dot dot4"></div>
        <div class="globe-dot dot5"></div>

    </div>


    <!-- Hero -->

    <section class="hero">

        <div class="eyebrow">
            QUANT TRADING TERMINAL
            <span>•</span>
            SESSION INTELLIGENCE
        </div>

        <h1>
            Smart Session<br>
            <span class="gradient">Anomaly Detector</span>
        </h1>

        <p class="description">
            Multi-asset market intelligence, anomaly detection,
            execution controls and systematic strategy research —
            all in one professional workspace.
        </p>


        <!-- Feature badges -->

        <div class="features">

            <div class="feature green">
                <span class="feature-icon">♧</span>
                ML ENGINE ONLINE
                <span>●</span>
            </div>

            <div class="feature blue">
                <span class="feature-icon">◎</span>
                MULTI-MARKET
            </div>

            <div class="feature purple">
                <span class="feature-icon">◷</span>
                MULTI-TIMEFRAME ANALYSIS
            </div>

            <div class="feature cyan">
                <span class="feature-icon">↗</span>
                FLEXIBLE EXECUTION
            </div>

        </div>


        <!-- Feature strip -->

        <div class="feature-strip">

            <div class="strip-item">
                <span class="strip-icon">◉</span>
                LIVE MARKET DATA
            </div>

            <div class="separator"></div>

            <div class="strip-item">
                <span class="strip-icon">♧</span>
                ANOMALY MONITORING
            </div>

            <div class="separator"></div>

            <div class="strip-item">
                <span class="strip-icon">◎</span>
                SIGNAL GENERATION
            </div>

        </div>

    </section>


    <div class="footer-message">
        <span>|</span>
        Smarter Analysis
        &nbsp; • &nbsp;
        Better Decisions
        &nbsp; • &nbsp;
        Greater Opportunities
    </div>

    <div class="wave"></div>

</main>


<script>

/* =========================
   LIVE UTC CLOCK
========================= */

function updateClock() {

    const now = new Date();

    const time = now.toLocaleTimeString("en-US", {
        timeZone: "UTC",
        hour12: false
    });

    document.getElementById("clock").textContent =
        time + " (UTC)";
}

updateClock();
setInterval(updateClock, 1000);


/* =========================
   SUBTLE MARKET CARD FLOAT
========================= */

const cards = document.querySelectorAll(".market-card");

cards.forEach((card, index) => {

    card.style.animation =
        `floatCard ${3 + index * .35}s ease-in-out infinite`;

    card.style.animationDelay =
        `${index * .15}s`;

});


const style = document.createElement("style");

style.innerHTML = `
@keyframes floatCard {

    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-4px);
    }

}
`;

document.head.appendChild(style);

</script>

</body>
</html>

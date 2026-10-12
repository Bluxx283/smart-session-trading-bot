# 📈 Smart Session Market Pattern & Anomaly Analyzer

> An autonomous AI trading system and market pattern analyzer that combines real-time session monitoring with unsupervised machine learning to detect volatility shifts and structural price anomalies across global trading sessions.

---

## 🌟 Key Features

- **Session-Aware Feature Extraction:** Dynamically computes rolling standard deviations, volatility expansions, volume surges, and dynamic Z-scores across Asian, London, and New York market hours.
- **Unsupervised Anomaly Detection:** Employs an Isolation Forest model to flag statistical outliers, liquidity spikes, and structural trend breaks without fixed indicator rules.
- **Zero-Install Web Dashboard:** Built with Streamlit for live evaluation, allowing instant access for non-technical users and evaluators.
- **Broker & Account Integration:** Designed to bridge detection signals directly to MetaTrader 5 (MT5) and broker execution endpoints.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (Isolation Forest)
- **Data Processing:** Pandas, NumPy
- **Frontend / Dashboard:** Streamlit
- **Computer Vision:** OpenCV (Screen-Watcher Module)

---

## 🚀 How to Run Locally

1. **Clone the repository:**


## Beginner-friendly mode

The app opens in **Beginner** mode by default. Use the **Experience level** selector in the sidebar to switch between:

- **Beginner:** simple navigation, a guided home page, a short trading glossary, and plain-language notes on charts, news, risk, and backtesting.
- **Advanced:** the full set of strategy, backtesting, Expert Advisor, broker, and prop-firm tools.

### Suggested first session

1. Open **Markets** to explore a chart and check whether quotes are live or delayed.
2. Open **News & Calendar** to review scheduled events and their impact labels.
3. Open **Risk Calculator** to estimate risk before considering a hypothetical trade.
4. Use **Test a Strategy** to explore historical results before using any strategy.

### Important safety notes

- The broker page currently displays a session-level demo/connection flag; it does not verify broker credentials or send live orders.
- Calendar caution alerts are informational. They do not automatically prevent trading.
- Market data can be delayed or unavailable. Forex Factory page markup/timezone behavior can change.
- Backtests are simulations, not forecasts or guarantees of future results.

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

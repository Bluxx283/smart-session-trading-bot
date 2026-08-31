import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Smart Session Trading Bot", layout="wide")

st.title("📈 Smart Session Market Pattern & Anomaly Analyzer")
st.markdown("Live AI session monitoring and volatility breakdown across **Asian, London, and NY** market hours.")

# Sidebar controls
session = st.sidebar.selectbox("Select Active Market Session", ["Asian Session", "London Session", "New York Session"])
sensitivity = st.sidebar.slider("Anomaly Sensitivity (Contamination %)", 0.01, 0.15, 0.05)

# Simulated Market Tick Data
np.random.seed(42)
n_points = 200
prices = 2000 + np.cumsum(np.random.randn(n_points) * 1.5)
prices[150:155] += np.array([8, 12, 15, 10, 6])
volumes = np.random.randint(100, 1000, size=n_points)

df = pd.DataFrame({
    'timestamp': pd.date_range(start='2026-01-01', periods=n_points, freq='1min'),
    'price': prices,
    'volume': volumes
})

# Feature Engineering
df['returns'] = df['price'].pct_change()
df['rolling_volatility'] = df['returns'].rolling(window=20).std()
df['rolling_mean'] = df['price'].rolling(window=20).mean()
df['rolling_std'] = df['price'].rolling(window=20).std()
df['z_score'] = (df['price'] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
df['vol_mean'] = df['volume'].rolling(window=20).mean()
df['volume_surge'] = df['volume'] / (df['vol_mean'] + 1e-8)
clean_df = df.dropna().copy()

# Isolation Forest Model
model = IsolationForest(n_estimators=100, contamination=sensitivity, random_state=42)
clean_df['is_anomaly'] = model.fit_predict(clean_df[['rolling_volatility', 'z_score', 'volume_surge']].values) == -1

# Metrics Layout
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"Price Action & Real-Time Monitoring ({session})")
    # Display clean dataframe line chart
    chart_data = clean_df.set_index('timestamp')[['price']]
    st.line_chart(chart_data)

with col2:
    st.metric("Total Data Points", len(clean_df))
    anomalies_detected = int(clean_df['is_anomaly'].sum())
    st.metric("Detected Anomalies", anomalies_detected)
    st.caption(f"Session Status: 🟢 **Active** ({session})")

st.subheader("Statistical Features & Anomaly Highlights")
st.dataframe(clean_df.tail(15), use_container_width=True)

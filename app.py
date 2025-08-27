import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("💼 Investmate")
st.caption("Your personal robo-advisor simulator • Educational demo only")

# ETFs
etfs = {
    "Stocks": "VTI",
    "Bonds": "AGG",
    "Cash": "BIL"
}

# Download data
data = yf.download(list(etfs.values()), start="2020-01-01", end="2025-01-01")["Close"]
data.columns = etfs.keys()
data = data.dropna()

st.subheader("ETF Prices (first 5 rows)")
st.write(data.head())

# Normalize prices
normalized = data / data.iloc[0] * 100

# Portfolio allocations
allocations = {
    "Conservative": {"Stocks": 0.2, "Bonds": 0.7, "Cash": 0.1},
    "Balanced":     {"Stocks": 0.5, "Bonds": 0.4, "Cash": 0.1},
    "Aggressive":   {"Stocks": 0.8, "Bonds": 0.15, "Cash": 0.05}
}

# Portfolio values
portfolio_values = pd.DataFrame(index=normalized.index)
for profile, weights in allocations.items():
    portfolio_values[profile] = sum(normalized[a] * w for a, w in weights.items())

# Plot
st.subheader("Portfolio Growth (2020–2025)")
fig, ax = plt.subplots(figsize=(10,6))
for col in portfolio_values.columns:
    ax.plot(portfolio_values.index, portfolio_values[col], label=col)
ax.set_title("Portfolio Value (Start = $100)")
ax.set_xlabel("Date")
ax.set_ylabel("Value ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Final values
st.subheader("Final Portfolio Values")
st.write(portfolio_values.tail(1).T)

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="Investmate", page_icon="💼", layout="wide")
st.title("💼 Investmate")
st.caption("Your personal robo-advisor simulator • Educational demo only")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
# Dates
default_start = date(2020, 1, 1)
start = st.sidebar.date_input("Start date", default_start, min_value=date(2010,1,1), max_value=date.today()-timedelta(days=3))
end   = st.sidebar.date_input("End date",   date.today())

# Starting amount
initial = st.sidebar.number_input("Starting value ($)", min_value=100, max_value=1_000_000, value=1000, step=100)

# ETFs universe
ETFS = {"Stocks":"VTI", "Bonds":"AGG", "Cash":"BIL"}
st.sidebar.markdown("**Assets**")
st.sidebar.write(", ".join(f"{k} ({v})" for k,v in ETFS.items()))

# Risk profiles
ALLOCATIONS = {
    "Conservative": {"Stocks":0.20, "Bonds":0.70, "Cash":0.10},
    "Balanced":     {"Stocks":0.50, "Bonds":0.40, "Cash":0.10},
    "Aggressive":   {"Stocks":0.80, "Bonds":0.15, "Cash":0.05},
}
profile = st.sidebar.selectbox("Risk profile", list(ALLOCATIONS.keys()), index=1)

show_data = st.sidebar.checkbox("Show raw prices table", value=False)

# ---------- Data download ----------
@st.cache_data(show_spinner=True)
def load_prices(tickers, start, end):
    df = yf.download(list(tickers), start=start, end=end, progress=False)
    # yfinance sometimes returns a column MultiIndex with 'Close'
    if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
        df = df["Close"]
    df = df.dropna(how="all")
    return df

prices = load_prices(list(ETFS.values()), start, end)
if prices.empty:
    st.error("No data returned for the selected dates. Try widening the date range.")
    st.stop()

# Rename cols to asset classes
prices.columns = list(ETFS.keys())
if show_data:
    st.subheader("ETF Prices (first 5 rows)")
    st.write(prices.head())

# ---------- Portfolio math ----------
normalized = prices / prices.iloc[0] * 100  # start at 100
portfolio_values = pd.DataFrame(index=normalized.index)
for name, weights in ALLOCATIONS.items():
    portfolio_values[name] = sum(normalized[a]*w for a, w in weights.items())

# Scale to starting value
portfolio_values = portfolio_values / 100 * initial

# ---------- Charts ----------
st.subheader("Portfolio Growth")
fig, ax = plt.subplots(figsize=(10,5))
for col in portfolio_values.columns:
    ax.plot(portfolio_values.index, portfolio_values[col], label=col)
ax.set_title(f"Portfolio Value (Start = ${initial:,})")
ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
ax.legend(); ax.grid(True)
st.pyplot(fig)

# ---------- Summary metrics ----------
st.subheader("Summary at End Date")
last_row = portfolio_values.tail(1).T
last_row.columns = ["Final Value ($)"]
years = max((prices.index[-1] - prices.index[0]).days / 365.25, 0.01)

cols = st.columns(3)
for i, name in enumerate(["Conservative","Balanced","Aggressive"]):
    final_val = float(last_row.loc[name, "Final Value ($)"])
    total_ret = final_val/initial - 1
    cagr = (final_val/initial)**(1/years) - 1
    cols[i].metric(
        label=name,
        value=f"${final_val:,.0f}",
        delta=f"{total_ret*100:+.1f}% (total) • {cagr*100:.1f}% CAGR"
    )

st.divider()
st.markdown("**Selected allocation**")
st.table(pd.Series(ALLOCATIONS[profile], name=profile).map(lambda x: f"{x:.0%}"))

# Download results
csv = portfolio_values.to_csv().encode()
st.download_button("Download portfolio values (CSV)", csv, "portfolio_values.csv", "text/csv")

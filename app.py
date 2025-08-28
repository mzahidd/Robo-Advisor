import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="Investmate", page_icon="💼", layout="wide")
st.title("💼 Investmate")
st.caption("Your personal robo-advisor simulator • Educational demo only")

# ---------------- Sidebar: basics ----------------
st.sidebar.header("Settings")

default_start = date(2020, 1, 1)
start = st.sidebar.date_input("Start date", default_start,
                              min_value=date(2010,1,1),
                              max_value=date.today()-timedelta(days=3))
end   = st.sidebar.date_input("End date",   date.today())

initial = st.sidebar.number_input("Starting value ($)", min_value=100, max_value=1_000_000, value=1000, step=100)
monthly = st.sidebar.number_input("Monthly contribution ($)", min_value=0, max_value=50_000, value=200, step=50)

# Assets
ETFS = {"Stocks":"VTI", "Bonds":"AGG", "Cash":"BIL"}
st.sidebar.markdown("**Assets**")
st.sidebar.write(", ".join(f"{k} ({v})" for k,v in ETFS.items()))

# Preset allocations
ALLOCATIONS = {
    "Conservative": {"Stocks":0.20, "Bonds":0.70, "Cash":0.10},
    "Balanced":     {"Stocks":0.50, "Bonds":0.40, "Cash":0.10},
    "Aggressive":   {"Stocks":0.80, "Bonds":0.15, "Cash":0.05},
}

# ---------------- Risk questionnaire (auto-pick) ----------------
with st.sidebar.expander("Risk questionnaire (auto-picks profile)", expanded=True):
    q1 = st.radio("1) If portfolio drops 20% in a year, you would…",
                  ["Sell to avoid more loss", "Hold steady", "Buy more"], index=1)
    q2 = st.radio("2) Your top priority is…",
                  ["Protecting savings", "Balanced growth", "Max growth"], index=1)
    q3 = st.radio("3) Experience with investing?",
                  ["New", "Some", "Experienced"], index=1)
    q4 = st.radio("4) Time horizon",
                  ["< 3 years", "3–7 years", "> 7 years"], index=1)
    q5 = st.radio("5) Comfort with fluctuations",
                  ["Low", "Medium", "High"], index=1)

# Simple scoring -> profile
score_map = {
    "Sell to avoid more loss":0, "Hold steady":1, "Buy more":2,
    "Protecting savings":0, "Balanced growth":1, "Max growth":2,
    "New":0, "Some":1, "Experienced":2,
    "< 3 years":0, "3–7 years":1, "> 7 years":2,
    "Low":0, "Medium":1, "High":2
}
score = sum(score_map[a] for a in [q1,q2,q3,q4,q5])
if score <= 3:
    auto_profile = "Conservative"
elif score <= 7:
    auto_profile = "Balanced"
else:
    auto_profile = "Aggressive"

# Allow manual override, but default to auto choice
profile = st.sidebar.selectbox("Risk profile (you can override)",
                               list(ALLOCATIONS.keys()),
                               index=list(ALLOCATIONS.keys()).index(auto_profile))

show_data = st.sidebar.checkbox("Show raw prices table", value=False)

# ---------------- Data ----------------
@st.cache_data(show_spinner=True)
def load_prices(tickers, start, end):
    df = yf.download(list(tickers), start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
        df = df["Close"]
    return df.dropna(how="all")

prices = load_prices(list(ETFS.values()), start, end)
if prices.empty:
    st.error("No data for that date range. Try widening it.")
    st.stop()

prices.columns = list(ETFS.keys())
if show_data:
    st.subheader("ETF Prices (first 5 rows)")
    st.write(prices.head())

# ---------------- Backtest (historical) ----------------
normalized = prices / prices.iloc[0] * 100  # each asset starts at 100

# Build comparison series for all preset profiles (lump sum only)
portfolio_values = pd.DataFrame(index=normalized.index)
for name, weights in ALLOCATIONS.items():
    portfolio_values[name] = sum(normalized[a]*w for a, w in weights.items())
portfolio_values = portfolio_values / 100 * initial  # scale to $initial

# Chart: all presets
st.subheader("Portfolio Growth (historical)")
fig, ax = plt.subplots(figsize=(10,5))
for col in portfolio_values.columns:
    ax.plot(portfolio_values.index, portfolio_values[col], label=col)
ax.set_title(f"Portfolio Value (Start = ${initial:,})")
ax.set_xlabel("Date"); ax.set_ylabel("Value ($)")
ax.legend(); ax.grid(True)
st.pyplot(fig)

# ---------------- Focus on selected profile ----------------
st.subheader(f"Details • {profile}")
weights = ALLOCATIONS[profile]

# Allocation pie
pie_fig, pie_ax = plt.subplots(figsize=(4,4))
pie_ax.pie(list(weights.values()), labels=list(weights.keys()), autopct='%1.0f%%')
pie_ax.set_title("Allocation")
col_pie, col_metrics = st.columns([1,2])
with col_pie:
    st.pyplot(pie_fig)

# Lump-sum metrics
sel_series = portfolio_values[profile]
years_hist = max((prices.index[-1] - prices.index[0]).days / 365.25, 0.01)
final_lump = float(sel_series.iloc[-1])
total_ret = final_lump/initial - 1
cagr = (final_lump/initial)**(1/years_hist) - 1
with col_metrics:
    st.metric("Lump-sum final value", f"${final_lump:,.0f}",
              delta=f"{total_ret*100:+.1f}% total • {cagr*100:.1f}% CAGR")

# ---------------- Dollar-Cost Averaging (monthly contributions) ----------------
# Build index (starts near 1.0) for selected profile
sel_index = sum((prices / prices.iloc[0])[a]*w for a,w in weights.items())
sel_index.name = "Index"

# Month-end trading days
month_end = prices.index.to_series().dt.to_period("M").ne(
    prices.index.to_series().shift(-1).dt.to_period("M")
)
shares_initial = initial / sel_index.iloc[0]
shares_added = (month_end.astype(int) * monthly) / sel_index
shares_cum = shares_initial + shares_added.cumsum()
value_dca = shares_cum * sel_index
value_dca.name = "With monthly contributions"

st.markdown("**Selected profile — Lump-sum vs. Monthly contributions**")
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(sel_series.index, sel_series.values, label="Lump sum only")
ax2.plot(value_dca.index, value_dca.values, label="With monthly contributions")
ax2.set_xlabel("Date"); ax2.set_ylabel("Value ($)")
ax2.grid(True); ax2.legend()
st.pyplot(fig2)

final_dca = float(value_dca.iloc[-1])
st.metric("Final value with monthly contributions", f"${final_dca:,.0f}",
          delta=f"+${final_dca-final_lump:,.0f} vs lump sum")

st.divider()

# ---------------- Monte Carlo (future simulation) ----------------
st.subheader("Monte Carlo Projection (future)")

colA, colB, colC = st.columns(3)
proj_years = colA.number_input("Years to project", min_value=1, max_value=40, value=10)
n_sims     = colB.number_input("Simulations", min_value=200, max_value=5000, value=1000, step=100)
goal       = colC.number_input("Goal at end ($, optional)", min_value=0, value=0, help="Leave 0 to skip")

# Build daily return series for the selected profile (historical)
returns = prices.pct_change().dropna()
port_ret = sum(returns[a]*w for a,w in weights.items()).dropna()

def mc_paths(start_value: float, daily_returns: pd.Series, years: int, sims: int, monthly_contrib: float, seed: int = 42):
    """Bootstrap daily returns; add monthly contributions at ~21 trading day steps."""
    np.random.seed(seed)
    values = np.zeros((sims, years*252 + 1), dtype=float)
    values[:, 0] = start_value
    dr = daily_returns.values
    for t in range(1, years*252 + 1):
        draw = np.random.choice(dr, size=sims, replace=True)
        values[:, t] = values[:, t-1] * (1 + draw)
        if t % 21 == 0 and monthly_contrib > 0:
            values[:, t] += monthly_contrib
    return values

start_for_projection = final_dca if monthly > 0 else final_lump
paths = mc_paths(start_for_projection, port_ret, proj_years, int(n_sims), monthly)
end_values = paths[:, -1]
p5, p50, p95 = np.percentile(end_values, [5, 50, 95])

cols = st.columns(3)
cols[0].metric("5th percentile (cautious)", f"${p5:,.0f}")
cols[1].metric("Median (50th)", f"${p50:,.0f}")
cols[2].metric("95th percentile (optimistic)", f"${p95:,.0f}")

# Distribution chart
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.hist(end_values, bins=40)
ax3.set_title(f"Distribution of ending values • {proj_years} years • {int(n_sims)} sims")
ax3.set_xlabel("Ending value ($)"); ax3.set_ylabel("Count")
st.pyplot(fig3)

# Goal probability
if goal and goal > 0:
    prob = (end_values >= goal).mean()
    st.success(f"Probability of reaching **${goal:,.0f}**: **{prob*100:.1f}%**")

st.divider()
st.markdown("**Selected allocation**")
st.table(pd.Series(weights, name=profile).map(lambda x: f"{x:.0%}"))

# Download
csv = pd.DataFrame({
    "date": sel_series.index,
    "lump_sum": sel_series.values,
    "dca_value": value_dca.reindex(sel_series.index, method="ffill").values
}).to_csv(index=False).encode()
st.download_button("Download historical values (CSV)", csv, "investmate_values.csv", "text/csv")


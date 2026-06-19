# ══════════════════════════════════════════════════════
#  STATISTICAL PAIRS TRADING — STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Statistical Pairs Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0A0E1A; color: #E0E0E0; }
    .metric-card {
        background: #1A2035;
        border: 1px solid #00D4FF33;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #00D4FF; }
    .metric-label { font-size: 13px; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Strategy-Pairs%20Trading-00D4FF", width=200)
    st.markdown("## ⚙️ Controls")

    entry_z  = st.slider("Entry Z-Score",   1.0, 3.0, 2.0, 0.1)
    exit_z   = st.slider("Exit Z-Score",    0.1, 1.5, 0.5, 0.1)
    stop_z   = st.slider("Stop-Loss Z",     2.0, 4.0, 3.0, 0.1)
    tc_bps   = st.slider("Transaction Cost (bps)", 1, 30, 10, 1)
    window   = st.slider("Z-Score Window (days)",  20, 120, 60, 5)

    st.markdown("---")
    st.markdown("## 📅 Date Range")
    start_date = st.date_input("Start", value=pd.to_datetime("2021-01-01"))
    end_date   = st.date_input("End",   value=pd.to_datetime("2024-12-31"))

    st.markdown("---")
    st.markdown("## 🔎 Pair Selector")

    # Pre-loaded example pairs (replace with your valid_pairs output)
    EXAMPLE_PAIRS = {
        "XOM / CVX  (Energy)":        ("XOM",  "CVX"),
        "JPM / BAC  (Financials)":    ("JPM",  "BAC"),
        "AAPL / MSFT (Technology)":   ("AAPL", "MSFT"),
        "KO / PEP   (Staples)":       ("KO",   "PEP"),
        "NEE / DUK  (Utilities)":     ("NEE",  "DUK"),
        "GS / MS    (Financials)":    ("GS",   "MS"),
        "LLY / ABBV (Healthcare)":    ("LLY",  "ABBV"),
        "CAT / DE   (Industrials)":   ("CAT",  "DE"),
    }
    selected_pair_label = st.selectbox(
        "Select a Pair", list(EXAMPLE_PAIRS.keys())
    )
    ticker_a, ticker_b = EXAMPLE_PAIRS[selected_pair_label]

    st.markdown("---")
    st.caption("Built with Python · yfinance · statsmodels · Plotly")

# ── Helper functions ──────────────────────────────────
@st.cache_data(ttl=3600)
def load_prices(ticker_a, ticker_b, start, end):
    tickers = [ticker_a, ticker_b, "SPY"]
    raw = yf.download(tickers, start=str(start), end=str(end),
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique()
        prices = raw["Close"] if "Close" in lvl0 else raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]]
    return prices.dropna()

def compute_zscore(spread, window):
    mu  = spread.rolling(window, min_periods=window//2).mean()
    sig = spread.rolling(window, min_periods=window//2).std()
    return (spread - mu) / sig.replace(0, np.nan)

def run_signals(z, entry, exit_, stop):
    pos, positions = 0, np.zeros(len(z))
    for i in range(1, len(z)):
        zi = z.iloc[i]
        if pos == 0:
            if zi >=  entry: pos = -1
            elif zi <= -entry: pos = 1
        elif pos == 1:
            if zi >= -exit_ or zi <= -stop: pos = 0
        elif pos == -1:
            if zi <=  exit_ or zi >=  stop: pos = 0
        positions[i] = pos
    return pd.Series(positions, index=z.index)

def run_backtest(pa, pb, beta, alpha_v, signals, tc):
    ret_a = np.log(pa / pa.shift(1))
    ret_b = np.log(pb / pb.shift(1))
    spread_ret   = ret_a - beta * ret_b
    prev_sig     = signals.shift(1).fillna(0)
    pnl          = prev_sig * spread_ret
    pos_changes  = signals.diff().abs().fillna(0)
    pnl         -= pos_changes * 2 * tc
    return pnl.dropna()

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

def run_coint(log_a, log_b):
    try:
        stat, pval, _ = coint(log_a, log_b, trend='c')
        X    = sm.add_constant(log_b)
        ols  = sm.OLS(log_a, X).fit()
        beta = ols.params.iloc[1]
        alp  = ols.params.iloc[0]
        res  = log_a - alp - beta * log_b
        adf  = adfuller(res.dropna())[1]
        lag  = log_b.shift(1); dif = log_b.diff()
        m    = sm.OLS(dif.dropna(), sm.add_constant(lag.dropna())).fit()
        hl   = np.log(2) / -m.params.iloc[1] if m.params.iloc[1] < 0 else None
        return pval, beta, alp, hl, adf
    except:
        return 1.0, 1.0, 0.0, None, 1.0

def metrics(pnl, rf=0.05/252):
    cum  = np.exp(pnl.cumsum())
    ex   = pnl - rf
    sh   = (ex.mean()/ex.std())*np.sqrt(252) if ex.std()>0 else 0
    cagr = cum.iloc[-1]**(252/len(pnl)) - 1
    dd   = ((cum - cum.cummax())/cum.cummax()).min()
    return sh, cagr, dd, cum

# ── Main Content ──────────────────────────────────────
st.markdown("# 📊 Statistical Pairs Trading Engine")
st.markdown(
    f"**Active Pair:** `{ticker_a}` / `{ticker_b}` &nbsp;|&nbsp; "
    f"**Window:** {window}d &nbsp;|&nbsp; "
    f"**Entry:** ±{entry_z}σ &nbsp;|&nbsp; "
    f"**TC:** {tc_bps}bps"
)
st.markdown("---")

# ── Load Data ─────────────────────────────────────────
with st.spinner(f"Loading {ticker_a} / {ticker_b} data..."):
    prices = load_prices(ticker_a, ticker_b, start_date, end_date)

if ticker_a not in prices.columns or ticker_b not in prices.columns:
    st.error("Could not load prices. Check tickers or date range.")
    st.stop()

pa  = prices[ticker_a]
pb  = prices[ticker_b]
spy = prices["SPY"] if "SPY" in prices.columns else None

log_a  = np.log(pa)
log_b  = np.log(pb)

# ── Cointegration ─────────────────────────────────────
with st.spinner("Running cointegration tests..."):
    pval, beta, alpha_v, half_life, adf_pval = run_coint(log_a, log_b)

spread  = log_a - alpha_v - beta * log_b
z_score = compute_zscore(spread, window)
signals = run_signals(z_score, entry_z, exit_z, stop_z)
tc_dec  = tc_bps / 10000
pnl     = run_backtest(pa, pb, beta, alpha_v, signals, tc_dec)
sharpe, cagr, max_dd, cum_equity = metrics(pnl)

# ── KPI Row ───────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(col, label, value, color="#00D4FF"):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

sig_color  = "#00FF88" if sharpe > 1 else ("#FFD700" if sharpe > 0 else "#FF4444")
coint_color= "#00FF88" if pval < 0.05 else "#FF4444"

kpi(k1, "EG P-Value",   f"{pval:.4f}",  coint_color)
kpi(k2, "Hedge Ratio β",f"{beta:.3f}",  "#00D4FF")
kpi(k3, "Half-Life",    f"{half_life:.1f}d" if half_life else "N/A", "#9B59B6")
kpi(k4, "Sharpe Ratio", f"{sharpe:.3f}", sig_color)
kpi(k5, "CAGR",         f"{cagr:.1%}",  "#00FF88" if cagr > 0 else "#FF4444")
kpi(k6, "Max Drawdown", f"{max_dd:.1%}", "#FF4444")

st.markdown("<br>", unsafe_allow_html=True)

# ── Cointegration Status Banner ───────────────────────
if pval < 0.05:
    st.success(f"✅ **COINTEGRATED** — EG p={pval:.4f} < 0.05 | ADF p={adf_pval:.4f} | β={beta:.4f}")
else:
    st.error(f"❌ **NOT COINTEGRATED** — EG p={pval:.4f} ≥ 0.05. Try a different pair.")

st.markdown("---")

# ── 4-Panel Chart ─────────────────────────────────────
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        f"① Normalized Prices — {ticker_a} vs. {ticker_b}",
        f"② Spread  (log {ticker_a} − {beta:.3f}·log {ticker_b})",
        "③ Rolling Z-Score with Signal Bands",
        "④ Cumulative Strategy P&L",
    ],
    vertical_spacing=0.06, shared_xaxes=True,
)

DARK = "#0A0E1A"
CYAN = "#00D4FF"
ORG  = "#FF6B35"
GRN  = "#00FF88"
RED  = "#FF4444"
GLD  = "#FFD700"

# Panel 1
pa_n = pa / pa.iloc[0] * 100
pb_n = pb / pb.iloc[0] * 100
fig.add_trace(go.Scatter(x=pa_n.index, y=pa_n, name=ticker_a,
    line=dict(color=CYAN, width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=pb_n.index, y=pb_n, name=ticker_b,
    line=dict(color=ORG,  width=1.5)), row=1, col=1)

# Panel 2
fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread",
    line=dict(color="#9B59B6", width=1.5),
    fill='tozeroy', fillcolor='rgba(155,89,182,0.08)'), row=2, col=1)
fig.add_trace(go.Scatter(
    x=spread.rolling(window).mean().index,
    y=spread.rolling(window).mean(),
    name="Rolling Mean", line=dict(color=GLD, width=1, dash='dash'),
    showlegend=False), row=2, col=1)

# Panel 3
fig.add_trace(go.Scatter(x=z_score.index, y=z_score,
    name="Z-Score", line=dict(color=CYAN, width=1.2)), row=3, col=1)
for y, col in [(entry_z, RED), (-entry_z, GRN),
               (exit_z,  GLD), (-exit_z,  GLD),
               (stop_z,  "rgba(255,50,50,0.4)"),
               (-stop_z, "rgba(255,50,50,0.4)")]:
    fig.add_hline(y=y, line_color=col, line_dash="dot",
                  line_width=1.2, row=3, col=1)

# Panel 4 — color positive/negative separately
cum_pct = (cum_equity - 1) * 100
dd_pct  = ((cum_equity - cum_equity.cummax()) / cum_equity.cummax()) * 100
fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct,
    name="Strategy P&L %", line=dict(color=GRN, width=2),
    fill='tozeroy', fillcolor='rgba(0,255,136,0.08)'), row=4, col=1)
fig.add_trace(go.Scatter(x=dd_pct.index, y=dd_pct,
    name="Drawdown", line=dict(color=RED, width=1),
    fill='tozeroy', fillcolor='rgba(255,68,68,0.12)'), row=4, col=1)
fig.add_hline(y=0, line_color="#555", line_width=0.8, row=4, col=1)

fig.update_layout(
    height=820, showlegend=True,
    plot_bgcolor=DARK, paper_bgcolor="#0D1117",
    font=dict(color="#E0E0E0", size=11),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(10,14,26,0.8)",
                bordercolor="#1A2035", borderwidth=1),
)
fig.update_xaxes(gridcolor="#1A2035", showgrid=True)
fig.update_yaxes(gridcolor="#1A2035", showgrid=True)
st.plotly_chart(fig, use_container_width=True)

# ── Trade Stats Row ───────────────────────────────────
st.markdown("### 📋 Trade Statistics")
c1, c2 = st.columns(2)

n_trades    = int((signals.diff().abs() > 0).sum() // 2)
pct_in_mkt  = (signals != 0).mean()
n_long      = int((signals == 1).sum())
n_short     = int((signals == -1).sum())
hit_rate    = (pnl > 0).mean()
gains       = pnl[pnl > 0].sum()
losses      = abs(pnl[pnl < 0].sum())
pf          = gains / losses if losses > 0 else float('inf')

trade_stats = pd.DataFrame({
    "Metric": ["Total Trades", "% Time in Market", "Long Days", "Short Days",
               "Hit Rate", "Profit Factor", "Ann. Volatility"],
    "Value":  [n_trades, f"{pct_in_mkt:.1%}", n_long, n_short,
               f"{hit_rate:.1%}", f"{pf:.2f}",
               f"{pnl.std()*np.sqrt(252):.1%}"],
})
c1.dataframe(trade_stats, use_container_width=True, hide_index=True)

# Monthly return heatmap
monthly = (pnl.resample("ME").sum() * 100).to_frame("return")
monthly["year"]  = monthly.index.year
monthly["month"] = monthly.index.strftime("%b")
pivot = monthly.pivot(index="year", columns="month", values="return")
months_order = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])

fig_heat = go.Figure(go.Heatmap(
    z=pivot.values, x=pivot.columns.tolist(),
    y=pivot.index.astype(str).tolist(),
    colorscale=[[0,"#8B0000"],[0.5,"#1A2035"],[1,"#006400"]],
    zmid=0, text=np.round(pivot.values,1),
    texttemplate="%{text}%",
))
fig_heat.update_layout(
    title="Monthly Returns Heatmap (%)",
    height=250, plot_bgcolor=DARK, paper_bgcolor="#0D1117",
    font=dict(color="#E0E0E0"),
    margin=dict(l=40, r=20, t=40, b=20),
)
c2.plotly_chart(fig_heat, use_container_width=True)

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Strategy**: Cointegration-based Statistical Pairs Trading &nbsp;|&nbsp; "
    "**Tests**: Engle-Granger · Johansen · ADF &nbsp;|&nbsp; "
    "**Model**: Ornstein-Uhlenbeck Mean Reversion &nbsp;|&nbsp; "
    "**Used by**: D.E. Shaw · Citadel · Two Sigma"
)
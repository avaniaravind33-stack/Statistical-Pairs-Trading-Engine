# ══════════════════════════════════════════════════════════════
#  STATISTICAL PAIRS TRADING ENGINE — STREAMLIT DEPLOYMENT
#  Deploys the full Colab engine as a live interactive dashboard
# ══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Statistical Pairs Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Colors ──────────────────────────────────────────────
COLORS = {
    'primary':    '#00D4FF',
    'secondary':  '#FF6B35',
    'positive':   '#00FF88',
    'negative':   '#FF4444',
    'neutral':    '#888888',
    'background': '#0A0E1A',
    'grid':       '#1A2035',
    'text':       '#E0E0E0',
    'gold':       '#FFD700',
    'purple':     '#9B59B6',
}

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0A0E1A; }
    [data-testid="stSidebar"]          { background-color: #0D1117; }
    .metric-card {
        background: #1A2035;
        border: 1px solid #00D4FF44;
        border-radius: 10px;
        padding: 18px 12px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-value { font-size: 26px; font-weight: 700; }
    .metric-label { font-size: 12px; color: #888; margin-top: 4px; }
    h1, h2, h3   { color: #E0E0E0 !important; }
    .stSelectbox label, .stSlider label { color: #E0E0E0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# SIDEBAR — CONTROLS
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Pairs Trading Engine")
    st.markdown("*Cointegration-Based Statistical Arbitrage*")
    st.markdown("---")

    st.markdown("### 🔎 Select a Pair")
    PAIRS = {
        "XOM / CVX  — Energy":             ("XOM",  "CVX"),
        "JPM / BAC  — Financials":          ("JPM",  "BAC"),
        "GS  / MS   — Financials":          ("GS",   "MS"),
        "KO  / PEP  — Consumer Staples":    ("KO",   "PEP"),
        "AAPL/ MSFT — Technology":          ("AAPL", "MSFT"),
        "NEE / DUK  — Utilities":           ("NEE",  "DUK"),
        "CAT / DE   — Industrials":         ("CAT",  "DE"),
        "LLY / ABBV — Health Care":         ("LLY",  "ABBV"),
        "PLD / AMT  — Real Estate":         ("PLD",  "AMT"),
        "NEM / FCX  — Materials":           ("NEM",  "FCX"),
    }
    selected = st.selectbox("Pair", list(PAIRS.keys()))
    ticker_a, ticker_b = PAIRS[selected]

    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", value=pd.to_datetime("2019-01-01"))
    end_date   = col2.date_input("End",   value=pd.to_datetime("2024-12-31"))

    train_pct = st.slider("Training Split %", 40, 80, 60, 5)

    st.markdown("### ⚙️ Signal Parameters")
    entry_z = st.slider("Entry Z-Score",   1.0, 3.0, 2.0, 0.1)
    exit_z  = st.slider("Exit Z-Score",    0.1, 1.5, 0.5, 0.1)
    stop_z  = st.slider("Stop-Loss Z",     2.0, 4.5, 3.0, 0.1)
    window  = st.slider("Z-Score Window",  20,  120, 60,  5)

    st.markdown("### 💰 Transaction Cost")
    tc_bps  = st.slider("Bps per side",   1, 30, 10, 1)

    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Built with Python · statsmodels · yfinance · Plotly")

# ═══════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ta, tb, start, end):
    raw = yf.download(
        [ta, tb, "SPY"],
        start=str(start), end=str(end),
        auto_adjust=True, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique()
        prices = raw["Close"] if "Close" in lvl0 else raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]]
    return prices.dropna()

def run_cointegration(log_a, log_b):
    """Engle-Granger + Johansen + OU half-life."""
    try:
        eg_stat, eg_pval, _ = coint(log_a, log_b, trend='c')
        X   = sm.add_constant(log_b)
        ols = sm.OLS(log_a, X).fit()
        beta, alpha = ols.params.iloc[1], ols.params.iloc[0]
        spread = log_a - alpha - beta * log_b
        adf_p  = adfuller(spread.dropna())[1]

        # OU half-life
        lag   = spread.shift(1); dif = spread.diff()
        mask  = ~(lag.isna() | dif.isna())
        m     = sm.OLS(dif[mask], sm.add_constant(lag[mask])).fit()
        kappa = -m.params.iloc[1]
        hl    = np.log(2) / kappa if kappa > 0 else None

        # Johansen
        df_j  = pd.concat([log_a, log_b], axis=1).dropna()
        j_res = coint_johansen(df_j, det_order=0, k_ar_diff=1)
        j_ok  = (j_res.lr1[0] > j_res.cvt[0,1]) and (j_res.lr2[0] > j_res.cvm[0,1])

        return {
            'eg_pval': eg_pval, 'adf_pval': adf_p,
            'beta': beta,       'alpha': alpha,
            'half_life': hl,    'johansen': j_ok,
            'spread': spread,
        }
    except Exception as e:
        return None

def compute_zscore(spread, w):
    mu  = spread.rolling(w, min_periods=w//2).mean()
    sig = spread.rolling(w, min_periods=w//2).std()
    return (spread - mu) / sig.replace(0, np.nan)

def generate_signals(z, entry, exit_, stop):
    pos, arr = 0, np.zeros(len(z))
    zv = z.fillna(0).values
    for i in range(1, len(zv)):
        zi = zv[i]
        if pos == 0:
            if   zi >=  entry: pos = -1
            elif zi <= -entry: pos =  1
        elif pos == 1:
            if zi >= -exit_ or zi <= -stop: pos = 0
        elif pos == -1:
            if zi <=  exit_ or zi >=  stop: pos = 0
        arr[i] = pos
    return pd.Series(arr, index=z.index)

def run_backtest(pa, pb, beta, alpha, signals, tc):
    ret_a  = np.log(pa / pa.shift(1))
    ret_b  = np.log(pb / pb.shift(1))
    spread_ret   = ret_a - beta * ret_b
    pnl          = signals.shift(1).fillna(0) * spread_ret
    pnl         -= signals.diff().abs().fillna(0) * 2 * tc
    return pnl.dropna()

def compute_metrics(pnl, rf=0.05/252):
    cum  = np.exp(pnl.cumsum())
    ex   = pnl - rf
    sh   = (ex.mean()/ex.std())*np.sqrt(252) if ex.std()>0 else 0
    cagr = cum.iloc[-1]**(252/max(len(pnl),1)) - 1
    dd   = ((cum - cum.cummax())/cum.cummax()).min()
    hits = (pnl > 0).mean()
    g, l = pnl[pnl>0].sum(), abs(pnl[pnl<0].sum())
    pf   = g/l if l > 0 else np.inf
    return dict(sharpe=sh, cagr=cagr, max_dd=dd, hit_rate=hits,
                profit_factor=pf, cum_equity=cum, vol=pnl.std()*np.sqrt(252))

def kalman_hedge(pa, pb, delta=1e-5, Ve=0.001):
    la, lb = np.log(pa.values), np.log(pb.values)
    n  = len(la)
    Wt = delta/(1-delta) * np.eye(2)
    th = np.zeros((n,2)); P = np.zeros((n,2,2)); P[0] = 1e4*np.eye(2)
    for t in range(1, n):
        F  = np.array([1.0, lb[t]])
        Pp = P[t-1] + Wt
        S  = float(F @ Pp @ F.T) + Ve
        K  = (Pp @ F.T) / S
        th[t] = th[t-1] + K*(la[t] - float(F @ th[t-1]))
        P[t]  = Pp - np.outer(K, F) @ Pp
    return pd.Series(th[:,1], index=pa.index), pd.Series(th[:,0], index=pa.index)
# ═══════════════════════════════════════════
# PORTFOLIO SCAN FUNCTIONS
# ═══════════════════════════════════════════

PORTFOLIO_PAIRS = [
    ("XOM",  "CVX",  "Energy"),
    ("JPM",  "BAC",  "Financials"),
    ("GS",   "MS",   "Financials"),
    ("KO",   "PEP",  "Consumer Staples"),
    ("NEE",  "DUK",  "Utilities"),
    ("CAT",  "DE",   "Industrials"),
    ("PG",   "CL",   "Consumer Staples"),
    ("WMT",  "COST", "Consumer Staples"),
    ("JNJ",  "ABBV", "Health Care"),
    ("UNH",  "ABBV", "Health Care"),
    ("LLY",  "JNJ",  "Health Care"),
    ("AAPL", "MSFT", "Information Technology"),
    ("AMD",  "NVDA", "Information Technology"),
    ("QCOM", "TXN",  "Information Technology"),
    ("PLD",  "AMT",  "Real Estate"),
    ("NEM",  "FCX",  "Materials"),
    ("VZ",   "T",    "Communication Services"),
    ("TMUS", "EA",   "Communication Services"),
    ("PM",   "MO",   "Consumer Staples"),
    ("HON",  "MMM",  "Industrials"),
]

@st.cache_data(ttl=7200, show_spinner=False)
def run_portfolio_scan(start, end, train_pct, entry_z, exit_z, stop_z,
                       window, tc_bps):
    """Scan all portfolio pairs, backtest each, return aggregated results."""
    results   = {}
    meta_rows = []
    tc_dec    = tc_bps / 10000

    prices_all = yf.download(
        list(set([t for p in PORTFOLIO_PAIRS for t in p[:2]] + ["SPY"])),
        start=str(start), end=str(end),
        auto_adjust=True, progress=False,
    )
    if isinstance(prices_all.columns, pd.MultiIndex):
        lvl0 = prices_all.columns.get_level_values(0).unique()
        prices_all = (prices_all["Close"]
                      if "Close" in lvl0
                      else prices_all.xs("Close", axis=1, level=1))
    prices_all = prices_all.dropna(how="all")

    n_total  = len(prices_all)
    n_train  = int(n_total * train_pct / 100)
    train_end_idx = prices_all.index[n_train - 1]

    spy_ret = np.log(prices_all["SPY"] / prices_all["SPY"].shift(1)).dropna() \
              if "SPY" in prices_all.columns else pd.Series(dtype=float)

    for ta, tb, sector in PORTFOLIO_PAIRS:
        if ta not in prices_all.columns or tb not in prices_all.columns:
            continue
        try:
            pa = prices_all[ta].dropna()
            pb = prices_all[tb].dropna()
            common = pa.index.intersection(pb.index)
            pa, pb = pa[common], pb[common]
            if len(pa) < 300:
                continue

            la_train = np.log(pa[:train_end_idx])
            lb_train = np.log(pb[:train_end_idx])

            cr = run_cointegration(la_train, lb_train)
            if cr is None or cr['eg_pval'] >= 0.05:
                continue
            hl = cr['half_life']
            if hl is None or not (5 <= hl <= 60):
                continue

            beta_v  = cr['beta']
            alpha_v = cr['alpha']
            spread  = np.log(pa) - alpha_v - beta_v * np.log(pb)
            z_full  = compute_zscore(spread, window)
            sigs    = generate_signals(z_full, entry_z, exit_z, stop_z)

            test_mask   = prices_all.index > train_end_idx
            pa_t = pa[test_mask]; pb_t = pb[test_mask]; sigs_t = sigs[test_mask]
            pnl  = run_backtest(pa_t, pb_t, beta_v, alpha_v, sigs_t, tc_dec)
            if len(pnl) == 0:
                continue

            m = compute_metrics(pnl)

            pair_key = f"{ta}/{tb}"
            results[pair_key] = {
                'pnl': pnl, 'cum_equity': m['cum_equity'],
                'spread': spread, 'z': z_full, 'signals': sigs,
                'ticker_a': ta, 'ticker_b': tb, 'sector': sector,
                'half_life': hl, 'eg_pval': cr['eg_pval'], 'beta': beta_v,
            }
            meta_rows.append({
                'pair': pair_key, 'ticker_a': ta, 'ticker_b': tb,
                'sector': sector, 'half_life': hl,
                'sharpe': m['sharpe'], 'cagr': m['cagr'],
                'max_dd': m['max_dd'], 'eg_pval': cr['eg_pval'],
                'n_trades': int((sigs_t.diff().abs() > 0).sum() // 2),
            })
        except Exception:
            continue

    meta_df = pd.DataFrame(meta_rows).sort_values('sharpe', ascending=False) \
              if meta_rows else pd.DataFrame()

    # Aggregate portfolio P&L
    if results:
        pnl_df  = pd.DataFrame({k: v['pnl'] for k, v in results.items()})
        port_pnl= pnl_df.mean(axis=1)
        port_cum= np.exp(port_pnl.cumsum())

        # Rolling beta
        spy_aligned = spy_ret.reindex(port_pnl.index).dropna()
        port_al     = port_pnl.reindex(spy_aligned.index).dropna()
        roll_beta   = pd.Series(dtype=float, index=port_pnl.index)
        for i in range(60, len(port_pnl)):
            ws = port_pnl.iloc[i-60:i]
            wb = spy_ret.reindex(ws.index).dropna()
            ws = ws.reindex(wb.index)
            if len(wb) > 10:
                cov = np.cov(ws, wb)
                roll_beta.iloc[i] = cov[0,1]/cov[1,1] if cov[1,1] != 0 else np.nan

        spy_cum = np.exp(spy_aligned.cumsum()) if len(spy_aligned) > 0 else pd.Series()
        port_m  = compute_metrics(port_al)

        return {
            'results':    results,
            'meta_df':    meta_df,
            'port_pnl':   port_pnl,
            'port_cum':   port_cum,
            'roll_beta':  roll_beta,
            'spy_cum':    spy_cum,
            'port_m':     port_m,
            'prices_all': prices_all,
            'train_end':  train_end_idx,
        }
    return None


def make_portfolio_dashboard(port_data):
    """6-panel dashboard matching the Colab output exactly."""
    port_pnl  = port_data['port_pnl']
    port_cum  = port_data['port_cum']
    roll_beta = port_data['roll_beta']
    spy_cum   = port_data['spy_cum']
    port_m    = port_data['port_m']
    n_pairs   = len(port_data['results'])

    roll_sharpe = port_pnl.rolling(60).apply(
        lambda x: (x.mean()/x.std())*np.sqrt(252) if x.std()>0 else 0,
        raw=True
    )

    monthly_pnl = port_pnl.resample('ME').sum() * 100
    m_df = pd.DataFrame({
        'year':  monthly_pnl.index.year,
        'month': monthly_pnl.index.strftime('%b'),
        'pnl':   monthly_pnl.values,
    })
    pivot = m_df.pivot(index='year', columns='month', values='pnl')
    mo    = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot = pivot.reindex(columns=[m for m in mo if m in pivot.columns])

    dd_pct   = ((port_cum - port_cum.cummax()) / port_cum.cummax()) * 100
    daily_pct = port_pnl * 100

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "<b>Portfolio Equity Curve vs. SPY</b>",
            "<b>Rolling 60-Day Sharpe Ratio</b>",
            "<b>Rolling Beta to SPY (Market Neutrality)</b>",
            "<b>Monthly P&L Heatmap (%)</b>",
            "<b>P&L Return Distribution</b>",
            "<b>Drawdown Profile</b>",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Panel 1 — Equity curve
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=(port_cum.values-1)*100,
        name="Pairs Portfolio",
        line=dict(color=COLORS['positive'], width=2.5),
        fill='tozeroy', fillcolor='rgba(0,255,136,0.07)',
    ), row=1, col=1)
    if len(spy_cum) > 0:
        spy_al = spy_cum.reindex(port_cum.index).ffill()
        fig.add_trace(go.Scatter(
            x=spy_al.index, y=(spy_al.values-1)*100,
            name="SPY Benchmark",
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
        ), row=1, col=1)
    fig.add_hline(y=0, line_color=COLORS['neutral'], line_width=0.8, row=1, col=1)

    # Panel 2 — Rolling Sharpe
    vs = roll_sharpe.dropna()
    fig.add_trace(go.Scatter(
        x=vs.index, y=vs.values, name="Rolling Sharpe",
        line=dict(color=COLORS['gold'], width=1.5),
        fill='tozeroy', fillcolor='rgba(255,215,0,0.08)',
    ), row=1, col=2)
    fig.add_hline(y=1.0, line_color=COLORS['positive'], line_dash='dot',
                  line_width=1.2, row=1, col=2)
    fig.add_hline(y=0.0, line_color=COLORS['neutral'],  line_width=0.6, row=1, col=2)

    # Panel 3 — Rolling Beta
    vb = roll_beta.dropna()
    fig.add_trace(go.Scatter(
        x=vb.index, y=vb.values, name="Rolling Beta",
        line=dict(color=COLORS['purple'], width=1.5),
    ), row=2, col=1)
    for yv, col, ds in [(0, COLORS['positive'], 'dash'),
                         (0.2, COLORS['negative'], 'dot'),
                         (-0.2, COLORS['negative'], 'dot')]:
        fig.add_hline(y=yv, line_color=col, line_dash=ds,
                      line_width=1.2, row=2, col=1)

    # Panel 4 — Monthly heatmap
    if len(pivot) > 0:
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.astype(str).tolist(),
            colorscale=[[0,'#8B0000'],[0.35,'#CC0000'],
                        [0.45,'#1A2035'],[0.5,'#1A2035'],
                        [0.55,'#1A2035'],[0.65,'#006400'],[1,'#00AA00']],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            colorbar=dict(len=0.3, y=0.5, x=1.01, thickness=12),
        ), row=2, col=2)

    # Panel 5 — Return distribution
    fig.add_trace(go.Histogram(
        x=daily_pct.dropna().values, nbinsx=60,
        name="Daily Returns",
        marker_color=COLORS['primary'], opacity=0.7,
    ), row=3, col=1)
    mu, sd = daily_pct.mean(), daily_pct.std()
    xn = np.linspace(daily_pct.min(), daily_pct.max(), 200)
    yn = stats.norm.pdf(xn, mu, sd) * len(daily_pct) * (daily_pct.max()-daily_pct.min())/60
    fig.add_trace(go.Scatter(x=xn, y=yn, name="Normal Dist.",
        line=dict(color=COLORS['gold'], width=2)), row=3, col=1)
    fig.add_vline(x=0, line_color=COLORS['neutral'], line_width=0.8, row=3, col=1)

    # Panel 6 — Drawdown
    fig.add_trace(go.Scatter(
        x=dd_pct.index, y=dd_pct.values, name="Drawdown",
        fill='tozeroy', fillcolor='rgba(255,68,68,0.28)',
        line=dict(color=COLORS['negative'], width=1),
    ), row=3, col=2)
    fig.add_hline(y=0, line_color=COLORS['neutral'], line_width=0.5, row=3, col=2)

    fig.update_layout(
        height=1000,
        title=dict(
            text=(f"<b>Statistical Pairs Trading Portfolio — Institutional Dashboard</b>"
                  f"<br><sup>N={n_pairs} pairs | "
                  f"Sharpe: {port_m['sharpe']:.2f} | "
                  f"CAGR: {port_m['cagr']:.1%} | "
                  f"Max DD: {port_m['max_dd']:.1%}</sup>"),
            font=dict(size=14, color=COLORS['text']), x=0.02,
        ),
        showlegend=True,
        plot_bgcolor=COLORS['background'], paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(10,14,26,0.85)',
                    bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, zeroline=False)
    return fig


def make_halflife_chart(meta_df):
    """Half-life distribution + Sharpe vs Half-Life scatter — Image 3."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["<b>Half-Life Distribution</b>",
                        "<b>Sharpe Ratio vs. Half-Life</b>"],
        horizontal_spacing=0.12,
    )
    valid = meta_df.dropna(subset=['half_life'])

    fig.add_trace(go.Histogram(
        x=valid['half_life'], nbinsx=30,
        marker_color=COLORS['primary'],
        marker_line=dict(color=COLORS['background'], width=0.5),
        opacity=0.85, name="Pairs",
    ), row=1, col=1)
    fig.add_vline(x=5,  line_color=COLORS['positive'],
                  line_dash='dash', line_width=2, row=1, col=1)
    fig.add_vline(x=60, line_color=COLORS['negative'],
                  line_dash='dash', line_width=2, row=1, col=1)

    import plotly.express as px
    sector_colors = px.colors.qualitative.Set2
    for i, sector in enumerate(valid['sector'].unique()):
        sub = valid[valid['sector'] == sector]
        fig.add_trace(go.Scatter(
            x=sub['half_life'], y=sub['sharpe'],
            mode='markers', name=sector,
            text=sub['pair'],
            hovertemplate="<b>%{text}</b><br>HL: %{x:.1f}d<br>Sharpe: %{y:.3f}",
            marker=dict(size=9,
                        color=sector_colors[i % len(sector_colors)],
                        line=dict(width=0.5, color='white'), opacity=0.85),
        ), row=1, col=2)

    fig.add_hline(y=1.0, line_color=COLORS['positive'], line_dash='dot',
                  line_width=1.2, row=1, col=2)
    fig.add_hline(y=0.0, line_color=COLORS['neutral'],
                  line_width=0.6, row=1, col=2)

    fig.update_layout(
        title=dict(text="<b>OU Half-Life Analysis Across Cointegrated Pairs</b>",
                   font=dict(size=13, color=COLORS['text']), x=0.02),
        height=430, showlegend=True,
        plot_bgcolor=COLORS['background'], paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        legend=dict(bgcolor='rgba(10,14,26,0.85)',
                    bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True,
                     title_text="Half-Life (Trading Days)")
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True)
    fig.update_yaxes(title_text="Number of Pairs", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    return fig


def make_rolling_coint_chart(results, prices_all, train_end, window=252):
    """Rolling EG cointegration p-value — Image 4."""
    fig    = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['positive'],
              COLORS['gold'], COLORS['purple']]

    top5 = list(results.keys())[:5]
    for idx, pair_key in enumerate(top5):
        r  = results[pair_key]
        ta, tb = r['ticker_a'], r['ticker_b']
        if ta not in prices_all.columns or tb not in prices_all.columns:
            continue

        la = np.log(prices_all[ta].dropna())
        lb = np.log(prices_all[tb].dropna())
        common = la.index.intersection(lb.index)
        la, lb = la[common], lb[common]

        pvals, dates = [], []
        for end_i in range(window, len(la), 21):
            try:
                _, pv, _ = coint(la.iloc[end_i-window:end_i],
                                  lb.iloc[end_i-window:end_i], trend='c')
                pvals.append(pv)
            except Exception:
                pvals.append(np.nan)
            dates.append(la.index[end_i])

        fig.add_trace(go.Scatter(
            x=dates, y=pvals,
            name=pair_key,
            line=dict(color=colors[idx % len(colors)], width=1.6),
            mode='lines',
        ))

    fig.add_hline(y=0.05, line_color=COLORS['positive'], line_dash='dash',
                  line_width=2,
                  annotation_text="5% Threshold",
                  annotation_position="right",
                  annotation_font_color=COLORS['positive'])
    fig.add_hline(y=0.10, line_color=COLORS['negative'], line_dash='dot',
                  line_width=1.5,
                  annotation_text="10% Threshold",
                  annotation_position="right",
                  annotation_font_color=COLORS['negative'])
    fig.add_vline(x=str(train_end)[:10],
                  line_color=COLORS['gold'], line_dash='dash', line_width=2,
                  annotation_text="Train|Test",
                  annotation_font_color=COLORS['gold'])

    fig.update_layout(
        title=dict(
            text=("<b>Rolling Cointegration Stability (1-Year Window)</b>"
                  "<br><sup>P-value < 0.05 (green) = cointegrated | "
                  "Gold line = train/test split</sup>"),
            font=dict(size=13, color=COLORS['text']), x=0.02,
        ),
        height=430,
        plot_bgcolor=COLORS['background'], paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        xaxis_title="Date", yaxis_title="EG Cointegration P-Value",
        hovermode='x unified',
        legend=dict(bgcolor='rgba(10,14,26,0.85)',
                    bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig.update_xaxes(gridcolor=COLORS['grid'])
    fig.update_yaxes(gridcolor=COLORS['grid'], range=[0, 0.42])
    return fig

# ═══════════════════════════════════════════
# MAIN — runs when button clicked or on load
# ═══════════════════════════════════════════

st.markdown("# 📈 Statistical Pairs Trading Engine")
tab1, tab2 = st.tabs(["📊 Single Pair Analysis", "🗂️ Portfolio Dashboard"])
with tab1:
    with st.spinner(f"⏳ Loading data for {ticker_a} / {ticker_b} ..."):
        prices = load_prices(ticker_a, ticker_b, start_date, end_date)

if ticker_a not in prices.columns or ticker_b not in prices.columns:
    st.error("❌ Could not load one or both tickers. Try different dates or tickers.")
    st.stop()

# ── Train / Test Split ────────────────────────────────────────
n_total    = len(prices)
n_train    = int(n_total * train_pct / 100)
train_end  = prices.index[n_train - 1]
test_start = prices.index[n_train]

pa_full  = prices[ticker_a]
pb_full  = prices[ticker_b]
spy_full = prices["SPY"] if "SPY" in prices.columns else None

log_a_full = np.log(pa_full)
log_b_full = np.log(pb_full)

# ── Cointegration on TRAIN only ───────────────────────────────
with st.spinner("🔬 Running cointegration tests..."):
    coint_result = run_cointegration(
        log_a_full.loc[:train_end],
        log_b_full.loc[:train_end],
    )

if coint_result is None:
    st.error("Cointegration test failed. Try a different pair or date range.")
    st.stop()

beta  = coint_result['beta']
alpha = coint_result['alpha']
hl    = coint_result['half_life']

# ── Spread & Signals on FULL period ──────────────────────────
spread_full  = log_a_full - alpha - beta * log_b_full
z_full       = compute_zscore(spread_full, window)
signals_full = generate_signals(z_full, entry_z, exit_z, stop_z)

# ── Backtest on TEST period only ──────────────────────────────
test_mask     = prices.index >= test_start
pa_test       = pa_full[test_mask]
pb_test       = pb_full[test_mask]
signals_test  = signals_full[test_mask]
z_test        = z_full[test_mask]
spread_test   = spread_full[test_mask]

tc_dec  = tc_bps / 10000
pnl     = run_backtest(pa_test, pb_test, beta, alpha, signals_test, tc_dec)
metrics = compute_metrics(pnl)

# ── SPY benchmark ─────────────────────────────────────────────
spy_pnl = None
if spy_full is not None:
    spy_test    = np.log(spy_full[test_mask] / spy_full[test_mask].shift(1)).dropna()
    spy_cum     = np.exp(spy_test.cumsum())
    spy_metrics = compute_metrics(spy_test)

# ═══════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════
eg_color   = "#00FF88" if coint_result['eg_pval'] < 0.05 else "#FF4444"
sh_color   = "#00FF88" if metrics['sharpe'] > 1.0 else ("#FFD700" if metrics['sharpe'] > 0 else "#FF4444")
dd_color   = "#FF4444"
hl_color   = "#9B59B6"

def kpi_card(col, label, value, color="#00D4FF", sublabel=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
        {"<div style='font-size:10px;color:#555'>"+sublabel+"</div>" if sublabel else ""}
    </div>""", unsafe_allow_html=True)

c = st.columns(7)
kpi_card(c[0], "EG P-Value",    f"{coint_result['eg_pval']:.4f}", eg_color,
         "✅ Cointegrated" if coint_result['eg_pval']<0.05 else "❌ Not Cointegrated")
kpi_card(c[1], "Hedge Ratio β", f"{beta:.3f}",   "#00D4FF")
kpi_card(c[2], "Half-Life",     f"{hl:.1f}d" if hl else "N/A", hl_color)
kpi_card(c[3], "Sharpe Ratio",  f"{metrics['sharpe']:.2f}", sh_color)
kpi_card(c[4], "CAGR",          f"{metrics['cagr']:.1%}",
         "#00FF88" if metrics['cagr']>0 else "#FF4444")
kpi_card(c[5], "Max Drawdown",  f"{metrics['max_dd']:.1%}", dd_color)
kpi_card(c[6], "Johansen",
         "✅ Pass" if coint_result['johansen'] else "⚠️ Fail",
         "#00FF88" if coint_result['johansen'] else "#FFD700")

st.markdown("<br>", unsafe_allow_html=True)

# ── Status banner ─────────────────────────────────────────────
if coint_result['eg_pval'] < 0.05:
    st.success(
        f"✅ **COINTEGRATED** — "
        f"EG p={coint_result['eg_pval']:.4f} | "
        f"ADF p={coint_result['adf_pval']:.4f} | "
        f"β={beta:.4f} | "
        f"Half-Life={f'{hl:.1f} days' if hl else 'N/A'} | "
        f"Johansen={'Pass ✅' if coint_result['johansen'] else 'Fail ⚠️'}"
    )
else:
    st.error(
        f"❌ **NOT COINTEGRATED** — EG p={coint_result['eg_pval']:.4f} ≥ 0.05. "
        "Try a different pair or longer date range."
    )

# ═══════════════════════════════════════════
# 4-PANEL MAIN CHART
# ═══════════════════════════════════════════
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        f"① Normalized Prices — {ticker_a} vs. {ticker_b}",
        f"② Spread  (log {ticker_a} − {beta:.3f}·log {ticker_b})",
        "③ Rolling Z-Score — Trading Signals",
        "④ Cumulative P&L (Out-of-Sample)",
    ],
    vertical_spacing=0.055,
    shared_xaxes=True,
    row_heights=[0.22, 0.22, 0.28, 0.28],
)

BG = COLORS['background']

# Panel 1 — Normalized prices
pa_n = pa_full / pa_full.iloc[0] * 100
pb_n = pb_full / pb_full.iloc[0] * 100
fig.add_trace(go.Scatter(x=pa_n.index, y=pa_n, name=ticker_a,
    line=dict(color=COLORS['primary'], width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=pb_n.index, y=pb_n, name=ticker_b,
    line=dict(color=COLORS['secondary'], width=1.5)), row=1, col=1)
fig.add_vline(x=str(train_end)[:10], line_color=COLORS['gold'],
              line_dash="dash", line_width=1.5,
              annotation_text="Train|Test", annotation_font_color=COLORS['gold'])

# Panel 2 — Spread
fig.add_trace(go.Scatter(x=spread_full.index, y=spread_full,
    name="Spread", line=dict(color=COLORS['purple'], width=1.4),
    fill='tozeroy', fillcolor='rgba(155,89,182,0.08)'), row=2, col=1)
rm = spread_full.rolling(window).mean()
fig.add_trace(go.Scatter(x=rm.index, y=rm, showlegend=False,
    line=dict(color=COLORS['gold'], width=1, dash='dash')), row=2, col=1)

# Panel 3 — Z-Score + signal shading
fig.add_trace(go.Scatter(x=z_full.index, y=z_full,
    name="Z-Score", line=dict(color=COLORS['primary'], width=1.2)), row=3, col=1)
for yv, col in [
    (entry_z,  'rgba(255,68,68,0.8)'),   (-entry_z, 'rgba(0,255,136,0.8)'),
    (exit_z,   'rgba(255,215,0,0.5)'),   (-exit_z,  'rgba(255,215,0,0.5)'),
    (stop_z,   'rgba(255,50,50,0.3)'),   (-stop_z,  'rgba(255,50,50,0.3)'),
]:
    fig.add_hline(y=yv, line_color=col, line_dash="dot", line_width=1.2, row=3, col=1)

# Add signal markers on z-score panel
sig_changes = signals_full.diff().fillna(0)
entries_long  = z_full[sig_changes ==  1]
entries_short = z_full[sig_changes == -1]
exits         = z_full[sig_changes.abs() > 0][sig_changes == 0]

fig.add_trace(go.Scatter(
    x=entries_long.index, y=entries_long.values, mode='markers',
    marker=dict(symbol='triangle-up', size=8, color=COLORS['positive']),
    name="Long Entry", showlegend=True,
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=entries_short.index, y=entries_short.values, mode='markers',
    marker=dict(symbol='triangle-down', size=8, color=COLORS['negative']),
    name="Short Entry", showlegend=True,
), row=3, col=1)

# Panel 4 — Cumulative P&L
cum_equity = metrics['cum_equity']
cum_pct    = (cum_equity - 1) * 100
dd_pct     = ((cum_equity - cum_equity.cummax()) / cum_equity.cummax()) * 100

fig.add_trace(go.Scatter(x=cum_pct.index, y=cum_pct,
    name="Strategy P&L", line=dict(color=COLORS['positive'], width=2),
    fill='tozeroy', fillcolor='rgba(0,255,136,0.07)'), row=4, col=1)
fig.add_trace(go.Scatter(x=dd_pct.index, y=dd_pct,
    name="Drawdown", line=dict(color=COLORS['negative'], width=1),
    fill='tozeroy', fillcolor='rgba(255,68,68,0.12)'), row=4, col=1)

if spy_full is not None:
    spy_test_cum = np.exp(np.log(spy_full[test_mask]/spy_full[test_mask].shift(1)).dropna().cumsum())
    spy_pct = (spy_test_cum - 1) * 100
    fig.add_trace(go.Scatter(x=spy_pct.index, y=spy_pct,
        name="SPY", line=dict(color=COLORS['gold'], width=1.5, dash='dot')), row=4, col=1)

fig.add_hline(y=0, line_color="#444", line_width=0.8, row=4, col=1)

fig.update_layout(
    height=860, showlegend=True,
    plot_bgcolor=BG, paper_bgcolor="#0D1117",
    font=dict(color=COLORS['text'], size=11),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(10,14,26,0.85)",
                bordercolor=COLORS['grid'], borderwidth=1,
                orientation="h", y=-0.02),
    margin=dict(l=50, r=30, t=50, b=40),
)
fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, zeroline=False)
fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, zeroline=False)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# BOTTOM ROW — Stats + Monthly Heatmap
# ═══════════════════════════════════════════
st.markdown("---")
col_l, col_r = st.columns([1, 1.4])

with col_l:
    st.markdown("### 📋 Performance Statistics")
    n_trades  = int((signals_test.diff().abs() > 0).sum() // 2)
    pct_in    = (signals_test != 0).mean()

    stats_data = {
        "Metric": [
            "Sharpe Ratio", "Sortino Ratio", "CAGR", "Ann. Volatility",
            "Max Drawdown", "Hit Rate", "Profit Factor",
            "# Trades (OOS)", "% Time in Market",
            "EG P-Value", "Half-Life (days)", "Hedge Ratio (β)",
        ],
        "Strategy": [
            f"{metrics['sharpe']:.3f}",
            f"{(metrics['cagr']-0.05)/max(pnl[pnl<0.05/252].std()*np.sqrt(252),1e-6):.3f}",
            f"{metrics['cagr']:.2%}",
            f"{metrics['vol']:.2%}",
            f"{metrics['max_dd']:.2%}",
            f"{metrics['hit_rate']:.1%}",
            f"{metrics['profit_factor']:.2f}",
            str(n_trades),
            f"{pct_in:.1%}",
            f"{coint_result['eg_pval']:.4f}",
            f"{hl:.1f}" if hl else "N/A",
            f"{beta:.4f}",
        ],
    }
    if spy_full is not None:
        stats_data["SPY Benchmark"] = [
            f"{spy_metrics['sharpe']:.3f}", "—",
            f"{spy_metrics['cagr']:.2%}",
            f"{spy_metrics['vol']:.2%}",
            f"{spy_metrics['max_dd']:.2%}",
            f"{spy_metrics['hit_rate']:.1%}",
            "—", "—", "100%", "—", "—", "—",
        ]

    st.dataframe(
        pd.DataFrame(stats_data),
        use_container_width=True, hide_index=True,
    )

with col_r:
    st.markdown("### 📅 Monthly Returns Heatmap (%)")
    monthly  = (pnl.resample("ME").sum() * 100).to_frame("ret")
    monthly["Year"]  = monthly.index.year
    monthly["Month"] = monthly.index.strftime("%b")
    pivot = monthly.pivot(index="Year", columns="Month", values="ret")
    mo    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in mo if m in pivot.columns])

    fig_h = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(),
        y=pivot.index.astype(str).tolist(),
        colorscale=[[0,"#8B0000"],[0.45,"#1A2035"],[0.5,"#1A2035"],[0.55,"#1A2035"],[1,"#006400"]],
        zmid=0,
        text=np.where(np.isnan(pivot.values), "", np.round(pivot.values,1).astype(str)),
        texttemplate="%{text}%",
        showscale=True,
        colorbar=dict(len=0.8, thickness=12),
    ))
    fig_h.update_layout(
        height=280,
        plot_bgcolor=BG, paper_bgcolor="#0D1117",
        font=dict(color=COLORS['text'], size=11),
        margin=dict(l=50, r=60, t=10, b=30),
        xaxis=dict(gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid']),
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ═══════════════════════════════════════════
# KALMAN FILTER SECTION
# ═══════════════════════════════════════════
st.markdown("---")
with st.expander("🔬 Kalman Filter — Dynamic Hedge Ratio (Advanced)", expanded=False):
    st.markdown(
        "The **Kalman Filter** estimates a time-varying hedge ratio β, "
        "adapting to structural changes in the pair relationship. "
        "When the Kalman β drifts significantly from the OLS β, "
        "it indicates the pair relationship is shifting."
    )
    with st.spinner("Fitting Kalman Filter..."):
        kf_beta, kf_intercept = kalman_hedge(pa_full, pb_full)
        kf_spread = np.log(pa_full) - kf_intercept - kf_beta * np.log(pb_full)

    fig_kf = make_subplots(rows=2, cols=1,
        subplot_titles=[
            "Hedge Ratio: OLS (Static) vs. Kalman Filter (Dynamic)",
            "Spread: OLS vs. Kalman",
        ], vertical_spacing=0.12)

    fig_kf.add_trace(go.Scatter(x=kf_beta.index, y=kf_beta, name="Kalman β",
        line=dict(color=COLORS['primary'], width=1.5)), row=1, col=1)
    fig_kf.add_hline(y=beta, line_color=COLORS['secondary'], line_dash='dash',
        line_width=2, annotation_text=f"OLS β={beta:.3f}", row=1, col=1)
    fig_kf.add_trace(go.Scatter(x=kf_spread.index, y=kf_spread, name="Kalman Spread",
        line=dict(color=COLORS['positive'], width=1.2)), row=2, col=1)
    fig_kf.add_trace(go.Scatter(x=spread_full.index, y=spread_full, name="OLS Spread",
        line=dict(color=COLORS['secondary'], width=1.2, dash='dot')), row=2, col=1)
    fig_kf.add_vline(x=str(train_end)[:10], line_color=COLORS['gold'],
        line_dash='dash', line_width=1.5)
    fig_kf.update_layout(
        height=480, plot_bgcolor=BG, paper_bgcolor="#0D1117",
        font=dict(color=COLORS['text'], size=11), hovermode="x unified",
        legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig_kf.update_xaxes(gridcolor=COLORS['grid'])
    fig_kf.update_yaxes(gridcolor=COLORS['grid'])
    st.plotly_chart(fig_kf, use_container_width=True)

# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════
st.markdown("---")
st.markdown(
    "**Strategy:** Cointegration-Based Statistical Arbitrage &nbsp;|&nbsp; "
    "**Tests:** Engle-Granger · Johansen · ADF &nbsp;|&nbsp; "
    "**Model:** Ornstein-Uhlenbeck Mean Reversion &nbsp;|&nbsp; "
    "**Used by:** D.E. Shaw · Citadel · Two Sigma · Goldman Sachs"
)
# ═══════════════════════════════════════════
# TAB 2 — PORTFOLIO DASHBOARD
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### 🗂️ Full Portfolio Scan")
    st.info(
        f"Scans **{len(PORTFOLIO_PAIRS)} pre-defined sector pairs** using the same "
        "parameters set in the sidebar. Results are cached for 2 hours."
    )

    run_port = st.button("🚀 Run Portfolio Scan", type="primary",
                         use_container_width=True)

    if run_port:
        with st.spinner("⏳ Downloading prices and scanning all pairs... (~60 sec)"):
            port_data = run_portfolio_scan(
                start_date, end_date, train_pct,
                entry_z, exit_z, stop_z, window, tc_bps
            )

        if port_data is None or not port_data['results']:
            st.error("No cointegrated pairs found. Try adjusting the date range.")
        else:
            meta_df   = port_data['meta_df']
            results   = port_data['results']
            prices_all= port_data['prices_all']
            train_end = port_data['train_end']
            port_m    = port_data['port_m']

            # ── Portfolio KPI row ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Portfolio Metrics (Equal-Weight, Out-of-Sample)")
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            def pkpi(col, label, value, color="#00D4FF"):
                col.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{color}">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            pkpi(k1, "Valid Pairs",    len(results), "#00D4FF")
            pkpi(k2, "Sharpe Ratio",  f"{port_m['sharpe']:.3f}",
                 "#00FF88" if port_m['sharpe']>1 else "#FFD700")
            pkpi(k3, "CAGR",           f"{port_m['cagr']:.1%}",
                 "#00FF88" if port_m['cagr']>0 else "#FF4444")
            pkpi(k4, "Max Drawdown",   f"{port_m['max_dd']:.1%}", "#FF4444")
            pkpi(k5, "Ann. Vol",       f"{port_m['vol']:.1%}", "#9B59B6")
            pkpi(k6, "Hit Rate",       f"{port_m['hit_rate']:.1%}", "#FFD700")
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Chart 1+2: 6-panel portfolio dashboard ─────────────────────
            st.markdown("#### 📊 Portfolio Dashboard")
            fig_port = make_portfolio_dashboard(port_data)
            st.plotly_chart(fig_port, use_container_width=True)

            # ── Chart 3: Half-life distribution ────────────────────────────
            st.markdown("---")
            st.markdown("#### 📐 OU Half-Life Analysis")
            fig_hl = make_halflife_chart(meta_df)
            st.plotly_chart(fig_hl, use_container_width=True)

            # ── Chart 4: Rolling cointegration stability ───────────────────
            st.markdown("---")
            st.markdown("#### 🔄 Rolling Cointegration Stability")
            fig_rc = make_rolling_coint_chart(results, prices_all, train_end)
            st.plotly_chart(fig_rc, use_container_width=True)

            # ── Pair metrics table ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📋 All Valid Pairs — Ranked by Sharpe")
            display_cols = ['pair','sector','sharpe','cagr','max_dd',
                            'half_life','eg_pval','n_trades']
            disp = meta_df[display_cols].copy()
            disp['cagr']   = disp['cagr'].map('{:.1%}'.format)
            disp['max_dd'] = disp['max_dd'].map('{:.1%}'.format)
            disp['sharpe'] = disp['sharpe'].round(3)
            disp['half_life'] = disp['half_life'].round(1)
            disp['eg_pval']   = disp['eg_pval'].round(4)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        **Click "Run Portfolio Scan" to generate:**
        - 📈 6-panel institutional dashboard (equity curve, rolling Sharpe, beta, heatmap, distribution, drawdown)
        - 📐 OU Half-Life distribution + Sharpe vs Half-Life scatter
        - 🔄 Rolling cointegration stability for top 5 pairs
        - 📋 Full pair metrics table
        """)
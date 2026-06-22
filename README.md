# Statistical-Pairs-Trading-Engine
Scans the S&amp;P 500 for cointegrated stock pairs using Engle-Granger and Johansen tests, constructs a mean-reverting spread, signals trades using z-scores of the spread, and backtests the long-short strategy with realistic transaction costs.

# Quantitative finance concepts explained
Cointegration vs. correlation 
Two stocks can be highly correlated without being cointegrated.Correlation measures how returns move together today. Cointegration means their price levels are tied by a long-run equilibrium — they can diverge in the short term but are "pulled back" together. If you trade on correlation alone, you have no mean-reversion guarantee.Cointegration gives you that.

I(1) and I(0). Stock prices are I(1) — they have a unit root, meaning they accumulate shocks permanently (a random walk). If you differenced them once (Δlog_P) you get I(0) (stationary returns). The brilliant insight of pairs trading: take a linear combination of two I(1) series and get an I(0) series — the spread. That's exactly what log(A) - β·log(B) is. 

The Ornstein-Uhlenbeck process. The continuous-time model of mean reversion: dX = κ(μ - X)dt + σdW. κ is the "pull" back to the mean μ. High κ = fast reversion = short trades. The discrete AR(1) regression in fit_ou_process() estimates κ directly from the data. The half-life formula ln(2)/κ tells you how long it takes for a deviation to halve — this is your expected trade duration.

The hedge ratio β. This is how you make the spread stationary. You can't just take P_A - P_B because the prices are in different units and scales. Instead you regress log(P_A) on log(P_B), and the slope β tells you: "for every $1 of log-price in A, I need β dollars of log-price in B to create a balanced spread." A β of 1.3 means you short 1.3x as much B as you long A.

Johansen vs. Engle-Granger. EG is asymmetric — it regresses A on B (not B on A). If you swap the dependent variable you might get a different answer. Johansen is symmetric; it uses maximum likelihood to find all cointegrating vectors simultaneously. Using both as a requirement (EG first, Johansen confirms) is more robust than either alone, which is why JOHANSEN_CONFIRM = True is the right default.

Z-score signal logic. The z-score measures how many standard deviations the current spread is from its recent mean. If z = +2.5, the spread is historically "expensive" — stock A is overpriced relative to B. You short the spread (short A, long B) and bet on reversion. If z = -2.5, the spread is "cheap" — long the spread (long A, short B). Exit at ±0.5 when you've captured most of the reversion.

# Three institutional-grade upgrades
Upgrade 1: Kalman Filter as the default hedge ratio (already sketched in Cell 17, but not used in the main pipeline). Replace the static OLS β with the time-varying Kalman estimate everywhere — in spread construction, in signals, and in the backtest. This single change typically reduces tracking error by 15–30% and improves Sharpe significantly on pairs where the relationship drifts. The math is already in the notebook; you just need to wire kalman_filter_hedge_ratio() into PairsTradingBacktester.run() before spread construction.

Upgrade 2: Regime-conditional trading using a Hidden Markov Model. The strategy works well in mean-reverting (low-volatility) regimes and loses money in trending/crisis regimes. Train a 2-state HMM on VIX levels, realized volatility, and cross-sectional correlation to classify each day as "mean-reverting" or "trending. " Only take new positions in the mean-reverting state; close existing positions when the model switches state. This directly addresses the strategy's biggest failure mode — pairs trading during the 2022 rate shock, for example.

Upgrade 3: Volatility-targeted position sizing instead of equal-weight. Currently every pair contributes 1/N of portfolio weight regardless of its individual volatility. A pair with 20% annualized vol gets the same weight as one with 5% vol, meaning the first pair contributes 16× more risk. Replace this with a volatility-targeting approach: size each pair so it contributes a fixed annualized volatility (e.g., 2%) to the portfolio. This produces a more stable Sharpe, reduces tail risk, and is the standard approach at Citadel and Millennium's stat-arb pods.

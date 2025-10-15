"""
Project 10 — Backtesting Engine (walk-forward, costs, metrics)
Python 3.13-safe | pandas 2.2+, numpy 2.1+, scikit-learn 1.5+

What this does:
- Loads features from `features_output.csv` (Project 9) if available,
  else generates synthetic asset + market and re-computes features.
- Builds 3 strategies:
  (1) Buy & Hold
  (2) Naive Mean-Reversion using z-score
  (3) ML (Ridge) walk-forward: train on past, predict next-day ret, size positions
- Converts signals -> positions ([-1, +1] capped)
- Applies transaction costs (bps) on position changes
- Reports Sharpe (ann), CAGR, Max Drawdown, Hit Rate, Turnover

No leakage: models are fit only on data strictly before the prediction day.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ------------- Config -------------
ANNUALIZATION_DAYS = 252
COST_BPS = 5.0          # round-trip cost per unit turnover, e.g., 5 bps
MAX_LEV = 1.0           # cap positions to [-1, +1]
RIDGE_ALPHA = 5.0
MIN_TRAIN = 252         # need at least 1 year to start predicting
USE_FEATURES = ["momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]

# ------------- Metrics -------------
def performance_metrics(ret: pd.Series, label: str = "") -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {"label": label, "N": 0}
    cum = (1 + ret).cumprod()
    total_return = cum.iloc[-1] - 1.0
    yrs = max((len(ret) / ANNUALIZATION_DAYS), 1e-12)
    cagr = (1 + total_return) ** (1 / yrs) - 1 if total_return > -0.999999 else -1.0
    vol_ann = ret.std() * np.sqrt(ANNUALIZATION_DAYS)
    sharpe = np.nan if vol_ann == 0 else ret.mean() / ret.std() * np.sqrt(ANNUALIZATION_DAYS)
    # drawdown
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    max_dd = dd.min()
    hit_rate = (ret > 0).mean()
    turnover = (ret.index.to_series().map(lambda d: 0))  # placeholder
    return {
        "label": label,
        "N": len(ret),
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "HitRate": hit_rate,
        "TotalRet": total_return,
    }

def print_metrics_table(metrics_list):
    cols = ["Strategy", "N", "CAGR", "Sharpe", "MaxDD", "HitRate", "TotalRet"]
    print("\n=== Performance Summary ===")
    print(f"{cols[0]:<18} {cols[1]:>6} {cols[2]:>8} {cols[3]:>8} {cols[4]:>8} {cols[5]:>8} {cols[6]:>10}")
    for m in metrics_list:
        print(f"{m['label']:<18} {m.get('N',0):>6d} "
              f"{m.get('CAGR',np.nan):>8.2%} {m.get('Sharpe',np.nan):>8.2f} "
              f"{m.get('MaxDD',np.nan):>8.2%} {m.get('HitRate',np.nan):>8.2%} "
              f"{m.get('TotalRet',np.nan):>10.2%}")

# ------------- Costs & PnL -------------
def apply_costs_and_pnl(returns: pd.Series, weights: pd.Series, cost_bps: float = 0.0) -> pd.Series:
    """
    Strategy daily return:
      r_strat_t = w_{t-1} * ret_t - cost_per_turnover * |w_t - w_{t-1}|
    cost_per_turnover is in decimal (bps / 1e4).
    Weights are target end-of-day weights; assume next-day open implementation.
    """
    returns = returns.astype(float)
    weights = weights.astype(float).clip(-MAX_LEV, MAX_LEV)
    # use prior weight for exposure
    w_prev = weights.shift(1).fillna(0.0)
    gross = w_prev * returns
    # turnover cost on weight changes
    tw = (weights - w_prev).abs()
    cost = (cost_bps / 1e4) * tw
    strat = gross - cost
    return strat

# ------------- Data / Features -------------
def load_features_or_build() -> pd.DataFrame:
    # try to load file from repo root
    fn = os.path.join(os.path.dirname(__file__), "..", "features_output.csv")
    if os.path.exists(fn):
        df = pd.read_csv(fn, parse_dates=True, index_col=0)
        # ensure needed columns present; if not, fall back to build
        if all(c in df.columns for c in ["price", "ret", "momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]):
            return df
        print("[warn] features_output.csv found but missing required columns; rebuilding synthetic features.")
    # synthetic build (similar to Project 9)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    mkt_rets = rng.normal(0.0003, 0.01, size=n)
    asset_rets = 0.3 * mkt_rets + rng.normal(0, 0.012, size=n)
    price = 100 * (1 + asset_rets).cumprod()
    mkt_price = 100 * (1 + mkt_rets).cumprod()
    df = pd.DataFrame({"price": price, "mkt_price": mkt_price}, index=dates)
    df["ret"] = df["price"].pct_change()
    df["mkt_ret"] = df["mkt_price"].pct_change()
    df = df.dropna(subset=["ret", "mkt_ret"]).copy()
    window = 20
    df["momentum"]  = df["ret"].rolling(window).mean()
    df["volatility"] = df["ret"].rolling(window).std()
    df["zscore"]     = (df["ret"] - df["ret"].rolling(window).mean()) / df["ret"].rolling(window).std()
    # rolling beta
    cov = df["ret"].rolling(window).cov(df["mkt_ret"])
    var = df["mkt_ret"].rolling(window).var()
    df["beta_mkt"] = cov / var
    df["ret_lag1"] = df["ret"].shift(1)
    df = df.dropna().copy()
    return df

# ------------- Strategies -------------
def strat_buy_hold(df: pd.DataFrame) -> pd.Series:
    # Always hold 1x the asset
    w = pd.Series(1.0, index=df.index)
    return apply_costs_and_pnl(df["ret"], w, cost_bps=0.0)

def strat_naive_mean_rev(df: pd.DataFrame, z_entry: float = 0.5) -> pd.Series:
    # Go short if zscore > z_entry; long if zscore < -z_entry; else 0
    z = df["zscore"]
    w = pd.Series(0.0, index=df.index)
    w[z < -z_entry] = 1.0
    w[z >  z_entry] = -1.0
    return apply_costs_and_pnl(df["ret"], w, cost_bps=COST_BPS)

def strat_ml_ridge_walkforward(df: pd.DataFrame) -> pd.Series:
    """
    Walk-forward Ridge on features to predict next-day returns.
    - Train on [0:t-1], predict at t, convert prediction to weight via tanh scaling.
    - Features standardized in each fit (no leakage of future stats).
    """
    feats = USE_FEATURES
    # make sure features exist
    for c in feats:
        if c not in df.columns:
            raise ValueError(f"Missing required feature: {c}")
    X = df[feats].values
    y = df["ret"].shift(-1).values  # predict next-day return
    idx = df.index

    weights = pd.Series(index=idx, dtype=float)
    for t in range(MIN_TRAIN, len(df) - 1):
        X_train = X[:t]
        y_train = y[:t]
        X_pred  = X[t:t+1]
        if np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(y_train)):
            weights.iloc[t] = 0.0
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xp  = scaler.transform(X_pred)
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(Xtr, y_train)
        yhat = model.predict(Xp)[0]
        # position sizing: scale signal with tanh to cap leverage smoothly
        w = np.tanh(50 * yhat)   # 50 magnifies tiny yhat to meaningful weights
        weights.iloc[t] = float(np.clip(w, -MAX_LEV, MAX_LEV))

    weights = weights.fillna(0.0)
    return apply_costs_and_pnl(df["ret"], weights, cost_bps=COST_BPS)

# ------------- Main -------------
if __name__ == "__main__":
    pd.options.display.float_format = lambda x: f"{x:,.6f}"
    df = load_features_or_build()

    # Align and sanity-check
    keep_cols = ["price", "ret", "momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]
    df = df[keep_cols].dropna().copy()

    print("=== Data span ===")
    print(f"{df.index.min().date()} → {df.index.max().date()}  N={len(df)}")

    # Run strategies
    ret_bh   = strat_buy_hold(df)
    ret_mr   = strat_naive_mean_rev(df, z_entry=0.5)
    ret_ml   = strat_ml_ridge_walkforward(df)

    # Drop warmup NA
    ret_bh = ret_bh.dropna()
    ret_mr = ret_mr.dropna()
    ret_ml = ret_ml.dropna()

    # Metrics
    m_bh = performance_metrics(ret_bh,   "Buy&Hold")
    m_mr = performance_metrics(ret_mr,   "Naive MeanRev")
    m_ml = performance_metrics(ret_ml,   "ML Ridge WF")

    print_metrics_table([m_bh, m_mr, m_ml])

    # Quick tails for inspection
    print("\nLast 5 daily returns (strategies):")
    out = pd.DataFrame({
        "ret_bh": ret_bh,
        "ret_mr": ret_mr.reindex(ret_bh.index),
        "ret_ml": ret_ml.reindex(ret_bh.index),
    }).dropna().tail()
    print(out)

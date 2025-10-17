"""
Project 11 — Portfolio Construction (clean 2-level MultiIndex version)
- Multi-asset walk-forward backtest
- Alpha: per-asset Ridge on engineered features (from Project 9 logic)
- Risk: ridge-shrunk rolling covariance
- Sizing: mean-variance style w ∝ Σ^-1 μ_hat with leverage cap
- Vol targeting to hit target annualized volatility
- Costs and metrics

Python 3.13 | pandas 2.2+ | numpy 2.1+ | scikit-learn 1.5+
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --------------------- CONFIG ---------------------
ANNUALIZATION_DAYS = 252
SEED = 42
N_ASSETS = 4
MIN_TRAIN = 252               # warmup before first prediction
FEAT_WIN = 20                 # rolling window for features
COV_WIN = 60                  # rolling window for covariance
RIDGE_ALPHA = 5.0             # alpha model L2
COV_RIDGE = 5e-6              # covariance ridge add-on (λI)
TARGET_VOL = 0.10             # annualized vol target
COST_BPS = 5.0                # cost per unit turnover (sum|Δw|), in bps
LEVERAGE_CAP = 1.5            # cap total gross exposure (L1)
FEATURES = ["momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]

# --------------------- UTILS ---------------------
def ann_factor() -> float:
    return np.sqrt(ANNUALIZATION_DAYS)

def realized_vol(ret: pd.Series, win: int) -> float:
    r = ret.tail(win).dropna()
    if len(r) == 0:
        return np.nan
    return float(r.std() * ann_factor())

def performance_metrics(ret: pd.Series, label: str) -> Dict[str, float]:
    ret = ret.dropna()
    if len(ret) == 0:
        return {"label": label, "N": 0}
    cum = (1 + ret).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0)
    years = max(len(ret) / ANNUALIZATION_DAYS, 1e-12)
    cagr = (1 + total_ret) ** (1 / years) - 1 if total_ret > -0.999999 else -1.0
    vol_ann = ret.std() * ann_factor()
    sharpe = np.nan if vol_ann == 0 else ret.mean() / ret.std() * ann_factor()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    max_dd = float(dd.min())
    hit = float((ret > 0).mean())
    return {"label": label, "N": len(ret), "CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "HitRate": hit, "TotalRet": total_ret}

def print_metrics_table(metrics_list: List[Dict[str, float]]):
    cols = ["Strategy", "N", "CAGR", "Sharpe", "MaxDD", "HitRate", "TotalRet"]
    print("\n=== Portfolio Performance ===")
    print(f"{cols[0]:<28} {cols[1]:>6} {cols[2]:>8} {cols[3]:>8} {cols[4]:>8} {cols[5]:>8} {cols[6]:>10}")
    for m in metrics_list:
        print(f"{m['label']:<28} {m.get('N',0):>6d} "
              f"{m.get('CAGR',np.nan):>8.2%} {m.get('Sharpe',np.nan):>8.2f} "
              f"{m.get('MaxDD',np.nan):>8.2%} {m.get('HitRate',np.nan):>8.2%} "
              f"{m.get('TotalRet',np.nan):>10.2%}")

def apply_costs(returns: pd.Series, weights: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    Portfolio daily return after costs:
      r_t = sum_i w_{i,t-1} * r_{i,t}  -  (bps/1e4) * sum_i |w_{i,t} - w_{i,t-1}|
    """
    w_prev = weights.shift(1).fillna(0.0)
    gross = (w_prev * returns.to_frame().reindex(columns=weights.columns, fill_value=0)).sum(axis=1)
    turnover = (weights - w_prev).abs().sum(axis=1)
    cost = (cost_bps / 1e4) * turnover
    return gross - cost

# --------------------- DATA (2-level MultiIndex: (field, asset)) ---------------------
def load_or_simulate_multi_asset() -> pd.DataFrame:
    """
    Simulate N_ASSETS correlated assets + market and compute features per asset.
    Columns are MultiIndex (field, asset). Example fields:
      'price','ret','mkt_ret','momentum','volatility','zscore','beta_mkt','ret_lag1'
    """
    rng = np.random.default_rng(SEED)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Market factor
    mkt_ret = rng.normal(0.0002, 0.010, size=n)
    mkt_price = 100 * (1 + mkt_ret).cumprod()

    # Simulate assets
    betas = np.linspace(0.2, 0.8, N_ASSETS)
    idio_vols = np.linspace(0.010, 0.016, N_ASSETS)

    cols = []
    data = {}

    for i in range(N_ASSETS):
        a = f"A{i+1}"
        r = betas[i] * mkt_ret + rng.normal(0.0000, idio_vols[i], size=n)
        p = 100 * (1 + r).cumprod()

        data[("price", a)] = p
        data[("ret", a)] = r
        data[("mkt_ret", a)] = mkt_ret

    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])

    # Features per asset (same 2-level structure)
    for i in range(N_ASSETS):
        a = f"A{i+1}"
        ret = df[("ret", a)]
        mret = df[("mkt_ret", a)]

        mom = ret.rolling(FEAT_WIN).mean()
        vol = ret.rolling(FEAT_WIN).std()
        zsc = (ret - mom) / vol
        cov = ret.rolling(FEAT_WIN).cov(mret)
        var = mret.rolling(FEAT_WIN).var().replace(0.0, np.nan)
        beta = cov / var
        lag1 = ret.shift(1)

        df[("momentum", a)] = mom
        df[("volatility", a)] = vol
        df[("zscore", a)] = zsc
        df[("beta_mkt", a)] = beta
        df[("ret_lag1", a)] = lag1

    df = df.dropna()  # drop initial rolling NaNs
    return df

# --------------------- CORE: PORTFOLIO LOOP ---------------------
def ridge_alpha_per_asset(X_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xp = scaler.transform(x_pred.reshape(1, -1))
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xtr, y_train)
    yhat = float(model.predict(Xp)[0])
    return yhat

def shrinked_cov(returns_window: pd.DataFrame, lam: float) -> np.ndarray:
    """
    Ridge shrinkage: Σ_shrunk = Σ + λI
    """
    R = returns_window.values
    R = R[~np.isnan(R).any(axis=1)]
    if R.shape[0] < 2:
        k = returns_window.shape[1]
        return np.eye(k) * 1e-4
    Sigma = np.cov(R.T, ddof=1)
    k = Sigma.shape[0]
    return Sigma + lam * np.eye(k)

def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, cap_gross: float) -> np.ndarray:
    """
    w ∝ Σ^-1 μ, then normalize to respect gross leverage cap (L1).
    """
    try:
        inv = np.linalg.pinv(Sigma, rcond=1e-10)
        w = inv @ mu
    except np.linalg.LinAlgError:
        w = np.zeros_like(mu)
    if np.allclose(w, 0):
        return w
    gross = np.sum(np.abs(w))
    if gross > cap_gross and gross > 0:
        w = w * (cap_gross / gross)
    return w

def backtest_portfolio(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    df: columns MultiIndex (field, asset)
    Steps per day t:
      - per asset: train Ridge on past FEATURES to predict next-day return (μ̂)
      - Σ_t from past COV_WIN days of returns (shrinked)
      - Raw w ∝ Σ^-1 μ̂, cap gross
      - Vol targeting to TARGET_VOL
      - Next-day PnL uses w_t and r_{t+1}; subtract trading costs
    """
    # assets = second-level unique labels
    assets = list(df.columns.get_level_values("asset").unique())
    dates = df.index

    # Panels
    ret_panel = df.xs("ret", axis=1, level="field")  # T x N
    # feature panel per asset: use xs(asset) to get a column Index of fields, then subset FEATURES
    feat_panel = {}
    for a in assets:
        feat_a = df.xs(a, axis=1, level="asset")
        # ensure all features exist
        missing = [f for f in FEATURES if f not in feat_a.columns]
        if missing:
            X = np.zeros((len(df), len(FEATURES)), dtype=float)
        else:
            X = feat_a[FEATURES].values
            # replace any non-finite with 0 to be safe
            X[~np.isfinite(X)] = 0.0
        feat_panel[a] = X

    weights = pd.DataFrame(0.0, index=dates, columns=assets)
    port_ret = pd.Series(0.0, index=dates, dtype=float)
    T = len(df)

    for t in range(MIN_TRAIN, T - 1):
        # ---- alpha vector μ_hat ----
        mu_hat = []
        for a in assets:
            X = feat_panel[a][:t, :]
            y = ret_panel[a].shift(-1).values[:t]  # next-day return target
            x_pred = feat_panel[a][t, :]

            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            Xc = X[mask]
            yc = y[mask]
            if len(yc) < 30 or not np.isfinite(x_pred).all():
                mu_hat.append(0.0)
                continue
            try:
                yhat = ridge_alpha_per_asset(Xc, yc, x_pred)
            except Exception:
                yhat = 0.0
            mu_hat.append(yhat)
        mu_hat = np.array(mu_hat, dtype=float)

        # ---- covariance Σ_t ----
        ret_win = ret_panel.iloc[t - COV_WIN:t, :]
        Sigma = shrinked_cov(ret_win, lam=COV_RIDGE)

        # ---- raw weights ----
        w_raw = mean_variance_weights(mu_hat, Sigma, cap_gross=LEVERAGE_CAP)

        # ---- volatility targeting ----
        # estimate vol either from historical simulated portfolio returns or Sigma
        if t > MIN_TRAIN:
            w_hist = weights.iloc[max(0, t - COV_WIN):t, :].shift(1).fillna(0.0)
            if not w_hist.empty:
                ret_hist = (ret_panel.iloc[max(0, t - COV_WIN):t, :] * w_hist).sum(axis=1)
                cur_vol = realized_vol(ret_hist, win=min(60, len(ret_hist)))
            else:
                cur_vol = np.nan
        else:
            cur_vol = np.nan

        if not np.isfinite(cur_vol) or cur_vol == 0:
            try:
                port_var = float(np.dot(w_raw, Sigma @ w_raw))
                cur_vol = np.sqrt(port_var) * ann_factor()
            except Exception:
                cur_vol = np.nan

        scale = TARGET_VOL / cur_vol if (np.isfinite(cur_vol) and cur_vol > 1e-8) else 1.0
        w_t = np.clip(w_raw * scale, -LEVERAGE_CAP, LEVERAGE_CAP)
        weights.iloc[t, :] = w_t

        # ---- next-day PnL ----
        r_next = ret_panel.iloc[t + 1, :].values
        port_ret.iloc[t + 1] = float(np.dot(w_t, r_next))

    # Apply costs on turnover and return the after-cost returns
    port_ret_after = apply_costs(port_ret, weights, cost_bps=COST_BPS)
    return port_ret_after, weights

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    pd.options.display.float_format = lambda x: f"{x:,.6f}"

    df = load_or_simulate_multi_asset()
    df = df.dropna().copy()

    # Backtest
    port_ret, weights = backtest_portfolio(df)

    # Metrics
    met = performance_metrics(port_ret, "MV + VolTarget (Multi-Asset, Ridge α & Σ)")
    print_metrics_table([met])

    print("\nLast 5 days — weights:")
    print(weights.tail().round(3))

    print("\nLast 5 days — portfolio returns:")
    print(port_ret.tail().to_frame("ret"))

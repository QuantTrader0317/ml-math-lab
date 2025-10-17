"""
Project 14 — Signal Stacking + Walk-Forward Cross-Validation (Weekly Rebalance)
Python 3.13 | numpy 2.1+ | pandas 2.2+ | scikit-learn 1.5+

What this does:
- Simulate multi-asset prices/returns and build features (momentum, zscore, vol, beta, lag1)
- For each weekly rebalance date:
    * Do a walk-forward CV on the past window per asset
    * Train base models (Ridge, Lasso, RF, GB) and get out-of-fold (OOF) predictions
    * Fit a meta-learner (Ridge) on those OOF predictions → stacked alpha
    * Refit base models on full past, predict for "today", stack via meta-learner → μ̂ per asset
- Risk model: EWMA covariance (RiskMetrics λ), small ridge add-on
- Portfolio: mean-variance weights with per-asset cap & gross cap
- Extras: beta-neutralization, vol targeting, μ̂ threshold, top-K selection, weekly trading costs
- Benchmarks: Buy&Hold equal-weight, Ridge-only alpha model

Outputs: performance table + last 5 lines samples
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import set_config
set_config(assume_finite=True)
from typing import Dict, List, Tuple
from dataclasses import dataclass

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --------------------- CONFIG ---------------------
ANNUALIZATION_DAYS = 252
SEED = 42
N_ASSETS = 4

# Features / data
FEAT_WIN = 20
MIN_TRAIN = 252              # warmup
CV_WINDOW = 252              # lookback window for CV/meta fit (~1 year)
PRICE_START = 15.0           # synthetic assets start near this price

FEATURES = ["momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]

# Alpha (stacking)
BASE_MODELS = {
    "ridge": lambda: Ridge(alpha=5.0, fit_intercept=True),
    "lasso": lambda: Lasso(alpha=0.0005, fit_intercept=True, max_iter=2000),
    "rf":    lambda: RandomForestRegressor(n_estimators=150, max_depth=4, random_state=SEED, n_jobs=-1),
    "gb":    lambda: GradientBoostingRegressor(random_state=SEED, n_estimators=200, max_depth=2, learning_rate=0.05)
}
META_MODEL = lambda: Ridge(alpha=1.0, fit_intercept=True)

MU_CLIP = 0.004              # clip |μ̂| <= 40 bps
MU_MIN  = 0.0005             # zero-out tiny μ̂ (less noise trading)
TOP_K   = 3                  # keep only top-K |μ̂| assets each rebalance

# Risk model & portfolio
EWMA_LAMBDA = 0.94
COV_RIDGE = 5e-6
TARGET_VOL = 0.12
ASSET_CAP = 0.30
GROSS_CAP = 1.50
BETA_NEUTRAL = True

# Rebalance & cost
REBALANCE_WEEKDAY = 4        # Friday
COST_BPS_PER_REBAL = 2.0     # 2 bps per "notional turnover" at rebalance

pd.options.display.float_format = lambda x: f"{x:,.6f}"


# --------------------- UTILS ---------------------
def ann_factor() -> float:
    return np.sqrt(ANNUALIZATION_DAYS)

def performance_metrics(ret: pd.Series, label: str) -> Dict[str, float]:
    ret = ret.dropna()
    if len(ret) == 0:
        return {"label": label, "N": 0}
    cum = (1 + ret).cumprod()
    total = float(cum.iloc[-1] - 1.0)
    years = max(len(ret) / ANNUALIZATION_DAYS, 1e-12)
    cagr = (1 + total) ** (1 / years) - 1 if total > -0.999999 else -1.0
    vol_ann = ret.std() * ann_factor()
    sharpe = np.nan if vol_ann == 0 else ret.mean() / ret.std() * ann_factor()
    dd = (cum / cum.cummax()) - 1.0
    maxdd = float(dd.min())
    hit = float((ret > 0).mean())
    return {"label": label, "N": len(ret), "CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "HitRate": hit, "TotalRet": total}

def print_metrics_table(metrics_list: List[Dict[str, float]]):
    cols = ["Strategy", "N", "CAGR", "Sharpe", "MaxDD", "HitRate", "TotalRet"]
    print("\n=== Performance Summary ===")
    print(f"{cols[0]:<24} {cols[1]:>6} {cols[2]:>8} {cols[3]:>8} {cols[4]:>8} {cols[5]:>8} {cols[6]:>10}")
    for m in metrics_list:
        print(f"{m['label']:<24} {m.get('N',0):>6d} "
              f"{m.get('CAGR',np.nan):>8.2%} {m.get('Sharpe',np.nan):>8.2f} "
              f"{m.get('MaxDD',np.nan):>8.2%} {m.get('HitRate',np.nan):>8.2%} "
              f"{m.get('TotalRet',np.nan):>10.2%}")


# --------------------- DATA ---------------------
def load_or_simulate_multi_asset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    mkt_ret = rng.normal(0.0002, 0.010, size=n)

    betas = np.linspace(0.2, 0.8, N_ASSETS)
    idio_vols = np.linspace(0.010, 0.016, N_ASSETS)

    data = {}
    for i in range(N_ASSETS):
        a = f"A{i+1}"
        r = betas[i] * mkt_ret + rng.normal(0.0000, idio_vols[i], size=n)
        p = PRICE_START * (1 + r).cumprod()
        data[("price", a)] = p
        data[("ret", a)] = r
        data[("mkt_ret", a)] = mkt_ret

    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])

    # Rolling features
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
        df[("momentum", a)]  = mom
        df[("volatility", a)] = vol
        df[("zscore", a)]     = zsc
        df[("beta_mkt", a)]   = beta
        df[("ret_lag1", a)]   = lag1

    df = df.dropna()
    return df


# --------------------- ALPHA STACKING ---------------------
def time_series_oof_preds(X: np.ndarray, y: np.ndarray, models: Dict[str, callable], n_splits: int = 5) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build out-of-fold (OOF) predictions for each base model using TimeSeriesSplit.
    Returns:
        preds_df: shape (n_oof, n_models) aligned to y_oof
        y_oof   : shape (n_oof,)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = {name: [] for name in models.keys()}
    y_oof = []
    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        # Scale linear models; trees/GB don't need scaling but it's harmless
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        for name, make_model in models.items():
            model = make_model()
            try:
                model.fit(Xtr_s, ytr)
                p = model.predict(Xte_s)
            except Exception:
                p = np.zeros_like(yte)
            preds[name].extend(list(p))
        y_oof.extend(list(yte))

    preds_df = pd.DataFrame(preds)
    y_oof = np.array(y_oof)
    return preds_df, y_oof


def stacked_mu_hat_for_today(X_hist: np.ndarray, y_hist: np.ndarray, x_today: np.ndarray) -> float:
    """
    Build OOF preds on the most recent CV_WINDOW subset (or all if shorter),
    fit META on OOF predictions, refit base models on *all past*, predict today,
    stack via META. Returns clipped μ̂.
    """
    if len(y_hist) < 120 or not np.isfinite(x_today).all():
        return 0.0

    # use last CV_WINDOW window
    if len(y_hist) > CV_WINDOW:
        Xw = X_hist[-CV_WINDOW:]
        yw = y_hist[-CV_WINDOW:]
    else:
        Xw = X_hist
        yw = y_hist

    # OOF predictions for meta fit
    preds_df, y_oof = time_series_oof_preds(Xw, yw, BASE_MODELS, n_splits=5)
    # Fit META on OOF preds
    meta = META_MODEL()
    try:
        meta.fit(preds_df.values, y_oof)
    except Exception:
        # fallback to simple linear blend if meta fails
        meta = LinearRegression().fit(preds_df.values, y_oof)

    # Refit base models on all past, predict today
    scaler = StandardScaler().fit(X_hist)
    X_all = scaler.transform(X_hist)
    x_t = scaler.transform(x_today.reshape(1, -1))

    base_preds = []
    for name, make_model in BASE_MODELS.items():
        model = make_model()
        try:
            model.fit(X_all, y_hist)
            p = float(model.predict(x_t)[0])
        except Exception:
            p = 0.0
        base_preds.append(p)

    mu = float(meta.predict(np.array(base_preds).reshape(1, -1))[0])
    # clip & threshold
    mu = float(np.clip(mu, -MU_CLIP, MU_CLIP))
    if abs(mu) < MU_MIN:
        mu = 0.0
    return mu


# --------------------- RISK & WEIGHTS ---------------------
def ewma_cov(returns: pd.DataFrame, lam: float, ridge: float) -> np.ndarray:
    R = returns.values
    R = R[~np.isnan(R).any(axis=1)]
    if R.size == 0:
        k = returns.shape[1]
        return np.eye(k) * 1e-4
    k = R.shape[1]
    Sigma = np.zeros((k, k), dtype=float)
    for r in R:
        r = r.reshape(-1, 1)
        Sigma = lam * Sigma + (1 - lam) * (r @ r.T)
    return Sigma + ridge * np.eye(k)

def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, gross_cap: float, asset_cap: float) -> np.ndarray:
    inv = np.linalg.pinv(Sigma, rcond=1e-10)
    w = inv @ mu
    # per-asset cap
    w = np.clip(w, -asset_cap, asset_cap)
    # gross cap
    gross = np.sum(np.abs(w))
    if gross > gross_cap and gross > 0:
        w *= (gross_cap / gross)
    return w

def project_beta_neutral(w: np.ndarray, betas: np.ndarray) -> np.ndarray:
    if not np.isfinite(betas).any():
        return w
    num = float(betas @ w)
    den = float(betas @ betas)
    if den > 1e-12:
        w = w - (num / den) * betas
    return w


# --------------------- BACKTEST ---------------------
def backtest_stacked(df: pd.DataFrame, use_stacking: bool = True, ridge_only: bool = False) -> pd.Series:
    """
    Build weekly weights from μ̂ (stacked or ridge-only) + EWMA Σ,
    then compute daily portfolio returns with simple cost on rebalance.
    """
    assets = list(df.columns.get_level_values("asset").unique())
    dates = df.index
    ret_panel = df.xs("ret", axis=1, level="field")
    beta_panel = df.xs("beta_mkt", axis=1, level="field").copy()

    # Prebuild features per asset
    featX = {}
    y_next = {}
    for a in assets:
        fa = df.xs(a, axis=1, level="asset").reindex(columns=FEATURES)
        X = fa.values
        y = df[("ret", a)].shift(-1).values  # predict next day's return
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        featX[a] = (X, mask)  # we'll subselect up to time t
        y_next[a] = y

    weights = pd.DataFrame(0.0, index=dates, columns=assets)
    w_prev = np.zeros(len(assets), float)

    for t in range(MIN_TRAIN, len(dates) - 1):
        d = dates[t]
        if d.weekday() != REBALANCE_WEEKDAY:
            # carry forward
            weights.iloc[t, :] = w_prev
            continue

        mus = []
        # make μ̂ per-asset
        for i, a in enumerate(assets):
            X_all, mask_all = featX[a]
            y_all = y_next[a]

            # restrict to past data up to t
            X_hist_full = X_all[:t]
            y_hist_full = y_all[:t]
            mask = mask_all[:t] & np.isfinite(y_hist_full)
            X_hist = X_hist_full[mask]
            y_hist = y_hist_full[mask]

            x_today = X_all[t]
            if ridge_only:
                # simpler baseline: ridge only (no stacking/meta)
                mu = simple_ridge_mu(X_hist, y_hist, x_today)
            else:
                mu = stacked_mu_hat_for_today(X_hist, y_hist, x_today) if use_stacking else 0.0
            mus.append(mu)

        mu_vec = np.array(mus, dtype=float)

        # Top-K selection by |μ̂| (others get μ̂=0)
        if TOP_K is not None and TOP_K < len(mu_vec):
            idx = np.argsort(-np.abs(mu_vec))  # descending by abs
            keep = set(idx[:TOP_K])
            for i in range(len(mu_vec)):
                if i not in keep:
                    mu_vec[i] = 0.0

        # Σ via EWMA on last ~1y
        start = max(0, t - 252)
        Sigma = ewma_cov(ret_panel.iloc[start:t, :], lam=EWMA_LAMBDA, ridge=COV_RIDGE)

        # MV weights
        w_raw = mean_variance_weights(mu_vec, Sigma, GROSS_CAP, ASSET_CAP)

        # Beta neutral (optional)
        if BETA_NEUTRAL:
            betas_now = beta_panel.iloc[t, :].reindex(index=assets).values
            betas_now[~np.isfinite(betas_now)] = 0.0
            w_raw = project_beta_neutral(w_raw, betas_now)
            w_raw = np.clip(w_raw, -ASSET_CAP, ASSET_CAP)

        # Vol targeting
        port_var = float(w_raw @ (Sigma @ w_raw))
        cur_vol = np.sqrt(max(port_var, 0.0)) * ann_factor() if port_var > 0 else np.nan
        scale = TARGET_VOL / cur_vol if (np.isfinite(cur_vol) and cur_vol > 1e-8) else 1.0
        w_t = np.clip(w_raw * scale, -ASSET_CAP, ASSET_CAP)

        weights.iloc[t, :] = w_t
        w_prev = w_t.copy()

    # forward-fill weights
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)

    # Strategy daily returns: r_p(t) = w(t-1)^T r(t)
    strat_ret = (weights.shift(1) * ret_panel).sum(axis=1).fillna(0.0)

    # Simple cost model: on rebalance days, subtract 2 bps * turnover (L1 diff of weights)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = np.where(dates.to_series().dt.weekday == REBALANCE_WEEKDAY,
                    (COST_BPS_PER_REBAL / 1e4) * turnover.values, 0.0)
    strat_ret_after_cost = strat_ret - cost

    return strat_ret_after_cost


def simple_ridge_mu(X_hist: np.ndarray, y_hist: np.ndarray, x_today: np.ndarray) -> float:
    if len(y_hist) < 120 or not np.isfinite(x_today).all():
        return 0.0
    scaler = StandardScaler()
    Xh = scaler.fit_transform(X_hist)
    xt = scaler.transform(x_today.reshape(1, -1))
    model = Ridge(alpha=5.0, fit_intercept=True)
    try:
        model.fit(Xh, y_hist)
        mu = float(model.predict(xt)[0])
    except Exception:
        mu = 0.0
    mu = float(np.clip(mu, -MU_CLIP, MU_CLIP))
    if abs(mu) < MU_MIN:
        mu = 0.0
    return mu


# --------------------- MAIN ---------------------
if __name__ == "__main__":
    df = load_or_simulate_multi_asset()
    prices = df.xs("price", axis=1, level="field")
    rets   = df.xs("ret", axis=1, level="field")

    # Buy & Hold equal weight baseline
    ew = pd.Series(1.0 / prices.shape[1], index=prices.columns)
    bh_ret = (rets * ew).sum(axis=1)

    # Ridge-only weekly MV
    ridge_ret = backtest_stacked(df, use_stacking=False, ridge_only=True)

    # Stacked weekly MV
    stacked_ret = backtest_stacked(df, use_stacking=True, ridge_only=False)

    # Summaries
    met = [
        performance_metrics(bh_ret, "Buy&Hold (EW)"),
        performance_metrics(ridge_ret, "Ridge-only MV"),
        performance_metrics(stacked_ret, "Stacked MV (Project14)")
    ]
    print_metrics_table(met)

    print("\nLast 5 daily returns (strategies):")
    out = pd.DataFrame({
        "ret_bh": bh_ret,
        "ret_ridge": ridge_ret,
        "ret_stacked": stacked_ret
    }).tail()
    print(out)

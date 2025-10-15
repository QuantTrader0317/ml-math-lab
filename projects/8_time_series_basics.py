"""
Project 8 — Time-Series Basics for Returns
Python 3.13-safe. NumPy 2.1+, pandas, statsmodels.

What this does:
- Simulates a price series with two regimes:
  (1) Geometric Brownian Motion (trend-like)
  (2) Mean-reverting (AR(1) on returns)
- Computes arithmetic & log returns (no look-ahead)
- Runs stationarity (ADF) and serial-correlation tests (ACF, Ljung–Box)
- Performs a proper time-based split and fits AR(1) on returns
- Evaluates out-of-sample performance (MSE, directional accuracy)

Why this matters:
- Markets are time-ordered: you must evaluate models in chronological order.
- Prices are usually non-stationary; models should work on returns/spreads/residuals.
- Stationarity & autocorr checks are the “pre-flight” for any trading signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Soft import: if missing, show a helpful message instead of crashing hard.
try:
    from statsmodels.tsa.stattools import adfuller, acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.ar_model import AutoReg
except Exception as e:
    raise SystemExit(
        "This project requires statsmodels:\n"
        "  pip install statsmodels\n"
        f"Import error: {e}"
    )


# ---------- utils ----------
def to_series(arr: np.ndarray, name: str) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(arr), freq="B")  # business days
    return pd.Series(arr, index=idx, name=name)

def returns_from_prices(p: pd.Series) -> pd.DataFrame:
    """Compute arithmetic and log returns from PRICE series (no leakage)."""
    arith = p.pct_change().rename("ret")
    logr = np.log(p / p.shift(1)).rename("logret")
    out = pd.concat([arith, logr], axis=1).dropna()
    return out

def adf_test(x: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test results."""
    res = adfuller(x, autolag="AIC")
    keys = ["test_stat", "pvalue", "lags_used", "n_obs"]
    vals = [res[0], res[1], res[2], res[3]]
    return dict(zip(keys, vals))

def acf_lags(x: pd.Series, nlags: int = 10) -> pd.DataFrame:
    """ACF up to nlags (excluding lag 0 in the display)."""
    acf_vals = acf(x, nlags=nlags, fft=True)
    df = pd.DataFrame({"lag": range(nlags + 1), "acf": acf_vals})
    return df.iloc[1:]  # drop lag 0 (always 1.0)

def ljung_box(x: pd.Series, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box test for no autocorrelation up to 'lags'."""
    lb = acorr_ljungbox(x, lags=lags, return_df=True)
    lb.index.name = "lag"
    return lb

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% times sign of prediction equals sign of actual (ignores zeros)."""
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    mask = s_true != 0
    if not np.any(mask):
        return np.nan
    return float((s_true[mask] == s_pred[mask]).mean())


# ---------- simulate two regimes ----------
def simulate_prices(n: int = 1500, seed: int = 0) -> pd.Series:
    """
    First half: GBM-like (price non-stationary; returns ~ i.i.d.)
    Second half: introduce weak mean reversion in returns via AR(1)
    """
    rng = np.random.default_rng(seed)

    # GBM params
    mu = 0.08 / 252      # daily drift
    sigma = 0.2 / np.sqrt(252)

    # Part 1: GBM log-price increments
    n1 = n // 2
    eps1 = rng.normal(0, 1, size=n1)
    r1 = mu + sigma * eps1              # daily log-returns
    logp1 = np.cumsum(r1) + np.log(100) # start at 100
    p1 = np.exp(logp1)

    # Part 2: add weak AR(1) structure to returns (mean reversion)
    n2 = n - n1
    eps2 = rng.normal(0, 1, size=n2)
    phi = -0.25                         # AR(1) coefficient on returns (negative => mean reversion)
    r2 = np.zeros(n2)
    for t in range(1, n2):
        r2[t] = phi * r2[t-1] + sigma * eps2[t]  # AR(1) on returns

    # stitch: continue price from p1 end
    logp2 = np.log(p1[-1]) + np.cumsum(r2)
    p2 = np.exp(logp2)

    prices = np.r_[p1, p2]
    return to_series(prices, name="price")


# ---------- AR(1) fit with time split ----------
def train_test_split_time(df: pd.DataFrame, split_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * split_frac)
    return df.iloc[:cut], df.iloc[cut:]

def fit_ar1_on_returns(ret: pd.Series, train_frac: float = 0.7) -> dict:
    train, test = train_test_split_time(ret.to_frame("ret"), split_frac=train_frac)
    # Fit AR(1): ret_t = c + phi * ret_{t-1} + e_t
    model = AutoReg(train["ret"], lags=1, old_names=False)
    res = model.fit()
    # Forecast 1-step ahead over test set (rolling AR with fixed params)
    # Align: we forecast at t using ret_{t-1}, so we shift by 1.
    test_lag = test["ret"].shift(1)
    # when lag is nan (first row), drop
    mask = ~test_lag.isna()
    y_true = test["ret"][mask].values
    y_pred = (res.params["const"] + res.params["ret.L1"] * test_lag[mask]).values

    mse = float(np.mean((y_true - y_pred) ** 2))
    dir_acc = directional_accuracy(y_true, y_pred)
    return {
        "params": res.params.to_dict(),
        "train_n": int(len(train)),
        "test_n": int(mask.sum()),
        "mse_test": mse,
        "diracc_test": dir_acc,
    }


# ---------- main ----------
if __name__ == "__main__":
    # Simulate price series
    price = simulate_prices(n=1500, seed=2)

    # 1) Returns (no leakage: shift-1)
    rets = returns_from_prices(price)
    ret = rets["ret"]     # arithmetic
    logret = rets["logret"]

    print("\n=== Basic shapes ===")
    print("n_prices:", len(price), "  n_returns:", len(ret))

    # 2) Stationarity tests (ADF)
    adf_price = adf_test(price.dropna())
    adf_ret = adf_test(ret.dropna())
    adf_logret = adf_test(logret.dropna())

    print("\n=== Stationarity (ADF test) ===")
    print("Price   : test_stat={:.3f}, pvalue={:.4f}  -> likely NON-stationary if p>0.05".format(
        adf_price["test_stat"], adf_price["pvalue"]))
    print("Returns : test_stat={:.3f}, pvalue={:.4f}  -> likely STATIONARY if p<0.05".format(
        adf_ret["test_stat"], adf_ret["pvalue"]))
    print("LogRet  : test_stat={:.3f}, pvalue={:.4f}  -> similar to returns".format(
        adf_logret["test_stat"], adf_logret["pvalue"]))

    # 3) Serial correlation: ACF and Ljung–Box on returns
    acf_df = acf_lags(ret, nlags=10)
    lb_df = ljung_box(ret, lags=10)

    print("\n=== Autocorrelation (ACF) on returns ===")
    print(acf_df.to_string(index=False, header=True, max_rows=20))

    print("\n=== Ljung–Box (H0: no autocorr up to lag k) ===")
    # show last row = joint test up to lag 10
    last_row = lb_df.iloc[-1]
    print(f"up to lag {int(last_row.name)}: Q={last_row['lb_stat']:.3f}, pvalue={last_row['lb_pvalue']:.4f}")

    # 4) Time split + AR(1) on returns
    ar1_res = fit_ar1_on_returns(ret, train_frac=0.7)
    print("\n=== AR(1) on returns (time-split) ===")
    print("params:", {k: round(float(v), 6) for k, v in ar1_res["params"].items()})
    print(f"train_n={ar1_res['train_n']}, test_n={ar1_res['test_n']}")
    print(f"test MSE={ar1_res['mse_test']:.8f}, test directional accuracy={ar1_res['diracc_test']:.3f}")

    # 5) Quick “what to look for” summary
    print("\nSummary:")
    print("- Prices usually fail stationarity (high p-value); returns typically pass (low p-value).")
    print("- Any significant ACF/Ljung–Box suggests predictable structure (mean reversion or momentum).")
    print("- AR(1) params: negative phi ~ mean reversion, positive phi ~ momentum.")
    print("- Always split by time; never leak the future into the past.")

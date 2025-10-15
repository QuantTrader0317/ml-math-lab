"""
Project 9 — Feature Engineering for Returns
Python 3.13 safe | NumPy 2.1+, pandas 2.2+, statsmodels 0.14+
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

# ---------- generate synthetic price + "market" ----------
np.random.seed(42)
n = 1500
dates = pd.date_range("2020-01-01", periods=n, freq="D")

# market drift + random noise
mkt_rets = np.random.normal(0.0003, 0.01, size=n)
asset_rets = 0.3 * mkt_rets + np.random.normal(0, 0.012, size=n)

prices = 100 * (1 + asset_rets).cumprod()
mkt_prices = 100 * (1 + mkt_rets).cumprod()

df = pd.DataFrame({
    "date": dates,
    "price": prices,
    "mkt_price": mkt_prices
}).set_index("date")

# compute daily returns
df["ret"] = df["price"].pct_change()
df["mkt_ret"] = df["mkt_price"].pct_change()

# after computing df["ret"] and df["mkt_ret"]
df = df.dropna(subset=["ret", "mkt_ret"]).copy()

# ---------- features ----------
window = 20

df["momentum"] = df["ret"].rolling(window).mean()
df["volatility"] = df["ret"].rolling(window).std()
df["zscore"] = (df["ret"] - df["ret"].rolling(window).mean()) / df["ret"].rolling(window).std()

# rolling beta vs market
def rolling_beta(y, x, w):
    betas = []
    for i in range(len(y)):
        if i < w:
            betas.append(np.nan)
            continue

        Y = y[i-w:i]
        X = x[i-w:i]

        # skip bad windows
        if (
            np.any(~np.isfinite(Y)) or
            np.any(~np.isfinite(X)) or
            np.std(X) < 1e-12
        ):
            betas.append(np.nan)
            continue

        model = OLS(Y, add_constant(X)).fit()
        betas.append(model.params[1])
    return np.array(betas)


df["beta_mkt"] = rolling_beta(df["ret"].values, df["mkt_ret"].values, window)

# lagged returns (1d)
df["ret_lag1"] = df["ret"].shift(1)

# drop NaN rows from rolling windows
df = df.dropna()

# ---------- correlation matrix ----------
corr = df[["ret", "momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]].corr()

# ---------- simple predictive check ----------
# next-day return
df["ret_fwd1"] = df["ret"].shift(-1)
corr_next = df[["ret_fwd1", "momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]].corr().iloc[0, 1:]

print("=== Feature Correlations with Next-Day Return ===")
print(corr_next.to_string(float_format=lambda x: f"{x: .3f}"))

print("\n=== Rolling-Feature Sample (last 5 days) ===")
print(df[["ret", "momentum", "volatility", "zscore", "beta_mkt"]].tail())

print("\n=== In-Sample Correlation Matrix ===")
print(corr.round(3))

# optional save
df.to_csv("features_output.csv", index=True)
print("\nSaved features_output.csv")


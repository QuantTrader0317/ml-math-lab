"""
Project 13 — Risk & Monitoring Dashboard (console-first)
- Reuses a simple multi-asset sim + weekly MV weights (like Project 12 upgrades).
- Builds daily risk/ops dashboard: turnover, realized vol, drawdown, exposures, PnL attribution.
- Saves CSVs: equity.csv, risk_timeseries.csv, pnl_by_asset.csv, exposures.csv

Python 3.13 | pandas 2.2+ | numpy 2.1+ | scikit-learn 1.5+
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------- CONFIG ----------
ANNUALIZATION_DAYS = 252
SEED = 42
N_ASSETS = 4

MIN_TRAIN = 252
FEAT_WIN = 20
RIDGE_ALPHA = 5.0
MU_CLIP = 0.003

EWMA_LAMBDA = 0.94
COV_RIDGE = 5e-6
TARGET_VOL = 0.12          # use more of the book
LEVERAGE_CAP = 1.5
ASSET_CAP = 0.30
TURNOVER_BLEND = 0.60

SLIPPAGE_BPS = 1.0
FEE_PER_TRADE = 0.00
STARTING_CASH = 500.0

ROUND_SHARES = False       # <- FRACTIONAL SHARES ON

FEATURES = ["momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]

# Optional: cheap price simulation
PRICE_START = 15.0        # set to 15.0 later for low-priced assets

# ---------- utils ----------
def ann_factor() -> float:
    return np.sqrt(ANNUALIZATION_DAYS)

def performance_metrics(ret: pd.Series) -> Dict[str, float]:
    ret = ret.dropna()
    if len(ret) == 0:
        return {"N": 0}
    cum = (1 + ret).cumprod()
    total = float(cum.iloc[-1] - 1.0)
    years = max(len(ret) / ANNUALIZATION_DAYS, 1e-12)
    cagr = (1 + total) ** (1 / years) - 1 if total > -0.999999 else -1.0
    vol_ann = ret.std() * ann_factor()
    sharpe = np.nan if vol_ann == 0 else ret.mean() / ret.std() * ann_factor()
    dd = (cum / cum.cummax()) - 1
    maxdd = float(dd.min())
    hit = float((ret > 0).mean())
    return {"N": len(ret), "CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd, "HitRate": hit, "TotalRet": total}

# ---------- data ----------
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

# ---------- alpha + risk ----------
def ridge_alpha(Xtr: np.ndarray, ytr: np.ndarray, xpred: np.ndarray) -> float:
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    xpred = sc.transform(xpred.reshape(1, -1))
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xtr, ytr)
    return float(model.predict(xpred)[0])

def ewma_cov(returns: pd.DataFrame, lam: float, ridge: float) -> np.ndarray:
    R = returns.values
    R = R[~np.isnan(R).any(axis=1)]
    if R.size == 0:
        k = returns.shape[1]
        return np.eye(k) * 1e-4
    k = R.shape[1]
    Sigma = np.zeros((k, k))
    for r in R:
        r = r.reshape(-1, 1)
        Sigma = lam * Sigma + (1 - lam) * (r @ r.T)
    return Sigma + ridge * np.eye(k)

def mv_weights(mu: np.ndarray, Sigma: np.ndarray, gross_cap: float, asset_cap: float) -> np.ndarray:
    inv = np.linalg.pinv(Sigma, rcond=1e-10)
    w = inv @ mu
    w = np.clip(w, -asset_cap, asset_cap)
    gross = np.sum(np.abs(w))
    if gross > gross_cap and gross > 0:
        w *= (gross_cap / gross_cap)  # (minor cap; we'll re-cap after blend anyway)
    return w

def beta_neutralize(w: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if not np.isfinite(beta).any():
        return w
    num = float(beta @ w)
    den = float(beta @ beta)
    if den > 1e-12:
        w = w - (num / den) * beta
    return w

def compute_weekly_weights(df: pd.DataFrame) -> pd.DataFrame:
    assets = list(df.columns.get_level_values("asset").unique())
    dates = df.index
    ret_panel = df.xs("ret", axis=1, level="field")
    beta_panel = df.xs("beta_mkt", axis=1, level="field", drop_level=False).droplevel("field", axis=1)
    feat_panel = {a: df.xs(a, axis=1, level="asset").reindex(columns=FEATURES).values for a in assets}
    for a in assets:
        X = feat_panel[a]
        X[~np.isfinite(X)] = 0.0

    w = pd.DataFrame(0.0, index=dates, columns=assets)
    w_prev = np.zeros(len(assets), float)

    for t in range(MIN_TRAIN, len(dates) - 1):
        d = dates[t]
        if d.weekday() != 4:  # Friday only
            w.iloc[t, :] = w_prev
            continue

        mu = []
        for i, a in enumerate(assets):
            X = feat_panel[a][:t, :]
            y = ret_panel[a].shift(-1).values[:t]
            xpred = feat_panel[a][t, :]
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            Xc, yc = X[mask], y[mask]
            if len(yc) < 60 or not np.isfinite(xpred).all():
                mu.append(0.0); continue
            try:
                m = ridge_alpha(Xc, yc, xpred)
            except Exception:
                m = 0.0
            mu.append(float(np.clip(m, -MU_CLIP, MU_CLIP)))
        mu = np.array(mu)

        start = max(0, t - 252)
        Sigma = ewma_cov(ret_panel.iloc[start:t, :], lam=EWMA_LAMBDA, ridge=COV_RIDGE)

        w_raw = mv_weights(mu, Sigma, LEVERAGE_CAP, ASSET_CAP)
        w_raw = (1 - TURNOVER_BLEND) * w_raw + TURNOVER_BLEND * w_prev
        w_raw = np.clip(w_raw, -ASSET_CAP, ASSET_CAP)

        betas_now = beta_panel.iloc[t, :].reindex(index=assets).values
        betas_now[~np.isfinite(betas_now)] = 0.0
        w_raw = beta_neutralize(w_raw, betas_now)
        w_raw = np.clip(w_raw, -ASSET_CAP, ASSET_CAP)

        port_var = float(w_raw @ (Sigma @ w_raw))
        cur_vol = np.sqrt(max(port_var, 0.0)) * ann_factor() if port_var > 0 else np.nan
        scale = TARGET_VOL / cur_vol if (np.isfinite(cur_vol) and cur_vol > 1e-8) else 1.0
        w_t = np.clip(w_raw * scale, -ASSET_CAP, ASSET_CAP)

        w.iloc[t, :] = w_t
        w_prev = w_t.copy()

    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w

# ---------- broker ----------
@dataclass
class Position:
    shares: float = 0.0

class PaperBroker:
    def __init__(self, slippage_bps=SLIPPAGE_BPS, fee_per_trade=FEE_PER_TRADE, starting_cash=STARTING_CASH):
        self.slippage_bps = slippage_bps
        self.fee_per_trade = fee_per_trade
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[Tuple[pd.Timestamp, str, float, float, float]] = []

    def equity(self, prices: Dict[str, float]) -> float:
        v = self.cash
        for a, p in self.positions.items():
            px = prices.get(a, np.nan)
            if np.isfinite(px):
                v += p.shares * px
        return v

    def place_order(self, date: pd.Timestamp, asset: str, qty: float, price: float):
        if qty == 0 or not np.isfinite(price):
            return
        slip = (self.slippage_bps / 1e4) * np.sign(qty)
        fill = price * (1 + slip)
        cost = fill * qty
        self.cash -= (cost + self.fee_per_trade)
        pos = self.positions.setdefault(asset, Position(0.0))
        pos.shares += qty
        self.trade_log.append((date, asset, float(qty), float(fill), float(self.fee_per_trade)))

    def rebalance(self, date: pd.Timestamp, target_w: Dict[str, float], prices: Dict[str, float]):
        for a in target_w.keys():
            self.positions.setdefault(a, Position(0.0))
        eq = self.equity(prices)
        desired = {a: target_w[a] * eq for a in target_w}
        current = {a: self.positions[a].shares * prices[a] for a in target_w}
        dollars = {a: desired[a] - current[a] for a in target_w}
        shares = {a: (dollars[a] / prices[a]) if prices[a] != 0 else 0.0 for a in target_w}
        if ROUND_SHARES:
            shares = {a: float(np.sign(q) * np.floor(abs(q))) for a, q in shares.items()}

        for a, q in shares.items():
            self.place_order(date, a, q, prices[a])

# ---------- run & dashboard ----------
if __name__ == "__main__":
    pd.options.display.float_format = lambda x: f"{x:,.6f}"

    df = load_or_simulate_multi_asset()
    assets = list(df.columns.get_level_values("asset").unique())
    prices = df.xs("price", axis=1, level="field")
    rets   = df.xs("ret", axis=1, level="field")

    weights = compute_weekly_weights(df)

    broker = PaperBroker()
    equity = []
    notional = []
    pnl_by_asset = {a: [] for a in assets}
    last_value = {a: 0.0 for a in assets}

    last_w = None
    for date in df.index:
        px = {a: float(prices.loc[date, a]) for a in assets}
        w_t = {a: float(weights.loc[date, a]) for a in assets}

        # Rebalance only if weights changed (weekly Fridays + ffill)
        if (last_w is None) or any(abs(w_t[a] - last_w[a]) > 1e-9 for a in assets):
            broker.rebalance(date, w_t, px)
        last_w = w_t

        eq = broker.equity(px)
        equity.append((date, eq))

        # Track per-asset notional & PnL contrib
        row_notional = {}
        for a in assets:
            pos = broker.positions.get(a, Position()).shares
            row_notional[a] = pos * px[a]
            # incremental PnL contrib today = pos * price_change
            # use returns for ease:
            r = float(rets.loc[date, a]) if date in rets.index else 0.0
            pnl = pos * px[a] * r
            pnl_by_asset[a].append((date, pnl))
        notional.append((date, row_notional))

    # Build series/tables
    equity = pd.Series({d: v for d, v in equity}).sort_index()
    ret = equity.pct_change().fillna(0.0)
    turn = (weights.diff().abs().sum(axis=1)).fillna(0.0)  # daily gross turnover (approx)
    realized_vol = ret.rolling(63).std() * ann_factor()
    cum = (1 + ret).cumprod()
    dd = cum / cum.cummax() - 1.0

    exposures = pd.DataFrame({d: row for d, row in notional}).T
    exposures.index.name = "date"
    exposures = exposures.fillna(0.0)

    pnl_asset_df = pd.DataFrame({a: pd.Series({d: p for d, p in pnl_by_asset[a]}) for a in assets}).fillna(0.0)
    pnl_asset_df.index.name = "date"

    # Metrics
    met = performance_metrics(ret)

    # Print dashboard
    print("\n=== Risk & Monitoring Dashboard ===")
    print(f"Span: {equity.index.min().date()} → {equity.index.max().date()}  N={len(equity)}")
    print(f"Equity: start=${equity.iloc[0]:,.2f}  end=${equity.iloc[-1]:,.2f}")
    print(f"CAGR={met['CAGR']:.2%}  Sharpe={met['Sharpe']:.2f}  MaxDD={met['MaxDD']:.2%}  HitRate={met['HitRate']:.2%}  TotalRet={met['TotalRet']:.2%}")
    print(f"Realized vol (63d) last={realized_vol.iloc[-1]:.2%}")
    print(f"Avg daily turnover={turn.mean():.3f}")

    print("\nLast 5 turnover:")
    print(turn.tail())

    print("\nLast 5 realized vol (ann.):")
    print(realized_vol.tail())

    print("\nLast 5 drawdown:")
    print(dd.tail())

    print("\nPnL attribution (last 5 rows):")
    print(pnl_asset_df.tail())

    print("\nExposures (last 5 rows):")
    print(exposures.tail())

    # Save CSVs
    equity.to_frame("equity").to_csv("equity.csv")
    pd.DataFrame({
        "return": ret,
        "realized_vol_63d_ann": realized_vol,
        "turnover": turn,
        "drawdown": dd
    }).to_csv("risk_timeseries.csv")
    pnl_asset_df.to_csv("pnl_by_asset.csv")
    exposures.to_csv("exposures.csv")
    print("\nSaved: equity.csv, risk_timeseries.csv, pnl_by_asset.csv, exposures.csv")

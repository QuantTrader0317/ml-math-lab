"""
Project 12 — Paper Trading Engine (Improved: weekly, clipped alpha, turnover penalty, EWMA Σ, beta-neutral)
Python 3.13 | pandas 2.2+ | numpy 2.1+ | scikit-learn 1.5+

Upgrades:
- Weekly rebalancing (Fridays) to cut turnover/costs
- Trading cost = 1 bps
- Turnover penalty: blend toward prior weights
- Clip predicted returns (mu_hat) to reduce noise
- Per-asset cap ±30%
- EWMA covariance (RiskMetrics λ=0.94) + ridge add-on
- Vol targeting to 8%
- Soft beta-neutralization (project weights to be market-neutral)

Start capital is configurable. For now we keep it at 100_000 so stats are stable;
you can change STARTING_CASH to 500 when you want to simulate small-account behavior.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --------------------- CONFIG ---------------------
ANNUALIZATION_DAYS = 252
SEED = 42
N_ASSETS = 4

MIN_TRAIN = 252
FEAT_WIN = 20
RIDGE_ALPHA = 5.0
MU_CLIP = 0.003

# Risk model
EWMA_LAMBDA = 0.94
COV_RIDGE = 5e-6
TARGET_VOL = 0.12          # <- a little higher so a $500 acct actually uses risk
LEVERAGE_CAP = 1.5
ASSET_CAP = 0.30

# Turnover penalty
TURNOVER_BLEND = 0.60

# Costs (fractional-friendly)
SLIPPAGE_BPS = 1.0         # lower assumed slippage
FEE_PER_TRADE = 0.00       # many brokers = $0 tickets; set yours realistically
COST_BPS = 1.0

# Account
STARTING_CASH = 500.0      # <- small account, fractional shares mode (“no rounding”)

# Optional: cheap price simulation (Option B – see below)
PRICE_START = 15.0        # leave at 100.0 for now; set 15.0 for cheap-price sim

FEATURES = ["momentum", "volatility", "zscore", "beta_mkt", "ret_lag1"]

# --------------------- UTILS ---------------------
def ann_factor() -> float:
    return np.sqrt(ANNUALIZATION_DAYS)

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
    print("\n=== Paper Trader Performance ===")
    print(f"{cols[0]:<34} {cols[1]:>6} {cols[2]:>8} {cols[3]:>8} {cols[4]:>8} {cols[5]:>8} {cols[6]:>10}")
    for m in metrics_list:
        print(f"{m['label']:<34} {m.get('N',0):>6d} "
              f"{m.get('CAGR',np.nan):>8.2%} {m.get('Sharpe',np.nan):>8.2f} "
              f"{m.get('MaxDD',np.nan):>8.2%} {m.get('HitRate',np.nan):>8.2%} "
              f"{m.get('TotalRet',np.nan):>10.2%}")

# --------------------- DATA (2-level MultiIndex: (field, asset)) ---------------------
def load_or_simulate_multi_asset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Market factor
    mkt_ret = rng.normal(0.0002, 0.010, size=n)
    mkt_price = 100 * (1 + mkt_ret).cumprod()

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

    # Features per asset
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

# --------------------- ALPHA + RISK ---------------------
def ridge_alpha_per_asset(X_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xp = scaler.transform(x_pred.reshape(1, -1))
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xtr, y_train)
    return float(model.predict(Xp)[0])

def ewma_cov(returns: pd.DataFrame, lam: float, ridge: float) -> np.ndarray:
    """
    RiskMetrics-style EWMA covariance:
        Σ_t = λ Σ_{t-1} + (1-λ) r_t r_t^T
    Computed over the window by iterating rows.
    """
    R = returns.values
    R = R[~np.isnan(R).any(axis=1)]
    nobs, k = R.shape if R.ndim == 2 else (0, returns.shape[1])
    if nobs == 0:
        return np.eye(k) * 1e-4
    Sigma = np.zeros((k, k), dtype=float)
    for r in R:
        r = r.reshape(-1, 1)
        Sigma = lam * Sigma + (1 - lam) * (r @ r.T)
    # Ridge add-on
    Sigma = Sigma + ridge * np.eye(k)
    return Sigma

def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, cap_gross: float, asset_cap: float) -> np.ndarray:
    inv = np.linalg.pinv(Sigma, rcond=1e-10)
    w = inv @ mu
    # Per-asset cap
    w = np.clip(w, -asset_cap, asset_cap)
    # Gross cap
    gross = np.sum(np.abs(w))
    if gross > cap_gross and gross > 0:
        w = w * (cap_gross / gross)
    return w

def project_beta_neutral(w: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """
    Soft beta-neutralization: w <- w - ((β^T w)/(β^T β)) β
    No change if β is all zeros.
    """
    if not np.isfinite(betas).any():
        return w
    num = float(betas @ w)
    den = float(betas @ betas)
    if den > 1e-12:
        w = w - (num / den) * betas
    return w

def compute_target_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly target weights (Fridays only).
    - Predict μ̂ with Ridge per asset (clipped).
    - Σ via EWMA covariance (λ=0.94) + ridge
    - Mean-variance weights with caps
    - Turnover blend toward previous weights
    - Beta-neutralization vs market beta
    - Vol targeting to TARGET_VOL
    """
    assets = list(df.columns.get_level_values("asset").unique())
    dates = df.index

    ret_panel = df.xs("ret", axis=1, level="field")
    beta_panel = df.xs("beta_mkt", axis=1, level="field", drop_level=False).droplevel("field", axis=1)

    # Prebuild feature arrays
    feat_panel = {}
    for a in assets:
        feat_a = df.xs(a, axis=1, level="asset")
        X = feat_a.reindex(columns=FEATURES).values
        X[~np.isfinite(X)] = 0.0
        feat_panel[a] = X

    weights = pd.DataFrame(0.0, index=dates, columns=assets)
    w_prev = np.zeros(len(assets), dtype=float)

    for t in range(MIN_TRAIN, len(dates) - 1):
        date = dates[t]
        # Rebalance Fridays only
        if date.weekday() != 4:  # 4 = Friday
            weights.iloc[t, :] = w_prev
            continue

        # ----- μ̂ per asset (clipped) -----
        mu_hat = []
        for i, a in enumerate(assets):
            X = feat_panel[a][:t, :]
            y = ret_panel[a].shift(-1).values[:t]
            x_pred = feat_panel[a][t, :]
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            Xc = X[mask]
            yc = y[mask]
            if len(yc) < 60 or not np.isfinite(x_pred).all():
                mu_hat.append(0.0)
                continue
            try:
                m = ridge_alpha_per_asset(Xc, yc, x_pred)
            except Exception:
                m = 0.0
            # clip forecasts to reduce noise → fewer spurious trades
            mu_hat.append(float(np.clip(m, -MU_CLIP, MU_CLIP)))
        mu_hat = np.array(mu_hat, dtype=float)

        # ----- Σ_t via EWMA on returns up to t -----
        # Use last 252 days to stabilize covariance
        start = max(0, t - 252)
        Sigma = ewma_cov(ret_panel.iloc[start:t, :], lam=EWMA_LAMBDA, ridge=COV_RIDGE)

        # ----- raw MV weights -----
        w_raw = mean_variance_weights(mu_hat, Sigma, cap_gross=LEVERAGE_CAP, asset_cap=ASSET_CAP)

        # ----- turnover penalty: blend toward previous weights -----
        w_raw = (1 - TURNOVER_BLEND) * w_raw + TURNOVER_BLEND * w_prev
        # re-apply per-asset and gross caps post-blend
        w_raw = np.clip(w_raw, -ASSET_CAP, ASSET_CAP)
        gross = np.sum(np.abs(w_raw))
        if gross > LEVERAGE_CAP and gross > 0:
            w_raw *= (LEVERAGE_CAP / gross)

        # ----- beta-neutralization -----
        betas_now = beta_panel.iloc[t, :].reindex(index=assets).values
        betas_now[~np.isfinite(betas_now)] = 0.0
        w_raw = project_beta_neutral(w_raw, betas_now)
        # re-cap after projection
        w_raw = np.clip(w_raw, -ASSET_CAP, ASSET_CAP)

        # ----- vol targeting -----
        port_var = float(w_raw @ (Sigma @ w_raw))
        cur_vol = np.sqrt(max(port_var, 0.0)) * ann_factor() if port_var > 0 else np.nan
        scale = TARGET_VOL / cur_vol if (np.isfinite(cur_vol) and cur_vol > 1e-8) else 1.0
        w_t = np.clip(w_raw * scale, -ASSET_CAP, ASSET_CAP)

        # store & carry forward
        weights.iloc[t, :] = w_t
        w_prev = w_t.copy()

    # forward fill (carry last decided weights to non-rebalance days & to end)
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)
    return weights

# --------------------- PAPER BROKER ---------------------
@dataclass
class Position:
    shares: float = 0.0

class PaperBroker:
    def __init__(self,
                 slippage_bps: float = SLIPPAGE_BPS,
                 fee_per_trade: float = FEE_PER_TRADE,
                 margin_leverage: float = LEVERAGE_CAP,
                 starting_cash: float = STARTING_CASH):
        self.slippage_bps = slippage_bps
        self.fee_per_trade = fee_per_trade
        self.margin_leverage = margin_leverage
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[Tuple[pd.Timestamp, str, float, float, float]] = []

    def equity(self, prices: Dict[str, float]) -> float:
        value = self.cash
        for a, pos in self.positions.items():
            px = prices.get(a, np.nan)
            if np.isfinite(px):
                value += pos.shares * px
        return value

    def place_order(self, date: pd.Timestamp, asset: str, qty: float, price: float):
        if qty == 0 or not np.isfinite(price):
            return
        slip = (self.slippage_bps / 1e4) * np.sign(qty)
        fill = price * (1 + slip)
        cost = fill * qty
        fee = self.fee_per_trade
        self.cash -= (cost + fee)
        pos = self.positions.setdefault(asset, Position(0.0))
        pos.shares += qty
        self.trade_log.append((date, asset, float(qty), float(fill), float(fee)))

    def rebalance_to_weights(self,
                             date: pd.Timestamp,
                             target_weights: Dict[str, float],
                             prices: Dict[str, float],
                             max_attempts: int = 4):
        # ensure all assets
        for a in target_weights.keys():
            self.positions.setdefault(a, Position(0.0))

        attempt = 0
        while attempt < max_attempts:
            eq = self.equity(prices)
            if eq <= 0:
                return

            desired_notional = {a: float(np.clip(w, -LEVERAGE_CAP, LEVERAGE_CAP)) * eq
                                for a, w in target_weights.items()}
            current_notional = {a: self.positions[a].shares * prices[a] for a in target_weights.keys()}
            order_dollars = {a: desired_notional[a] - current_notional[a] for a in target_weights.keys()}
            order_shares = {a: (order_dollars[a] / prices[a]) if prices[a] != 0 else 0.0
                            for a in target_weights.keys()}

            sim_cash = self.cash
            sim_positions = {a: self.positions[a].shares for a in target_weights.keys()}
            for a, q in order_shares.items():
                if q == 0:
                    continue
                slip = (self.slippage_bps / 1e4) * np.sign(q)
                fill = prices[a] * (1 + slip)
                cost = fill * q
                sim_cash -= cost + self.fee_per_trade
                sim_positions[a] = sim_positions.get(a, 0.0) + q

            sim_gross = sum(abs(sim_positions[a] * prices[a]) for a in target_weights.keys())
            sim_equity = sim_cash + sum(sim_positions[a] * prices[a] for a in target_weights.keys())
            if sim_equity <= 0:
                scale = 0.5
            else:
                sim_leverage = sim_gross / sim_equity if sim_equity != 0 else np.inf
                if sim_cash >= 0 and sim_leverage <= LEVERAGE_CAP:
                    for a, q in order_shares.items():
                        self.place_order(date, a, q, prices[a])
                    return
                scale = 0.5

            target_weights = {a: w * scale for a, w in target_weights.items()}
            attempt += 1
        # give up today if still infeasible

# --------------------- RUN ---------------------
if __name__ == "__main__":
    pd.options.display.float_format = lambda x: f"{x:,.6f}"

    # 1) Data
    df = load_or_simulate_multi_asset()  # columns: (field, asset)
    assets = list(df.columns.get_level_values("asset").unique())
    prices = df.xs("price", axis=1, level="field")
    rets   = df.xs("ret", axis=1, level="field")

    # 2) Target weights (weekly, improved)
    weights = compute_target_weights(df)

    # 3) Paper broker
    broker = PaperBroker(slippage_bps=SLIPPAGE_BPS,
                         fee_per_trade=FEE_PER_TRADE,
                         margin_leverage=LEVERAGE_CAP,
                         starting_cash=STARTING_CASH)

    # Daily loop: rebalance only on days where weights changed (Fridays or ffill days)
    equity_curve = []
    last_w = None
    for date in df.index:
        px = {a: float(prices.loc[date, a]) for a in assets}
        w_t = {a: float(weights.loc[date, a]) for a in assets}
        if (last_w is None) or any(abs(w_t[a] - last_w[a]) > 1e-9 for a in assets):
            broker.rebalance_to_weights(date, w_t, px)
        equity_curve.append((date, broker.equity(px)))
        last_w = w_t

    equity = pd.Series({d: v for d, v in equity_curve}).sort_index()
    ret = equity.pct_change().fillna(0.0)

    # 4) Metrics & samples
    met = performance_metrics(ret, "PaperTrader (weekly, clipped μ̂, EWMA Σ, β-neutral)")
    print_metrics_table([met])

    trades_df = pd.DataFrame(broker.trade_log, columns=["date", "asset", "qty", "fill_px", "fee"])
    if not trades_df.empty:
        trades_df = trades_df.set_index("date").sort_index()
        print("\nLast 5 trades:")
        print(trades_df.tail())

    print("\nLast positions (shares):")
    print({a: broker.positions.get(a, Position()).shares for a in assets})

    print("\nLast 5 equity points:")
    print(equity.tail().to_frame("equity"))

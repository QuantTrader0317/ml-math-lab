# projects/15_alpaca_live_stack.py
# Real-time (daily/weekly) execution with stacked alpha (Project 14 style) -> MV portfolio -> Alpaca paper trades.

from __future__ import annotations
import os, datetime as dt
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.trading.enums import QueryOrderStatus

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

# ===================== CONFIG =====================
UNIVERSE = ["SPY", "QQQ", "IWM", "TLT"]
BENCH = "SPY"

# Rebalance cadence
REBALANCE_MODE = "daily"          # "daily" or "weekly"
REBALANCE_WEEKDAY = 4             # 0=Mon..4=Fri (used if weekly)
FORCE_REBALANCE_TODAY = False     # True = force trade this run

# Budget / position sizing
BUDGET = 500.0                    # slice of equity to deploy
POSITION_CAP = 0.40               # per-asset weight cap
ALLOW_SHORTS = False              # keep long-only

# Min trade threshold (skip tiny)
NOTIONAL_THRESHOLD = 0.02         # as % of BUDGET (2%)

# Execution mode
DRY_RUN = False                   # True = print only; False = submit to Alpaca
PAPER = True                      # use paper account

# Features windows
MOM_WIN = 21
VOL_WIN = 21
ZSCORE_WIN = 63
BETA_WIN = 63
RET_LAG = 1

# Stacker / CV
N_SPLITS = 3
ALPHA_RIDGE = 5.0                 # meta ridge
ALPHA_MV = 5.0                    # covariance ridge

# --- Slippage / limit-order guard ---
USE_LIMIT_ORDERS = False
LIMIT_SLIPPAGE = 0.002            # 0.20% around last close

# --- Risk gates ---
TURNOVER_CAP = 0.30               # skip if sum|delta|/BUDGET > 30%
VOL_LOOKBACK = 21                 # ~1m
VOL_CAP_ANNUAL = 0.40             # skip if SPY vol > 40%

# ===================== ENV & CLIENTS =====================
def load_env():
    load_dotenv()
    for k in ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL"]:
        if not os.environ.get(k):
            raise RuntimeError(f"Missing env var: {k}")

def trading_client() -> TradingClient:
    return TradingClient(
        os.environ["APCA_API_KEY_ID"],
        os.environ["APCA_API_SECRET_KEY"],
        paper=PAPER
    )

def data_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        os.environ["APCA_API_KEY_ID"],
        os.environ["APCA_API_SECRET_KEY"],
    )

# ===================== TIME & REBALANCE =====================
def now_et() -> dt.datetime:
    return dt.datetime.now().astimezone()

def is_rebalance_today(today: dt.date) -> bool:
    if FORCE_REBALANCE_TODAY:
        return True
    if REBALANCE_MODE == "daily":
        return True
    if REBALANCE_MODE == "weekly":
        return today.weekday() == REBALANCE_WEEKDAY
    return False

# ===================== DATA =====================
def fetch_daily_closes(symbols: List[str], n_days: int = 800) -> pd.DataFrame:
    end = now_et()
    start = end - dt.timedelta(days=int(n_days * 1.7))

    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,           # free plan
        limit=10000,
    )
    dc = data_client()
    bars = dc.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError("No bars returned (check symbols/feed/window).")

    closes = bars["close"].unstack(0).sort_index().dropna(how="any")
    return closes

# ===================== FEATURES =====================
def rolling_beta(ret: pd.Series, mkt: pd.Series, win: int) -> pd.Series:
    cov = ret.rolling(win).cov(mkt)
    var = mkt.rolling(win).var()
    return cov / (var.replace(0, np.nan))

def build_features(prices: pd.DataFrame, bench: str) -> Dict[str, pd.DataFrame]:
    rets = prices.pct_change()
    feats_by_asset = {}
    mkt = rets[bench]

    for s in prices.columns:
        rs = rets[s]
        mom = prices[s].pct_change(MOM_WIN)
        vol = rs.rolling(VOL_WIN).std()
        mean = prices[s].rolling(ZSCORE_WIN).mean()
        std = prices[s].rolling(ZSCORE_WIN).std()
        zsc = (prices[s] - mean) / std.replace(0, np.nan)
        beta_s = rolling_beta(rs, mkt, BETA_WIN)
        lag1 = rs.shift(RET_LAG)

        X = pd.DataFrame({
            "momentum": mom,
            "volatility": vol,
            "zscore": zsc,
            "beta_mkt": beta_s,
            "ret_lag1": lag1,
        }, index=prices.index).dropna()

        y = rs.reindex_like(X).shift(-1)
        XY = X.copy()
        XY["target"] = y
        feats_by_asset[s] = XY.dropna()
    return feats_by_asset

# ===================== STACKED ALPHA =====================
def oof_preds_time_series(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    preds = np.zeros((len(y), 2))
    y_oof = np.full_like(y, np.nan, dtype=float)

    for tr, va in tscv.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        r = Ridge(alpha=1.0).fit(Xtr_s, ytr)
        l = Lasso(alpha=0.001, max_iter=5000).fit(Xtr_s, ytr)

        preds[va, 0] = r.predict(Xva_s)
        preds[va, 1] = l.predict(Xva_s)
        y_oof[va] = yva

    return preds, y_oof

def stacked_mu_today(X: np.ndarray, y: np.ndarray, x_today: np.ndarray) -> float:
    preds_oof, y_oof = oof_preds_time_series(X, y)
    ok = ~np.isnan(y_oof)
    Z, yt = preds_oof[ok], y_oof[ok]

    meta = Ridge(alpha=ALPHA_RIDGE).fit(Z, yt)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    xt = scaler.transform(x_today.reshape(1, -1))

    r = Ridge(alpha=1.0).fit(Xs, y)
    l = Lasso(alpha=0.001, max_iter=5000).fit(Xs, y)
    z_today = np.column_stack([r.predict(xt), l.predict(xt)])

    return float(meta.predict(z_today)[0])

def forecast_mu_vector(feats_by_asset: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    mu = {}
    for s, df in feats_by_asset.items():
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        x_today = df.drop(columns=["target"]).values[-1]
        if len(np.unique(y)) < 3 or len(y) < 60:
            scaler = StandardScaler().fit(X)
            r = Ridge(alpha=1.0).fit(scaler.transform(X), y)
            mu[s] = float(r.predict(scaler.transform(x_today.reshape(1, -1)))[0])
        else:
            mu[s] = stacked_mu_today(X, y, x_today)
    return mu

# ===================== RISK MODEL & WEIGHTS =====================
def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    cols = returns.columns
    m = len(cols)
    S = np.zeros((m, m))
    for i in range(len(returns)):
        r = returns.iloc[i].values.reshape(-1, 1)
        S = lam * S + (1 - lam) * (r @ r.T)
    S = S + ALPHA_MV * np.eye(m)
    return S

def mv_weights(mu_vec: np.ndarray, cov: np.ndarray, long_only=True) -> np.ndarray:
    w = np.linalg.solve(cov, mu_vec)
    if long_only:
        w = np.clip(w, 0, None)
    if w.sum() == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    w = np.clip(w, -POSITION_CAP, POSITION_CAP)
    s = w.sum()
    if abs(s) > 1e-12:
        w = w / s
    return w

def compute_target_weights(prices: pd.DataFrame, bench: str) -> Dict[str, float]:
    feats = build_features(prices, bench)
    mu_hat = forecast_mu_vector(feats)
    syms = list(prices.columns)
    mu_vec = np.array([mu_hat[s] for s in syms])

    rets = prices.pct_change().dropna()
    recent = rets.tail(252) if len(rets) > 252 else rets
    cov = ewma_cov(recent[syms], lam=0.94)

    w = mv_weights(mu_vec, cov, long_only=(not ALLOW_SHORTS))
    return {s: float(w[i]) for i, s in enumerate(syms)}

# ===================== EXECUTION HELPERS =====================
def get_equity_and_positions(tc: TradingClient) -> Tuple[float, Dict[str, float]]:
    acct = tc.get_account()
    equity = float(acct.equity)
    cur_mv = {p.symbol: float(p.market_value) for p in tc.get_all_positions()}
    return equity, cur_mv

def realized_vol_annualized(prices: pd.Series, lookback=21) -> float:
    rets = prices.pct_change().dropna()
    if len(rets) < lookback:
        return 0.0
    return float(rets.tail(lookback).std() * np.sqrt(252))

def compute_turnover(targets: dict[str, float], current: dict[str, float], budget: float) -> float:
    deltas = []
    for s in set(list(targets.keys()) + list(current.keys())):
        t = targets.get(s, 0.0)
        c = current.get(s, 0.0)
        deltas.append(abs(t - c))
    return float(sum(deltas)) / float(budget) if budget > 0 else 0.0

def log_fills_for_today(tc: TradingClient):
    LOG_DIR = Path("live_logs"); LOG_DIR.mkdir(exist_ok=True)
    today = dt.date.today()
    start_dt = dt.datetime.combine(today, dt.time(0, 0, 0)).astimezone()
    end_dt   = dt.datetime.combine(today, dt.time(23, 59, 59)).astimezone()

    req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=start_dt, until=end_dt, nested=True)
    orders = tc.get_orders(filter=req)

    rows = []
    for o in orders:
        rows.append({
            "id": o.id,
            "symbol": o.symbol,
            "side": str(o.side).split(".")[-1],
            "notional": float(o.notional) if o.notional else None,
            "qty": float(o.qty) if o.qty else None,
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "status": str(o.status).split(".")[-1],
            "submitted_at": str(o.submitted_at),
            "filled_at": str(o.filled_at),
            "updated_at": str(o.updated_at),
        })
    pd.DataFrame(rows).to_csv(LOG_DIR / f"fills_{today.isoformat()}.csv", index=False)

    pos = tc.get_all_positions()
    pos_df = pd.DataFrame([{
        "date": today.isoformat(),
        "symbol": p.symbol,
        "qty": float(p.qty),
        "market_price": float(p.asset_current_price),
        "market_value": float(p.market_value)
    } for p in pos])
    pos_df.to_csv(LOG_DIR / f"positions_{today.isoformat()}.csv", index=False)

    acct = tc.get_account()
    eq_row = {"date": today.isoformat(), "equity": float(acct.equity), "cash": float(acct.cash)}
    eqp = LOG_DIR / "equity_live.csv"
    pd.DataFrame([eq_row]).to_csv(eqp, mode="a", header=not eqp.exists(), index=False)

# ===================== MAIN =====================
if __name__ == "__main__":
    load_env()
    tc = trading_client()

    today = now_et().date()
    if not is_rebalance_today(today):
        print(f"{today} not a rebalance day ({REBALANCE_MODE}). Set FORCE_REBALANCE_TODAY=True to test.")
        raise SystemExit(0)

    # 1) Data
    closes = fetch_daily_closes(UNIVERSE, n_days=800)
    closes = closes[UNIVERSE]              # ensure col order
    prices = closes.copy()

    # 2) Alpha -> weights (stacked)
    weights = compute_target_weights(prices, BENCH)

    # 3) Budget slice & notionals
    acct_equity, current_notional = get_equity_and_positions(tc)
    budget = float(BUDGET)
    target_notional = {s: weights.get(s, 0.0) * budget for s in UNIVERSE}

    print(f"Account equity: ${acct_equity:,.2f} | Strategy budget: ${budget:,.2f} | Mode={REBALANCE_MODE}")
    print("Weights:", {k: round(v, 4) for k, v in weights.items()})
    print("Target notionals:", {k: round(v, 2) for k, v in target_notional.items()})
    print("Current market values:", {k: round(v, 2) for k, v in current_notional.items()})

    # --- RISK GATES ---
    turnover = compute_turnover(target_notional, current_notional, budget)
    spy_vol = realized_vol_annualized(prices["SPY"], lookback=VOL_LOOKBACK) if "SPY" in prices.columns else 0.0

    # NEW: if we have no positions yet, allow the initial seed without turnover block
    no_positions = (len(current_notional) == 0)

    if not FORCE_REBALANCE_TODAY and not no_positions:
        if turnover > TURNOVER_CAP:
            print(f"Skip: turnover {turnover:.1%} > cap {TURNOVER_CAP:.1%}")
            print("Done.");
            raise SystemExit(0)
        if spy_vol > VOL_CAP_ANNUAL:
            print(f"Skip: vol {spy_vol:.1%} > cap {VOL_CAP_ANNUAL:.1%}")
            print("Done.");
            raise SystemExit(0)

    min_trade = NOTIONAL_THRESHOLD * budget
    print(f"Min trade (${min_trade:.2f}) with DRY_RUN={DRY_RUN}")

    # 4) Submit orders (Limit preferred; Market fallback)
    results = []
    last_close = {s: float(prices[s].iloc[-1]) for s in UNIVERSE}

    for sym in UNIVERSE:
        tgt = target_notional.get(sym, 0.0)
        cur = current_notional.get(sym, 0.0)
        delta = tgt - cur

        # long-only guard
        if not ALLOW_SHORTS and tgt < 0:
            results.append((sym, delta, "SKIP: long-only"))
            continue

        if abs(delta) < min_trade:
            results.append((sym, delta, "SKIP: below threshold"))
            continue

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        notional_abs = round(abs(delta), 2)

        if DRY_RUN:
            results.append((sym, delta, "DRY-RUN"))
            continue

        try:
            if USE_LIMIT_ORDERS:
                # Limit orders require qty (notional is for market orders only).
                px = last_close[sym]
                limit_price = px * (1 + LIMIT_SLIPPAGE) if side == OrderSide.BUY else px * (1 - LIMIT_SLIPPAGE)
                qty = round(notional_abs / limit_price, 6)  # fractional shares allowed

                req = LimitOrderRequest(
                    symbol=sym,
                    side=side,
                    qty=str(qty),
                    limit_price=round(limit_price, 4),
                    time_in_force=TimeInForce.DAY
                )
                o = tc.submit_order(req)
                status = f"OK (LIMIT {limit_price:.4f}, qty={qty}, id={o.id})"
            else:
                req = MarketOrderRequest(
                    symbol=sym,
                    side=side,
                    notional=str(notional_abs),
                    time_in_force=TimeInForce.DAY
                )
                o = tc.submit_order(req)
                status = f"OK (MKT notional={notional_abs:.2f}, id={o.id})"

        except Exception as e:
            status = f"ERROR: {e}"

        results.append((sym, delta, status))

    print("Orders:")
    for sym, delta, status in results:
        print(f"  {sym:4s}  delta=${delta:.2f} -> {status}")

    # 5) Log fills / positions / equity snapshot
    log_fills_for_today(tc)
    print("Done.")

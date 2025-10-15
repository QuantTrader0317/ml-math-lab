"""
Project 7 — Optimizers (SGD, Momentum, Adam) for Logistic Regression
Python 3.13-safe. NumPy 2.1+.

What this does:
- Builds a binary classification dataset (two Gaussian blobs)
- Trains logistic regression with three optimizers: SGD, Momentum, Adam
- Uses mini-batch training, same random init for fair comparisons
- Reports train/test log-loss and accuracy, plus loss progress

Why this matters:
- The loss surface is the same; the optimizer determines how fast and how stably you get to a good solution.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss as sk_log_loss
from sklearn.linear_model import LogisticRegression as SKLogReg


# ---------- math helpers ----------
def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]

def log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def accuracy(y_true: np.ndarray, p: np.ndarray, thresh: float = 0.5) -> float:
    yhat = (p >= thresh).astype(int)
    return float(accuracy_score(y_true, yhat))

def batches(Xb: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = Xb.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield Xb[j], y[j]


# ---------- gradient ----------
def grad_logreg_batch(Xb: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """
    Gradient of average log-loss for logistic regression with optional L2 (no penalty on bias).
    """
    n = Xb.shape[0]
    p = sigmoid(Xb @ w)
    g = (Xb.T @ (p - y)) / n
    if l2 > 0.0:
        reg = np.r_[0.0, w[1:]] * (l2 / n)   # don't penalize bias
        g = g + reg
    return g


# ---------- optimizers ----------
class SGD:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, w: np.ndarray, g: np.ndarray, t: int) -> np.ndarray:
        return w - self.lr * g


class Momentum:
    def __init__(self, lr: float, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.v = None  # velocity

    def step(self, w: np.ndarray, g: np.ndarray, t: int) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.beta * self.v + g
        return w - self.lr * self.v


class Adam:
    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = None
        self.v = None

    def step(self, w: np.ndarray, g: np.ndarray, t: int) -> np.ndarray:
        # t starts at 1
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        m_hat = self.m / (1 - self.b1 ** t)
        v_hat = self.v / (1 - self.b2 ** t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------- training loop ----------
def train_logreg_with_optimizer(
    X: np.ndarray,
    y: np.ndarray,
    optimizer_name: str,
    lr: float = 0.1,
    batch_size: int = 64,
    epochs: int = 50,
    l2: float = 0.0,
    seed: int = 0,
    momentum_beta: float = 0.9,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
) -> dict:
    rng = np.random.default_rng(seed)
    Xb = add_bias(X)
    # same init for all optimizers (we'll pass the same seed at the call site)
    w = rng.normal(scale=0.01, size=Xb.shape[1])

    # select optimizer
    if optimizer_name.lower() == "sgd":
        opt = SGD(lr=lr)
    elif optimizer_name.lower() == "momentum":
        opt = Momentum(lr=lr, beta=momentum_beta)
    elif optimizer_name.lower() == "adam":
        opt = Adam(lr=lr, b1=adam_b1, b2=adam_b2)
    else:
        raise ValueError("optimizer_name must be 'sgd', 'momentum', or 'adam'.")

    loss_hist = []
    t = 0  # global step counter for Adam bias correction
    for epoch in range(1, epochs + 1):
        # mini-batch loop
        for Xb_mb, y_mb in batches(Xb, y, batch_size=batch_size, rng=rng):
            t += 1
            g = grad_logreg_batch(Xb_mb, y_mb, w, l2=l2)
            w = opt.step(w, g, t)

        # epoch-end metrics (full-batch probabilities)
        p = sigmoid(Xb @ w)
        loss = log_loss(y, p)
        loss_hist.append(loss)

    return {"w": w, "loss_hist": loss_hist}


# ---------- demo ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # make two blob classes (moderate overlap so optimizers matter)
    n_per = 600
    mean0, mean1 = np.array([-1.0, 0.0]), np.array([0.9, 0.8])
    X0 = rng.normal(mean0, 0.95, size=(n_per, 2))
    X1 = rng.normal(mean1, 0.95, size=(n_per, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_per), np.ones(n_per)]

    # train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)

    # shared hyperparams
    l2 = 0.0          # set >0 to add ridge-like regularization
    batch_size = 64
    epochs = 60

    # For fairness: same init → pass different seeds to each run? No: pass SAME seed so initial w is identical.
    seed = 123

    # --- SGD ---
    res_sgd = train_logreg_with_optimizer(
        Xtr, ytr, optimizer_name="sgd", lr=0.2, batch_size=batch_size,
        epochs=epochs, l2=l2, seed=seed
    )
    w_sgd = res_sgd["w"]

    # --- Momentum ---
    res_mom = train_logreg_with_optimizer(
        Xtr, ytr, optimizer_name="momentum", lr=0.15, batch_size=batch_size,
        epochs=epochs, l2=l2, seed=seed, momentum_beta=0.9
    )
    w_mom = res_mom["w"]

    # --- Adam ---
    res_adam = train_logreg_with_optimizer(
        Xtr, ytr, optimizer_name="adam", lr=0.05, batch_size=batch_size,
        epochs=epochs, l2=l2, seed=seed, adam_b1=0.9, adam_b2=0.999
    )
    w_adam = res_adam["w"]

    # evaluate
    def eval_weights(name: str, w: np.ndarray, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray):
        Xtrb = add_bias(Xtr); Xteb = add_bias(Xte)
        ptr = sigmoid(Xtrb @ w); pte = sigmoid(Xteb @ w)
        print(f"\n=== {name} ===")
        print("weights [bias, w1, w2]:", np.round(w, 4))
        print(f"train: loss={log_loss(ytr, ptr):.4f}  acc={accuracy(ytr, ptr):.3f}")
        print(f" test: loss={log_loss(yte, pte):.4f}  acc={accuracy(yte, pte):.3f}")

    eval_weights("SGD", w_sgd, Xtr, ytr, Xte, yte)
    eval_weights("Momentum", w_mom, Xtr, ytr, Xte, yte)
    eval_weights("Adam", w_adam, Xtr, ytr, Xte, yte)

    # compare to scikit-learn (reference)
    clf = SKLogReg(solver="lbfgs").fit(Xtr, ytr)
    pte_sk = clf.predict_proba(Xte)[:, 1]
    print("\n=== scikit-learn LogisticRegression (lbfgs) ===")
    print("intercept_, coef_:", round(float(clf.intercept_[0]), 4), np.round(clf.coef_[0], 4))
    print(f" test: loss={sk_log_loss(yte, pte_sk):.4f}  acc={accuracy_score(yte, (pte_sk>=0.5).astype(int)):.3f}")

    # show rough loss progress per optimizer
    print("\nLoss progress (first -> last):")
    print("SGD     :", f"{res_sgd['loss_hist'][0]:.4f} -> {res_sgd['loss_hist'][-1]:.4f}  (epochs={len(res_sgd['loss_hist'])})")
    print("Momentum:", f"{res_mom['loss_hist'][0]:.4f} -> {res_mom['loss_hist'][-1]:.4f}  (epochs={len(res_mom['loss_hist'])})")
    print("Adam    :", f"{res_adam['loss_hist'][0]:.4f} -> {res_adam['loss_hist'][-1]:.4f}  (epochs={len(res_adam['loss_hist'])})")

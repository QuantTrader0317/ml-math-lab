"""
Project 3 - Logistic Regression (from scratch)
Python 3.13-safe. Numpy 2.1+.

What this does:
- Generates a simple two-class dataset (two Gaussian blobs)
- Implements logistic regression trained by gradient descent
- Uses stable log-loss, early stopping, optional L2 regularization
- Compares accuracy with scikit-learn's LogisticRegression
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# -------- math helpers --------
def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -40, 40)
    return 1 / (1 + np.exp(-z))

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]

def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def grad_logreg(Xb: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """
    Gradient of average log-loss with optional L2 (ridge) penalty (excluding bias term).
    L2 term: (l2 / n) * [0, w1, w2, ...]  (no penalty on intercept)
    """
    n = Xb.shape[0]
    p = sigmoid(Xb @ w)
    g = (Xb.T @ (p - y)) / n
    # L2 penalty (do not regularize bias)
    if l2 > 0:
        reg = np.r_[0.0, w[1:]] * (l2 / n)
        g = g + reg
    return g

def fit_logreg_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-7,
    l2: float = 0.0,
    verbose_every: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, list]:
    """
    Train logistic regression with full-batch gradient descent.
    Returns: (weights including bias, loss_history)
    """
    rng = np.random.default_rng(seed)
    Xb = add_bias(X)
    w = rng.normal(scale=0.01, size=Xb.shape[1])
    loss_hist = []

    prev_loss = np.inf
    for t in range(max_iter):
        p = sigmoid(Xb @ w)
        loss = log_loss(y, p)
        loss_hist.append(loss)

        g = grad_logreg(Xb, y, w, l2=l2)
        w -= lr * g

        # Early stopping on loss improvement
        if abs(prev_loss - loss) < tol:
            if verbose_every:
                print(f"Early stop at iter {t}, loss={loss:.6f}")
            break
        prev_loss = loss

        if verbose_every and (t % verbose_every == 0):
            print(f"iter {t:5d}  loss={loss:.6f}")

    return w, loss_hist


# ---------- data & training ----------
if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # make two blob classes
    n_per = 400
    mean0, mean1 = np.array([-1.2, -0.2]), np.array([1.0, 0.8])
    X0 = rng.normal(mean0, 0.9, size=(n_per, 2))
    X1 = rng.normal(mean1, 0.9, size=(n_per, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_per), np.ones(n_per)]

    # train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # ---- scratch model (GD) ----
    w, history = fit_logreg_gd(
        Xtr, ytr,
        lr=0.4,            # learning rate
        max_iter=4000,
        tol=1e-8,
        l2=0.0,            # try e.g. 0.1 to add ridge penalty
        verbose_every=0,   # set to 200 to watch the loss drop
        seed=0
    )

    # predictions
    Xtrb = add_bias(Xtr)
    Xteb = add_bias(Xte)
    ptr = sigmoid(Xtrb @ w)
    pte = sigmoid(Xteb @ w)
    yhat_tr = (ptr >= 0.5).astype(int)
    yhat_te = (pte >= 0.5).astype(int)

    # metrics
    loss_tr = log_loss(ytr, ptr)
    loss_te = log_loss(yte, pte)
    acc_tr = accuracy_score(ytr, yhat_tr)
    acc_te = accuracy_score(yte, yhat_te)

    print("\n=== Scratch Logistic Regression (Gradient Descent) ===")
    print("weights [bias, w1, w2]:", np.round(w, 4))
    print(f"loss (train) = {loss_tr:.4f}  |  acc (train) = {acc_tr:.3f}")
    print(f"loss (test)  = {loss_te:.4f}  |  acc (test)  = {acc_te:.3f}")
    if len(history) > 0:
        print(f"first/last loss: {history[0]:.4f} -> {history[-1]:.4f}  (iters={len(history)})")

    # ---- sklearn comparison ----
    clf = LogisticRegression(solver="lbfgs").fit(Xtr, ytr)
    yhat_te_sk = clf.predict(Xte)
    acc_te_sk = accuracy_score(yte, yhat_te_sk)
    print("\n=== scikit-learn LogisticRegression ===")
    print("intercept_ and coef_:", round(float(clf.intercept_[0]), 4), np.round(clf.coef_[0], 4))
    print(f"acc (test) = {acc_te_sk:.3f}")
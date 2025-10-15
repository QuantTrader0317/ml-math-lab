"""
Project 6 - Regularized Regression (Ridge & Lasso)
Python 3.13-safe. Numpy 2.1+

What this does:
- Generates a regression dataset with correlated features
- Implements Ridge (closed-form) and Lasso (coordinate descent)
- Compare to scikit-learn Ridge/Lasso
- Runs a small hyperparameter sweep over lambda and prints R^2 + coefficients
"""

import numpy as np
from sklearn.linear_model import Ridge as SKRidge, Lasso as SKLasso
from sklearn.metrics import r2_score
from numpy.linalg import inv

# -----------------
# Utilities
# -----------------

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]

def standardize(X: np.ndarray):
    """
    Standardize columns (mean 0, std 1). Returns Xz, means, stds.
    Regularization is more stable/fair when features are on the same scale.
    """
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd

def unstandardize_beta(beta_z: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """
    Convert coefficients learned on standardized features back to original scale.
    beta_z is [b0, b1, ..., bd] for standardized X; return [b0_orig, b1_orig, ...].
    """
    b0 = beta_z[0] - np.sum((beta_z[1:] * (mu.flatten() / sd.flatten())))
    bj = beta_z[1:] / sd.flatten()
    return np.r_[b0, bj]

# -------------------------
# Ridge (closed-form)
# -------------------------
def ridge_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge with intercept: we standardize X (not y), fit ridge on [1, Xz],
    then un-standardize coefficients to original X scale.
    """
    Xz, mu, sd = standardize(X)
    Xb = add_bias(Xz)
    # no penalty on the bias term → build penalty matrix with 0 in top-left
    d = Xb.shape[1]
    I = np.eye(d)
    I[0, 0] = 0.0
    beta_z = inv(Xb.T @ Xb + lam * I) @ (Xb.T @ y)
    beta_orig = unstandardize_beta(beta_z, mu, sd)
    return beta_orig

# -------------------------
# Lasso (coordinate descent)
# -------------------------
def soft_threshold(z: float, g: float) -> float:
    """Soft-thresholding operator: S(z, g) = sign(z) * max(|z| - g, 0)."""
    if z > g:
        return z - g
    if z < -g:
        return z + g
    return 0.0

def lasso_coordinate_descent(X: np.ndarray, y: np.ndarray, lam: float,
                             max_iter: int = 5000, tol: float = 1e-6, verbose=False, seed=0) -> np.ndarray:
    """
    Lasso with intercept via coordinate descent on standardized X (no penalty on intercept).
    We standardize X, fit, then unstandardize back to original scale.
    """
    rng = np.random.default_rng(seed)
    Xz, mu, sd = standardize(X)
    n, d = Xz.shape
    # add bias
    Xb = add_bias(Xz)  # shape (n, d+1); columns: [1, Xz]
    w = np.zeros(d + 1)  # [bias, w1, ..., wd]

    # Precompute column norms for speed (ignore bias col)
    col_norm2 = np.sum(Xz ** 2, axis=0)  # length d

    lam2 = lam / n  # average loss formulation

    prev_w = w.copy()
    for it in range(max_iter):
        # --- update bias (no penalty) ---
        # residual with current weights
        r = y - Xb @ w
        # optimal bias is mean of residual added to current bias
        w[0] += r.mean()

        # --- update each coefficient with soft-thresholding ---
        for j in range(d):
            # remove current feature contribution from residual
            r = y - (w[0] + (Xz @ w[1:])) + Xz[:, j] * w[1 + j]
            rho_j = np.dot(Xz[:, j], r)  # correlation term
            # coordinate update with soft-thresholding
            if col_norm2[j] == 0:
                w[1 + j] = 0.0
            else:
                w[1 + j] = soft_threshold(rho_j, lam2) / col_norm2[j]

        # stopping check
        if np.linalg.norm(w - prev_w, ord=2) < tol:
            if verbose:
                print(f"[lasso] early stop at iter {it}")
            break
        prev_w = w.copy()

    # unstandardize to original X scale
    beta_orig = unstandardize_beta(w, mu, sd)
    return beta_orig

# -------------------------
# Demo / Comparison
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Create correlated features
    n, d = 500, 6
    X = rng.normal(size=(n, d))
    # inject correlation: x2 ~= x1, x4 ~= x3
    X[:, 1] = 0.85 * X[:, 0] + 0.3 * rng.normal(size=n)
    X[:, 3] = 0.75 * X[:, 2] - 0.2 * rng.normal(size=n)

    # Sparse-ish true beta (intercept + 6 features)
    beta_true = np.array([1.5, 2.0, 0.0, -1.2, 0.0, 0.8, 0.0])  # b0, b1..b6
    noise = rng.normal(0, 1.0, size=n)
    y = add_bias(X) @ beta_true + noise

    print("\n=== True coefficients ===")
    print("beta_true:", np.round(beta_true, 3))

    # baseline OLS (for reference)
    Xb = add_bias(X)
    beta_ols, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    r2_ols = r2_score(y, Xb @ beta_ols)
    print("\n=== OLS (baseline) ===")
    print("beta_ols :", np.round(beta_ols, 3))
    print("R^2      :", round(float(r2_ols), 4))

    # lambdas to sweep
    lambdas = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0]

    print("\n=== Ridge: closed-form vs sklearn ===")
    print("lam\tR2_scratch\tR2_sklearn\tbeta_scratch[:4]\t\tbeta_sklearn[:4]")
    for lam in lambdas:
        beta_ridge = ridge_closed_form(X, y, lam)
        r2_ridge = r2_score(y, add_bias(X) @ beta_ridge)

        sk_ridge = SKRidge(alpha=lam, fit_intercept=True).fit(X, y)
        beta_skr = np.r_[sk_ridge.intercept_, sk_ridge.coef_]
        r2_skr = r2_score(y, sk_ridge.predict(X))

        print(f"{lam:.1f}\t{r2_ridge:.4f}\t\t{r2_skr:.4f}\t\t{np.round(beta_ridge[:4],3)}\t{np.round(beta_skr[:4],3)}")

    print("\n=== Lasso: coordinate descent vs sklearn ===")
    print("lam\tR2_scratch\tR2_sklearn\tbeta_scratch\t\t\t\tbeta_sklearn")
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
        beta_lasso = lasso_coordinate_descent(X, y, lam=lam, max_iter=8000, tol=1e-7)
        r2_lasso = r2_score(y, add_bias(X) @ beta_lasso)

        sk_lasso = SKLasso(alpha=lam, fit_intercept=True, max_iter=10000).fit(X, y)
        beta_skl = np.r_[sk_lasso.intercept_, sk_lasso.coef_]
        r2_skl = r2_score(y, sk_lasso.predict(X))

        print(f"{lam:.2f}\t{r2_lasso:.4f}\t\t{r2_skl:.4f}\t\t{np.round(beta_lasso,3)}\t{np.round(beta_skl,3)}")

    # Short takeaway
    print("\nNote:")
    print("- Ridge shrinks coefficients smoothly as lambda increases (none go exactly to zero).")
    print("- Lasso pushes some coefficients to exactly zero as lambda grows (feature selection).")

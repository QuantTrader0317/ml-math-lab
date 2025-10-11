"""
Project 2 - Ordinary Least Squares (OLS) Regression
Python 3.13-safe. NumPy 2.1+

What this does:
- Builds synthetic data from a known linear model with noise
- Fits OLS three ways: Normal Equation, Least Squares (SVD), scikit-learn
- Prints coefficients, R^2, residual diagnostics
"""


import numpy as np
from sklearn.linear_model import LinearRegression

# -------helpers--------
def add_bias(X: np.ndarray) -> np.ndarray:
    """Add intercept column of 1s to X."""
    return np.c_[np.ones((X.shape[0], 1)), X]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def normal_eq_beta(Xb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed - form OLS: (X^t X)^{-1} X^T y.
    Warning: can be numerically unstable if X^T X is ill-conditioned.
    """

    XtX = Xb.T @ Xb
    XtX_inv = np.linalg.inv(XtX) # for teaching; real code should prefer Lstsq/QR/SVD
    beta = XtX_inv @ Xb.T @ y
    return beta

def lstsq_beta(Xb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Stable OLS via least squares (SVD under the hood).
    This is the correct way for production.
    """

    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta

def residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res = y_true - y_pred
    return {
        "mean": float(res.mean()),
        "std": float(res.std(ddof=1)),
        "max_abs": float(np.max(np.abs(res))),
    }

#---------- data ---------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n, d = 400, 3            # 400 samples, 3 features
    X = rng.normal(size=(n, d))    # design matrix
    beta_true = np.array([2.0, -1.5, 0.5, 3.0])  # [intercept, b1, b2, b3]
    noise = rng.normal(0, 0.7, size=n)

    Xb = add_bias(X)      # add intercept
    y = Xb @ beta_true + noise   # target with noise


    # --------- OLS via Normal Equation ----------
    beta_ne = normal_eq_beta(Xb, y)
    yhat_ne = Xb @ beta_ne
    r2_ne = r2_score(y, yhat_ne)
    res_ne = residual_stats(y, yhat_ne)

    #---------OLS via Least Squares (SVD) ------------
    beta_ls = lstsq_beta(Xb, y)
    yhat_ls = Xb @ beta_ls
    r2_ls = r2_score(y, yhat_ls)
    res_ls = residual_stats(y, yhat_ls)

    #----------- OLS via scikit-Learn------------
    lr = LinearRegression(fit_intercept=True).fit(X, y)
    beta_skl = np.r_[lr.intercept_, lr.coef_]  # [intercept, coef...]
    yhat_skl = lr.predict(X)
    r2_skl = r2_score(y, yhat_skl)
    res_skl = residual_stats(y, yhat_skl)

    # -------------- Diagnostics --------------
    # Condition number (bigger = more collinearity = normal equation gets sketchy
    _, svals, _ = np.linalg.svd(Xb, full_matrices=False)
    cond_number = float(svals[0] / svals[-1])

    print("\n=== True coefficients ===")
    print("beta_true        :", np.round(beta_true, 4))

    print("\n=== OLS (Normal Equation) ===")
    print("beta_ne          :", np.round(beta_ne, 4))
    print("R^2         :", np.round(r2_ne, 4))
    print("residuals {mean, std, max_abs}:", {k: round(v, 4) for k, v in res_ne.items()})

    print("\n=== OLS (Least Squares / SVD) ===")
    print("beta_ls           :", np.round(beta_ls, 4))
    print("R^2               :", round(r2_ls, 4))
    print("residuals {mean, std, max_abs}:", {k: round(v, 4) for k, v in res_ls.items()})

    print("\n=== OLS (scikit-learn) ===")
    print("beta_skl          :", np.round(beta_skl, 4))
    print("R^2               :", round(r2_skl, 4))
    print("residuals {mean, std, max_abs}:", {k: round(v, 4) for k, v in res_skl.items()})

    print("\n=== Design matrix health ===")
    print("Xb SVD condition number:", round(cond_number, 2))
    if cond_number > 1e6:
        print("Warning: ill-conditioned design; prefer SVD/QR over (X^T X)^(-1).")
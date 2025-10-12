"""
Project 5 — PCA via SVD (from scratch)
Python 3.13-safe. NumPy 2.1+.

What this does:
- Generates a 3D dataset with correlated features
- Implements PCA using SVD on centered data
- Reports explained variance ratios and reconstruction error
- Verifies results against scikit-learn PCA
"""

import numpy as np
from sklearn.decomposition import PCA as SKPCA

# ---------- PCA (scratch) ----------
def pca_svd(X: np.ndarray, k: int):
    """
    PCA via SVD on centered data.
    Returns:
      Z      : projected data (n, k)
      comps  : principal components (k, d) row-wise
      ratios : explained variance ratios (k,)
      mu     : mean used for centering (1, d)
      S      : singular values (min(n, d),) for optional diagnostics
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu                      # center
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]                   # rows = PCs
    n = X.shape[0]
    var_explained = (S**2) / (n - 1) # eigenvalues of covariance
    ratios = var_explained / var_explained.sum()
    Z = Xc @ comps.T                 # projection to top-k
    return Z, comps, ratios[:k], mu, S

def reconstruct_from_pcs(Z: np.ndarray, comps: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Reconstruct data from projected coordinates."""
    return Z @ comps + mu

def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean((A - B)**2)))


# ---------- Demo ----------
if __name__ == "__main__":
    rng = np.random.default_rng(3)

    # make correlated 3D data
    n = 600
    X = rng.normal(size=(n, 3))
    # induce correlation structure
    X[:, 1] = 2.0 * X[:, 0] + 0.35 * X[:, 1]
    X[:, 2] = -0.6 * X[:, 0] + 0.3 * X[:, 1] + X[:, 2]

    # choose how many PCs to keep
    k = 2

    # ---- scratch PCA
    Z, comps, ratios, mu, S = pca_svd(X, k=k)
    X_recon = reconstruct_from_pcs(Z, comps, mu)
    err = rmse(X, X_recon)

    print("=== PCA (scratch) ===")
    print(f"keep k = {k} components")
    print("explained variance ratios:", np.round(ratios, 4))
    print("reconstruction RMSE:", round(err, 6))
    print("top PCs (rows):\n", np.round(comps, 4))

    # ---- cumulative explained variance (scratch)
    # Use all singular values to compute cumulative variance curve
    _, _, ratios_all, _, _ = pca_svd(X, k=min(X.shape))
    cume = np.cumsum(ratios_all)
    # find smallest k explaining >= 95%
    k95 = int(np.searchsorted(cume, 0.95) + 1)
    print("\ncoverage: PCs to reach 95% variance =", k95)

    # ---- sklearn check
    sk = SKPCA(n_components=k, svd_solver="full")
    Z_sk = sk.fit_transform(X)
    X_recon_sk = sk.inverse_transform(Z_sk)
    err_sk = rmse(X, X_recon_sk)

    print("\n=== PCA (scikit-learn) ===")
    print("explained variance ratios:", np.round(sk.explained_variance_ratio_, 4))
    print("reconstruction RMSE:", round(err_sk, 6))
    print("components (rows):\n", np.round(sk.components_, 4))

    # consistency checks
    # components are unique up to sign; compare absolute cosine similarity
    for i in range(k):
        num = abs(np.dot(comps[i], sk.components_[i]))
        den = np.linalg.norm(comps[i]) * np.linalg.norm(sk.components_[i])
        cos = float(np.clip(num / den, -1.0, 1.0))
        print(f"\nPC{i+1} alignment (abs cos): {cos:.6f}")

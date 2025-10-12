"""
Project 4 — k-Means Clustering (from scratch)
Python 3.13-safe. NumPy 2.1+. Compares to scikit-learn.

What this does:
- Generates 3 Gaussian blobs in 2D
- Implements k-means: random init -> assign -> update -> repeat
- Handles empty clusters robustly
- Compares inertia (SSE) against sklearn.KMeans
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def kmeans_scratch(X: np.ndarray, k: int, max_iter: int = 200, tol: float = 1e-6, seed: int = 0):
    """
    Run Lloyd's algorithm (k-means) from scratch.
    Returns: labels (n,), centers (k,d), inertia (float), n_iters (int)
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # ----- init: pick k distinct points as initial centers
    idx0 = rng.choice(n, size=k, replace=False)
    centers = X[idx0].copy()

    prev_inertia = np.inf
    for it in range(1, max_iter + 1):
        # ----- assign: nearest center for each point
        # Efficiently compute argmin over centers (uses sklearn helper for speed)
        labels = pairwise_distances_argmin(X, centers, axis=1)

        # ----- update: mean of assigned points
        new_centers = centers.copy()
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                # empty cluster → re-seed randomly
                new_centers[j] = X[rng.integers(0, n)]
            else:
                new_centers[j] = pts.mean(axis=0)

        # ----- compute inertia (within-cluster SSE)
        inertia = float(np.sum((X - new_centers[labels]) ** 2))

        # stop if centers barely moved or inertia improved tiny amount
        center_shift = float(np.linalg.norm(new_centers - centers))
        if abs(prev_inertia - inertia) < tol and center_shift < np.sqrt(tol):
            centers = new_centers
            break

        centers = new_centers
        prev_inertia = inertia

    # final labels/inertia with final centers
    labels = pairwise_distances_argmin(X, centers, axis=1)
    inertia = float(np.sum((X - centers[labels]) ** 2))
    return labels, centers, inertia, it


if __name__ == "__main__":
    rng = np.random.default_rng(2)

    # ----- make synthetic 3-cluster data
    A = rng.normal([-2.0, 0.0], 0.55, size=(160, 2))
    B = rng.normal([ 2.0, 0.2], 0.60, size=(170, 2))
    C = rng.normal([ 0.2, 2.0], 0.55, size=(165, 2))
    X = np.vstack([A, B, C])

    # ----- scratch k-means
    k = 3
    labels, centers, inertia, n_iter = kmeans_scratch(X, k=k, max_iter=300, tol=1e-6, seed=0)
    print("=== k-means (scratch) ===")
    print("iters:", n_iter)
    print("centers:\n", np.round(centers, 3))
    print("inertia:", round(inertia, 2))

    # ----- sklearn comparison
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(X)
    print("\n=== k-means (scikit-learn) ===")
    print("centers:\n", np.round(km.cluster_centers_, 3))
    print("inertia:", round(float(km.inertia_), 2))

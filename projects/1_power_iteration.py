import numpy as np

def power_iteration(A, max_iter=5000, tol=1e-10, seed=0):
    """
    Dominant eigenpair of symmetric (or any) matrix A via power iteration.
    Uses residual ||A x - lambda x|| as the stopping test.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        x[0] = 1.0
        x_norm = 1.0
    x = x / x_norm

    lam = 0.0
    for _ in range(max_iter):
        y = A @ x
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            # x is in the nullspace; re-initialize
            x = rng.normal(size=n)
            x = x / np.linalg.norm(x)
            continue

        # Rayleigh quotient (best scalar along x)
        lam_new = float(x @ y)

        # Next iterate
        x_new = y / y_norm

        # Residual test: how close is A x_new to lam_new x_new?
        res = np.linalg.norm(A @ x_new - lam_new * x_new)

        x = x_new
        lam = lam_new

        if res < tol:
            break

    return lam, x

def make_psd_matrix(n=6, seed=42, jitter=1e-6):
    """
    Symmetric positive semidefinite test matrix with tiny diagonal jitter
    to avoid zero eigenvalues.
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    A = M.T @ M
    if jitter:
        A = A + jitter * np.eye(n)
    return A

if __name__ == "__main__":
    A = make_psd_matrix(6, seed=42)

    # Power iteration (robust)
    lam, vec = power_iteration(A, max_iter=10000, tol=1e-12, seed=0)

    # Ground truth using symmetric solver (eigh)
    w, V = np.linalg.eigh(A)                  # eigenvalues sorted ascending
    idx = int(np.argmax(w))
    lam_true = float(w[idx])
    v_true = V[:, idx]

    # Angle between vectors (ignore sign)
    cosang = abs(np.dot(vec, v_true)) / (np.linalg.norm(vec) * np.linalg.norm(v_true))
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = float(np.arccos(cosang))

    print("Power iteration λ* :", round(lam, 6))
    print("NumPy top λ*       :", round(lam_true, 6))
    print("Angle difference (rad):", round(angle, 10))

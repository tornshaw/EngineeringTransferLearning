import numpy as np
from scipy.linalg import eigh

def tca(Xs, Xt, kern=None, hyp=None, mu=1.0, k=10):
    """
    Transfer component analysis

    Inputs:
    Xs: source features (ns, d)
    Xt: target features (nt, d)
    kern: kernel function
    hyp: kernel hyperparameters
    mu: trade-off parameter
    k: number of eigenvectors (dimension of embedding <= d)

    Outputs:
    Zs: transformed source (ns, k)
    Zt: transformed target (nt, k)
    W: embedding matrix (n, k)
    mmd: MMD distance
    """
    ns, d = Xs.shape
    nt = Xt.shape[0]

    X = np.vstack([Xs, Xt])
    n = X.shape[0]

    if k > d:
        raise ValueError('k must be less than d')

    # Calculate M - constants from MMD biased v-statistic
    M = np.full((n, n), -1 / (ns * nt))
    M[:ns, :ns] = 1 / ns**2
    M[ns:, ns:] = 1 / nt**2

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Kernel embedding
    if kern:
        K, _ = kern(hyp, X, X)
    else:
        K = X
        n = d

    # Generalized eigenvalue problem
    A = mu * np.eye(n) + K.T @ M @ K
    B = K.T @ H @ K

    # Add small regularization to B
    eps = 1e-6
    B_reg = B + eps * np.eye(B.shape[0])

    eigvals, W_full = eigh(A, B_reg)
    k_use = min(k, W_full.shape[1])
    idx = np.argsort(eigvals)[:k_use]
    W = W_full[:, idx]

    Z = K @ W

    # Extract transformed source and target
    Zs = Z[:ns]
    Zt = Z[ns:]

    # MMD of transfer space
    mmd = np.sum(np.diag(W.T @ K @ M @ K @ W))

    return Zs, Zt, W, mmd
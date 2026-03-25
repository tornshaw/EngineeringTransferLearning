import numpy as np
from scipy.linalg import eigh
from util.f1score import f1score

def bda(Xs, Ys, Xt, kern=None, hyp=None, mu=1.0, k=10, lambda_=0.5, classifier=None, iter=10, mode=0, Yt=None):
    """
    Balanced distribution adaptation

    Inputs:
    Xs: source features
    Ys: source labels
    Xt: target features
    kern: kernel function
    hyp: hyperparameters
    mu: trade-off
    k: dimensions
    lambda_: balance factor
    classifier: classifier function
    iter: iterations
    mode: 0 for BDA, 1 for WBDA
    Yt: target labels (optional)

    Outputs:
    Zs, Zt, Ytp, W, cls, fscore, mmd
    """
    ns, nt = len(Xs), len(Xt)
    X = np.vstack([Xs, Xt])
    n, d = X.shape

    if k > d:
        raise ValueError('k must be less than d')

    M0 = np.full((n, n), -1 / (ns * nt))
    M0[:ns, :ns] = 1 / ns**2
    M0[ns:, ns:] = 1 / nt**2

    H = np.eye(n) - np.ones((n, n)) / n

    if kern:
        K, hyp = kern(hyp, X, X)
    else:
        K = X
        n = d

    # Initial pseudo-labels
    Ytp, cls = classifier(Xs, Ys, Xt)

    fs_best = 0
    fscore_list = []
    cls_best = None
    W_best = None
    Zs_best = None
    Zt_best = None
    Ytp_best = None
    failed = 0

    for i in range(iter):
        M = np.zeros_like(M0)
        for c in np.unique(Ys):
            cYs = np.where(Ys == c)[0]
            cYtp = np.where(Ytp == c)[0]

            nsc = len(cYs)
            ntc = len(cYtp)

            if mode == 1:
                Py_s = nsc / ns
                Py_t = ntc / nt
            else:
                Py_s = 1
                Py_t = 1

            if nsc > 0 and ntc > 0:
                M[np.ix_(cYs, cYs)] += Py_s / nsc**2
                M[np.ix_(ns + cYtp, ns + cYtp)] += Py_t / ntc**2
                M[np.ix_(cYs, ns + cYtp)] -= np.sqrt(Py_s * Py_t) / (nsc * ntc)
                M[np.ix_(ns + cYtp, cYs)] -= np.sqrt(Py_s * Py_t) / (nsc * ntc)

        M_balanced = (1 - lambda_) * M0 + lambda_ * M

        A = mu * np.eye(n) + K.T @ M_balanced @ K
        B = K.T @ H @ K

        # Solve generalized eigenvalue problem A w = lambda B w with dense solver for stability on small n
        # Add a small regularization term to B to avoid singularities.
        eps = 1e-6
        B_reg = B + eps * np.eye(B.shape[0])

        eigvals, W_full = eigh(A, B_reg)
        k_use = min(k, W_full.shape[1])
        idx = np.argsort(eigvals)[:k_use]
        W = W_full[:, idx]

        if np.isreal(W).all():
            Z = K @ W
            Zs = Z[:ns]
            Zt = Z[ns:]

            Ytp, cls = classifier(Zs, Ys, Zt)

            if Yt is not None:
                f1, _ = f1score(Yt, Ytp)
                fscore_list.append(f1)
                if f1 > fs_best:
                    cls_best = cls
                    W_best = W
                    Zs_best = Zs
                    Zt_best = Zt
                    Ytp_best = Ytp
                    fs_best = f1
        else:
            failed += 1

    if Yt is not None and failed != iter:
        W = W_best
        Zs = Zs_best
        Zt = Zt_best
        cls = cls_best
        Ytp = Ytp_best
        fscore = fs_best
    elif failed == iter:
        Zs = np.nan
        Zt = np.nan
        Ytp = np.nan
        cls = np.nan
        fscore = np.nan

    mmd = np.sum(np.diag(W.T @ K @ M_balanced @ K @ W))

    return Zs, Zt, Ytp, W, cls, fscore, mmd
import numpy as np
from scipy.linalg import eigh
from util.f1score import f1score
from .metricInformedLabels import metricInformedLabels

def mjda(Xs, Ys, Xt, kern=None, hyp=None, mu=1.0, k=10, classifier=None, iter=10, nf=None, ne=1000, Yt=None):
    """
    Metric-informed joint distribution adaptation

    Inputs: similar to JDA, with nf, ne for metric-informed labels

    Outputs: Zs, Zt, Ytp, W, cls, fscore, mmd
    """
    ns, nt = len(Xs), len(Xt)
    X = np.vstack([Xs, Xt])
    n, d = X.shape

    if k > d:
        raise ValueError('k must be less than d')

    if nf is None:
        if n < d + 1:
            nf = int(np.sqrt(n))
            ne = 1000
        else:
            nf = d
            ne = 1

    # Metric-informed labels
    Ytp, _ = metricInformedLabels(Xs, Ys, Xt, nf, ne)

    M0 = np.full((n, n), -1 / (ns * nt))
    M0[:ns, :ns] = 1 / ns**2
    M0[ns:, ns:] = 1 / nt**2

    H = np.eye(n) - np.ones((n, n)) / n

    if kern:
        K = kern(hyp, X, X)
    else:
        K = X
        n = d

    fs_best = 0
    fscore_list = []
    cls_best = None
    W_best = None
    Zs_best = None
    Zt_best = None
    Ytp_best = None
    failed = 0

    for i in range(iter):
        M = M0.copy()
        for c in np.unique(Ys):
            cYs = np.where(Ys == c)[0]
            cYtp = np.where(Ytp == c)[0]

            nsc = len(cYs)
            ntc = len(cYtp)

            if nsc > 0 and ntc > 0:
                M[np.ix_(cYs, cYs)] += 1 / nsc**2
                M[np.ix_(ns + cYtp, ns + cYtp)] += 1 / ntc**2
                M[np.ix_(cYs, ns + cYtp)] -= 1 / (nsc * ntc)
                M[np.ix_(ns + cYtp, cYs)] -= 1 / (nsc * ntc)

        A = mu * np.eye(n) + K.T @ M @ K
        B = K.T @ H @ K

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
    else:
        fscore = None

    mmd = np.sum(np.diag(K @ M0))

    return Zs, Zt, Ytp, W, cls, fscore, mmd
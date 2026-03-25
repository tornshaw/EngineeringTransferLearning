import numpy as np
from scipy.linalg import eigs
from util.f1score import f1score

def jda(Xs, Ys, Xt, kern=None, hyp=None, mu=1.0, k=10, classifier=None, iter=10, Yt=None):
    """
    Joint distribution adaptation

    Inputs:
    Xs: source features
    Ys: source labels
    Xt: target features
    kern: kernel function
    hyp: kernel hyperparameters
    mu: trade-off
    k: dimensions
    classifier: classifier function
    iter: iterations
    Yt: target labels (optional)

    Outputs:
    Zs: transformed source
    Zt: transformed target
    Ytp: pseudo labels
    W: embedding matrix
    cls: classifier
    fscore: f1 scores
    mmd: MMD
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
        K = kern(hyp, X, X)
    else:
        K = X
        n = d

    M = M0.copy()
    fscore_list = []
    fs_best = 0
    cls_best = None
    W_best = None
    Zs_best = None
    Zt_best = None

    for i in range(iter):
        A = mu * np.eye(n) + K.T @ M @ K
        B = K.T @ H @ K
        eigvals, W = eigs(A, B, k=k, which='SM')

        if np.isreal(W).all():
            Z = K @ W
            Zs = Z[:ns]
            Zt = Z[ns:]

            Ytp, cls = classifier(Zs, Ys, Zt)

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

            if Yt is not None:
                f1, _ = f1score(Yt, Ytp)
                fscore_list.append(f1)
                if f1 > fs_best:
                    cls_best = cls
                    W_best = W
                    Zs_best = Zs
                    Zt_best = Zt
                    fs_best = f1

    if Yt is not None:
        cls = cls_best
        W = W_best
        Zs = Zs_best
        Zt = Zt_best
        fscore = fs_best
    else:
        fscore = None

    mmd = np.sum(np.diag(K @ M0))

    return Zs, Zt, Ytp, W, cls, fscore, mmd
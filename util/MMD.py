import numpy as np

def MMD(Xs, Xt, kern=None, hyp=None, Ys=None, Yt=None):
    """
    Maximum mean discrepancy distance

    Inputs:
    Xs: source data (ns, d)
    Xt: target data (nt, d)
    kern: kernel function or None
    hyp: kernel hyperparameters
    Ys: source labels (ns,)
    Yt: target labels (nt,)

    Outputs:
    mmd: marginal MMD
    mmd_c: conditional MMD (if labels provided)
    """
    X = np.vstack([Xs, Xt])
    ns, nt = len(Xs), len(Xt)
    n = ns + nt

    # M0
    M0 = np.full((n, n), -1 / (ns * nt))
    M0[:ns, :ns] = 1 / ns**2
    M0[ns:, ns:] = 1 / nt**2

    if kern is not None:
        K = kern(hyp, X, X)
        K += np.eye(n) * 1e-6
    else:
        K = X
        n = X.shape[1]

    # Marginal MMD
    mmd = np.sum(np.diag(K @ M0))

    M = M0.copy()
    mmd_c = None

    if Ys is not None and Yt is not None:
        for c in np.unique(Ys):
            cYs = np.where(Ys == c)[0]
            cYt = np.where(Yt == c)[0]

            nsc = len(cYs)
            ntc = len(cYt)

            if nsc > 0 and ntc > 0:
                M[np.ix_(cYs, cYs)] += 1 / nsc**2
                M[np.ix_(ns + cYt, ns + cYt)] += 1 / ntc**2
                M[np.ix_(cYs, ns + cYt)] -= 1 / (nsc * ntc)
                M[np.ix_(ns + cYt, cYs)] -= 1 / (nsc * ntc)

        mmd_c = np.sum(np.diag(K @ M))

    return mmd, mmd_c
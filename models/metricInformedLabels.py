import numpy as np
from scipy.linalg import cholesky

def metricInformedLabels(Xs, Ys, Xt, nf=None, ne=10, cb=99, nMC=10000):
    """
    Metric-informed pseudo-labels using MSD distances

    Inputs:
    Xs, Ys: source data and labels
    Xt: target data
    nf: number of features
    ne: number of ensembles
    cb: confidence bound
    nMC: Monte Carlo samples

    Outputs:
    Ytp: target pseudo-labels
    Ysp: source pseudo-labels
    """
    classes = np.unique(Ys)
    C = len(classes)
    ns, d = Xs.shape
    nt = len(Xt)
    X = np.vstack([Xs, Xt])

    if nf is None:
        nf = int(np.sqrt(ns + nt))

    dc = np.full((ns + nt, C), np.nan)
    dc_norm = np.full((ns + nt, C), np.nan)

    for i, c in enumerate(classes):
        Xs_c = Xs[Ys == c]
        df = Xs_c.shape[1]

        if df == nf:
            mu = np.mean(Xs_c, axis=0)
            sig = np.cov(Xs_c.T)
            dc[:, i] = MSD(X, mu, sig)
        else:
            dm = np.full((ns + nt, ne), np.nan)
            for j in range(ne):
                feat_ind = np.random.choice(df, nf, replace=False)
                fb_s_c = Xs_c[:, feat_ind]
                fb = X[:, feat_ind]
                mu = np.mean(fb_s_c, axis=0)
                sig = np.cov(fb_s_c.T)
                dm[:, j] = MSD(fb, mu, sig)
            dc[:, i] = np.mean(dm, axis=1)

        # Monte Carlo threshold
        thres = []
        for _ in range(nMC):
            x_zeta = np.random.randn(len(Xs_c), nf)
            thres.append(np.max(MSD(x_zeta, np.zeros(nf), np.eye(nf))))
        thres = np.sort(thres)
        thresh = thres[int(nMC * cb / 100)]

        dc_norm[:, i] = dc[:, i] / thresh
        dc_norm[dc[:, i] < thresh, i] = 0

    # Pseudo-labels
    ind = np.argmin(dc_norm, axis=1)
    Ypseudo = classes[ind]
    Ysp = Ypseudo[:ns]
    Ytp = Ypseudo[ns:]

    return Ytp, Ysp

def MSD(X, Xmu, Xcov):
    """
    Mahalanobis squared distance
    """
    try:
        R = cholesky(Xcov)
        res = X - Xmu
        d = np.sum((np.linalg.solve(R.T, res.T))**2, axis=0)
    except:
        Xcov += np.eye(Xcov.shape[0]) * 1e-12
        R = cholesky(Xcov)
        res = X - Xmu
        d = np.sum((np.linalg.solve(R.T, res.T))**2, axis=0)
    return d
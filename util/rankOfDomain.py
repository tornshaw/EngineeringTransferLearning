import numpy as np

def rankOfDomain(Ps, Pt, Xs, Xt, d):
    """
    Rank of domain

    Inputs:
    Ps: source subspace basis
    Pt: target subspace basis
    Xs: source dataset
    Xt: target dataset
    d: dimension of PCA subspaces

    Output:
    ROD: rank of domain
    """
    # Principal angles
    U, S, Vt = np.linalg.svd(Ps[:, :d].T @ Pt[:, :d])
    th = np.arccos(np.diag(S).real)

    # Principal vectors
    s = Ps[:, :d] @ U
    t = Pt[:, :d] @ Vt.T

    # Center data
    Xss = Xs - np.mean(Xs, axis=0)
    Xtt = Xt - np.mean(Xt, axis=0)

    # PCA covariances
    Sig2s = np.diag(1 / Xss.shape[0] * (s.T @ Xss.T @ Xss @ s))
    Sig2t = np.diag(1 / Xtt.shape[0] * (t.T @ Xtt.T @ Xtt @ t))

    ROD = (1 / d) * np.sum(th * (0.5 * Sig2s / Sig2t + 0.5 * Sig2t / Sig2s - 1))
    return ROD
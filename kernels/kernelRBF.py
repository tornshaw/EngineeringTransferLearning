import numpy as np
from scipy.spatial.distance import cdist

def kernelRBF(hyp, xp, xq):
    """
    Radial basis function kernel

    Inputs:
    hyp: sigma, or nan for median heuristic
    xp, xq: data

    Output:
    K: kernel matrix
    hyp: updated hyp
    """
    if np.isnan(hyp):
        Z = np.vstack([xp, xq])
        dist_sq = cdist(Z, Z, 'sqeuclidean')
        upper_tri = dist_sq[np.triu_indices_from(dist_sq, k=1)]
        hyp = np.sqrt(0.5 * np.median(upper_tri))
    d = cdist(xp / np.sqrt(2 * hyp**2), xq / np.sqrt(2 * hyp**2), 'sqeuclidean')
    K = np.exp(-d)
    return K, hyp
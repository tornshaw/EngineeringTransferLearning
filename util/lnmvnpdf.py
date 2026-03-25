from scipy.stats import multivariate_normal
import numpy as np

def lnmvnpdf(X, Mu=None, Sigma=None):
    """
    Log multivariate normal probability density function.

    Inputs:
    X: N-by-D matrix of observations
    Mu: mean vector (1-by-D) or None for zero mean
    Sigma: covariance matrix (D-by-D) or None for identity

    Output:
    y: log probabilities (N-by-1)
    """
    if Mu is None:
        Mu = np.zeros(X.shape[1])
    if Sigma is None:
        Sigma = np.eye(X.shape[1])
    return multivariate_normal.logpdf(X, mean=Mu, cov=Sigma)
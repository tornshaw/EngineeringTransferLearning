import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.plot_gaussian_2d import plot_gaussian_2d
from scipy.spatial.distance import cdist

def k_means(X, K, tol=0.1, plt=0):
    """
    k-means clustering

    Inputs
    X = inputs (n*d)
    K = no. of components
    tol = convergence tolerance on means
    plt = 0 - no plot

    Outputs
    classifier = structure of k-means parameters
        classifier.mu = means of clusters (K*d)
    Y = label predictions
    """
    n, d = X.shape

    # random initialise
    ind = np.random.permutation(n)
    mu = X[ind[:K], :]  # randomly initialise means

    mu_old = mu * 10  # initialise previous mu
    while np.sum((mu_old - mu) ** 2) > tol:
        mu_old = mu

        # Label data according to distance from means
        D = cdist(X, mu)
        ind = np.argmin(D, axis=1)
        Y = ind + 1  # 1-based

        # Calculate means based on labelled data
        for k in range(K):
            mu[k, :] = np.mean(X[Y == k+1, :], axis=0)

        # plot if 2D and plot on
        if plt != 0 and d == 2:
            # Note: plotting not implemented
            pass

    # pack up classifier parameters
    classifier = {
        'mu': mu,
        'k': K
    }

    return classifier
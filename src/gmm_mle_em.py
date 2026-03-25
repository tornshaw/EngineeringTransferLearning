import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.lnmvnpdf import lnmvnpdf
from src.k_means import k_means

def gmm_mle_em(X, K, tol=1e-6, method=0, plt=0):
    """
    Unsupervised Gaussian mixture model using EM and maximum likelihood estimates

    Inputs
    X = inputs (n*d)
    K = no. of components
    tol = convergence tolerance on means
    method = initialisation method, 0 - random, 1 - k-means
    plt = 0 - no plot

    Outputs
    classifier = structure of GMM parameters
        classifier.mu = means (k*d)
        classifier.sigma = covariances (d*d*k)
        classifier.lambda = mixing proportions (k)
        classifier.lml = complete log likelihood
    Y = label predictions
    """
    n, d = X.shape

    # initialise GMM
    if method == 0:
        # random initialise
        ind = np.random.permutation(n)
        mu = X[ind[:K], :]
    else:
        # k-means initialise
        kmeans_result = k_means(X, K, 1)
        mu = kmeans_result['mu']

    sigma = np.tile(np.eye(d), (K, 1, 1))  # initial unity covariance
    lambda_ = np.ones(K) / K  # mixing proportion

    lml = []
    while len(lml) < 3 or abs(np.sum(lml[-1] - lml[-3:])) > tol:
        # E-step
        ln_lik = np.zeros((n, K))
        for k in range(K):
            ln_lik[:, k] = lnmvnpdf(X, mu[k, :], sigma[k, :, :])

        ln_lik = np.log(lambda_) + ln_lik  # mixture likelihood

        max_ln_lik = np.max(ln_lik, axis=1, keepdims=True)
        ln_r = ln_lik - np.log(np.sum(np.exp(ln_lik - max_ln_lik), axis=1, keepdims=True)) - max_ln_lik
        r = np.exp(ln_r)  # responsibilities

        Y = np.argmax(r, axis=1) + 1  # label predictions (1-based)

        lml.append(np.sum(np.log(np.sum(np.exp(ln_lik), axis=1))))

        # M-step
        lambda_ = np.mean(r, axis=0)

        Nk = np.sum(r, axis=0)

        mu = (r.T @ X) / Nk[:, np.newaxis]

        for k in range(K):
            diff = X - mu[k, :]
            sigma[k, :, :] = (r[:, k:k+1] * diff).T @ diff / Nk[k]

        # plot if plt on
        if plt != 0:
            # Note: plotting not implemented in Python version
            pass

    # pack classifier
    classifier = {
        'mu': mu,
        'sigma': sigma,
        'lambda': lambda_,
        'K': K,
        'lml': lml
    }

    return classifier, Y
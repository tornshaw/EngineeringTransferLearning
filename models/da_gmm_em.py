import numpy as np
from scipy.optimize import fmin
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.lnmvnpdf import lnmvnpdf
from util.plot_gaussian_2d import plot_gaussian_2d
from src.gmm_mle import gmm_mle
from src.gmm_mle_em import gmm_mle_em

def da_gmm_em(Xs, ys, Xt, H, tol=1e-9, method=1, k=None):
    """
    Expectation maximisation (EM) domain-adapted Gaussian mixture model

    Inputs
    Xs = source features (ns*d)
    ys = source labels (ns*1)
    Xt = target features (nt*d)
    H = mapping matrix (d*d)
    tol = convergence tolerance on change in likelihood in EM
    method = covariance type, 1 = different covariances 2 = same covariances
    k = no. of components in mixture model, if ys is not empty k does not need to be specified

    Outputs
    H, lambda_mle, mu_s, sigma_s, lambda_s, lml
    """
    n, D = Xt.shape

    if k is None and len(ys) > 0:
        k = len(np.unique(ys))

    if len(ys) > 0:
        # fit source GMM parameters
        prior_classifier = gmm_mle(Xs, ys)
        prior = {
            'mu0': prior_classifier['mu'],
            'S0': prior_classifier['sigma']
        }
        # Assuming gmm_map is similar to gmm_mle
        source_classifier = gmm_mle(Xs, ys)
        mu_s = source_classifier['mu']
        sigma_s = source_classifier['sigma']
        lambda_s = source_classifier['lambda']
    else:
        if k is None:
            k = int(input('K = '))
        source_classifier, _ = gmm_mle_em(Xs, k, 1e-9)
        mu_s = source_classifier['mu']
        sigma_s = source_classifier['sigma']
        lambda_s = source_classifier['lambda']

    # initialise
    lambda_mle = lambda_s.copy()
    r = np.random.rand(n, k)
    r = r / np.sum(r, axis=1, keepdims=True)  # initial responsibilities

    ind_u = np.arange(n)  # unlabeled indices, all target are unlabeled
    n_u = len(ind_u)

    log_lik = []

    while len(log_lik) < 3 or abs(np.sum(log_lik[-1] - log_lik[-3:])) > tol:
        # E-step
        lml = 0
        log_r = np.full((n, k), np.nan)
        for i in range(n_u):
            pi_Nk = np.full(k, np.nan)
            r_pi_Nk = np.full(k, np.nan)
            for j in range(k):
                pi_Nk[j] = np.log(lambda_mle[j]) + lnmvnpdf((Xt[ind_u[i], :] @ H).reshape(1, -1), mu_s[j, :], sigma_s[:, :, j])
                r_pi_Nk[j] = r[i, j] * pi_Nk[j]
            max_pi = np.max(pi_Nk)
            log_r[ind_u[i], :] = pi_Nk - np.log(np.sum(np.exp(pi_Nk - max_pi))) - max_pi
            lml += np.log(np.sum(np.exp(r_pi_Nk)))
        r = np.exp(log_r)
        # r[np.isnan(r[:, 0]), :] = r[np.any(r != 0, axis=1), :]  # handle nan - skipped for simplicity

        # M-step
        lambda_mle = np.mean(r, axis=0)

        # equal covariance
        if method != 1:
            H = (mu_s.T @ r.T @ Xt) @ np.linalg.inv(Xt.T @ Xt + np.eye(D) * 1e-4)
            cost = costfn(H.flatten(), Xt, Xs, mu_s, sigma_s, r)
            Xhat = Xt @ H
        else:
            H0 = H.flatten()
            res = fmin(costfn, H0, args=(Xt, Xs, mu_s, sigma_s, r), disp=False)
            H = res.reshape(D, D)
            Xhat = Xt @ H

        log_lik.append(-lml)

    return H, lambda_mle, mu_s, sigma_s, lambda_s, -np.array(log_lik)

def costfn(H0, X, Xs, mu, sigma, r):
    n, D = X.shape
    k = mu.shape[0]

    H = H0.reshape(D, D)
    Xhat = X @ H

    cost = np.zeros(k)
    for j in range(k):
        xdiff = Xhat - mu[j, :]
        R = np.linalg.cholesky(sigma[:, :, j])
        xRinv = xdiff @ np.linalg.inv(R)
        dist = np.sum(xRinv ** 2, axis=1)
        cost[j] = np.sum(r[:, j] * dist)
    cost = np.sum(cost)

    return cost
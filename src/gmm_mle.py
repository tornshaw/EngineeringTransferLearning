import numpy as np

def gmm_mle(X, y, plt=False):
    """
    Supervised Gaussian mixture model using maximum likelihood estimates

    Inputs:
    X: inputs (n, d)
    y: labels (n,)
    plt: plot flag

    Outputs:
    classifier: dict with mu, sigma, lambda, K, classes
    """
    n, d = X.shape
    labs = np.unique(y)
    K = len(labs)
    r = np.array([y == lab for lab in labs]).T.astype(float)  # (n, K)

    rk = np.sum(r, axis=0)  # (K,)
    lambda_ = rk / n

    mu = (r.T @ X) / rk[:, None]  # (K, d)

    sigma = np.full((d, d, K), np.nan)
    for k in range(K):
        sigma[:, :, k] = (1 / rk[k]) * ((r[:, k:k+1] * X).T @ X) - mu[k:k+1].T @ mu[k:k+1]

    # Plot if needed (omitted)

    classifier = {
        'mu': mu,
        'sigma': sigma,
        'lambda': lambda_,
        'K': K,
        'classes': labs
    }
    return classifier
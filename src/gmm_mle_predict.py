import numpy as np
from scipy.stats import multivariate_normal

def gmm_mle_predict(classifier, Xtest, plt=False):
    """
    Predict using GMM

    Inputs:
    classifier: GMM parameters
    Xtest: test data
    plt: plot flag

    Outputs:
    Ytest: predictions
    r: responsibilities
    """
    n, d = Xtest.shape
    K = len(classifier['lambda'])

    ln_lik = np.full((n, K), np.nan)
    for k in range(K):
        ln_lik[:, k] = multivariate_normal.logpdf(Xtest, mean=classifier['mu'][k], cov=classifier['sigma'][k])

    ln_lik += np.log(classifier['lambda'])

    # Log responsibilities
    max_lik = np.max(ln_lik, axis=1, keepdims=True)
    ln_r = ln_lik - np.log(np.sum(np.exp(ln_lik - max_lik), axis=1, keepdims=True)) - max_lik
    r = np.exp(ln_r)

    Ytest = np.argmax(r, axis=1)
    if 'classes' in classifier:
        Ytest = classifier['classes'][Ytest]

    # Plot if needed (omitted for simplicity)

    return Ytest, r
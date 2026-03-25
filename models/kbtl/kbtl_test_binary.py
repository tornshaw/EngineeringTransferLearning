import numpy as np
from scipy.stats import norm
import sys
sys.path.append('../..')
from util.f1score import f1score

def kbtl_test_binary(K, params, y=None):
    """
    Binary kernelised bayesian transfer learning - testing

    Inputs:
    K: list of kernels
    params: hyperparameters
    y: list of labels (optional)

    Outputs:
    pred: predictive variables
    """
    N = [Ki.shape[1] for Ki in K]  # number of points

    # Check for no test data
    no_pts = [n == 0 for n in N]
    if any(no_pts):
        tt = next(i for i, val in enumerate(no_pts) if not val)
        for t in range(tt):
            K[t] = np.full((N[t], N[t]), np.nan)

    # Predict dimensionality reduction subspace
    pred = {}
    pred['H'] = {'mu': [K[t].T @ params['A']['mu'][t] for t in range(len(K))]}

    # Predictive function
    pred['f'] = {
        'mu': [np.hstack([np.ones((N[t], 1)), pred['H']['mu'][t]]) @ params['bw']['mu'] for t in range(len(K))],
        'sig': [1 + np.diag(np.hstack([np.ones((N[t], 1)), pred['H']['mu'][t]]) @ params['bw']['sig'] @ np.hstack([np.ones((N[t], 1)), pred['H']['mu'][t]]).T) for t in range(len(K))]
    }

    # Probability of y = +1
    prob_plus = [norm.cdf((pred['f']['mu'][t] - params['margin']) / np.sqrt(pred['f']['sig'][t])) for t in range(len(K))]
    prob_minus = [1 - norm.cdf((pred['f']['mu'][t] + params['margin']) / np.sqrt(pred['f']['sig'][t])) for t in range(len(K))]
    pred['py'] = [p_plus / (p_plus + p_minus) for p_plus, p_minus in zip(prob_plus, prob_minus)]

    # MAP estimate of labels
    map_minus = [py < 0.5 for py in pred['py']]
    pred['ymap'] = [np.ones(N[t]) - 2 * np.array(map_minus[t]) for t in range(len(K))]

    # Performance metrics
    if y is not None:
        pred['acc'] = [100 * np.sum(pred['ymap'][t] == y[t]) / len(y[t]) for t in range(len(K))]
        pred['f1'] = [f1score(y[t], pred['ymap'][t])[0] for t in range(len(K))]

    return pred
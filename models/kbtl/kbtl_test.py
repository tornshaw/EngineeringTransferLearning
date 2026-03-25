import numpy as np
from scipy.stats import norm
import sys
sys.path.append('../..')
from util.accuracy import accuracy
from util.f1score import f1score

def kbtl_test(K, params, y=None):
    """
    Multi-class kernelised bayesian transfer learning - testing

    Inputs:
    K: list of kernels from each domain
    params: structure of hyperparameters from training
    y: list of class numeric labels for each domain (optional)

    Outputs:
    pred: structure containing predictive variables
        pred.H.mu = mean predictive latent space
        pred.f = mean predictive function (.mu .sig)
        pred.py = probability of +1 class
        pred.ymap = MAP estimate of class label
        if y is known: pred.acc, pred.f1
    """
    # Size of test domains
    T = len(K)  # no. of tasks
    N = [Ki.shape[1] for Ki in K]  # no. of points per task

    if y is not None:
        Ls = [yi.shape[1] if yi.ndim > 1 else 1 for yi in y]
        if all(L == 1 for L in Ls):
            # Convert labels from numerics to N x L matrix of +1 and -1
            all_labels = np.concatenate([yi.flatten() for yi in y])
            labs = np.unique(all_labels)
            L = len(labs)  # number of labels
            Y = [np.ones((N[t], L)) for t in range(T)]  # initialize label matrix

            # Convert to (N x L) matrix of [-1 +1]'s
            for t in range(T):
                for l in range(L):
                    Y[t][y[t].flatten() != labs[l], l] = -1
        else:
            Y = y  # labels in N x L matrix form
            L = Y[0].shape[1]  # number of labels
            labs = np.arange(L)  # assume labels are 0 to L-1
    else:
        L = params['bw']['mu'].shape[1]  # number of classes from training
        labs = np.arange(L)

    # Check if no test data
    no_pts = [np.any(N[t] == 0) for t in range(T)]
    if any(no_pts):
        tt = next((t for t, has_pts in enumerate(no_pts) if not has_pts), T)
        for t in range(tt):
            K[t] = np.full((N[t], N[t]), np.nan)

    # Predict dimensionality reduction subspace
    pred = {
        'H': {
            'mu': [params['A']['mu'][t].T @ K[t] for t in range(T)]
        }
    }

    # Predictive function
    pred['f'] = {
        'mu': [np.vstack([np.ones((1, N[t])), pred['H']['mu'][t]]).T @ params['bw']['mu'] for t in range(T)]
    }

    # Predictive function variance
    pred['f']['sig'] = []
    for t in range(T):
        fsig_t = np.zeros((N[t], L))
        for ll in range(L):
            fsig_t[:, ll] = 1 + np.diag(np.vstack([np.ones((1, N[t])), pred['H']['mu'][t]]).T @
                                       params['bw']['sig'][:, :, ll] @
                                       np.vstack([np.ones((1, N[t])), pred['H']['mu'][t]]))
        pred['f']['sig'].append(fsig_t)

    # Probability of y = +1
    prob_plus = [norm.cdf((pred['f']['mu'][t] - params['margin']) / pred['f']['sig'][t]) for t in range(T)]
    prob_minus = [1 - norm.cdf((pred['f']['mu'][t] - params['margin']) / pred['f']['sig'][t]) for t in range(T)]
    pred['py'] = [prob_plus[t] / (prob_plus[t] + prob_minus[t]) for t in range(T)]

    # MAP estimate
    pred['ymap'] = []
    for t in range(T):
        _, yind = np.max(pred['py'][t], axis=1), np.argmax(pred['py'][t], axis=1)
        pred['ymap'].append(labs[yind])

    # If labels are known, compute accuracy and F1
    if y is not None:
        pred['acc'] = []
        pred['f1'] = []
        for t in range(T):
            if N[t] > 0:
                true_labels = np.argmax(Y[t], axis=1) if Y[t].shape[1] > 1 else y[t].flatten()
                pred_labels = pred['ymap'][t]
                pred['acc'].append(accuracy(true_labels, pred_labels))
                pred['f1'].append(f1score(true_labels, pred_labels))

    return pred
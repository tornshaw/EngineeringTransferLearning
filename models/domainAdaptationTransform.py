import numpy as np

def domainAdaptationTransform(Xtest, Xs, Xt, W, kern, hyp):
    """
    Transforms inputs based on learnt weights and kernel

    Inputs:
    Xtest: new data
    Xs, Xt: training data
    W: weights
    kern: kernel function
    hyp: hyperparameters

    Output:
    Z: transformed data
    """
    X = np.vstack([Xs, Xt])
    K = kern(hyp, Xtest, X)
    Z = K @ W
    return Z
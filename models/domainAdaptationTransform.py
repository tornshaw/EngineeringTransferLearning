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
    K_out = kern(hyp, Xtest, X)
    K = K_out[0] if isinstance(K_out, tuple) else K_out
    Z = K @ W
    return Z

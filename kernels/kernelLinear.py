import numpy as np

def kernelLinear(hyp, xp, xq):
    """
    Linear kernel

    Inputs:
    hyp: unused
    xp, xq: data

    Output:
    K: kernel matrix
    """
    return xp @ xq.T
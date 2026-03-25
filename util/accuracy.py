import numpy as np

def accuracy(yp, yt):
    """
    Accuracy of classifier

    Inputs:
    yp: predicted labels
    yt: true labels

    Output:
    acc: accuracy (percentage)
    """
    return 100 * np.sum(yp == yt) / len(yt)
from sklearn.metrics import f1_score
import numpy as np

def f1score(yt, yp):
    """
    Macro F1 score of classifier

    Inputs:
    yt: true labels
    yp: predicted labels

    Outputs:
    f1: macro F1 score
    f1_c: F1 per class
    """
    classes = np.unique(yt)
    f1_c = []
    for c in classes:
        tp = np.sum((yp == c) & (yt == c))
        fn = np.sum((yp != c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_class = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_c.append(f1_class)
    f1_c = np.array(f1_c)
    f1 = np.mean(f1_c)
    return f1, f1_c
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.gmm_mle import gmm_mle
from src.gmm_mle_predict import gmm_mle_predict

def classifierGMM(Zs, Ys, Zt, classifier=None):
    """
    (Supervised) Gaussian mixture model (maximum likelihood estimates)

    Inputs
    Zs = source data
    Ys = source labels
    Zt = target data
    classifier = pretrained classifier

    Outputs
    Ytp = target label predictions
    classifier = trained classifier
    """
    if classifier is None:
        # Train supervised GMM Classifier
        classifier = gmm_mle(Zs, Ys)

    # Predict
    Ytp, _ = gmm_mle_predict(classifier, Zt)

    return Ytp, classifier

    return Ytp, classifier
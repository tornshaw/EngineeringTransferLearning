from sklearn.mixture import GaussianMixture
import numpy as np

def classifierGMM_EM(Zs, ks, Zt, classifier=None):
    """
    (Unsupervised) EM Gaussian mixture model

    Inputs:
    Zs: source data
    ks: number of components
    Zt: target data
    classifier: pretrained classifier

    Outputs:
    Ytp: target predictions
    classifier: trained classifier
    """
    if classifier is None:
        classifier = GaussianMixture(n_components=ks, random_state=0)
        classifier.fit(Zs)
    Ytp = classifier.predict(Zt)
    return Ytp, classifier
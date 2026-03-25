from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

def classifierKMeans(Zs, ks, Zt, classifier=None):
    """
    Unsupervised k-means classifier

    Inputs:
    Zs: source data
    ks: number of clusters
    Zt: target data
    classifier: pretrained classifier

    Outputs:
    Ytp: target predictions
    classifier: trained classifier
    """
    if classifier is None:
        kmeans = KMeans(n_clusters=ks, random_state=0)
        Ys = kmeans.fit_predict(Zs)
        classifier = {'mu': kmeans.cluster_centers_}

    # Predict
    D = cdist(Zt, classifier['mu'])
    Ytp = np.argmin(D, axis=1) + 1  # MATLAB labels from 1

    return Ytp, classifier
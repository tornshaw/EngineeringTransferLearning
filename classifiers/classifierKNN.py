import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def classifierKNN(Zs, Ys, Zt, classifier=None):
    """
    KNN with number of neighbours equal to the number of classes - 1

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
        noNeigh = int(np.max(Ys) - 1)
        # Train KNN
        classifier = KNeighborsClassifier(n_neighbors=noNeigh)
        classifier.fit(Zs, Ys)

    # Predict
    Ytp = classifier.predict(Zt)

    return Ytp, classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def classifierKNN_cv(Zs, Ys, Zt, classifier=None):
    """
    KNN with cross-validated number of neighbors

    Inputs:
    Zs: source data
    Ys: source labels
    Zt: target data
    classifier: pretrained classifier

    Outputs:
    Ytp: target predictions
    classifier: trained classifier
    """
    if classifier is None:
        param_grid = {'n_neighbors': np.arange(1, min(20, len(np.unique(Ys))) + 1)}
        knn = KNeighborsClassifier()
        classifier = GridSearchCV(knn, param_grid, cv=5)
        classifier.fit(Zs, Ys)

    Ytp = classifier.predict(Zt)
    return Ytp, classifier
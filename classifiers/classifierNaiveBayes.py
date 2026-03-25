from sklearn.naive_bayes import GaussianNB
import numpy as np

def classifierNaiveBayes(Zs, Ys, Zt, classifier=None):
    """
    Naive Bayes classifier

    Inputs:
    Zs: source data
    Ys: source labels
    Zt: target data
    classifier: pretrained classifier

    Outputs:
    Ytp: target predictions
    classifier: trained classifier
    posteriors: posterior probabilities
    """
    if classifier is None:
        classifier = GaussianNB()
        classifier.fit(Zs, Ys)

    Ytp = classifier.predict(Zt)
    posteriors = classifier.predict_proba(Zt)

    return Ytp, classifier, posteriors
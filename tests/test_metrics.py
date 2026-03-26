import numpy as np

from util.accuracy import accuracy
from util.f1score import f1score


def test_accuracy_returns_percentage():
    yt = np.array([1, 1, 2, 2, 3])
    yp = np.array([1, 2, 2, 2, 3])
    assert accuracy(yp, yt) == 80.0


def test_f1score_macro_and_per_class():
    yt = np.array([1, 1, 2, 2, 3, 3])
    yp = np.array([1, 2, 2, 2, 3, 1])

    macro_f1, class_f1 = f1score(yt, yp)

    expected_class_f1 = np.array([0.5, 0.8, 2 / 3])
    assert np.allclose(class_f1, expected_class_f1)
    assert np.isclose(macro_f1, expected_class_f1.mean())
    assert isinstance(macro_f1, float)

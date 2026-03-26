import numpy as np

from kernels.kernelRBF import kernelRBF


def test_kernel_rbf_symmetric_and_diagonal_one():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    k, hyp = kernelRBF(np.nan, x, x)

    assert hyp > 0
    assert np.allclose(k, k.T)
    assert np.allclose(np.diag(k), np.ones(x.shape[0]))


def test_kernel_rbf_with_explicit_hyperparameter_matches_expected():
    xp = np.array([[0.0], [1.0]])
    xq = np.array([[0.0], [2.0]])

    k, hyp = kernelRBF(1.0, xp, xq)

    expected = np.exp(-np.array([[0.0, 2.0], [0.5, 0.5]]))
    assert hyp == 1.0
    assert np.allclose(k, expected)

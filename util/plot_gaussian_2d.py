import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian_2d(mu, sigma, npts=50, stds=2):
    """
    Plots k-Gaussian distributions given mean and covariances in 2D

    Inputs:
    mu: k-means (k, d)
    sigma: k-covariances (d, d, k)
    npts: number of points on plot
    stds: number of standard deviations

    Outputs:
    X: x-coordinates (k, npts+1)
    Y: y-coordinates (k, npts+1)
    """
    k = mu.shape[0]
    theta = 2 * np.pi / npts * np.arange(npts + 1)

    plt.plot(mu[:, 0], mu[:, 1], 'k+')
    X = np.full((k, npts + 1), np.nan)
    Y = np.full((k, npts + 1), np.nan)

    for i in range(k):
        eigvals, eigvecs = np.linalg.eigh(sigma[:, :, i])
        alpha = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        xy = stds * np.sqrt(eigvals)[:, None] * np.array([np.cos(theta), np.sin(theta)])
        rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        ellip = rotation @ xy

        X[i, :] = mu[i, 0] + ellip[0, :]
        Y[i, :] = mu[i, 1] + ellip[1, :]

        plt.plot(X[i, :], Y[i, :], 'k-')

    plt.legend([str(i+1) for i in range(k)])
    plt.show()

    return X, Y
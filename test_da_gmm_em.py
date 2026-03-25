import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.da_gmm_em import da_gmm_em
from util.accuracy import accuracy
from util.f1score import f1score

def test_da_gmm_em():
    """
    Test the DA-GMM-EM algorithm with synthetic data
    """
    print("Testing DA-GMM-EM algorithm...")

    # Generate synthetic data
    np.random.seed(42)  # for reproducibility

    # Source domain: two classes
    n_s = 100
    mu_s1 = np.array([0, 0])
    mu_s2 = np.array([2, 2])
    sigma_s = np.eye(2) * 0.5

    X_s1 = np.random.multivariate_normal(mu_s1, sigma_s, n_s//2)
    X_s2 = np.random.multivariate_normal(mu_s2, sigma_s, n_s//2)
    Xs = np.vstack([X_s1, X_s2])
    ys = np.hstack([np.ones(n_s//2), np.ones(n_s//2) * 2]).astype(int)

    # Target domain: shifted and scaled
    n_t = 80
    mu_t1 = np.array([1, -1])
    mu_t2 = np.array([3, 1])
    sigma_t = np.eye(2) * 0.8

    X_t1 = np.random.multivariate_normal(mu_t1, sigma_t, n_t//2)
    X_t2 = np.random.multivariate_normal(mu_t2, sigma_t, n_t//2)
    Xt = np.vstack([X_t1, X_t2])
    yt = np.hstack([np.ones(n_t//2), np.ones(n_t//2) * 2]).astype(int)

    print(f"Source data shape: {Xs.shape}, labels: {np.unique(ys)}")
    print(f"Target data shape: {Xt.shape}, labels: {np.unique(yt)}")

    # Initial transformation matrix (identity)
    H_init = np.eye(2)

    # Run DA-GMM-EM
    print("\nRunning DA-GMM-EM...")
    H_opt, lambda_mle, mu_s_opt, sigma_s_opt, lambda_s_opt, lml = da_gmm_em(
        Xs, ys, Xt, H_init, tol=1e-6, method=1
    )

    print("\nOptimized transformation matrix H:")
    print(H_opt)
    print(f"\nMixing proportions: {lambda_mle}")
    print(f"\nSource GMM means:\n{mu_s_opt}")
    print(f"\nLog marginal likelihood: {lml[-1]:.4f} (final)")
    print(f"Convergence: {len(lml)} iterations")

    # Apply transformation to target data
    Xt_transformed = Xt @ H_opt

    print(f"\nTransformed target data shape: {Xt_transformed.shape}")
    print("Sample transformed points:")
    print(Xt_transformed[:5])

    print("\nDA-GMM-EM test completed successfully!")
    print("The algorithm learned a transformation matrix to adapt the target domain to the source domain.")

    return {
        'H': H_opt,
        'lambda_mle': lambda_mle,
        'mu_s': mu_s_opt,
        'sigma_s': sigma_s_opt,
        'lml': lml
    }

if __name__ == "__main__":
    results = test_da_gmm_em()
    print("\nTest completed successfully!")
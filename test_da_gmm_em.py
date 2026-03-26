import numpy as np

from models.da_gmm_em import da_gmm_em


def test_da_gmm_em_learns_valid_mapping_and_improves_likelihood():
    """DA-GMM-EM should return finite parameters and a improved log-likelihood and stable output."""
    rng = np.random.default_rng(42)

    n_s = 100
    mu_s1 = np.array([0.0, 0.0])
    mu_s2 = np.array([2.0, 2.0])
    sigma_s = np.eye(2) * 0.5

    x_s1 = rng.multivariate_normal(mu_s1, sigma_s, n_s // 2)
    x_s2 = rng.multivariate_normal(mu_s2, sigma_s, n_s // 2)
    xs = np.vstack([x_s1, x_s2])
    ys = np.hstack([np.ones(n_s // 2), np.ones(n_s // 2) * 2]).astype(int)

    n_t = 80
    mu_t1 = np.array([1.0, -1.0])
    mu_t2 = np.array([3.0, 1.0])
    sigma_t = np.eye(2) * 0.8

    x_t1 = rng.multivariate_normal(mu_t1, sigma_t, n_t // 2)
    x_t2 = rng.multivariate_normal(mu_t2, sigma_t, n_t // 2)
    xt = np.vstack([x_t1, x_t2])

    h_init = np.eye(2)

    h_opt, lambda_mle, mu_s_opt, sigma_s_opt, lambda_s_opt, lml = da_gmm_em(
        xs, ys, xt, h_init, tol=1e-6, method=1
    )

    assert h_opt.shape == (2, 2)
    assert np.isfinite(h_opt).all()

    assert lambda_mle.shape == (2,)
    assert np.isclose(np.sum(lambda_mle), 1.0)
    assert np.all(lambda_mle > 0)

    assert mu_s_opt.shape == (2, 2)
    assert sigma_s_opt.shape == (2, 2, 2)
    assert lambda_s_opt.shape == (2,)

    assert len(lml) >= 3
    assert np.isfinite(lml).all()
    assert lml[-1] > lml[0]
    assert np.max(lml) >= lml[-1] - 1e-6

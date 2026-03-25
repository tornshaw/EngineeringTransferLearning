import numpy as np
from scipy.stats import norm
from scipy.linalg import inv
import sys
sys.path.append('../..')

def kbtl_train_binary(K, y, params):
    """
    Binary kernelised bayesian transfer learning - training

    Inputs:
    K: list of kernels from each domain (T x N_t x N_t)
    y: list of binary labels ([-1 or 1]) for each domain
    params: structure of hyperparameters

    Outputs:
    params: updated hyperparameters
    """
    # Check labels
    for yi in y:
        unique_labs = np.unique(yi)
        if not np.array_equal(unique_labs, [-1, 1]):
            raise ValueError('Labels must be -1 or 1')

    T = len(K)  # number of tasks
    N = [Ki.shape[0] for Ki in K]  # number of points per task

    R = params['R']
    Hsig2 = params['Hsigma2']

    # Initialize Lambda
    Lambda = {
        'kappa': [params['lambda']['kappa'] + 0.5 * np.ones((n, R)) for n in N],
        'theta': [params['lambda']['theta'] * np.ones((n, R)) for n in N]
    }

    # Initialize A and H
    A = {
        'mu': [np.random.randn(n, R) for n in N],
        'sig_diag': [np.ones((n, R)) for n in N],
        'sig': [np.tile(np.eye(n), (R, 1, 1)).transpose(1, 2, 0) for n in N]
    }
    H = {
        'mu': [np.random.randn(R, n) for n in N],
        'sig': [np.eye(R) for _ in range(T)]
    }

    # Initialize classifier part
    gam = {
        'kappa': params['gamma']['kappa'] + 0.5,
        'theta': params['gamma']['theta']
    }
    eta = {
        'kappa': (params['eta']['kappa'] + 0.5) * np.ones(R),
        'theta': params['eta']['theta'] * np.ones(R)
    }

    # Weights and bias
    bw = {
        'mu': np.concatenate([[0], np.random.randn(R)]),
        'sig': np.eye(R + 1)
    }

    # Initialize discriminative function
    f = {
        'mu': [np.abs(np.random.randn(n)) + params['margin'] * np.sign(yi) for yi, n in zip(y, N)],
        'sig': [np.ones(n) for n in N]
    }

    # Get cross product K*K'
    KKt = [Ki @ Ki.T for Ki in K]

    # Margins
    labs = [[yi < 0, yi > 0] for yi in y]
    margin_plus = [[-1e40 if lab[0][i] else params['margin'] for i in range(n)] for lab, n in zip(labs, N)]
    margin_minus = [[-params['margin'] if lab[1][i] else 1e40 for i in range(n)] for lab, n in zip(labs, N)]
    lower_margin = [np.array([mp[i] for i in range(n)]) for mp, n in zip(margin_plus, N)]
    upper_margin = [np.array([mm[i] for i in range(n)]) for mm, n in zip(margin_minus, N)]

    # Variational inference
    for i in range(params['iter']):
        # Update dimensionality reduction part

        # Update Lambda theta
        Lambda['theta'] = [1 / (1 / params['lambda']['theta'] + 0.5 * (A['mu'][t]**2 + A['sig_diag'][t])) for t in range(T)]

        # Update A
        for t in range(T):
            for s in range(R):
                A['sig'][t][:, :, s] = inv(np.diag(Lambda['kappa'][t][:, s] * Lambda['theta'][t][:, s]) + KKt[t] / Hsig2)
                A['sig_diag'][t][:, s] = np.diag(A['sig'][t][:, :, s])
                A['mu'][t][:, s] = A['sig'][t][:, :, s] @ (K[t] @ H['mu'][t][s, :].T / Hsig2)

        # Update H
        H['sig'] = [inv(np.eye(R) / Hsig2 + bw['mu'][1:] @ bw['mu'][1:].T + bw['sig'][1:, 1:]) for _ in range(T)]
        H['mu'] = [H['sig'][t] @ (A['mu'][t].T @ K[t] / Hsig2 + (bw['mu'][1:, None] @ f['mu'][t][None, :]) - np.tile(bw['mu'][1:] * bw['mu'][0] + bw['sig'][1:, 0], (N[t], 1)).T) for t in range(T)]

        # Update classifier part

        # Update gamma theta
        gam['theta'] = 1 / (1 / params['gamma']['theta'] + 0.5 * (bw['mu'][0]**2 + bw['sig'][0, 0]))

        # Update eta theta
        eta['theta'] = 1 / (1 / params['eta']['theta'] + 0.5 * (bw['mu'][1:]**2 + np.diag(bw['sig'][1:, 1:])))

        # Update bias and weights
        bw['sig'] = np.block([[gam['kappa'] * gam['theta'], np.zeros((1, R))], [np.zeros((R, 1)), np.diag(eta['kappa'] * eta['theta'])]])
        bwsig = [np.block([[np.array([[N[t]]]), np.sum(H['mu'][t], axis=1).reshape(1, -1)],
                           [np.sum(H['mu'][t], axis=1).reshape(-1, 1), H['mu'][t] @ H['mu'][t].T + N[t] * H['sig'][t]]]) for t in range(T)]
        bw['sig'] = inv(bw['sig'] + np.sum(np.array(bwsig), axis=0))
        bwmu = [np.vstack([np.ones(N[t]), H['mu'][t]]) @ f['mu'][t] for t in range(T)]
        bw['mu'] = bw['sig'] @ np.sum(np.array(bwmu), axis=0)

        # Update f
        q_f_mu = [np.hstack([np.ones((N[t], 1)), H['mu'][t].T]) @ bw['mu'] for t in range(T)]
        alpha_tn = [lower_margin[t] - q_f_mu[t] for t in range(T)]
        beta_tn = [upper_margin[t] - q_f_mu[t] for t in range(T)]
        Z = [norm.cdf(beta_tn[t]) - norm.cdf(alpha_tn[t]) for t in range(T)]
        Z = [z + (z == 0) for z in Z]
        f['mu'] = [q_f_mu[t] + (norm.pdf(alpha_tn[t]) - norm.pdf(beta_tn[t])) / Z[t] for t in range(T)]
        f['sig'] = [1 + (alpha_tn[t] * norm.pdf(alpha_tn[t]) - beta_tn[t] * norm.pdf(beta_tn[t])) / Z[t] - ((norm.pdf(alpha_tn[t]) - norm.pdf(beta_tn[t])) / Z[t])**2 for t in range(T)]

        if i % 10 == 0:
            print(f'Iteration: {i}')

    # Pack parameters
    params['Lambda'] = Lambda
    params['A'] = A
    params['gamma'] = gam
    params['eta'] = eta
    params['bw'] = bw

    return params
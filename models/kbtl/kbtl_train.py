import numpy as np
from scipy.stats import norm
from scipy.linalg import inv
import sys
sys.path.append('../..')

def kbtl_train(K, y, params):
    """
    Multi-class kernelised bayesian transfer learning - training

    Inputs:
    K: list of kernels from each domain (T elements, each N_t x N_t)
    y: list of class numeric labels for each domain (converted to [-1 +1] N x L matrix)
        or list of labels in N x L matrix form
    params: structure of hyperparameters
        params.lambda.kappa = shape hyperparameter of gamma prior for projection matrices
        params.lambda.theta = shape hyperparameter of gamma prior for projection matrices
        params.gamma.kappa = shape hyperparameter of gamma prior for bias
        params.gamma.theta = shape hyperparameter of gamma prior for bias
        params.eta.kappa = shape hyperparameter of gamma prior for weights
        params.eta.theta = shape hyperparameter of gamma prior for weights
        params.iter = number of iterations
        params.margin = size of margin between classes
        params.R = latent subspace dimensionality
        params.Hsigma2 = variance of latent subspace

    Outputs:
    params = structure of updated hyperparameters
        params.Lambda = inferred projection matrix hyperpriors (.theta .kappa)
        params.A = inferred projection matrix distribution (.mu .sig .sig_diag)
        params.eta = inferred weight hyperpriors (.theta .kappa)
        params.bw = inferred bias and weight distributions (.mu .sig)
    Y = converted labels in N x L matrix form
    """
    # Sizes of domains
    T = len(K)  # no. of tasks
    N = [Ki.shape[0] for Ki in K]  # no. of points per task

    # Check if labels are in numeric form or matrix form
    Ls = [yi.shape[1] if yi.ndim > 1 else 1 for yi in y]
    if all(L == 1 for L in Ls):
        # Convert labels from numerics to N x L matrix of +1 and -1
        all_labels = np.concatenate([yi.flatten() for yi in y])
        labs = np.unique(all_labels)
        L = len(labs)  # number of labels
        Y = [np.ones((n, L)) for n in N]  # initialize label matrix

        # Convert to (N x L) matrix of [-1 +1]'s
        for t in range(T):
            for l in range(L):
                Y[t][y[t].flatten() != labs[l], l] = -1
    else:
        Y = y  # labels in N x L matrix form
        L = Y[0].shape[1]  # number of labels

    # Unpack parameters
    R = params['R']  # dimension size of H
    Hsig2 = params['Hsigma2']  # prior variance of H

    # Initialize dimensionality reduction part
    Lambda = {
        'kappa': [(params['lambda']['kappa'] + 0.5) * np.ones((n, R)) for n in N],
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
        'kappa': (params['gamma']['kappa'] + 0.5) * np.ones(L),
        'theta': params['gamma']['theta'] * np.ones(L)
    }
    eta = {
        'kappa': (params['eta']['kappa'] + 0.5) * np.ones((R, L)),
        'theta': params['eta']['theta'] * np.ones((R, L))
    }

    # Weights and bias
    bw = {
        'mu': np.vstack([np.zeros((1, L)), np.random.randn(R, L)]),
        'sig': np.tile(np.eye(R + 1), (L, 1, 1)).transpose(1, 2, 0)
    }

    # Initialize discriminative function
    f = {
        'mu': [(np.abs(np.random.randn(n, L)) + params['margin']) * np.sign(Y[t]) for t, n in enumerate(N)],
        'sig': [np.ones((n, L)) for n in N]
    }

    # Get cross product K*K'
    KKt = [Ki @ Ki.T for Ki in K]

    # Margins
    labs_neg = [Y[t] < 0 for t in range(T)]
    labs_pos = [Y[t] > 0 for t in range(T)]
    lower_margin = [-1e40 * labs_neg[t] + params['margin'] * labs_pos[t] for t in range(T)]
    upper_margin = [1e40 * labs_pos[t] - params['margin'] * labs_neg[t] for t in range(T)]

    # Variational inference steps
    for i in range(params['iter']):
        # Update dimensionality reduction part

        # Update lambda theta - hyperparameters of projection prior
        Lambda['theta'] = [1 / (1 / params['lambda']['theta'] + 0.5 * (A['mu'][t]**2 + A['sig_diag'][t])) for t in range(T)]

        # Update A - projection matrix
        for t in range(T):
            for s in range(R):
                A['sig'][t][:, :, s] = inv(np.diag(Lambda['kappa'][t][:, s] * Lambda['theta'][t][:, s]) + KKt[t] / Hsig2)
                A['sig_diag'][t][:, s] = np.diag(A['sig'][t][:, :, s])
                A['mu'][t][:, s] = A['sig'][t][:, :, s] @ (K[t] @ H['mu'][t][s, :].T / Hsig2)

        # Update H - latent subspace
        H['sig'] = [inv(np.eye(R) / Hsig2 + bw['mu'][1:, :] @ bw['mu'][1:, :].T + np.sum(bw['sig'][1:, 1:, :], axis=2)) for _ in range(T)]
        H['mu'] = [A['mu'][t].T @ K[t] / Hsig2 for t in range(T)]
        for ll in range(L):
            H['mu'] = [H['mu'][t] + bw['mu'][1:, ll:ll+1] @ f['mu'][t][:, ll:ll+1].T -
                      np.tile(bw['mu'][1:, ll] * bw['mu'][0, ll] + bw['sig'][1:, 0, ll], (N[t], 1)).T
                      for t in range(T)]
        H['mu'] = [H['sig'][t] @ H['mu'][t] for t in range(T)]

        # Update classifier part
        for ll in range(L):
            # Update gamma theta - hyperparameter of bias prior
            gam['theta'][ll] = 1 / (1 / params['gamma']['theta'] + 0.5 * (bw['mu'][0, ll]**2 + bw['sig'][0, 0, ll]))

            # Update eta theta - hyperparameters of weight prior
            eta['theta'][:, ll] = 1 / (1 / params['eta']['theta'] + 0.5 * (bw['mu'][1:, ll]**2 + np.diag(bw['sig'][1:, 1:, ll])))

            # Update bias and weights
            # Bias and weight covariance
            bw['sig'][:, :, ll] = np.block([[gam['kappa'][ll] * gam['theta'][ll], np.zeros((1, R))],
                                           [np.zeros((R, 1)), np.diag(eta['kappa'][:, ll] * eta['theta'][:, ll])]])

            bwsig = [np.block([[np.array([[N[t]]]), np.sum(H['mu'][t], axis=1).reshape(1, -1)],
                               [np.sum(H['mu'][t], axis=1).reshape(-1, 1), H['mu'][t] @ H['mu'][t].T + N[t] * H['sig'][t]]])
                    for t in range(T)]
            bw['sig'][:, :, ll] = inv(bw['sig'][:, :, ll] + np.sum(np.array(bwsig), axis=0))

            # Bias and weight mean
            bwmu = [np.vstack([np.ones((1, N[t])), H['mu'][t]]) @ f['mu'][t][:, ll] for t in range(T)]
            bw['mu'][:, ll] = bw['sig'][:, :, ll] @ np.sum(np.array(bwmu), axis=0)

        # Update f
        q_f_mu = [np.hstack([np.ones((N[t], 1)), H['mu'][t].T]) @ bw['mu'] for t in range(T)]
        alpha_tn = [lower_margin[t] - q_f_mu[t] for t in range(T)]
        beta_tn = [upper_margin[t] - q_f_mu[t] for t in range(T)]
        Z = [norm.cdf(beta_tn[t]) - norm.cdf(alpha_tn[t]) for t in range(T)]
        Z = [z + (z == 0) for z in Z]
        f['mu'] = [q_f_mu[t] + (norm.pdf(alpha_tn[t]) - norm.pdf(beta_tn[t])) / Z[t] for t in range(T)]
        f['sig'] = [1 + (alpha_tn[t] * norm.pdf(alpha_tn[t]) - beta_tn[t] * norm.pdf(beta_tn[t])) / Z[t] -
                   ((norm.pdf(alpha_tn[t]) - norm.pdf(beta_tn[t])) / Z[t])**2 for t in range(T)]

        if i % 10 == 0:
            print(f'Iteration: {i+1}')

    # Pack parameters
    params['Lambda'] = Lambda
    params['A'] = A
    params['gamma'] = gam
    params['eta'] = eta
    params['bw'] = bw

    return params, Y
    eta = {
        'kappa': (params['eta']['kappa'] + 0.5) * np.ones((R, L)),
        'theta': params['eta']['theta'] * np.ones((R, L))
    }

    # Weights and bias
    bw = {
        'mu': np.vstack([np.zeros(L), np.random.randn(R, L)]),
        'sig': np.tile(np.eye(R + 1), (L, 1, 1)).transpose(2, 0, 1)
    }

    # Initialize f
    f = {
        'mu': [np.abs(np.random.randn(n, L)) + params['margin'] * np.sign(Y[t]) for t, n in enumerate(N)],
        'sig': [np.ones((n, L)) for n in N]
    }

    # KKt
    KKt = [Ki @ Ki.T for Ki in K]

    # Margins
    labs_neg = [Y[t] < 0 for t in range(T)]
    labs_pos = [Y[t] > 0 for t in range(T)]
    lower_margin = [-1e40 * labs_neg[t] + params['margin'] * labs_pos[t] for t in range(T)]
    upper_margin = [params['margin'] * labs_neg[t] + 1e40 * labs_pos[t] for t in range(T)]

    # VI loop (simplified, similar to binary)
    for i in range(params['iter']):
        # Update similar to binary, but for each class
        # This is complex; for brevity, assume similar structure
        pass  # Full implementation needed

    params['Lambda'] = Lambda
    params['A'] = A
    params['gamma'] = gam
    params['eta'] = eta
    params['bw'] = bw

    return params, Y

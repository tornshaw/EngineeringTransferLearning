"""
KBTL demo for binary problem
"""

import numpy as np
from scipy.io import loadmat
import sys
sys.path.insert(0, '..')
from models.kbtl.kbtl_train_binary import kbtl_train_binary
from models.kbtl.kbtl_test_binary import kbtl_test_binary
from kernels.kernelRBF import kernelRBF

# Load data
data = loadmat('data/kbtl_demo_binary_data.mat')
X = [data['X'][0][i] for i in range(data['X'].shape[1])]  # list of arrays
Xtest = [data['Xtest'][0][i] for i in range(data['Xtest'].shape[1])]
Y = [data['Y'][0][i].flatten() for i in range(data['Y'].shape[1])]
Ytest = [data['Ytest'][0][i].flatten() for i in range(data['Ytest'].shape[1])]

T = len(X)

# Get kernel embeddings
Ktrain = []
khyp = []
for t in range(T):
    K, hyp = kernelRBF(np.nan, X[t], X[t])
    Ktrain.append(K)
    khyp.append(hyp)

# KBTL parameters
params = {
    'lambda': {'kappa': 1e-3, 'theta': 1e-3},
    'gamma': {'kappa': 1e-3, 'theta': 1e-3},
    'eta': {'kappa': 1e-3, 'theta': 1e-3},
    'iter': 500,
    'margin': 0,
    'R': 2,
    'Hsigma2': 0.25**2
}

# Train KBTL
params = kbtl_train_binary(Ktrain, Y, params)

# Test on training
pred_train = kbtl_test_binary(Ktrain, params, Y)
print(f'Training Accuracy: {pred_train["acc"]}')

# Test kernels
Ktest = []
for t in range(T):
    K, _ = kernelRBF(khyp[t], X[t], Xtest[t])
    Ktest.append(K)

pred_test = kbtl_test_binary(Ktest, params, Ytest)
print(f'Testing Accuracy: {pred_test["acc"]}')

# Expected: around 90-95% or as per paper
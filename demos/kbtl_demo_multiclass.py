"""
Multi-class Kernelised Bayesian Transfer Learning Demo

This script runs the multi-class problem from:
"On the application of kernelised Bayesian transfer learning to
population-based structural health monitoring"

Paul Gardner, University of Sheffield 2021

Note: Results might vary slightly from the paper due to different random
seeds in the random initialization of KBTL
"""

import numpy as np
import scipy.io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.kbtl.kbtl_train import kbtl_train
from models.kbtl.kbtl_test import kbtl_test
from kernels.kernelRBF import kernelRBF

def main():
    print("KBTL Multi-class Demo")
    print("=" * 50)

    # Load data
    try:
        data = scipy.io.loadmat(ROOT / 'data' / 'kbtl_demo_multiclass_data.mat')
        X = data['X'][0]  # Training data
        Y = data['Y'][0]  # Training labels
        Xtest = data['Xtest'][0]  # Test data
        Ytest = data['Ytest'][0]  # Test labels
    except:
        print("Data file not found. Creating synthetic multi-class data...")

        # Create synthetic multi-class data
        np.random.seed(42)
        T = 3  # number of tasks
        n_train = 50
        n_test = 30
        n_classes = 3
        d = 4  # feature dimension

        X = []
        Y = []
        Xtest = []
        Ytest = []

        for t in range(T):
            # Training data
            X_t = np.random.randn(n_train, d) + np.random.randn(1, d) * 2
            y_t = np.random.randint(0, n_classes, n_train)
            X.append(X_t)
            Y.append(y_t)

            # Test data
            Xtest_t = np.random.randn(n_test, d) + np.random.randn(1, d) * 2
            ytest_t = np.random.randint(0, n_classes, n_test)
            Xtest.append(Xtest_t)
            Ytest.append(ytest_t)

    # KBTL hyperprior parameters
    params = {
        # Shape and scale hyperparameters of gamma prior for projection matrices
        'lambda': {'kappa': 1e-3, 'theta': 1e-3},
        # Shape and scale hyperparameters of gamma prior for bias
        'gamma': {'kappa': 1e-3, 'theta': 1e-3},
        # Shape and scale hyperparameters of gamma prior for weights
        'eta': {'kappa': 1e-3, 'theta': 1e-3},
        # No. of iterations
        'iter': 500,  # Reduced from paper
        # Margin
        'margin': 1,
        # Latent subspace dimensionality
        'R': 2,
        # Variance of latent subspace
        'Hsigma2': 6**2
    }

    print(f"Number of tasks: {len(X)}")
    print(f"Training samples per task: {[len(x) for x in X]}")
    print(f"Test samples per task: {[len(x) for x in Xtest]}")
    print(f"Number of classes: {len(np.unique(np.concatenate(Y)))}")
    print(f"Latent subspace dimension: {params['R']}")
    print(f"Training iterations: {params['iter']}")

    # Train KBTL
    print("\nTraining KBTL...")
    T = len(X)

    # Get kernel embeddings
    Ktrain = []  # training kernels
    khyp = []  # kernel hyperparameters
    for t in range(T):
        K_t, hyp_t = kernelRBF(np.nan, X[t], X[t])  # kernel embedding with median heuristic
        Ktrain.append(K_t)
        khyp.append(hyp_t)

    hyp, Y_converted = kbtl_train(Ktrain, Y, params)  # train kbtl
    pred_train = kbtl_test(Ktrain, hyp, Y)  # predict training data

    print("\nTraining Results:")
    print(f"Training accuracies: {pred_train['acc']}")
    print(f"Mean training accuracy: {np.mean(pred_train['acc']):.4f}")

    # KBTL Prediction
    print("\nTesting KBTL...")

    # Get kernel embeddings for test
    Ktest = []  # test kernels
    for t in range(T):
        Ktest_t, _ = kernelRBF(khyp[t], X[t], Xtest[t])  # kernel embedding
        Ktest.append(Ktest_t)

    pred_test = kbtl_test(Ktest, hyp, Ytest)  # predict testing data

    print("\nTesting Results:")
    print(f"Test accuracies: {pred_test['acc']}")
    print(f"Mean test accuracy: {np.mean(pred_test['acc']):.4f}")

    print("\nDemo completed successfully!")
    return hyp, pred_train, pred_test

if __name__ == "__main__":
    main()

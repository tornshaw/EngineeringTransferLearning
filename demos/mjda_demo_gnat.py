"""
MJDA demo for Gnat repair problem
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mjda import mjda
from classifiers.classifierKNN_cv import classifierKNN_cv
from kernels.kernelRBF import kernelRBF
from util.accuracy import accuracy
from util.f1score import f1score

np.random.seed(0)

# Load data
data = loadmat(ROOT / 'data' / 'gnat_repair.mat')
Xs = data['Xs']
Ys = data['Ys'].flatten()
Xt = data['Xt']
Yt = data['Yt'].flatten()

# Training and testing splits
N = len(Xs)
ind_rand = np.random.permutation(N)
ntr = 500
ntr_ind = ind_rand[:ntr]
ntst_ind = ind_rand[ntr:]

Xs_tr = Xs[ntr_ind]
Ys_tr = Ys[ntr_ind]
Xs_tst = Xs[ntst_ind]
Ys_tst = Ys[ntst_ind]

Xt_tr = Xt[ntr_ind]
Yt_tr = Yt[ntr_ind]
Xt_tst = Xt[ntst_ind]
Yt_tst = Yt[ntst_ind]

# Normalize
scaler_s = StandardScaler()
Xs_tr = scaler_s.fit_transform(Xs_tr)
Xs_tst = scaler_s.transform(Xs_tst)

scaler_t = StandardScaler()
Xt_tr = scaler_t.fit_transform(Xt_tr)
Xt_tst = scaler_t.transform(Xt_tst)

# MJDA (aligned with MATLAB demo defaults)
Zs, Zt, Ytp, W, cls, fscore, mmd = mjda(
    Xs_tr, Ys_tr, Xt_tr,
    kernelRBF, np.nan,
    0.1, 2, classifierKNN_cv, 1, 8, 1000, Yt_tr
)

# Predict on test
# Assuming domainAdaptationTransform for test
from models.domainAdaptationTransform import domainAdaptationTransform
Zs_tst = domainAdaptationTransform(Xs_tst, Xs_tr, Xt_tr, W, kernelRBF, np.nan)
Zt_tst = domainAdaptationTransform(Xt_tst, Xs_tr, Xt_tr, W, kernelRBF, np.nan)

Ytp_tst, _ = classifierKNN_cv(Zs_tst, Ys_tr, Zt_tst, cls)

# Evaluate
acc = accuracy(Ytp_tst, Yt_tst)
f1, _ = f1score(Yt_tst, Ytp_tst)

print(f'Accuracy: {acc}')
print(f'F1: {f1}')

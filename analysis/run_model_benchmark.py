import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

from kernels.kernelRBF import kernelRBF
from models.kbtl.kbtl_test import kbtl_test
from models.kbtl.kbtl_test_binary import kbtl_test_binary
from models.kbtl.kbtl_train import kbtl_train
from models.kbtl.kbtl_train_binary import kbtl_train_binary
from models.mjda import mjda
from models.domainAdaptationTransform import domainAdaptationTransform
from classifiers.classifierKNN import classifierKNN
from classifiers.classifierKNN_cv import classifierKNN_cv
from models.bda import bda
from util.accuracy import accuracy
from util.f1score import f1score


def run_all_benchmarks(root: Path) -> dict:
    results = {}

    d = loadmat(root / 'data' / 'kbtl_demo_binary_data.mat')
    x = [d['X'][0][i] for i in range(d['X'].shape[1])]
    xtest = [d['Xtest'][0][i] for i in range(d['Xtest'].shape[1])]
    y = [d['Y'][0][i].flatten() for i in range(d['Y'].shape[1])]
    ytest = [d['Ytest'][0][i].flatten() for i in range(d['Ytest'].shape[1])]
    ktrain, khyp = [], []
    for i in range(len(x)):
        k_i, h_i = kernelRBF(np.nan, x[i], x[i])
        ktrain.append(k_i)
        khyp.append(h_i)

    params_binary = {
        'lambda': {'kappa': 1e-3, 'theta': 1e-3},
        'gamma': {'kappa': 1e-3, 'theta': 1e-3},
        'eta': {'kappa': 1e-3, 'theta': 1e-3},
        'iter': 500,
        'margin': 0,
        'R': 2,
        'Hsigma2': 0.25**2,
    }
    np.random.seed(0)
    hyp_binary = kbtl_train_binary(ktrain, y, params_binary)
    pred_train = kbtl_test_binary(ktrain, hyp_binary, y)
    ktest = [kernelRBF(khyp[i], x[i], xtest[i])[0] for i in range(len(x))]
    pred_test = kbtl_test_binary(ktest, hyp_binary, ytest)
    results['kbtl_binary'] = {
        'train_acc': [float(v) for v in pred_train['acc']],
        'test_acc': [float(v) for v in pred_test['acc']],
        'mean_train_acc': float(np.mean(pred_train['acc'])),
        'mean_test_acc': float(np.mean(pred_test['acc'])),
        'curve': {
            'x': list(range(1, len(pred_train['acc']) + 1)),
            'train_acc': [float(v) for v in pred_train['acc']],
            'test_acc': [float(v) for v in pred_test['acc']],
        },
    }

    d = loadmat(root / 'data' / 'kbtl_demo_multiclass_data.mat')
    x = [d['X'][0][i] for i in range(d['X'].shape[1])]
    y = [d['Y'][0][i].flatten() for i in range(d['Y'].shape[1])]
    xtest = [d['Xtest'][0][i] for i in range(d['Xtest'].shape[1])]
    ytest = [d['Ytest'][0][i].flatten() for i in range(d['Ytest'].shape[1])]
    ktrain, khyp = [], []
    for i in range(len(x)):
        k_i, h_i = kernelRBF(np.nan, x[i], x[i])
        ktrain.append(k_i)
        khyp.append(h_i)

    params_mc = {
        'lambda': {'kappa': 1e-3, 'theta': 1e-3},
        'gamma': {'kappa': 1e-3, 'theta': 1e-3},
        'eta': {'kappa': 1e-3, 'theta': 1e-3},
        'iter': 500,
        'margin': 1,
        'R': 2,
        'Hsigma2': 6**2,
    }
    np.random.seed(0)
    hyp_mc, _ = kbtl_train(ktrain, y, params_mc)
    pred_train = kbtl_test(ktrain, hyp_mc, y)
    ktest = [kernelRBF(khyp[i], x[i], xtest[i])[0] for i in range(len(x))]
    pred_test = kbtl_test(ktest, hyp_mc, ytest)
    results['kbtl_multiclass'] = {
        'train_acc': [float(v) for v in pred_train['acc']],
        'test_acc': [float(v) for v in pred_test['acc']],
        'mean_train_acc': float(np.mean(pred_train['acc'])),
        'mean_test_acc': float(np.mean(pred_test['acc'])),
        'curve': {
            'x': list(range(1, len(pred_train['acc']) + 1)),
            'train_acc': [float(v) for v in pred_train['acc']],
            'test_acc': [float(v) for v in pred_test['acc']],
        },
    }

    np.random.seed(0)
    d = loadmat(root / 'data' / 'gnat_repair.mat')
    xs, ys = d['Xs'], d['Ys'].flatten()
    xt, yt = d['Xt'], d['Yt'].flatten()

    n = len(xs)
    ind = np.random.permutation(n)
    ntr = 500
    xs_tr, ys_tr = xs[ind[:ntr]], ys[ind[:ntr]]
    xs_tst, ys_tst = xs[ind[ntr:]], ys[ind[ntr:]]
    xt_tr, yt_tr = xt[ind[:ntr]], yt[ind[:ntr]]
    xt_tst, yt_tst = xt[ind[ntr:]], yt[ind[ntr:]]

    scaler_s = StandardScaler()
    xs_tr = scaler_s.fit_transform(xs_tr)
    xs_tst = scaler_s.transform(xs_tst)
    scaler_t = StandardScaler()
    xt_tr = scaler_t.fit_transform(xt_tr)
    xt_tst = scaler_t.transform(xt_tst)

    _, _, _, w, cls, _, _, mjda_history = mjda(
        xs_tr, ys_tr, xt_tr, kernelRBF, np.nan, 0.1, 2, classifierKNN_cv, 1, 8, 1000, yt_tr, return_history=True
    )
    zs_tst = domainAdaptationTransform(xs_tst, xs_tr, xt_tr, w, kernelRBF, np.nan)
    zt_tst = domainAdaptationTransform(xt_tst, xs_tr, xt_tr, w, kernelRBF, np.nan)
    ytp_tst, _ = classifierKNN_cv(zs_tst, ys_tr, zt_tst, cls)
    results['mjda'] = {
        'test_acc': float(accuracy(ytp_tst, yt_tst)),
        'test_f1': float(f1score(yt_tst, ytp_tst)[0]),
        'curve': {
            'x': list(range(1, len(mjda_history['f1_per_iteration']) + 1)),
            'f1': mjda_history['f1_per_iteration'],
        },
    }

    feat = loadmat(root / 'data' / 'gnat_piper_preprocessed_features.mat')
    lbl = loadmat(root / 'data' / 'gnat_piper_preprocessed_labels.mat')
    xs_cells = feat['Xs_tr']
    ys_cells = lbl['Ys_tr']
    xs_tr = np.vstack([xs_cells[i, 0] for i in range(xs_cells.shape[0])])
    ys_tr = np.hstack([ys_cells[i, 0].flatten() for i in range(ys_cells.shape[0])])
    xt_tr = feat['Xt_tr']
    yt_tr = lbl['Yt_tr'].flatten()

    xs_tr = StandardScaler().fit_transform(xs_tr)
    xt_tr = StandardScaler().fit_transform(xt_tr)
    _, _, ytp, _, _, fscore_bda, mmd, bda_history = bda(
        xs_tr,
        ys_tr,
        xt_tr,
        kern=kernelRBF,
        hyp=np.nan,
        mu=1.0,
        k=10,
        lambda_=0.5,
        classifier=classifierKNN,
        iter=2,
        mode=0,
        Yt=yt_tr,
        return_history=True,
    )
    train_target_acc = float(accuracy(yt_tr, ytp))
    train_target_f1 = float(f1score(yt_tr, ytp)[0])

    results['bda_gnat_piper'] = {
        'train_target_acc': train_target_acc,
        'train_target_f1': train_target_f1,
        'mmd': float(mmd),
        'fscore': float(fscore_bda),
        'curve': {
            'x': list(range(1, len(bda_history['f1_per_iteration']) + 1)),
            'f1': bda_history['f1_per_iteration'],
        },
    }
    return results


def build_comparison_table(results: dict) -> list[dict]:
    rows = [
        {
            'metric': 'KBTL Binary mean test acc (%)',
            'python': results['kbtl_binary']['mean_test_acc'],
            'matlab_ref': 90.0,
            'criterion': '>= 90.0 (paper-level expected range from KBTL docs)',
        },
        {
            'metric': 'KBTL Multiclass mean test acc (%)',
            'python': results['kbtl_multiclass']['mean_test_acc'],
            'matlab_ref': 85.0,
            'criterion': '>= 85.0 (paper-level expected range from KBTL docs)',
        },
        {
            'metric': 'MJDA test acc (%)',
            'python': results['mjda']['test_acc'],
            'matlab_ref': 100.0,
            'criterion': '>= 100.0 (documented MATLAB/demo value)',
        },
        {
            'metric': 'BDA target-train F1 (%)',
            'python': results['bda_gnat_piper']['train_target_f1'] * 100,
            'matlab_ref': 100.0,
            'criterion': '>= 100.0 (best MCS case in docs/paper summary)',
        },
    ]

    for row in rows:
        row['status'] = '达标' if row['python'] >= row['matlab_ref'] else '未达标'
    return rows






def plot_training_curves(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    kb = results['kbtl_binary']['curve']
    axes[0, 0].plot(kb['x'], kb['train_acc'], marker='o', label='Train Acc')
    axes[0, 0].plot(kb['x'], kb['test_acc'], marker='s', label='Test Acc')
    axes[0, 0].set_title('KBTL Binary (per task)')
    axes[0, 0].set_xlabel('Task Index')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()

    km = results['kbtl_multiclass']['curve']
    axes[0, 1].plot(km['x'], km['train_acc'], marker='o', label='Train Acc')
    axes[0, 1].plot(km['x'], km['test_acc'], marker='s', label='Test Acc')
    axes[0, 1].set_title('KBTL Multiclass (per task)')
    axes[0, 1].set_xlabel('Task Index')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()

    mj = results['mjda']['curve']
    axes[1, 0].plot(mj['x'], mj['f1'], marker='o')
    axes[1, 0].set_title('MJDA F1 per iteration')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('F1')

    bd = results['bda_gnat_piper']['curve']
    axes[1, 1].plot(bd['x'], bd['f1'], marker='o', color='tab:orange')
    axes[1, 1].set_title('BDA F1 per iteration')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('F1')

    plt.tight_layout()
    return fig, axes


if __name__ == '__main__':
    root = ROOT
    results = run_all_benchmarks(root)
    rows = build_comparison_table(results)
    print(json.dumps({'results': results, 'comparison': rows}, indent=2, ensure_ascii=False))

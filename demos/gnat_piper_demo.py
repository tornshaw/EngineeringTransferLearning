import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from models.bda import bda
from classifiers.classifierKNN import classifierKNN
from kernels.kernelRBF import kernelRBF
from util.accuracy import accuracy
from util.f1score import f1score


def run_gnat_piper_demo():
    print('Running gnat_piper_demo.py')

    data_feat = loadmat('data/gnat_piper_preprocessed_features.mat')
    data_lbl = loadmat('data/gnat_piper_preprocessed_labels.mat')

    # Source/target PCA features are stored as 4x1 cell arrays in MATLAB
    Xs_cells = data_feat['Xs_tr']
    Ys_cells = data_lbl['Ys_tr']

    Xs_tr = np.vstack([Xs_cells[i,0] for i in range(Xs_cells.shape[0])])
    Ys_tr = np.hstack([Ys_cells[i,0].flatten() for i in range(Ys_cells.shape[0])])

    Xt_tr = data_feat['Xt_tr']
    Yt_tr = data_lbl['Yt_tr'].flatten()

    # Normalization
    scaler_s = StandardScaler().fit(Xs_tr)
    scaler_t = StandardScaler().fit(Xt_tr)
    Xs_tr_norm = scaler_s.transform(Xs_tr)
    Xt_tr_norm = scaler_t.transform(Xt_tr)

    # BDA 训练/测试
    Zs, Zt, Ytp, W, cls, fscore, mmd = bda(
        Xs_tr_norm,
        Ys_tr,
        Xt_tr_norm,
        kern=kernelRBF,
        hyp=np.nan,
        mu=1.0,
        k=10,
        lambda_=0.5,
        classifier=classifierKNN,
        iter=2,
        mode=0,
        Yt=Yt_tr
    )

    print('BDA output:')
    print(f'Zs shape: {Zs.shape}, Zt shape: {Zt.shape}')
    print(f'fscore: {fscore:.4f}, mmd: {mmd:.4f}')

    # 预测结果: Ytp 已包含目标预测类别
    act_acc = accuracy(Yt_tr, Ytp)
    act_f1, _ = f1score(Yt_tr, Ytp)

    print(f'Accuracy on target train labels: {act_acc:.4f}')
    print(f'F1 on target train labels: {act_f1:.4f}')

    return {
        'Zs': Zs,
        'Zt': Zt,
        'Ytp': Ytp,
        'fscore': fscore,
        'mmd': mmd,
        'acc': act_acc,
        'f1': act_f1,
    }


if __name__ == '__main__':
    results = run_gnat_piper_demo()
    print('gnat_piper_demo.py done')

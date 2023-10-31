# -*- coding: utf-8 -*- 
# @Time : 2022/10/13 22:14 
# @Author : lepold
# @File : statistics.py


"""
To process bold signal and compute the correlation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def norm(data, dim=1):
    return (data - np.mean(data, axis=dim, keepdims=True)) / np.std(data, axis=dim, keepdims=True)


def cal_pearson(data1, data2, transient=30, shift=1, write_path="./", name=''):
    """

    Parameters
    ----------
    data1: 2D, ndarray
        dimension 1 is voxel_idx, and dimension 2 is time
    data2 : 2D, ndarray
    -------

    """
    assert data1.shape == data2.shape
    processed_data1 = data1[transient:-shift, :]
    processed_data2 = data2[transient + shift:, :]
    t, n = processed_data1.shape
    corrceff = []
    for i in range(n):
        r = np.corrcoef(processed_data1[:, i], processed_data2[:, i])[0, 1]
        corrceff.append(r)
    corrceff = np.array(corrceff, dtype=np.float)

    meann, maxx, minn = corrceff.mean(), corrceff.max(), corrceff.min()
    weights = np.ones_like(corrceff) / float(len(corrceff))
    fig = plt.figure(figsize=(5, 3), dpi=300)
    fig.gca().hist(corrceff, bins=30, weights=weights)
    fig.gca().set_xlabel("corrcoeff")
    fig.gca().set_title("mean:%.2f, max=%.2f, min=%.2f" % (meann, maxx, minn))
    fig.savefig(os.path.join(write_path, name + "pearson_v1_trans_%d_shift_%d.png" % (transient, shift)))
    return corrceff


def main(assimilation_bold_path, experimental_bold_path, if_rearange=True, write_path="./"):
    bold_assim = np.load(assimilation_bold_path)
    print(f"bold_assim: t:{bold_assim.shape[0]}; voxels: {bold_assim.shape[1]}")
    if if_rearange:
        rearange_index = np.load(r"E:\PycarmProjects2\Digital_twin_brain\data\rearange_index.npy")
        temp = np.argsort(rearange_index)
        invert_index = np.arange(len(rearange_index))[temp]
        bold_assim = bold_assim[:, invert_index]
    bold_exp = loadmat(experimental_bold_path)["nii_ts_bandzscore"]
    tt, n = bold_assim.shape
    bold_exp = bold_exp[:tt, :]
    bold_exp = norm(bold_exp, dim=0)
    bold_assim = norm(bold_assim, dim=0)
    corr_coef = cal_pearson(bold_exp, bold_assim, write_path=write_path, shift=2, name="voxel")

# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 14:48 
# @Author : lepold
# @File : helpers.py

"""
Here are some simple Numpy functions or torch functions that help to achieve simulation or assimilation.

"""


import os
import torch
import numpy as np
import argparse


def np_move_avg(a, n=10, mode="valid"):
    """
    Calculate the moving average of the signal

    Parameters
    ----------
    a: ndarray
        1-dim or 2-dim ndarray, like (time_dim, variable_dim)
    n: int
        window lenght
    mode: str
        the options in ``np.convolve``.

    Returns
    -------

    """
    if a.ndim > 1:
        tmp = []
        for i in range(a.shape[1]):
            tmp.append(np.convolve(a[:, i], np.ones((n,)) * 1000 / n, mode=mode))
        tmp = np.stack(tmp, axis=1)
    else:
        tmp = np.convolve(a, np.ones((n,)) * 1000 / n, mode=mode)
    return tmp


def torch_2_numpy(u, is_cuda=True):
    """
    Convert ``torch.Tensor`` to ``numpy.ndarray`` in cpu memory.

    """
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def load_if_exist(func, *args, **kwargs):
    """
    Load npy if exist else generate it by ``func``.
    """
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

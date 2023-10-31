# -*- coding: utf-8 -*- 
# @Time : 2022/9/9 16:20 
# @Author : lepold
# @File : getpsd.py
import numpy as np
from scipy.signal import welch, get_window


def get_psd(data, fs, resolution=0.5):
    win = np.int(fs / resolution)
    the_window = get_window('boxcar', win)
    freqs, power = welch(data,
                         fs=fs,
                         window=the_window,
                         noverlap=int(0.75 * win),
                         nfft=win,
                         axis=0,
                         scaling='density',
                         detrend=False)
    return freqs, power


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_features = 1
    fs = 1000
    x_iter = np.arange(0, 10, 1 / fs)
    X = np.sin(5 * np.pi * (x_iter - 0.05)) + 0.3 * np.random.random(len(x_iter))
    X[1000:3000] = X[1000:3000] + 0.5 * np.sin(15 * np.pi * np.arange(1., 3., 1 / fs))

    freqs, power = get_psd(X, fs=fs, resolution=0.5)
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax.plot(x_iter, X)
    ax.set_xlabel("time(s)")
    ax.set_ylabel("signal")
    ax2.plot(freqs[:40], power[:40])
    ax2.set_xlabel("freqs")
    ax2.set_ylabel("psd")
    fig.show()

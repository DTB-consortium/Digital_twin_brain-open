import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import welch
# import pywt


def mean_firing_rate(spike_train):
    """

    Parameters
    ----------
    spike_train： numpy.ndarray
                2 dimensional ndarray, the first axis is time dimension

    Returns
    -------

    """
    acitvate_idx = np.where(spike_train.mean(axis=0) > 0.001)[0]
    if len(acitvate_idx) > 10:
        return np.mean(spike_train[:, acitvate_idx])
    else:
        return 0.


def instantaneous_rate(spike_train, bin_width=5):
    """
    Parameters
    ----------
    spike_train: numpy.ndarray
                2 dimensional ndarray, the first axis is time dimension
    bin_width: window width to count spikes

    Returns
    -------

    """
    spike_rate = spike_train.mean(axis=1)
    out = uniform_filter1d(spike_rate.astype(np.float32), size=bin_width)
    return out


def gaussian_kernel_inst_rate(spike_train, bin_width=5, kernel_width=20):
    """
    Parameters
    ----------
    spike_train: numpy.ndarray
                2 dimensional ndarray, the first axis is time dimension

    bin_width: window width to count spikes

    kernel_width: window width of gaussian kernel

    Returns
    -------

    """
    inst_rate = instantaneous_rate(spike_train, bin_width)
    rate_time_series_auto_kernel = gaussian_filter1d(inst_rate, kernel_width, axis=-1)
    return rate_time_series_auto_kernel


def spike_psd(spike_train, sampling_rate=1000, subtract_mean=False):
    """

    spike spectrum of instantaneous fire rate

    """
    st = instantaneous_rate(spike_train, bin_width=5)
    if subtract_mean:
        data = st - np.mean(st)
    else:
        data = st
    N_signal = data.size
    fourier_transform = np.abs(np.fft.rfft(data))
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))
    return frequency, power_spectrum


def pearson_cc(spike_train, pairs=20):
    """
    compute pearson correlation of instantaneous_rate of parir neurons
    Parameters
    ----------
    spike_train
    pairs: num pairs to compute correlation

    Returns
    -------

    """
    valid_idx = np.where(spike_train.mean(axis=0) > 0.001)[0]
    # print(valid_idx)
    if len(valid_idx) < 10:
        return 0.
    correlation = []
    for _ in range(pairs):
        a, b = np.random.choice(valid_idx, size=(2,), replace=False)
        x1 = uniform_filter1d(spike_train[:, a].astype(np.float32), size=50, )
        x2 = uniform_filter1d(spike_train[:, b].astype(np.float32), size=50, )
        correlation.append(np.corrcoef(x1, x2)[0, 1])
    correlation = np.array(correlation)
    return np.mean(correlation)


def correlation_coefficent(spike_train):
    ins_rate = instantaneous_rate(spike_train, bin_width=5)
    return np.std(ins_rate) / np.mean(ins_rate)


def coefficient_of_variation(spike_train):
    activate_idx = (spike_train.sum(0) > 5).nonzero()[0]
    if len(activate_idx) < 10:
        return np.nan
    else:
        cvs = []
        for i in activate_idx:
            out = spike_train[:, i].nonzero()[0]
            fire_interval = out[1:] - out[:-1]
            cvs.append(fire_interval.std() / fire_interval.mean())
        return np.array(cvs).mean()


def spike_spectrum(spike):
    ins_fr = instantaneous_rate(spike)
    freqs, psd = welch(ins_fr, 1., return_onesided=True, scaling='density')
    return freqs, psd


def morlet_wavelet_transform(x, fs, scales, dim=0):
    """
    Takes the complex Morlet wavelet transform of time series data and plots spectrogram.

    Parameters
    ----------
    x : ndarray
        1-dim vector time series or multi-dimensional vector with time specified in dimension DIM.

    fs: float
        sampling frequency

    scales: ndarray
        scales

    dim: int
        Specify the dimension of time in x.

    Returns
    -------

    coefs: ndarray
         FxT matrix of complex Morlet wavelet coefficients, where F is the number of centre frequencies.
         If X is multi-dimensional, CFS will be (F, ) + tuple(x.shape).

    frequencies: ndarray

    """
    dt = 1 / fs
    if np.ndim(x) == 1:
        pass
    else:
        perm_order = (dim,) + tuple(np.arange(0, dim)) + tuple(np.arange(dim + 1, np.ndim(x)))
        x = np.transpose(x, perm_order)
        x = x.reshape((x.shape[0], -1))
    coefs, frequencies = pywt.cwt(x, scales, 'morl', dt, axis=0)
    return coefs, frequencies


if __name__ == '__main__':
    t = np.linspace(0, 2, 2000, endpoint=False)
    fs = 1000
    sig = np.cos(2 * np.pi * 7 * t) + np.cos(2 * np.pi * 15 * t) + np.random.random(2000) * 0.2
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, sig, lw=1., c="k")
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("amplitude")

    coef, freqs = morlet_wavelet_transform(sig, fs=fs, scales=np.arange(1, 20, 0.05))
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(coef, origin="lower")
    fig.show()




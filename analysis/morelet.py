# -*- coding: utf-8 -*- 
# @Time : 2022/9/9 13:27 
# @Author : lepold
# @File : morelet.py

"""
Copyright 2019 Jacob M. Graving <jgraving@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import multiprocessing
import warnings

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
    warnings.warn('CuPy was not found, so GPU functionality is unavailable. '
                  'See https://github.com/cupy/cupy#installation '
                  'for installation instructions')

__all__ = ['wavelet_transform']


class Parallel:
    def __init__(self, n_jobs):

        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + n_jobs + 1
        elif n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        self.n_jobs = n_jobs
        self.pool = multiprocessing.Pool(n_jobs)

    def process(self, job, arg, asarray=False):

        processed = self.pool.map(job, arg)

        if asarray:
            processed = np.array(processed)

        return processed

    def close(self):

        self.pool.close()
        self.pool.terminate()
        self.pool.join()


def _morlet_conj_ft(omegas, omega0=5.0, gpu=False):
    """
    Returns a Fourier conjugate Morlet wavelet

    Conjugate wavelet is given by the formula:
        pi**(-0.25) * exp(-0.5 * (omegas-omega0)**2)

    Parameters:
    ===========
    omegas : int
        omegas to calculate wavelets for
    omega0 : float (default=5.0)
        Dimensionless omega0 parameter for wavelet transform
    Returns:
    ========
    ft_wavelet : ndarray
        array of Fourier conjugate wavelets
    """
    if gpu:
        backend = cp
    else:
        backend = np

    ft_wavelet = backend.pi ** (-0.25) * backend.exp(-0.5 * (omegas - omega0) ** 2)

    return ft_wavelet


def _morlet_fft_convolution(X, freqs, scales, dtime, omega0=5.0, gpu=False):
    """
    Calculates a Morlet continuous wavelet transform
    for a given signal across a range of frequencies

    Parameters:
    ===========
    X : array_like, shape (n_samples)
        Signal of interest
    freqs : array_like, shape (n_freqs)
        A list of frequencies
    scales : array_like, shape (n_freqs)
        A list of scales
    omega0 : float
        Dimensionless omega0 parameter for wavelet transform
    dtime : float
        Change in time per sample. The inverse of the sampling frequency.

    Returns:
    ========
    X_new : ndarray, shape (n_samples, n_freqs)
        The transformed signal.
    """
    if gpu:
        backend = cp
    else:
        backend = np

    n_samples = X.shape[0]
    # n_freqs = freqs.shape[0]

    # allocate memory for result
    # X_new = cp.zeros((n_freqs, n_samples))
    X = backend.asarray(X)

    # Test whether to add an extra zero
    if backend.mod(n_samples, 2) == 1:
        X = backend.concatenate((X, backend.zeros(1, dtype=backend.float32)))
        n_samples = X.shape[0]
        pad_test = True
    else:
        pad_test = False

    # zero pad the array
    # padding = (np.zeros((n_samples // 2)), X, np.zeros((n_samples // 2)))
    # X = np.concatenate(padding)
    X = backend.pad(X, pad_width=n_samples // 2,
                    mode='constant', constant_values=0)
    n_padded = X.shape[0]

    # calculate the omega values
    omegas = backend.arange(-n_padded // 2, n_padded // 2) / (n_padded * dtime)
    omegas *= 2 * backend.pi

    # Fourier transform the padded signal
    X_hat = backend.fft.fft(X)
    X_hat = backend.fft.fftshift(X_hat)

    # Set index to remove the extra zero if added
    if pad_test:
        idx0 = (n_samples // 2)
        idx1 = (n_samples // 2 + n_samples - 1)
    else:
        idx0 = (n_samples // 2)
        idx1 = (n_samples // 2 + n_samples)

    # Perform the wavelet transform
    scale = backend.asarray(scales)[None,]
    # for idx, scale in enumerate(scales):

    # calculate the wavelet
    morlet = -omegas[..., None] * scale
    morlet = _morlet_conj_ft(morlet, omega0, gpu=gpu)

    # convolve the wavelet
    convolved = backend.fft.ifft(morlet * X_hat[..., None], axis=0)
    convolved *= backend.sqrt(scale)

    convolved = convolved[idx0:idx1]  # remove zero padding
    convolved = backend.abs(convolved)  # use the norm of the complex values

    # scale power to account for disproportionally
    # large wavelet response at low frequencies
    power_scale = backend.pi ** -0.25
    power_scale *= backend.exp(0.25 * (omega0 - backend.sqrt(omega0 ** 2 + 2)) ** 2)
    power_scale = power_scale / backend.sqrt(2 * scale)
    convolved = convolved * power_scale

    if gpu:
        convolved = backend.asnumpy(convolved)
    return convolved


def _morlet_fft_convolution_parallel(feed_dict):
    return _morlet_fft_convolution(**feed_dict)


def wavelet_transform(X, n_freqs, fsample, fmin, fmax,
                      prob=True, omega0=5.0, log_scale=True,
                      n_jobs=1, gpu=False):
    """
    Applies a Morlet continuous wavelet transform to a data set
    across a range of frequencies.

    This is an implementation of the continuous wavelet transform
    described in Berman et al. 2014 [1],
    The output is adjusted for disproportionally large wavelet response
    at low frequencies by normalizing the response to a sine wave
    of the same frequency. Amplitude fluctuations are removed by
    normalizing the power spectrum at each sample.

    Parameters:
    ===========
    X : array_like, shape (n_samples, n_features)
        Data to transform
    n_freqs : int
        Number of frequencies to consider from fmin to fmax (inclusive)
    fsample : float
        Sampling frequency of the data (in Hz)
    fmin : float
        Minimum frequency of interest for a wavelet transform (in Hz)
    fmax : float
        Maximum frequency of interest for the wavelet transform (in Hz)
        Typically the Nyquist frequency of the signal (0.5 * fsample).
    prob : bool (default = True)
        Whether to normalize the power such that each sample sums to one.
        This effectively removes amplitude fluctuations.
    log_scale : bool (default = True)
        Whether to sample the frequencies on a log scale.
    omega0 : float (default = 5.0)
        Dimensionless omega0 parameter for wavelet transform.
    n_jobs : int (default = 1)
        Number of jobs to use for performing the wavelet transform.
        If -1, all CPUs are used. If 1 is given, no parallel computing is
        used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        Thus for n_jobs = -2, all CPUs but one are used.
    gpu : bool (default = False)
        Whether to use the gpu for calculating the wavelet transform.
        If True, cupy is used in place of numpy to perform the
        wavelet calculations.

    Returns:
    ========
    freqs : ndarray, shape (n_freqs)
        The frequencies used for the wavelet transform
    power : ndarray, shape (n_samples)
        The total power for each row in X_new
    X_new : ndarray, shape (n_samples, n_features*n_freqs)
        Continuous wavelet transformed X

    References:
    ===========
    [1] Berman, G. J., Choi, D. M., Bialek, W., & Shaevitz, J. W. (2014).
        Mapping the stereotyped behaviour of freely moving fruit flies.
        Journal of The Royal Society Interface, 11(99), 20140672.

    Notes:
    ======
    Based on code from Gordon J. Berman et al.
    (https://github.com/gordonberman/MotionMapper)
    """

    if gpu is True and cp is None:
        gpu = False
        warnings.warn('`gpu` set to True, but CuPy was not found, '
                      'using CPU with {:+.0f} thread(s). '
                      'See https://github.com/cupy/cupy#installation '
                      'for installation instructions'.format(n_jobs))

    X = X.astype(np.float32)
    # n_samples = X.shape[0]
    # n_features = X.shape[1]

    dtime = 1. / fsample

    # tmin = 1. / fmax
    # tmax = 1. / fmin

    # exponent = np.arange(0, n_freqs, dtype=np.float64)
    # exponent *= np.log(tmax / tmin)
    # exponent /= (np.log(2) * (n_freqs - 1))

    # periods = tmin * 2**exponent
    # freqs = np.flip(1. / periods, axis=0)

    if log_scale:
        fmin_log2 = np.log(fmin) / np.log(2)
        fmax_log2 = np.log(fmax) / np.log(2)
        freqs = np.logspace(fmin_log2, fmax_log2,
                            n_freqs, base=2)
    else:
        freqs = np.linspace(fmin, fmax, n_freqs)

    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * freqs)

    feed_dicts = [{"X": feature,
                   "freqs": freqs,
                   "scales": scales,
                   "dtime": dtime,
                   "omega0": omega0,
                   "gpu": gpu}
                  for feature in X.T]

    if n_jobs != 1 and not gpu:
        pool = Parallel(n_jobs)
        convolved = pool.process(_morlet_fft_convolution_parallel, feed_dicts)
        pool.close()
    else:
        convolved = list(map(_morlet_fft_convolution_parallel, feed_dicts))

    X_new = np.concatenate(convolved, axis=1)

    # for idx, conv in enumerate(convolved):
    #    X_new[:, (n_freqs * idx):(n_freqs * (idx + 1))] = conv.T

    power = X_new.sum(axis=1, keepdims=True)

    if prob:
        X_new /= power

    if gpu:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    return freqs, power.flatten(), X_new


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_features = 1
    fs = 1000
    x_iter = np.arange(0, 2, 1 / fs)
    X = np.sin(5 * np.pi * (x_iter - 0.05)) + 0.3 * np.random.random(len(x_iter))
    X[1000:1500] = X[1000:1500] + 0.5 * np.sin(15 * np.pi * np.arange(1., 1.5, 1 / fs))
    X = X[:, None]

    freqs, power, X_new = wavelet_transform(X, n_freqs=100, fsample=fs, fmin=1., fmax=50.)
    fig = plt.figure(figsize=(8, 5), dpi=100)
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax.plot(x_iter, X)
    ax.set_xlabel("time(s)")
    ax.set_ylabel("signal")
    ax2.imshow(X_new[:, :].T, aspect="auto", origin="lower")
    ax2.set_xlabel("time(ms)")
    ax2.set_yticks(np.linspace(0, 100, 4, endpoint=False, dtype=np.uint))
    ax2.set_yticklabels(np.round(freqs[np.linspace(0, 100, 4, endpoint=False, dtype=np.uint)], 2))
    ax2.set_ylabel("freqs")
    fig.show()




# -*- coding: utf-8 -*- 
# @Time : 2022/8/27 14:31 
# @Author : lepold
# @File : simulation.py

import os

import torch

from analysis.spike_statistics import *
from generation.read_block import connect_for_block
from models.block import block
from models.block_new import block as block_new


def simulation(block_path="small_block_d100", Time=3000, noise_rate=0.003, delta_t=0.1, **kwargs):
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    show_result = kwargs.get("show_result", False)
    save_result = kwargs.get("save_result", False)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")
    B = block(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_time = int(iter_time / 2)
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    log_all = np.zeros(shape=(document_iter_time, 250), dtype=np.uint8)
    synaptic_current = np.zeros(shape=(document_iter_time, 4), dtype=np.float32)
    for time in range(iter_time):
        print(time, end='\r')
        B.run(noise_rate=noise_rate, isolated=False)
        if time >= document_start:
            log_all[time - document_start] = B.active.data.cpu().numpy()[8000:8250]
            synaptic_current[time - document_start] = B.I_ui.mean(axis=-1).cpu().numpy()
    windows = int(1 / delta_t)
    print('window,', windows)
    log_all = log_all.reshape((-1, windows, 250))
    sub_log = log_all.sum(axis=1)
    times, peaks = sub_log.nonzero()
    if save_result:
        np.savez(os.path.join(block_path, "spike.npz"), times=times, peaks=peaks)
    print("log.max", sub_log.max())
    torch.cuda.empty_cache()
    if show_result:
        pcc_log = pearson_cc(sub_log[:, :200], pairs=200)
        cc_log = correlation_coefficent(sub_log[:, :200])
        mean_fr = mean_firing_rate(sub_log[:, :200])
        print("mean_fr", mean_fr)
        print("cc: ", cc_log)
        print("pcc: ", pcc_log)
        fr = instantaneous_rate(sub_log, bin_width=5)
        rate_time_series_auto_kernel = gaussian_kernel_inst_rate(sub_log, 5, 20)
        fig = plt.figure(figsize=(10, 5), dpi=100)
        ax = fig.add_axes([0.1, 0.6, 0.8, 0.35])
        ax.grid(False)
        x, y = sub_log[-1000:, :200].nonzero()
        ax.scatter(x, y, marker='.', color="black", s=1)
        x, y = sub_log[-1000:, 200:].nonzero()
        ax.scatter(x, y + 200, marker='.', color="red", s=1)
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 250])
        ax.set_xticks([])
        ax.set_ylabel('neuron')
        ax.invert_yaxis()
        ax.set_aspect(1)
        ax = fig.add_axes([0.1, 0.32, 0.8, 0.23])
        ax.grid(False)
        ax.plot(rate_time_series_auto_kernel[-1000:], color='0.2')
        ax.set_xlim([0, 1000])
        ax.plot(fr, color='0.8')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_ylabel("fr(KHz)")
        ax.set_xticks([])
        ax.set_xlim([0, 1000])
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.23])
        synaptic_current = synaptic_current[-10000:, ]
        epsc = np.sum(synaptic_current[:, :2], axis=1)
        ipsc = np.sum(synaptic_current[:, 2:], axis=1)
        psc = np.sum(synaptic_current, axis=1)
        ax.plot(np.arange(len(synaptic_current)), epsc, c="b", alpha=0.8, label="Exc")
        ax.plot(np.arange(len(synaptic_current)), ipsc, c="r", alpha=0.8, label="Inh")
        ax.plot(np.arange(len(synaptic_current)), psc, c="k", alpha=0.7, label="Net")
        ax.set_xlim([0, len(synaptic_current)])
        ax.set_xticks(np.linspace(0, len(synaptic_current), 5))
        ax.set_xticklabels(np.linspace(0, document_time, 5))
        ax.set_xlabel("time(ms)")
        ax.set_ylabel("current")
        ax.legend(loc="best")
        return fig
    else:
        return None


def simulation_via_background_current(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    show_result = kwargs.get("show_result", False)
    save_result = kwargs.get("save_result", False)
    # specified_gui = [0., 0., 0., 0.]
    property, w_uij = connect_for_block(block_path)
    w_uij = w_uij / torch.tensor(255., dtype=torch.float32)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)
    # property[:, 5] = 5.

    property = property.cuda("cuda:1")
    w_uij = w_uij.cuda("cuda:1")
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=0.6,
        i_sigma=0.2,
        tau_i=10.
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_time = int(iter_time / 2)
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    log_all = np.zeros(shape=(document_iter_time, 250), dtype=np.uint8)
    synaptic_current = np.zeros(shape=(document_iter_time, 4), dtype=np.float32)
    for time in range(iter_time):
        print(time, end='\r')
        B.run(debug_mode=False)
        if time >= document_start:
            log_all[time - document_start] = B.active.data.cpu().numpy()[8000:8250]
            synaptic_current[time - document_start] = B.I_ui.mean(axis=-1).cpu().numpy()
    windows = int(1 / delta_t)
    print('window,', windows)
    log_all = log_all.reshape((-1, windows, 250))
    sub_log = log_all.sum(axis=1)
    times, peaks = sub_log.nonzero()
    if save_result:
        np.savez(os.path.join(block_path, "spike.npz"), times=times, peaks=peaks)
    print("log.max", sub_log.max())
    torch.cuda.empty_cache()
    if show_result:
        pcc_log = pearson_cc(sub_log[:, :200], pairs=200)
        cc_log = correlation_coefficent(sub_log[:, :200])
        mean_fr = mean_firing_rate(sub_log[:, :200])
        print("mean_fr", mean_fr)
        print("cc: ", cc_log)
        print("pcc: ", pcc_log)
        fr = instantaneous_rate(sub_log, bin_width=5)
        rate_time_series_auto_kernel = gaussian_kernel_inst_rate(sub_log, 5, 20)
        fig = plt.figure(figsize=(10, 5), dpi=100)
        ax = fig.add_axes([0.1, 0.6, 0.8, 0.35])
        ax.grid(False)
        x, y = sub_log[-1000:, :200].nonzero()
        ax.scatter(x, y, marker='.', color="black", s=1)
        x, y = sub_log[-1000:, 200:].nonzero()
        ax.scatter(x, y + 200, marker='.', color="red", s=1)
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 250])
        ax.set_xticks([])
        ax.set_ylabel('neuron')
        ax.invert_yaxis()
        ax.set_aspect(1)
        ax = fig.add_axes([0.1, 0.32, 0.8, 0.23])
        ax.grid(False)
        ax.plot(rate_time_series_auto_kernel[-1000:], color='0.2')
        ax.set_xlim([0, 1000])
        ax.plot(fr, color='0.8')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_ylabel("fr(KHz)")
        ax.set_xticks([])
        ax.set_xlim([0, 1000])
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.23])
        synaptic_current = synaptic_current[-10000:, ]
        epsc = np.sum(synaptic_current[:, :2], axis=1)
        ipsc = np.sum(synaptic_current[:, 2:], axis=1)
        psc = np.sum(synaptic_current, axis=1)
        ax.plot(np.arange(len(synaptic_current)), epsc, c="b", alpha=0.8, label="Exc")
        ax.plot(np.arange(len(synaptic_current)), ipsc, c="r", alpha=0.8, label="Inh")
        ax.plot(np.arange(len(synaptic_current)), psc, c="k", alpha=0.7, label="Net")
        ax.set_xlim([0, len(synaptic_current)])
        ax.set_xticks(np.linspace(0, len(synaptic_current), 5))
        ax.set_xticklabels(np.linspace(0, document_time, 5))
        ax.set_xlabel("time(ms)")
        ax.set_ylabel("current")
        ax.legend(loc="best")
        return fig
    else:
        return None


if __name__ == '__main__':
    path = "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks/d1000_ou/uint8"
    # gui = np.array([0.00286671, 0.00041940, 0.02096406, 0.00133286], dtype=np.float_)
    gui = np.array([0.0007572339964099228, 0.0001108432697947137, 0.0037877766881138086, 0.00028210534946992993])
    fig = simulation_via_background_current(block_path=path, Time=4000, delta_t=1, specified_gui=gui,
                                            show_result=True)
    fig.savefig("./block_d1000.png")

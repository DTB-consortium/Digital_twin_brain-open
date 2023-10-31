# -*- coding: utf-8 -*- 
# @Time : 2023/3/13 10:10 
# @Author : lepold
# @File : simulation_singlecolumn.py


import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from cuda_develop.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def torch_2_numpy(u, is_cuda=True):
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def run_simulation_column(ip, block_path, write_path):
    os.makedirs(write_path, exist_ok=True)
    v_th = -50
    block_model = block_gpu(ip, block_path, 0.1, route_path=None)

    total_neurons = int(block_model.total_neurons)
    neurons_per_population = block_model.neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    populations = block_model.subblk_id.cpu().numpy()
    print(populations)  # 2 3 4 .. 9
    sample_size = np.array([80, 20, 80, 20, 20, 10, 60, 10]) * 3
    chunks = np.add.accumulate(sample_size)[:-1]
    sample_idx = []
    for i, size in enumerate(sample_size):
        sample_idx.append(
            np.random.choice(np.arange(neurons_per_population_base[i], neurons_per_population_base[i + 1]), size,
                             replace=False))
    sample_idx = np.concatenate(sample_idx)
    sample_number = sample_idx.shape[0]
    print("sample_num:", sample_number)
    assert sample_idx.max() < total_neurons
    sample_idx = torch.from_numpy(sample_idx.astype(np.int64)).cuda()
    block_model.set_samples(sample_idx)

    block_model.update_ou_background_stimuli(10., 0.5, 0.2)

    def _update_gui(_gui_laminar):
        for i, idx in enumerate(np.arange(10, 14)):
            population_info = np.stack(np.meshgrid(populations, idx, indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            param_info = gui_laminar[:, i].astype(np.float32)
            param_info = torch.from_numpy(param_info).cuda()
            block_model.assign_property_by_subblk(population_info, param_info)

    gui_laminar = np.array([
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001],
                            [0.001, 0.0001, 0.003, 0.001]])
    gui_laminar = np.broadcast_to(gui_laminar, (2, 8, 4)).reshape((-1, 4))
    print("gui", gui_laminar)
    _update_gui(gui_laminar)

    _ = block_model.run(30000, freqs=True, vmean=False, sample_for_show=False)
    temp_spike = []
    for return_info in block_model.run(10000, freqs=False, vmean=False, iou=True, sample_for_show=True):
        spike, vi = return_info
        # spike &= (torch.abs(vi - v_th) / 50 < 1e-5)
        temp_spike.append(spike)
    temp_spike = torch.stack(temp_spike, dim=0)
    Spike = torch_2_numpy(temp_spike)
    logs = np.split(Spike, chunks, axis=1)
    fr = np.array([log.mean() * 10000 for log in logs])
    print("firing", fr)
    block_model.shutdown()
    print("Done")

    fig = plt.figure(figsize=(6, 7))
    Spike = Spike[-10000:, :]
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.75], frameon=True)
    spike_events = [Spike[:, i].nonzero()[0] for i in range(Spike.shape[1])]
    s = 0
    colors = ["tab:blue", "tab:red"]
    total = np.sum(sample_size)
    names = ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    for i, size in enumerate(sample_size):
        e = s + size
        color = colors[i % 2]
        y_inter = (s + e) / 2 / total-0.02
        fr = Spike[:, s:e].mean() * 10000
        ax.eventplot(spike_events[s:e], lineoffsets=np.arange(s, e), colors=color)
        # xx, yy = Spike[:, s:e].nonzero()
        # yy = yy + s
        # ax.scatter(xx, yy, marker=',', s=1., color=color)
        s = e
        ax.text(0.8, y_inter, names[i] + f": {fr:.1f}Hz", color=color, fontsize=9, transform=ax.transAxes)
        ax.set_ylim([0, 900])
        ax.set_yticks([0, 900])
        ax.set_yticklabels([0, 900])
        ax.set_ylabel("Neuron")
        ax.set_xlim([0, 10000])
        ax.set_xticks([0, 500, 9999])
        ax.set_xticklabels([0, 500, 1000])
        ax.set_xlabel("Time (ms)")
        # ax.invert_yaxis()
    fig.savefig(os.path.join(write_path, "column.png"))

def run_simulation_voxel(ip, block_path, write_path):
    os.makedirs(write_path, exist_ok=True)
    block_model = block_gpu(ip, block_path, 0.1, route_path=None)

    total_neurons = int(block_model.total_neurons)
    neurons_per_population = block_model.neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    populations = block_model.subblk_id.cpu().numpy()
    print(populations)  # 2 3 4 .. 9
    sample_size = np.array([240, 60])
    chunks = np.add.accumulate(sample_size)[:-1]
    sample_idx = []
    for i, size in enumerate(sample_size):
        sample_idx.append(
            np.random.choice(np.arange(neurons_per_population_base[i], neurons_per_population_base[i + 1]), size,
                             replace=False))
    sample_idx = np.concatenate(sample_idx)
    sample_number = sample_idx.shape[0]
    print("sample_num:", sample_number)
    assert sample_idx.max() < total_neurons
    sample_idx = torch.from_numpy(sample_idx.astype(np.int64)).cuda()
    block_model.set_samples(sample_idx)

    block_model.update_ou_background_stimuli(10., 0.5, 0.2)

    def _update_gui(_gui_laminar):
        for i, idx in enumerate(np.arange(10, 14)):
            population_info = np.stack(np.meshgrid(populations, idx, indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            param_info = gui_laminar[:, i].astype(np.float32)
            param_info = torch.from_numpy(param_info).cuda()
            block_model.assign_property_by_subblk(population_info, param_info)

    gui_laminar = np.array([
                            [0.0007, 0.0001, 0.003, 0.0003],
                            [0.0007, 0.0001, 0.003, 0.0003]])
    print("gui", gui_laminar)
    _update_gui(gui_laminar)

    _ = block_model.run(30000, freqs=True, vmean=False, sample_for_show=False)
    temp_spike = []
    for return_info in block_model.run(10000, freqs=False, vmean=False, iou=True, sample_for_show=True):
        spike, vi = return_info
        # spike &= (torch.abs(vi - v_th) / 50 < 1e-5)
        temp_spike.append(spike)
    temp_spike = torch.stack(temp_spike, dim=0)
    Spike = torch_2_numpy(temp_spike)
    logs = np.split(Spike, chunks, axis=1)
    fr = np.array([log.mean() * 10000 for log in logs])
    print("firing", fr)
    block_model.shutdown()
    print("Done")

    fig = plt.figure(figsize=(6, 7))
    Spike = Spike[-10000:, :]
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.75], frameon=True)
    spike_events = [Spike[:, i].nonzero()[0] for i in range(Spike.shape[1])]
    s = 0
    colors = ["tab:blue", "tab:red"]
    total = np.sum(sample_size)
    names = ["E", "I"]
    for i, size in enumerate(sample_size):
        e = s + size
        color = colors[i % 2]
        y_inter = (s + e) / 2 / total-0.02
        fr = Spike[:, s:e].mean() * 10000
        ax.eventplot(spike_events[s:e], lineoffsets=np.arange(s, e), colors=color)
        # xx, yy = Spike[:, s:e].nonzero()
        # yy = yy + s
        # ax.scatter(xx, yy, marker=',', s=1., color=color)
        s = e
        ax.text(0.8, y_inter, names[i] + f": {fr:.1f}Hz", color=color, fontsize=9, transform=ax.transAxes)
        ax.set_ylim([0, 900])
        ax.set_yticks([0, 900])
        ax.set_yticklabels([0, 900])
        ax.set_ylabel("Neuron")
        ax.set_xlim([0, 10000])
        ax.set_xticks([0, 500, 9999])
        ax.set_xticklabels([0, 500, 1000])
        ax.set_xlabel("Time (ms)")
        # ax.invert_yaxis()
    fig.savefig(os.path.join(write_path, "voxel.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CUDA Data siulation")
    parser.add_argument("--ip", type=str, default="11.5.4.3:50051")
    parser.add_argument("--block_path", type=str,
                        default="/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/two_columns_d1000_iter0.1/uint8")
    parser.add_argument("--write_path", type=str, default="../results/two_columns_result")
    args = parser.parse_args()
    run_simulation_column(args.ip, args.block_path, args.write_path)

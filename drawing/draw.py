# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 16:25 
# @Author : lepold
# @File : draw.py

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from utils.helpers import np_move_avg


def draw(log_path, freqs, block_size, sample_idx, write_path, name_path, bold_path, real_bold_path, vmean_path,
         vsample):
    """
    Draw some valid figures for returned info from :ref:`block simulation <simulation>`
    and more detail ref to source code.

    """
    log = np.load(log_path)
    Freqs = np.load(freqs)
    vmean = np.load(vmean_path)
    vsample = np.load(vsample)
    if len(log.shape) > 2:
        log = log.reshape([-1, log.shape[-1]])
        vsample = vsample.reshape([-1, vsample.shape[-1]])
        Freqs = Freqs.reshape([-1, Freqs.shape[-1]])
        vmean = vmean.reshape((-1, vmean.shape[-1]))
    os.makedirs(write_path, exist_ok=True)
    block_size = np.load(block_size)
    name = loadmat(name_path)['AAL']
    bold_simulation = np.load(bold_path)
    # bold_simulation = bold_simulation[30:, ]
    T, voxels = bold_simulation.shape
    bold_y = loadmat(real_bold_path)["nii_ts_bandzscore"]
    rearange_index = np.load("/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/rearange_index.npy")
    bold_y = bold_y[:, rearange_index]
    bold_y = 0.01 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
    real_bold = bold_y[:T, :voxels]

    property = np.load(sample_idx)
    unique_voxel = np.unique(property[:, 1])

    def run_voxel(i):
        print("draw voxel: ", i)
        idx = np.where(property[:, 1] == i)[0]
        subpopu_idx = np.unique(property[idx, 2])
        sub_log = log[:, idx]
        sub_vsample = vsample[:, idx]
        region = property[idx[0], 3]
        if region < 90:
            sub_name = name[region // 2][0][0] + '-' + ['L', 'R'][region % 2]
        elif region == 91:
            sub_name = 'LGN-L'
        else:
            sub_name = 'LGN-R'
        if real_bold is None:
            sub_real_bold = None
        else:
            sub_real_bold = real_bold[:, i]
        sub_sim_bold = bold_simulation[:, i]
        sub_vmean = vmean[:, subpopu_idx]

        _, split = np.unique(property[idx, 2], return_counts=True)
        split = np.add.accumulate(split)
        split = np.insert(split, 0, 0)
        index = np.unique(property[idx, 2])
        fire_rate = Freqs[:, index]
        return write_path, block_size, i, sub_log, split, sub_name, fire_rate, index, sub_real_bold, sub_sim_bold, sub_vsample, sub_vmean

    n_nlocks = [process_block(*run_voxel(i)) for i in unique_voxel]

    table = pd.DataFrame({'Name': [b[1] for b in n_nlocks],
                          'Visualization': ['\includegraphics[scale=0.04125]{IMG%i.JPG}' % (info[0] + 1,) for info in
                                            n_nlocks],
                          'Neuron Sample': ['\includegraphics[scale=0.275]{log_%i.png}' % (info[0],) for info in
                                            n_nlocks],
                          'Layer fr': ['\includegraphics[scale=0.275]{frpoup_%i.png}' % (info[0],) for info in
                                       n_nlocks],
                          'Bold': ['\includegraphics[scale=0.275]{bold_%i.png}' % (info[0],) for info in
                                   n_nlocks],
                          'FR pdf': ['\includegraphics[scale=0.275]{statis_%i.png}' % (info[0],) for info in
                                     n_nlocks],
                          'CV': ['\includegraphics[scale=0.275]{cv_%i.png}' % (info[0],) for info in
                                 n_nlocks],
                          })
    column_format = '|l|c|c|c|c|c|c|c|'

    with open(os.path.join(write_path, 'chart.tex'), 'w') as f:
        f.write("""
                    \\documentclass[varwidth=25cm]{standalone}
                    \\usepackage{graphicx}
                    \\usepackage{longtable,booktabs}
                    \\usepackage{multirow}
                    \\usepackage{multicol}
                    \\begin{document}
                """)
        f.write(table.to_latex(bold_rows=True, longtable=True, multirow=True, multicolumn=True, escape=False,
                               column_format=column_format))

        f.write("""
                    \\end{document}
                """)

    print('-')


def process_block(write_path, real_block, block_i, log, split, name, fire_rate, subblk_index, bold_real=None,
                  bold_sim=None, sub_vsample=None, sub_vmean=None, time=1200, slice_window=800, stride=200):
    """
    This function provides a number of drawing options that can be optionally commented out if not required.

    """
    block_size = log.shape[-1]
    real_block_size = real_block[subblk_index]
    names = ['L2/3', 'L4', 'L5', 'L6']

    # frequence = log.sum() * 1000 / log.shape[0] / activate_idx.shape[0]
    frequence_map = torch.from_numpy(log.astype(np.float32)).transpose(0, 1).unsqueeze(1)
    frequence_map = 1000 / slice_window * torch.conv1d(frequence_map, torch.ones([1, 1, slice_window]),
                                                       stride=stride).squeeze().transpose(0, 1).numpy()
    fig_fre = plt.figure(figsize=(4, 4), dpi=500)
    fig_fre.gca().hist(frequence_map.reshape([-1]), 100, density=True)
    fig_fre.gca().set_yscale('log')
    fig_fre.savefig(os.path.join(write_path, "statis_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_fre)

    # fire_rate_ = (fire_rate[-time:, ::2] + fire_rate[-time:, 1::2]) / (real_block_size[::2] + real_block_size[1::2])
    # fire_rate_ = np_move_avg(fire_rate_, n=5, mode="same")
    # fig_frequence = plt.figure(figsize=(4, 4), dpi=500)
    # ax1 = fig_frequence.add_subplot(1, 1, 1)
    # ax1.grid(False)
    # ax1.set_xlabel('time(ms)')
    # ax1.set_ylabel('Instantaneous fr(hz)')
    # for i in range(fire_rate_.shape[1]):
    #     ax1.plot(fire_rate_[:, i], label=names[i])
    # ax1.legend(loc='best')
    # fig_frequence.savefig(os.path.join(write_path, "fr_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    # plt.close(fig_frequence)

    activate_idx = (log.sum(0) > 0).nonzero()[0]  # log.shape=[100*800, 300]
    cvs = []
    for i in activate_idx:
        out = log[:, i].nonzero()[0]
        if out.shape[0] >= 3:
            fire_interval = out[1:] - out[:-1]
            cvs.append(fire_interval.std() / fire_interval.mean())

    cv = np.array(cvs)
    fig_cv = plt.figure(figsize=(4, 4), dpi=500)
    fig_cv.gca().hist(cv, 100, range=(0, 2), density=True)
    fig_cv.savefig(os.path.join(write_path, "cv_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_cv)

    fig = plt.figure(figsize=(4, 4), dpi=500)
    axes = fig.add_subplot(1, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    fire_rate_ = np_move_avg(fire_rate[-2000:, ], n=10, mode="valid")
    if len(split) > 3:
        df = pd.DataFrame(fire_rate_, columns=['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I'])
    else:
        df = pd.DataFrame(fire_rate_, columns=['E', 'I'])
    df.plot.box(vert=False, showfliers=False, widths=0.2, color=color, ax=axes)
    fig.savefig(os.path.join(write_path, "frpopu_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    valid_idx = np.where(log.mean(axis=0) > 0.001)[0]
    instanous_fr = log[-2000:-800, valid_idx].mean(axis=1)
    instanous_fr = np_move_avg(instanous_fr, 10, mode="valid")
    length = len(instanous_fr)
    fig = plt.figure(figsize=(8, 4), dpi=500)
    ax1 = fig.add_subplot(1, 1, 1, frameon=False)
    ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax1.grid(False)
    ax1.set_xlabel('time(ms)')
    axes = fig.add_subplot(2, 1, 1)
    if len(split) > 3:
        sub_vmean = sub_vmean[-2000:-800, :] * np.array(
            [0.24355972, 0.05152225, 0.25995317, 0.07025761, 0.11709602, 0.03512881, 0.18501171, 0.03747072])
        sub_vmean = sub_vmean.sum(axis=-1)
        for t in range(8):
            x, y = log[-2000:-800, split[t]:split[t + 1]].nonzero()
            if t % 2 == 0:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="blue")
            else:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="red")
        names = ['L2/3', 'L4', 'L5', 'L6']
        names_loc = split[:-1][::2]
    else:
        sub_vmean = sub_vmean[-2000:-800, :] * np.array([0.8, 0.2])
        sub_vmean = sub_vmean.sum(axis=-1)
        for t in range(2):
            x, y = log[-2000:-800, split[t]:split[t + 1]].nonzero()

            if t % 2 == 0:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="blue")
            else:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="red")
        names = ["E", "I"]
        names_loc = split[:-1]
    axes.set_title("fre of spiking neurons: %.2f" % instanous_fr.mean())
    axes.set_xlim((0, length))
    axes.set_ylim((0, block_size))
    plt.yticks(names_loc, names)
    axes.invert_yaxis()
    axes.set_aspect(aspect=1)
    axes = fig.add_subplot(2, 1, 2)
    axes.plot(instanous_fr, c="black")
    fig.savefig(os.path.join(write_path, "log_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # fig_vi = plt.figure(figsize=(8, 4), dpi=500)
    # ax1 = fig_vi.add_subplot(1, 1, 1, frameon=False)
    # ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # ax1.grid(False)
    # ax1.set_xlabel('time(ms)')
    # axes = fig_vi.add_subplot(2, 1, 1)
    # sub_vsample = sub_vsample[-2000:-800, :]
    # axes.imshow(sub_vsample.T, vmin=-65, vmax=-50, cmap='jet', origin="lower")
    # axes = fig_vi.add_subplot(2, 1, 2)
    # axes.plot(sub_vmean, c="r")
    # fig_vi.savefig(os.path.join(write_path, "vi_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    # plt.close(fig_vi)

    fig_bold = plt.figure(figsize=(4, 4), dpi=500)
    if bold_real is not None:
        fig_bold.gca().plot(np.arange(len(bold_real)), bold_real, "r-", label="real")
    fig_bold.gca().plot(np.arange(len(bold_sim)), bold_sim, "b-", label="sim")
    fig_bold.gca().set_ylim((0., 0.08))
    fig_bold.gca().legend(loc="best")
    fig_bold.gca().set_xlabel('time')
    fig_bold.gca().set_ylabel('bold')
    fig_bold.savefig(os.path.join(write_path, "bold_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_bold)

    return block_i, name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model simulation")
    parser.add_argument("--log_path", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/spike_after_assim_1.npy")
    parser.add_argument("--freqs", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/freqs_after_assim_1.npy")
    parser.add_argument("--block_size", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/blk_size.npy")
    parser.add_argument("--sample_idx", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/sample_idx.npy")
    parser.add_argument("--name_path", type=str,
                        default="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/aal_names.mat")
    parser.add_argument("--bold_path", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/bold_after_assim.npy")
    parser.add_argument("--real_bold_path", type=str,
                        default="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz")
    parser.add_argument("--vmean_path", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/vmean_after_assim_1.npy")
    parser.add_argument("--vsample", type=str,
                        default="../data/subject1/rest_da_ampa/simulation_aug_13th/vi_after_assim_1.npy")
    parser.add_argument("--write_path", type=str, default="../data/subject1/rest_da_ampa/simulation_aug_13th/fig")
    args = parser.parse_args()
    draw(args.log_path, args.freqs, args.block_size, args.sample_idx, args.write_path, args.name_path, args.bold_path,
         args.real_bold_path, args.vmean_path,
         args.vsample)

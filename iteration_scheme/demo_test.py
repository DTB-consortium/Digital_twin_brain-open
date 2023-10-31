# -*- coding: utf-8 -*- 
# @Time : 2022/9/28 17:20 
# @Author : lepold
# @File : demo_test.py
import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import gridspec

import lifnn2
from default_params import bold_params
from generation.read_block import connect_for_block
from models.bold_model_pytorch import BOLD


class Testcase(unittest.TestCase):
    def test_compare_01ms_schemes(self):
        os.makedirs(os.path.join("Results2", "big_fig"), exist_ok=True)
        os.makedirs(os.path.join("Results2", "small_fig"), exist_ok=True)
        scaling = 0.8  # 0.3 for stable fire in weak coupling
        # tau_list = np.linspace(np.power(1, 1 / 10),  np.power(3, 1/ 10), 10, endpoint=True) ** 10
        tau_list = np.arange(0.5, 5., 2.)
        chunk_list = [10, 30]
        for id1, chunk in enumerate(chunk_list):
            for id2, tau in enumerate(tau_list):
                print(f"===============\nprocessing {id1} and {id2}\n=====================")
                info = dict()
                # gui = torch.tensor([0.01217722, 0.00106456, 0.18987342, 0.01022925], dtype=torch.float32)
                property, w_uij = connect_for_block(
                    "../small_blocks/normal_d20")
                property[:, 10:14] = property[:, 10:14] * scaling
                N = property.shape[0]
                property[:, 5] = tau
                property = property.cuda("cuda:2")
                w_uij = w_uij.cuda("cuda:2")
                model2 = lifnn2.NN01ms01ms_asusual(property, w_uij, delta_t=0.1)
                model3 = lifnn2.NN01ms1ms(property, w_uij, delta_t=0.1, chunk=chunk)
                bold_params['delta_t'] = 1e-4
                bold2 = BOLD(**bold_params)
                bold3 = BOLD(**bold_params)
                neuron_id = 148
                v2 = []
                j1 = []
                j2 = []
                j3 = []
                j4 = []
                log = []
                bold = []
                raster = []
                T_all = 25000
                T = 5000
                start = time.time()
                seed = neuron_id + T_all
                torch.manual_seed(seed)
                print("T_all", T_all)
                for i in range(T_all):
                    print(i, "in", T_all, end="\r")
                    act = model2.run(noise_rate=0.0005, isolated=False)
                    bold.append(bold2.run(act.float() / N * 10))
                    v2.append(model2.V_i[neuron_id].cpu().numpy())
                    j1.append(model2.I_ui[0, neuron_id].cpu().numpy())
                    j2.append(model2.I_ui[1, neuron_id].cpu().numpy())
                    j3.append(model2.I_ui[2, neuron_id].cpu().numpy())
                    j4.append(model2.I_ui[3, neuron_id].cpu().numpy())
                    log.append(model2.active.float().mean().cpu().numpy())
                    if i >= (T_all - 5000):
                        raster.append(model2.active[200:300].cpu().numpy())
                raster = np.array(raster)
                print("raster shape", raster.shape)
                # ====================
                # raster = np.array(raster)
                # print("fr", np.array(log).mean() * 10000)
                # fig_temp = plt.figure(figsize=(5, 2), dpi=100)
                # xx, yy = raster.nonzero()
                # print(xx[:20])
                # print(yy[:20])
                # fig_temp.gca().scatter(xx, yy, marker=',', c='k', s=1.)
                # fig_temp.savefig("./test_raster.png", dpi=100, bbox_inches='tight')
                # plt.close(fig_temp)
                # exit()
                # =====================

                print("\n")
                torch.manual_seed(seed)
                for i in range(T_all):
                    print(i, "in", T_all, end="\r")
                    act = model3.run(noise_rate=0.0005, isolated=False)
                    bold.append(bold3.run(act.float() / N * 10))
                    v2.append(model3.V_i[neuron_id].cpu().numpy())
                    j1.append(model3.I_ui[0, neuron_id].cpu().numpy())
                    j2.append(model3.I_ui[1, neuron_id].cpu().numpy())
                    j3.append(model3.I_ui[2, neuron_id].cpu().numpy())
                    j4.append(model3.I_ui[3, neuron_id].cpu().numpy())
                    log.append(model3.active.float().mean().cpu().numpy())
                v2 = np.stack(v2).reshape((2, -1))[:, -T:]
                j1 = np.stack(j1).reshape((2, -1))[:, -T:]
                j2 = np.stack(j2).reshape((2, -1))[:, -T:]
                j3 = np.stack(j3).reshape((2, -1))[:, -T:]
                j4 = np.stack(j4).reshape((2, -1))[:, -T:]
                bold = torch.stack(bold).cpu().numpy()
                bold = bold.reshape((2, -1))[:, -T:]
                info['bold'] = bold.mean(axis=-1)
                log = np.stack(log).reshape((2, -1))[:, 10000:]
                fr = log.mean(axis=1) / 0.1 * 1000
                print("fr", fr)
                info["fr"] = fr
                print(f"run time {time.time() - start:.2f}")

                fig = plt.figure(figsize=(10, 12), dpi=100)
                gs1 = gridspec.GridSpec(6, 1)
                gs1.update(left=0.06, right=0.96, top=0.94, bottom=0.07, hspace=0.18)
                ax = dict()
                for i in range(6):
                    ax[i] = plt.subplot(gs1[i, 0], frameon=True)
                names = ['Standard', 'Efficient']
                info["names"] = names
                info["configuration"] = [f"weak_coupling_Scaling{scaling}_Tref{tau:.2f}_Chunk{chunk}"] * 2
                df = pd.DataFrame(info)
                if id1 == 0 and id2 == 0:
                    df.to_csv('Results2/noise_dealy_stable_state_info.csv', mode='a', header=True, index=True)
                else:
                    df.to_csv('Results2/noise_delay_stable_state_info.csv', mode='a', header=False, index=True)
                ylabels = ["V", r"$I_{ampa}$", r"$I_{nmda}$", r"$I_{gabaA}$", r"$I_{gabaB}$", "Bold", "V"]

                fig_v = plt.figure(figsize=(5, 2), dpi=100)
                ax[6] = fig_v.gca()
                for i in range(2):
                    ax[6].plot(np.arange(T_all - T, T_all), v2[i], label=names[i] + f": {fr[i]:.2f} Hz")

                fig_raster = plt.figure(figsize=(5, 2), dpi=100)
                ax[7] = fig_raster.gca()
                ax[7].scatter(*raster.nonzero(), marker='.', s=1., c='k')
                ax[7].set_xlim([0, 5000])
                ax[7].set_ylim([0, 100])
                ax[7].set_xticks([0, 2500, 500])
                ax[7].set_xticklabels([0, 250, 500])
                ax[7].set_xlabel("Time(ms)")
                ax[7].set_ylabel("Index")

                for i in range(2):
                    ax[0].plot(np.arange(T_all - T, T_all), v2[i], label=names[i] + f": {fr[i]:.2f} Hz")
                    ax[1].plot(np.arange(T_all - T, T_all), j1[i])
                    ax[2].plot(np.arange(T_all - T, T_all), j2[i])
                    ax[3].plot(np.arange(T_all - T, T_all), j3[i])
                    ax[4].plot(np.arange(T_all - T, T_all), j4[i])
                    ax[5].plot(np.arange(T_all - T, T_all), bold[i])
                for i in range(7):
                    ax[i].set_xlim([T_all - T, T_all])
                    ax[i].set_xticks(np.linspace(T_all - T, T_all, 4, endpoint=False, dtype=np.int_))
                    ax[i].set_xticklabels(
                        np.linspace(int((T_all - T) / 10), int(T_all / 10), 4, endpoint=False, dtype=np.int_))
                    ax[i].set_ylabel(ylabels[i])
                ax[0].legend(loc="best")
                ax[0].set_title(f"scale_{scaling},Tref_{tau}")
                ax[4].set_xlabel("Time(ms)")
                fig.savefig(f"Results2/big_fig/stable_state_coupling_Scaling_{scaling}_Tref_{tau:.2f}_chunk_{chunk}.png",
                            dpi=300)
                fig_v.savefig(f"Results2/small_fig/v_for_stable_state_tref_{id2}_chunk_{id1}.png", dpi=300, bbox_inches='tight')
                fig_raster.savefig(f"Results2/small_fig/raster_for_stable_state_tref_{id2}_chunk_{id1}.png", dpi=300, bbox_inches='tight')
                plt.close(fig_raster)
                plt.close(fig_v)
                plt.close(fig)
        # fig.show()
        print("Done")

    def _test_compare_3_schemes(self):
        plt.rcParams['figure.figsize'] = (12, 9)
        fig = plt.gcf()
        axes = {}
        gs1 = gridspec.GridSpec(3, 1)
        gs1.update(left=0.06, right=0.36, top=0.94, bottom=0.07, hspace=0.1)
        for j in range(3):
            axes[j] = plt.subplot(gs1[j, 0], frameon=True)

        gs2 = gridspec.GridSpec(3, 1)
        gs2.update(left=0.46, right=0.96, top=0.94, bottom=0.07, hspace=0.1)
        for j in range(3):
            axes[j + 3] = plt.subplot(gs2[j, 0], frameon=True)
            axes[j + 3].spines['right'].set_color(None)
            axes[j + 3].spines['top'].set_color(None)

        for i in range(6):
            if i % 3 == 2:
                axes[i].set_xlabel("Time(ms)")
            if i < 3:
                axes[i].set_ylabel("Neuron")
            if i >= 3:
                axes[i].set_ylabel("V")

        scaling1 = 0.4
        scaling2 = 0.4
        tau = 0.3
        property, w_uij = connect_for_block(
            r"C:\Users\dell\Documents\WeChat Files\wxid_yv8ys00jk82222\FileStorage\File\2022-09\Code_for_Longbin\Code_for_Longbin\runjobs\single_small\single")
        property[:, 10:12] = property[:, 10:12] * scaling1
        property[:, 12:14] = property[:, 12:14] * scaling2
        property[:, 5] = tau
        property = property.cuda()
        w_uij = w_uij.cuda()
        model1 = lifnn2.NN1ms1ms(property, w_uij, delta_t=1)
        model2 = lifnn2.NN01ms01ms_asusual(property, w_uij, delta_t=0.1)
        model3 = lifnn2.NN01ms1ms(property, w_uij, delta_t=0.1)
        neuron_id = 450

        # ================================================
        torch.manual_seed(10241)
        log1 = []
        v1 = []
        T = 500
        T_all = 800
        for i in range(T_all):
            print(i, end="\r")
            model1.run(noise_rate=0.01, isolated=False)
            log1.append(model1.active.cpu().numpy())
            v1.append(model1.V_i[neuron_id].cpu().numpy())
        log1 = np.stack(log1, axis=0)[T:T_all, 100:200]
        v1 = np.stack(v1)[T:T_all]
        axes[0].scatter(*log1.nonzero(), marker=",", c='k', s=1.)
        axes[0].set_xticks(np.linspace(0, T_all - T, 3, dtype=np.int_))
        axes[0].set_xticklabels(np.linspace(T, T_all, 3, dtype=np.int_))
        axes[0].set_xlim([0, T_all - T])
        axes[0].set_ylim([0, 100])
        axes[3].plot(np.arange(T, T_all), v1, color="r", lw=1.)
        axes[3].set_xticks(np.linspace(T, T_all, 3, dtype=np.int_))

        # ==================================================
        torch.manual_seed(10241)
        log2 = []
        v2 = []
        for i in range(T_all * 10):
            print(i, end="\r")
            model2.run(noise_rate=0.001, isolated=False)
            log2.append(model2.active.cpu().numpy())
            v2.append(model2.V_i[neuron_id].cpu().numpy())
        log2 = np.stack(log2, axis=0)[T * 10: T_all * 10, 100:200]
        log2 = log2.reshape((-1, 10, 100))
        log2 = log2.sum(axis=1)
        v2 = np.stack(v2)[T * 10:T_all * 10]
        axes[1].scatter(*log2.nonzero(), marker=",", c='k', s=1.)
        axes[1].set_xticks(np.linspace(0, T_all - T, 3, dtype=np.int_))
        axes[1].set_xticklabels(np.linspace(T, T_all, 3, dtype=np.int_))
        axes[1].set_xlim([0, T_all - T])
        axes[1].set_ylim([0, 100])
        axes[4].plot(np.arange(T * 10, T_all * 10), v2, color="r", lw=1.)
        axes[4].set_xticks(np.linspace(T * 10, T_all * 10, 3, dtype=np.int_))
        axes[4].set_xticklabels(np.linspace(T, T_all, 3, dtype=np.int_))

        # =====================================================
        torch.manual_seed(10241)
        log3 = []
        v3 = []
        for i in range(T_all * 10):
            print(i, end="\r")
            model3.run(noise_rate=0.001, isolated=False)
            log3.append(model3.active.cpu().numpy())
            v3.append(model3.V_i[neuron_id].cpu().numpy())
        log3 = np.stack(log3, axis=0)[T * 10: T_all * 10, 100:200]
        log3 = log3.reshape((-1, 10, 100))
        log3 = log3.sum(axis=1)
        v3 = np.stack(v3)[T * 10:T_all * 10]
        axes[2].scatter(*log3.nonzero(), marker=",", c='k', s=1.)
        axes[2].set_xticks(np.linspace(0, T_all - T, 3, dtype=np.int_))
        axes[2].set_xticklabels(np.linspace(T, T_all, 3, dtype=np.int_))
        axes[2].set_xlim([0, T_all - T])
        axes[2].set_ylim([0, 100])

        axes[5].plot(np.arange(T * 10, T_all * 10), v3, color="r", lw=1.)
        axes[5].set_xticks(np.linspace(T * 10, T_all * 10, 3, dtype=np.int_))
        axes[5].set_xticklabels(np.linspace(T, T_all, 3, dtype=np.int_))
        info = ["scheme 1", "scheme 2", "scheme 3"]
        for i in np.arange(3, 6):
            axes[i].text(0.7, 1.05, r'\bfseries{}' + info[i - 3],
                         fontdict={'fontsize': 13, 'weight': 'bold',
                                   'horizontalalignment': 'left', 'verticalalignment':
                                       'bottom'}, transform=axes[i].transAxes)
        fig.savefig("iteration_three_scheme.png", dpi=100)


if __name__ == '__main__':
    unittest.main()

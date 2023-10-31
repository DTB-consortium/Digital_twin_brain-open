# -*- coding: utf-8 -*- 
# @Time : 2022/8/10 14:25 
# @Author : lepold
# @File : initialzie_params.py

import os
import re
import time

import numpy as np
import torch

from generation.make_block import connect_for_multi_sparse_block
from generation.read_block import connect_for_block
from models.block import block
from models.block_new import block as block_new


def initilalize_gui_in_homo_block(delta_t: float = 1., default_Hz=20, max_output_Hz=100, T_ref=5, degree=100, g_Li=0.03,
                                  V_L=-75,
                                  V_rst=-65, V_th=-50, path="./initialize_gui"):
    """
    A heuristic method to find appropriate parameter for single neuron.

    """

    gap = V_th - V_rst

    noise_rate = 1 / (1000 / default_Hz)
    max_delta_raise = gap / (1000 / max_output_Hz - T_ref)
    default_delta_raise = gap / (1000 / default_Hz - T_ref)

    leaky_compensation = g_Li * ((V_th + V_rst) / 2 - V_L)

    label = torch.tensor([0.3 * (max_delta_raise + leaky_compensation),
                          0.7 * (max_delta_raise + leaky_compensation),
                          0.5 * (max_delta_raise - default_delta_raise),
                          0.5 * (max_delta_raise - default_delta_raise)])
    print("label", label.tolist())

    gui = torch.ones(4) * 0.01

    def _test_gui(delta_t, time, noise_rate, path, document_time=None):
        property, w_uij = connect_for_block(path)
        property = property.cuda("cuda:1")
        w_uij = w_uij.cuda("cuda:1")
        B = block(
            node_property=property,
            w_uij=w_uij,
            delta_t=delta_t,
        )
        psc_list = []
        Hz_list = []
        max_iter = int(time / delta_t)
        noise_rate = noise_rate * delta_t
        if document_time is None:
            document_iter = int((1000 / delta_t))
        else:
            assert document_time < time
            document_iter = int(document_time / delta_t)
        for k in range(max_iter):
            B.run(noise_rate=noise_rate, isolated=True)
            psc_list.append(B.I_ui.mean(-1).abs())
            Hz_list.append(float(B.active.sum()) / property.shape[0])
        out = torch.stack(psc_list[-document_iter:]).mean(0)
        Hz = torch.tensor(Hz_list[-document_iter:]).mean() / delta_t
        return out.cpu().numpy(), Hz.cpu().numpy()

    block_path = os.path.join(path, "blocks")
    os.makedirs(block_path, exist_ok=True)
    total_info = np.zeros((20, 10))
    start = time.time()

    for i in range(20):
        block_name = re.compile('block_[0-9]*.npz')
        blocks = [name for name in os.listdir(block_path) if block_name.fullmatch(name)]
        if len(blocks) > 0:
            for name in blocks:
                os.remove(os.path.join(block_path, name))
        prob = torch.tensor([[0.8, 0.2], [0.8, 0.2]], dtype=torch.float32)
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': gui,
                              "V_reset": -65,
                              "noise_rate": noise_rate * delta_t,
                              'tao_ui': (2, 40, 10, 50),
                              'size': size} for size in [8000, 2000]]
        connect_for_multi_sparse_block(prob, population_kwards, degree=degree, init_min=0,
                                       init_max=1, prefix=block_path)
        out, Hz = _test_gui(delta_t=delta_t, time=3000, noise_rate=noise_rate, path=block_path, document_time=2000)
        total_info[i, 0] = i
        total_info[i, 1:5] = gui
        total_info[i, 5:9] = out
        total_info[i, 9] = Hz
        print(f"\niteration | {i} ")
        print("input spike: ", noise_rate)
        print("gui: ", gui.tolist())
        print("\n")
        print("psc: ", out.tolist())
        print("fr", Hz)

        gui = gui * label / out
    end = time.time()
    np.save(os.path.join(path, "iteration_info.npy"), total_info)
    np.savetxt(os.path.join(path, "iteration.txt"), total_info,
               fmt=["%d", ] + ["%.8f", ] * 4 + ["%.2f", ] * 4 + ["%.3f"], header="iteration | gui | pscs | fr",
               footer=f"cost time {end - start:.2f}")
    print("Done")


def initilalize_gui_single_neruonal_model(delta_t=0.1, initial_input=0.001, i_mean=0.8, i_sigma=0.1, tau_i=5.,
                                          degree=100, path="./initialize_gui"):
    """
    A heuristic method to find appropriate parameter for single neuron.

    """

    def _test_gui(delta_t, Hz, time, path):
        property, w_uij = connect_for_block(path)
        property = property.cuda("cuda:1")
        w_uij = w_uij.cuda("cuda:1")
        B = block_new(
            node_property=property,
            w_uij=w_uij,
            delta_t=delta_t,
            i_mean=i_mean,
            i_sigma=i_sigma,
            tau_i=tau_i
        )
        psc_list = []
        Hz_list = []
        max_iter = int(time / delta_t)
        for k in range(max_iter):
            B.run(debug_mode=True, noise_rate=Hz)
            psc_list.append(B.I_ui.mean(-1).abs())
            Hz_list.append(float(B.active.sum()) / property.shape[0])
        out = torch.stack(psc_list[-1000:]).mean(0)
        Hz = torch.tensor(Hz_list[-1000:]).mean() / delta_t
        return out.cpu().numpy(), Hz.cpu().numpy()

    block_path = os.path.join(path, "blocks")
    os.makedirs(block_path, exist_ok=True)
    Hz = initial_input
    gui = np.array([0.00286671, 0.00041940, 0.02096406, 0.00133286], dtype=np.float_)
    psc_compare = []
    total_info = np.zeros((20, 10))
    start = time.time()
    for i in range(10):
        block_name = re.compile('block_[0-9]*.npz')
        blocks = [name for name in os.listdir(block_path) if block_name.fullmatch(name)]
        if len(blocks) > 0:
            for name in blocks:
                os.remove(os.path.join(block_path, name))
        prob = torch.tensor([[0.8, 0.2], [0.8, 0.2]], dtype=torch.float32)
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': gui,
                              "V_reset": -65,
                              "noise_rate": initial_input,
                              'tao_ui': (2, 40, 10, 50),
                              'size': size} for size in [8000, 2000]]
        connect_for_multi_sparse_block(prob, population_kwards, degree=degree, init_min=0,
                                       init_max=1, prefix=block_path)
        out, Hz = _test_gui(delta_t=delta_t, Hz=float(Hz), time=2000, path=block_path)
        total_info[i, 0] = i
        total_info[i, 1:5] = gui
        total_info[i, 5:9] = out
        total_info[i, 9] = Hz
        print(f"\niteration | {i} ")
        print("input spike: ", Hz)
        print("gui: ", gui.tolist())
        print("psc: ", out.tolist())
        print("fr", Hz)
        psc_compare.append(out)
        if len(psc_compare) > 1:
            gui = gui / (psc_compare[1] / psc_compare[0])
            psc_compare.pop(0)
    end = time.time()
    np.save(os.path.join(path, "iteration_info.npy"), total_info)
    np.savetxt(os.path.join(path, "iteration.txt"), total_info,
               fmt=["%d", ] + ["%.8f", ] * 4 + ["%.2f", ] * 4 + ["%.3f"], header="iteration | gui | pscs | fr",
               footer=f"cost time {end - start:.2f}")
    print("Done")


def initilalize_gui_in_new_background_current(delta_t: float = 1., default_Hz=20, i_mean=0.8, i_sigma=0.1, tau_i=5.,
                                              max_output_Hz=100, T_ref=5, degree=100, g_Li=0.03,
                                              V_L=-75,
                                              V_rst=-65, V_th=-50, path="./initialize_gui"):
    """
    A heuristic method to find appropriate parameter for single neuron.

    """

    gap = V_th - V_rst

    noise_rate = 1 / (1000 / default_Hz)
    max_delta_raise = gap / (1000 / max_output_Hz - T_ref)
    default_delta_raise = gap / (1000 / default_Hz - T_ref)

    leaky_compensation = g_Li * ((V_th + V_rst) / 2 - V_L)
    print(f"max: {max_delta_raise}, default: {default_delta_raise}, leaky: {leaky_compensation}")
    assert max_delta_raise > 0 and default_delta_raise > 0

    label = torch.tensor([0.3 * (max_delta_raise + leaky_compensation - i_mean),
                          0.7 * (max_delta_raise + leaky_compensation - i_mean),
                          0.5 * (max_delta_raise - default_delta_raise),
                          0.5 * (max_delta_raise - default_delta_raise)])
    print("label", label.tolist())

    gui = label

    def _test_gui(delta_t, noise_rate, time, path, document_time=None):
        property, w_uij = connect_for_block(path)
        property = property.cuda("cuda:1")
        w_uij = w_uij.cuda("cuda:1")
        B = block_new(
            node_property=property,
            w_uij=w_uij,
            delta_t=delta_t,
            i_mean=i_mean,
            i_sigma=i_sigma,
            tau_i=tau_i
        )
        psc_list = []
        Hz_list = []
        noise_rate = noise_rate * delta_t
        max_iter = int(time / delta_t)
        if document_time is None:
            document_iter = int((1000 / delta_t))
        else:
            assert document_time < time
            document_iter = int(document_time / delta_t)
        for k in range(max_iter):
            B.run(noise_rate=noise_rate, debug_mode=True)
            psc_list.append(B.I_ui.mean(-1).abs())
            Hz_list.append(float(B.active.sum()) / property.shape[0])
        out = torch.stack(psc_list[-document_iter:]).mean(0)
        Hz = torch.tensor(Hz_list[-document_iter:]).mean() / delta_t
        return out.cpu().numpy(), Hz.cpu().numpy()

    block_path = os.path.join(path, "blocks")
    os.makedirs(block_path, exist_ok=True)
    total_info = np.zeros((20, 10))
    start = time.time()

    for i in range(20):
        block_name = re.compile('block_[0-9]*.npz')
        blocks = [name for name in os.listdir(block_path) if block_name.fullmatch(name)]
        if len(blocks) > 0:
            for name in blocks:
                os.remove(os.path.join(block_path, name))
        prob = torch.tensor([[0.8, 0.2], [0.8, 0.2]], dtype=torch.float32)
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': gui,
                              "V_reset": -65,
                              "noise_rate": noise_rate * delta_t,
                              'tao_ui': (2, 40, 10, 50),
                              'size': size} for size in [8000, 2000]]
        connect_for_multi_sparse_block(prob, population_kwards, degree=degree, init_min=0,
                                       init_max=1, prefix=block_path)
        out, Hz = _test_gui(delta_t=delta_t, noise_rate=noise_rate, time=3000, path=block_path, document_time=2000)
        total_info[i, 0] = i
        total_info[i, 1:5] = gui
        total_info[i, 5:9] = out
        total_info[i, 9] = Hz
        print(f"\niteration | {i} ")
        print("input spike: ", noise_rate)
        print("gui: ", gui.tolist())
        print("psc: ", out.tolist())
        print("fr", Hz)

        gui = gui * label / out
    end = time.time()
    np.save(os.path.join(path, "iteration_info.npy"), total_info)
    np.savetxt(os.path.join(path, "iteration.txt"), total_info,
               fmt=["%d", ] + ["%.8f", ] * 4 + ["%.2f", ] * 4 + ["%.3f"], header="iteration | gui | pscs | fr",
               footer=f"cost time {end - start:.2f}")
    print("Done")


def initilalize_gui_in_new_background_current_of_column(delta_t: float = 1., default_Hz=20, i_mean=0.8, i_sigma=0.1,
                                                        tau_i=5.,
                                                        max_output_Hz=100, T_ref=5, degree=100, g_Li=0.03,
                                                        V_L=-75,
                                                        V_rst=-65, V_th=-50, path="./initialize_gui", only_test=False):
    """
    A heuristic method to find appropriate parameter for single neuron.

    """
    start = time.time()
    gap = V_th - V_rst

    noise_rate = 1 / (1000 / default_Hz)
    max_delta_raise = gap / (1000 / max_output_Hz - T_ref)
    default_delta_raise = gap / (1000 / default_Hz - T_ref)

    leaky_compensation = g_Li * ((V_th + V_rst) / 2 - V_L)
    print(f"max: {max_delta_raise}, default: {default_delta_raise}, leaky: {leaky_compensation}")
    assert max_delta_raise > 0 and default_delta_raise > 0

    label = torch.tensor([0.3 * (max_delta_raise + leaky_compensation - i_mean),
                          0.7 * (max_delta_raise + leaky_compensation - i_mean),
                          0.5 * (max_delta_raise - default_delta_raise),
                          0.5 * (max_delta_raise - default_delta_raise)])
    label = torch.broadcast_to(label[None, :], (8, 4))
    # print("label", label.tolist())

    gui = label

    def _test_gui(delta_t, noise_rate, time, path, gui, document_time=None):
        property, w_uij = connect_for_block(path)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)
        for i in np.arange(2, 10):
            index = property[:, 3] == i
            property[index, 10:14] = gui[i - 2]
        property = property.cuda("cuda:0")
        w_uij = w_uij.cuda("cuda:0")
        B = block_new(
            node_property=property,
            w_uij=w_uij,
            delta_t=delta_t,
            i_mean=i_mean,
            i_sigma=i_sigma,
            tau_i=tau_i
        )
        psc_list = []
        Hz_list = []
        noise_rate = noise_rate * delta_t
        max_iter = int(time / delta_t)
        if document_time is None:
            document_iter = int((1000 / delta_t))
        else:
            assert document_time < time
            document_iter = int(document_time / delta_t)
        for k in range(max_iter):
            mean_act, mean_psc = B.run(noise_rate=noise_rate, debug_mode=True)
            psc_list.append(mean_psc.abs())  # 4
            Hz_list.append(mean_act)
        out = torch.stack(psc_list[-document_iter:]).mean(0)
        Hz = torch.stack(Hz_list[-document_iter:]).mean(0) / delta_t
        return out.cpu(), Hz.cpu().numpy()

    block_path = os.path.join(path, "uint8")
    os.makedirs(block_path, exist_ok=True)
    if not only_test:
        for i in range(10):
            out, Hz = _test_gui(delta_t=delta_t, noise_rate=noise_rate, time=1500, path=block_path, gui=gui,
                                document_time=1000)
            print(f"\niteration | {i} ")
            print("input spike: ", noise_rate)
            print("gui: ", gui.tolist())
            print("psc: ", out.tolist())
            print("fr", Hz)

            gui = gui * label / out

    if only_test:
        import matplotlib.pyplot as plt
        gui = torch.tensor([[0.005781320855021477, 0.0008467677398584783, 0.06233028694987297, 0.0035223669838160276],
                        [0.011737029999494553, 0.0017201988957822323, 0.12761932611465454, 0.006095761898905039],
                        [0.007711557671427727, 0.0011271658586338162, 0.051886193454265594, 0.003028653794899583],
                        [0.013564364984631538, 0.001982832560315728, 0.09460494667291641, 0.004822766408324242],
                        [0.0060959202237427235, 0.00089932611444965, 0.051885660737752914, 0.0030518490821123123],
                        [0.014037152752280235, 0.002034058328717947, 0.14434434473514557, 0.006506194360554218],
                        [0.007972839288413525, 0.0011654605623334646, 0.03358250483870506, 0.0020530037581920624],
                        [0.028504343703389168, 0.004141360521316528, 0.04464590176939964, 0.0023937018122524023]]) / torch.tensor([11])
        property, w_uij = connect_for_block(block_path)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)
        for i in np.arange(2, 10):
            index = property[:, 3] == i
            property[index, 10:14] = gui[i - 2]
        property = property.cuda("cuda:0")
        w_uij = w_uij.cuda("cuda:0")
        B = block_new(
            node_property=property,
            w_uij=w_uij,
            delta_t=delta_t,
            i_mean=i_mean,
            i_sigma=i_sigma,
            tau_i=tau_i
        )
        Hz_list = []
        log = []
        for k in range(3000):
            mean_act, _ = B.run(debug_mode=False)
            Hz_list.append(mean_act)
            log.append(B.active.cpu().numpy())
        log = np.stack(log, axis=0)
        Hz = torch.stack(Hz_list[-1000:]).mean(0) / delta_t
        print("fr\n", Hz)
        spike_event = [np.nonzero(log[:, i])[0] for i in np.arange(2000, 3000)]
        fig = plt.figure(figsize=(8, 4))
        fig.gca().eventplot(spike_event, lineoffsets=np.arange(1000), colors="tab:blue")
        fig.savefig("./test.png")
    print(f"Done, {time.time() - start:.2f} s ")


if __name__ == '__main__':
    # initilalize_gui_in_homo_block(delta_t=1., default_Hz=20, max_output_Hz=100, T_ref=5, degree=1000, g_Li=0.03,
    #                               V_L=-75,
    #                               V_rst=-65, V_th=-50, path="../data/initialize_gui_d1000")

    # initilalize_gui_single_neruonal_model(delta_t=1., initial_input=0.001, degree=1000, path="../data/initialize_gui_d1000")

    # initilalize_gui_in_new_background_current(delta_t=1., default_Hz=20, max_output_Hz=100, T_ref=5, degree=1000,
    #                                           g_Li=0.03, i_mean=0.8, i_sigma=0.1, tau_i=5.,
    #                                           V_L=-75,
    #                                           V_rst=-65, V_th=-50, path="../data/initialize_gui_d1000")

    initilalize_gui_in_new_background_current_of_column(delta_t=1., default_Hz=20, max_output_Hz=100, T_ref=5,
                                                        degree=100,
                                                        g_Li=0.03, i_mean=0.6, i_sigma=0.1, tau_i=2.,
                                                        V_L=-75,
                                                        V_rst=-65, V_th=-50, path="../data/isolated_column_scale1e4",
                                                        only_test=True)

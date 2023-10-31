# -*- coding: utf-8 -*- 
# @Time : 2023/5/10 14:34 
# @Author : lepold
# @File : auto_test.py

import os
import re
import subprocess as sp
import sys
import time
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import torch


class TestBlock(unittest.TestCase):
    @staticmethod
    def open_server(slurm_name: str, nodes: int, single_slots: int = 4):
        f1 = open("/public/home/ssct004t/project/Digital_twin_brain/test/server.slurm", 'r+', encoding='utf-8')
        content = f1.read()
        # print("Original current:" + '\n' + content)
        f1.close()
        content = re.sub(r"-J\s.*?\n", "-J %s\n" % slurm_name, content)
        content = re.sub(r"-N\s.*?\n", "-N %s\n" % nodes, content)
        content = re.sub(r"(==2.*?)=\d+", r"\1=%d"%single_slots, content)

        # print("After modify", '\n' + content)

        with open("/public/home/ssct004t/project/Digital_twin_brain/test/server.slurm", "w",
                  encoding='utf-8') as f2:
            f2.write(content)
            print("modify server.slurm！")

        command = "sbatch /public/home/ssct004t/project/Digital_twin_brain/test/server.slurm 2>&1 | tr -cd '[0-9]'"
        job_id = sp.check_output(command, shell=True)
        job_id = str(job_id, "utf-8")
        print(f"server job id is {job_id}")

        echos = 0
        ip = None
        while echos < 10:
            try:
                with open(f"/public/home/ssct004t/project/Digital_twin_brain/test/log/{job_id}.o", "r+",
                          encoding='utf-8') as out:
                    lines = out.read()
                ip = re.findall(r"\d+\.\d+\.\d+\.\d+:\d+", lines)[0]
                break
            except:
                time.sleep(15)
                echos += 1
                continue
        print(f"server ip is {ip}")
        return ip

    def _test_open_server(self):
        ip = self.open_server("openserver", 2)
        self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")

    # Testing in the case of inserting small models into large models,
    # Keywords: uint8, 1ms, steps=1, ou driven.
    def _test_basic_compare(self, print_detail=False):
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_release/python")
        from cuda_release.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block
        from generation.read_block import connect_for_block

        ip = self.open_server("server_debug", 2, single_slots=4)
        self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")
        path = '/public/home/ssct004t/project/Digital_twin_brain/data/dti_distribution_0.10m_d100_with_debug/module'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper(ip, os.path.join(path, 'uint8'), delta_t, route_path=route_path,
                                  allow_metric=False)

        sample_path = torch.from_numpy(np.load(os.path.join(path, "debug_selection_idx.npy")).astype(np.int64)).cuda()
        sample_idx = dist_block._neurons_thrush[sample_path[:, 0]] + sample_path[:, 1]
        assert (sample_idx < dist_block._neurons_thrush[sample_path[:, 0] + 1]).all()

        assert (dist_block._neurons_thrush[sample_path[:, 0]] <= sample_idx).all()
        run_number = 500

        with open(os.path.join(path, "debug_dir.txt"), "r") as f:
            cpu_path = f.readline()
            cpu_path = cpu_path.strip("\n")
        cpu_path = cpu_path.strip("block_0.npz")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)

        dist_block.set_samples(sample_idx)

        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        for j, (freqs, spike, vi, i_sy, iouu) in enumerate(
                dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=True)):
            sample_spike[j, :] = spike
            sample_vi[j, :] = vi
            sample_ou[j, :] = iouu.cpu()
            sample_syn_current[j, :] = i_sy.cpu()
            sample_freqs[j, :] = freqs
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()

        dist_block.shutdown()

        iteration = 0

        def _ou(self, other):
            nonlocal iteration
            iteration += 1
            return sample_ou[iteration - 1]

        with mock.patch('models.block_new.block.update_i_background', _ou) as n:
            cpu_block = block(node_property=property, w_uij=w_uij, delta_t=1.)

            log = np.zeros_like(sample_spike)
            V_i = np.zeros_like(sample_vi)
            syn_current = np.zeros_like(sample_vi)
            iou = np.zeros_like(sample_vi)

            bug_all = 0
            for k in range(run_number):
                cpu_block.run(debug_mode=False)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()
                syn_current[k] = cpu_block.I_syn.numpy()
                iou[k] = cpu_block.i_background.numpy()

                V_i_error = np.nanmax(np.abs(sample_vi[k] - V_i[k]) / np.abs(V_i[k]))
                ou_error = np.nanmax(np.abs(sample_ou.numpy()[k] - iou[k]) / np.abs(iou[k]))
                if print_detail:
                    print("\niteration:", k, "======>>>")
                    print(f"syn current comparison, now fire {log[k].sum()} at neuron {log[k].nonzero()[0]}")
                index = np.nonzero(syn_current[k])[0]
                index2 = np.nonzero(sample_syn_current[k])[0]
                if print_detail:
                    if len(index) == len(index2) and (index == index2).all():
                        print("  post-synaptic current are aligned, differ at")
                        index_star = np.where(syn_current[k] != sample_syn_current[k])[0]
                        print("    in cuda:", sample_syn_current[k][index_star])
                        print("    in python:", syn_current[k][index_star])
                    else:
                        print("  syn current is not aligned.")
                syn_i_error = np.nanmax(np.abs(sample_syn_current[k] - syn_current[k]) / np.abs(syn_current[k]))
                if print_detail:
                    print("syn related error", syn_i_error, "ou_error", ou_error, "v related error:", V_i_error)

                bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                bug_all += bug_idx.shape[0]
                if print_detail:
                    if bug_idx.shape[0] > 0:
                        print("")
                        print("error in: ", k, " cpu log: ", log[k, bug_idx], " cuda log: ", sample_spike[k, bug_idx])

        self.assertEqual(bug_all, 0, msg="Error in spike!")
        self.assertLess(V_i_error, 1e-3, msg="Related error of membrane potential exceeds limit!")

    # Testing in the case of small network, in which all are sampled. mainly to check the graph representation.
    # Keywords: uint8, 1ms, steps=1, ou driven.
    def _test_small_network_and_check_graph(self, print_detail=True):
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_release/python")
        from cuda_release.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block as block_ou
        from models.block import block as block_noise
        from generation.read_block import connect_for_block

        ip = self.open_server("server_debug", 2, single_slots=1)
        self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")


        route_path = None
        delta_t = 1.
        single_card = True
        ou_driven = True

        bases = [0]
        totals = []

        if single_card:
            cards = 1
            path = '/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/d4_ou_iter1_samll_block/d4_ou_1card/'
        else:
            cards = 4
            path = '/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/d4_ou_iter1_samll_block/d4_ou/'

        dist_block = BlockWrapper(ip, os.path.join(path, 'uint8'), delta_t, route_path=route_path)

        for i in range(cards):
            file = np.load(os.path.join(path, "uint8/block_%d.npz" % i))
            prop = file['property']
            bases.append(bases[-1] + prop.shape[0])
        bases = np.array(bases)

        for i in range(cards):
            file = np.load(os.path.join(path, "uint8/block_%d.npz" % i))
            output_neuron_idx = file['output_neuron_idx']
            input_neuron_idx = file['input_neuron_idx']
            input_block_idx = file['input_block_idx']
            input_channel_offset = file['input_channel_offset']
            w = file['weight']
            output_neuron_idx = output_neuron_idx + bases[i]
            input_neuron_idx = input_neuron_idx + bases[i + input_block_idx]
            total = np.concatenate(
                [output_neuron_idx[:, None], input_neuron_idx[:, None], input_channel_offset[:, None], w], axis=1)
            totals.append(total)

        totals = np.concatenate(totals, axis=0)
        if print_detail:
            print("output_neuron| input_neuron| input_channel")
            print(totals[:, :-2].astype(np.int64))

        N = bases[-1]
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 1000

        dist_block.set_samples(sample_idx)

        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        if ou_driven:
            for j, (freqs, spike, vi, i_sy, iouu) in enumerate(
                    dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_ou[j, :] = iouu.cpu()
                sample_syn_current[j, :] = i_sy.cpu()
                sample_freqs[j, :] = freqs
            if print_detail:
                print('mean firing rate (transition time))',
                      torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                dist_block.neurons_per_subblk.float()))
        else:
            for j, (freqs, spike, vi) in enumerate(
                    dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_freqs[j, :] = freqs
            if print_detail:
                print('mean firing rate (transition time))',
                      torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                dist_block.neurons_per_subblk.float()))
            sample_log = (sample_vi == -50)
            assert (torch.logical_and(sample_log, sample_spike) == sample_log).all()
            sample_log = sample_log.cpu().numpy()
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()
        dist_block.shutdown()

        cpu_path = os.path.join(path, "uint8")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)

        iteration = 0

        def _ou(self, other):
            nonlocal iteration
            iteration += 1
            return sample_ou[iteration - 1]

        iter = 0

        def _or(self, other):
            nonlocal iter
            # print(sample_spike[iter].shape, sample_spike[iter].astype(np.float32).mean() * 1000)
            sample = torch.from_numpy(sample_spike[iter]).to(torch.bool)
            iter += 1
            return sample

        if ou_driven:
            with mock.patch('models.block_new.block.update_i_background', _ou) as n:
                cpu_block = block_ou(node_property=property, w_uij=w_uij, delta_t=1.)

                log = np.zeros_like(sample_spike)
                V_i = np.zeros_like(sample_vi)
                syn_current = np.zeros_like(sample_vi)
                iou = np.zeros_like(sample_vi)

                bug_all = 0
                for k in range(run_number):
                    cpu_block.run(debug_mode=False)
                    log[k] = cpu_block.active.numpy()
                    V_i[k] = cpu_block.V_i.numpy()
                    syn_current[k] = cpu_block.I_syn.numpy()
                    iou[k] = cpu_block.i_background.numpy()
                    if print_detail:
                        print("\niteration", k)
                        print("cuda vi", sample_vi[k])
                        print("cpu vi", V_i[k])
                        if np.count_nonzero(sample_spike[k]) > 0:
                            print("    cuda firing neuron", list(sample_spike[k].nonzero()[0]))
                            print("    cpu firing neuron", list(log[k].nonzero()[0]))
                        print("cuda syn current", sample_syn_current[k])
                        print("cpu syn current", syn_current[k])
                    bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                    bug_all += bug_idx.shape[0]
        else:
            with mock.patch('torch.Tensor.__or__', _or) as n:
                cpu_block = block_noise(node_property=property, w_uij=w_uij, delta_t=1.)
                log = np.zeros_like(sample_spike)
                V_i = np.zeros_like(sample_vi)
                syn_current = np.zeros_like(sample_vi)
                bug_all = 0
                for k in range(run_number):
                    cpu_block.run(noise_rate=0.01)
                    log[k] = cpu_block.active.numpy()
                    V_i[k] = cpu_block.V_i.numpy()
                    syn_current[k] = cpu_block.I_syn.numpy()

                    if print_detail:
                        print("\niteration", k)
                        print("cuda vi", sample_vi[k])
                        print("cpu vi", V_i[k])
                        if np.count_nonzero(sample_log[k]) > 0:
                            print("    cuda firing neuron", list(sample_log[k].nonzero()[0]))
                            print("    cpu firing neuron", list(log[k].nonzero()[0]))
                    bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                    bug_all += bug_idx.shape[0]
        self.assertEqual(bug_all, 0, msg="Error in spike!")

    # Testing in the case of small network, in which all are sampled.
    # check the situation of 1ms communication and 10 steps iteration.
    # Keywords: uint8, 1ms, steps=10, ou driven.
    def _test_efficient_10steps_iteration(self, print_detail=True):
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_release/python")
        from cuda_release.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block
        from generation.read_block import connect_for_block

        class block_efficient(block):
            def __init__(self, node_property, w_uij, delta_t=1, chunk=10):
                super(block_efficient, self).__init__(node_property, w_uij, delta_t)
                K, self.N = self.w_uij.shape[0], self.w_uij.shape[2]
                self.chunk = chunk
                self.refractory_steps = (self.T_ref * chunk).type(torch.int8)
                print("self.refractory_steps", self.refractory_steps)
                self.t = self.t = torch.tensor(0, device=self.w_uij.device, dtype=torch.int64)
                self.t_ik_last = - torch.ones([N], device=self.w_uij.device, dtype=torch.int64) * self.refractory_steps  # shape [N]

            def update_Vi(self, delta_t, i_background):
                main_part = -self.g_Li * (self.V_i - self.V_L)
                C_diff_Vi = main_part + self.I_syn + self.I_extern_Input + i_background  # + self.I_T

                delta_Vi = delta_t / self.C * C_diff_Vi
                Vi_normal = self.V_i + delta_Vi

                is_not_saturated = (self.t >= self.t_ik_last + self.refractory_steps)
                V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
                active = torch.ge(V_i, self.V_th)

                self.V_i = torch.min(V_i, self.V_th)
                # self.V_i[active] = - 40
                return active

            def update_t_ik_last(self, active):
                self.t_ik_last[active] = self.t

            def run(self, debug_mode, noise_rate=None):
                inner_loop = 0
                valid_active = torch.zeros(self.N, dtype=torch.uint8, device=self.w_uij.device)
                while inner_loop < self.chunk - 1:
                    # self.t += self.delta_t / self.chunk
                    self.t += 1
                    self.i_background = self.update_i_background(self.delta_t / self.chunk)
                    self.active = self.update_Vi(self.delta_t / self.chunk, self.i_background)
                    if debug_mode:
                        new_active = torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate
                    else:
                        new_active = self.active
                    valid_active = valid_active | new_active
                    self.update_J_ui(self.delta_t / self.chunk,
                                     torch.zeros(self.N, dtype=torch.uint8, device=self.w_uij.device))
                    self.update_I_syn()
                    # self.update_calcium_current()
                    self.update_t_ik_last(self.active)
                    inner_loop += 1

                self.t += 1
                self.i_background = self.update_i_background(self.delta_t / self.chunk)
                self.active = self.update_Vi(self.delta_t / self.chunk, self.i_background)
                if debug_mode:
                    new_active = torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate
                else:
                    new_active = self.active
                valid_active = valid_active | new_active
                self.update_J_ui(self.delta_t / self.chunk, valid_active)
                self.update_I_syn()
                # self.update_calcium_current()
                self.update_t_ik_last(self.active)
                inner_loop += 1
                assert inner_loop == self.chunk

        ip = self.open_server("auto_ser", 2, single_slots=4)
        self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")
        path = '/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/d4_ou_iter1_samll_block/d4_ou'
        route_path = None
        delta_t = 1.
        t_steps = 10
        external_current = 0.05  # or 0 represent no current.
        dist_block = BlockWrapper(ip, os.path.join(path, 'uint8'), delta_t, route_path=route_path)

        N_1 = np.load(os.path.join(path, "uint8/block_0.npz"))['property'].shape[0]
        N_2 = np.load(os.path.join(path, "uint8/block_1.npz"))['property'].shape[0]
        N_3 = np.load(os.path.join(path, "uint8/block_2.npz"))['property'].shape[0]
        N_4 = np.load(os.path.join(path, "uint8/block_3.npz"))['property'].shape[0]
        N = N_1 + N_2 + N_3 + N_4
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 2000
        cpu_path = os.path.join(path, "uint8")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)
        dist_block.set_samples(sample_idx)

        population_info = torch.stack(
            torch.meshgrid(dist_block.subblk_id, torch.tensor([2], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        param = torch.ones(dist_block.total_subblks, device="cuda:0") * external_current
        dist_block.assign_property_by_subblk(population_info, param)
        if print_detail:
            print('sample_idx.size()', sample_idx.size())
            print("total subblks", dist_block.subblk_id.shape[0])
        sample_spike = torch.empty([run_number, t_steps, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, t_steps, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, t_steps, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, t_steps, sample_idx.shape[0]], dtype=torch.float32)
        for j in range(run_number):
            for idx, (freqs, spike, vi, i_sy, iouu) in enumerate(
                    dist_block.run(1, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=True,
                                   t_steps=t_steps, equal_sample=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_ou[j, :] = iouu.cpu()
                sample_syn_current[j, :] = i_sy.cpu()
                sample_freqs[j, :] = freqs
        if print_detail:
            print('mean firing rate (transition time))\n',
                  torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                            dist_block.neurons_per_subblk.float()))
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()
        # print(sample_syn_current[99, :])

        dist_block.shutdown()

        iteration = 0
        small_loop = 0

        def _ou(self, other):
            nonlocal iteration
            nonlocal small_loop
            if small_loop % 10 == 0:
                iteration += 1
                small_loop = 0
            small_loop += 1
            # print("iteration, small_loop", iteration-1, small_loop - 1)
            return sample_ou[iteration - 1, small_loop - 1]

        with mock.patch('models.block_new.block.update_i_background', _ou) as n:
            property[:, 2] = external_current
            cpu_block = block_efficient(node_property=property, w_uij=w_uij, delta_t=1., chunk=10)

            log = np.zeros((run_number, sample_idx.shape[0]))
            V_i = np.zeros((run_number, sample_idx.shape[0]))
            syn_current = np.zeros((run_number, sample_idx.shape[0]))
            iou = np.zeros((run_number, sample_idx.shape[0]))

            bug_all = 0
            for k in range(run_number):
                cpu_block.run(debug_mode=False)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()
                syn_current[k] = cpu_block.I_syn.numpy()
                # print("cpu: syn current", syn_current[k])
                iou[k] = cpu_block.i_background.numpy()

                V_i_error = np.nanmax(np.abs(sample_vi[k, -1] - V_i[k]) / np.abs(V_i[k]))
                ou_error = np.nanmax(np.abs(sample_ou.numpy()[k, -1] - iou[k]) / np.abs(iou[k]))
                if print_detail:
                    print("\niteration:", k, "======>>>")
                    print(f"syn current comparison, now fire {log[k].sum()} at neuron {log[k].nonzero()[0]}")
                index = np.nonzero(syn_current[k])[0]
                index2 = np.nonzero(sample_syn_current[k, -1])[0]
                if print_detail:
                    if len(index) == len(index2) and (index == index2).all():
                        print("  post-synaptic current are aligned, differ at")
                        index_star = np.where(syn_current[k] != sample_syn_current[k, -1])[0]
                        print("    in cuda:", sample_syn_current[k, -1, index_star])
                        print("    in python:", syn_current[k][index_star])
                    else:
                        print("  syn current is not aligned.")
                syn_i_error = np.nanmax(np.abs(sample_syn_current[k, -1] - syn_current[k]))
                if print_detail:
                    print("syn error", syn_i_error, "ou related error", ou_error, "v related error:", V_i_error)

                bug_idx = (log[k] != sample_spike[k, -1]).nonzero()[0]
                bug_all += bug_idx.shape[0]
                if print_detail:
                    if bug_idx.shape[0] > 0:
                        print("")
                        print("error in: ", k, " cpu log: ", log[k, bug_idx], " cuda log: ",
                              sample_spike[k, -1, bug_idx])

        self.assertEqual(iteration, run_number, msg="Not run finished!")
        self.assertEqual(bug_all, 0, msg="Error in spike!")
        self.assertLess(V_i_error, 1e-3, msg="Related error of membrane potential exceeds limit!")

    # Testing in the case of small network, in which all are sampled.
    # check the situation of 1ms communication and 10 steps iteration. check the statistical api "freqs" whether is right.
    # Keywords: uint8, 1ms, steps=10, ou driven.
    def _test_freqs_concide_with_spike(self, print_detail=True):
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_release/python")
        from cuda_release.python.dist_blockwrapper_pytorch import BlockWrapper

        ip = self.open_server("server_debug", 2, single_slots=4)
        self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")

        path = '/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/d4_ou_iter1_samll_block/d4_ou/'
        route_path = None
        delta_t = 1.
        single_card = False
        ou_driven = True
        checked = False
        dist_block = BlockWrapper(ip, os.path.join(path, 'uint8'), delta_t, route_path=route_path)

        bases = [0]
        totals = []

        if single_card:
            cards = 1
        else:
            cards = 4
        for i in range(cards):
            file = np.load(os.path.join(path, "uint8/block_%d.npz" % i))
            prop = file['property']
            bases.append(bases[-1] + prop.shape[0])
        bases = np.array(bases)

        for i in range(cards):
            file = np.load(os.path.join(path, "uint8/block_%d.npz" % i))
            output_neuron_idx = file['output_neuron_idx']
            input_neuron_idx = file['input_neuron_idx']
            input_block_idx = file['input_block_idx']
            input_channel_offset = file['input_channel_offset']
            w = file['weight']
            output_neuron_idx = output_neuron_idx + bases[i]
            input_neuron_idx = input_neuron_idx + bases[i + input_block_idx]
            total = np.concatenate(
                [output_neuron_idx[:, None], input_neuron_idx[:, None], input_channel_offset[:, None], w], axis=1)
            totals.append(total)

        totals = np.concatenate(totals, axis=0)
        if print_detail:
            print("output_neuron| input_neuron| input_channel")
            print(totals[:, :-2].astype(np.int64))

        N = bases[-1]
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 200

        dist_block.set_samples(sample_idx)
        t_steps = 10
        if checked:
            sample_spike = torch.empty([run_number, t_steps, sample_idx.shape[0]], dtype=torch.uint8).cuda()
            sample_vi = torch.empty([run_number, t_steps, sample_idx.shape[0]], dtype=torch.float32).cuda()
        else:
            sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
            sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        if ou_driven:
            for j, (freqs, spike, vi) in enumerate(
                    dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=False, t_steps=10, equal_sample=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_freqs[j, :] = freqs
            if print_detail:
                print('mean firing rate (transition time))',
                      torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                dist_block.neurons_per_subblk.float()))
        else:
            for j, (freqs, spike, vi) in enumerate(
                    dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_freqs[j, :] = freqs
            if print_detail:
                print('mean firing rate (transition time))',
                      torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                dist_block.neurons_per_subblk.float()))
            sample_log = (sample_vi == -50)
            assert (torch.logical_and(sample_log, sample_spike) == sample_log).all()
            sample_log = sample_log.cpu().numpy()
        sample_spike = sample_spike.cpu().numpy()
        sample_freqs = sample_freqs.cpu().numpy()
        dist_block.shutdown()
        for i in range(run_number):
            total_spike = sample_freqs[i].sum()
            spike_sum = sample_spike[i].sum(-1)
            if np.ndim(spike_sum) > 0:
                spike_sum = spike_sum.sum(-1)
            if print_detail:
                print("freq vs spike_sum", total_spike, " | ", spike_sum)
            assert total_spike == spike_sum
        self.assertEqual(i, run_number-1)

    def _test_estimate_speed_among_different_steps(self, print_detail=True):
        block_path = ["/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam/dti_distribution_200m_d100_1April/module/uint8",
                      "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/mutli_d1k_inluding_brainstem/dti_distribution_5000m_d1000_27April/module/uint8",
                      "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/mutli_d1k_inluding_brainstem/dti_distribution_8000m_d1000_27April/module/uint8",
                      "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/mutli_d1k_inluding_brainstem/dti_distribution_12000m_d1000/module/uint8"]
        total_nodes = [6, 206, 328, 492]
        remaining_gpus = [4, 2, 4, 2]
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_develop/python")
        from cuda_develop.python.dist_blockwrapper_pytorch import BlockWrapper
        import matplotlib.pyplot as plt

        t_steps = 10
        run_number = 1000
        times = []
        frs = []
        for i in range(3):
            ou_mean = 0.62 + 0.1 * i
            for path, nodes, slots in zip(block_path[:-3], total_nodes[:-3], remaining_gpus[:-3]):
                # test 1, 1ms with 10 steps
                ip = self.open_server("speed", nodes, single_slots=slots)
                self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")
                route_path = None
                delta_t = 1.
                dist_block = BlockWrapper(ip, path, delta_t, route_path=route_path,
                                          allow_metric=False)
                dist_block.update_ou_background_stimuli(10, ou_mean, 0.2)
                sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32)
                start = time.time()
                for j, (freqs,) in enumerate(
                        dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=False, iou=True, checked=False,
                                       t_steps=t_steps, equal_sample=True)):
                    sample_freqs[j, :] = freqs.cpu()
                dist_block.shutdown()
                hz1 = torch.mean(torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                          dist_block.neurons_per_subblk.float().cpu())).item()
                print('mean firing rate (transition time))', hz1)
                end = time.time()
                cost_time1 = end -start
                print("cost time ", int(cost_time1), " seconds")

                # test 2, canonical 1ms
                ip = self.open_server("speed", nodes, single_slots=slots)
                self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")
                route_path = None
                delta_t = 1.
                dist_block = BlockWrapper(ip, path, delta_t, route_path=route_path,
                                          allow_metric=False)
                dist_block.update_ou_background_stimuli(10, ou_mean, 0.2)
                sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32)
                start = time.time()
                for j, (freqs,) in enumerate(
                        dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=False, iou=True, checked=False,
                                       t_steps=1, equal_sample=True)):
                    sample_freqs[j, :] = freqs.cpu()
                dist_block.shutdown()
                hz2 = torch.mean(torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                          dist_block.neurons_per_subblk.float().cpu())).item()
                print('mean firing rate (transition time))', hz2)
                end = time.time()
                cost_time2 = end - start
                print("cost time ", int(cost_time2), " seconds")

                # test 3, 0.1 ms
                ip = self.open_server("speed", nodes, single_slots=slots)
                self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")
                route_path = None
                delta_t = 0.1
                dist_block = BlockWrapper(ip, path, delta_t, route_path=route_path,
                                          allow_metric=False)
                dist_block.update_ou_background_stimuli(10, ou_mean, 0.2)
                sample_freqs = torch.empty([run_number*10, dist_block.subblk_id.shape[0]], dtype=torch.int32)
                start = time.time()
                for j, (freqs,) in enumerate(
                        dist_block.run(run_number*10, freqs=True, vmean=False, sample_for_show=False, iou=True, checked=False,
                                       t_steps=1, equal_sample=True)):
                    sample_freqs[j, :] = freqs.cpu()
                dist_block.shutdown()
                hz3 = torch.mean(torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                                           dist_block.neurons_per_subblk.float().cpu())).item()
                print('mean firing rate (transition time))', hz3)
                end = time.time()
                cost_time3 = end - start
                print("cost time ", int(cost_time3), " seconds")
                times.append([cost_time1, cost_time2, cost_time3])
                frs.append([hz1, hz2, hz3])
                print("\n\n")
                time.sleep(10)
        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.gca()
        times = np.array(times)
        ax.plot(np.arange(3), times[:, 0], "*-", lw=1.5, label="1ms_10steps")
        ax.plot(np.arange(3), times[:, 1], "*-", lw=1.5, label="1ms_1step")
        ax.plot(np.arange(3), times[:, 2], "*-", lw=1.5, label="0.1ms")
        for i in range(len(frs)):
            ax.text(i+0.05, frs[i][0], f"{frs[i][0]:.0f} Hz")
            ax.text(i+0.05, frs[i][1], f"{frs[i][1]:.0f} Hz")
            ax.text(i+0.05, frs[i][2], f"{frs[i][2]:.0f} Hz")
        ax.set_xticks([0, 1, 2])
        ax.set_ylim([0, 180])
        ax.set_xticklabels(["0.62", "0.63", "0.64"])
        ax.set_ylabel("Seconds")
        ax.set_xlabel("OU mean")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(f"velocity_compare.png")
        plt.close(fig)
        print("Done!")

    # test whether the same result after implementing assigning property to the spike network.
    def test_small_network_after_assign(self, print_detail=True):
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain")
        sys.path.append("/public/home/ssct004t/project/Digital_twin_brain/cuda_release/python")
        from cuda_release.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block as block_ou
        from generation.read_block import connect_for_block

        ip = self.open_server("server_debug", 2, single_slots=1)
        self.assertIsInstance(ip, str, msg="server open failed (maybe set more waiting time).")


        route_path = None
        delta_t = 0.1

        path = '/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain-newest/data/small_blocks_debug/2100d100_exact_EI6_ou_f_modify'
        dist_block = BlockWrapper(ip, os.path.join(path, 'single'), delta_t, route_path=route_path)

        N = dist_block.total_neurons
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 1000
        compare_number = 1000

        dist_block.set_samples(sample_idx)

        # update tau_gaba to 20.
        population_info = torch.stack(
            torch.meshgrid(dist_block.subblk_id, torch.tensor([20], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        param = torch.ones(dist_block.total_subblks, device="cuda:0") * 10.
        dist_block.assign_property_by_subblk(population_info, param)
        # # update ou
        # dist_block.update_ou_background_stimuli(0.8, 0.7498, 0.001)

        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)

        for j, (freqs, spike, vi, i_sy, iouu) in enumerate(
                dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=True)):
            sample_spike[j, :] = spike
            sample_vi[j, :] = vi
            sample_ou[j, :] = iouu.cpu()
            sample_syn_current[j, :] = i_sy.cpu()
            sample_freqs[j, :] = freqs
        if print_detail:
            print('mean firing rate (transition time))',
                  torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                            dist_block.neurons_per_subblk.float()))
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()
        dist_block.shutdown()

        cpu_path = os.path.join(path, "single")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        property[:, 20] = 10.

        iteration = 0

        def _ou(self, other):
            nonlocal iteration
            iteration += 1
            return sample_ou[iteration - 1]

        iter = 0

        def _or(self, other):
            nonlocal iter
            # print(sample_spike[iter].shape, sample_spike[iter].astype(np.float32).mean() * 1000)
            sample = torch.from_numpy(sample_spike[iter]).to(torch.bool)
            iter += 1
            return sample

        with mock.patch('models.block_new.block.update_i_background', _ou) as n:
            cpu_block = block_ou(node_property=property, w_uij=w_uij, delta_t=delta_t) # i_mean=0.7498, i_sigma=0.003, tau_i=0.8

            log = np.zeros((compare_number, sample_idx.shape[0]), dtype=np.uint8)
            V_i = np.zeros((compare_number, sample_idx.shape[0]), dtype=np.float32)
            syn_current = np.zeros((compare_number, sample_idx.shape[0]), dtype=np.float32)
            iou = np.zeros((compare_number, sample_idx.shape[0]), dtype=np.float32)

            bug_all = 0
            for k in range(compare_number):
                cpu_block.run(debug_mode=False)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()
                syn_current[k] = cpu_block.I_syn.numpy()
                iou[k] = cpu_block.i_background.numpy()
                if print_detail:
                    print("\niteration", k)
                    print("cuda vi", sample_vi[k])
                    print("cpu vi", V_i[k])
                    if np.count_nonzero(sample_spike[k]) > 0:
                        print("    cuda firing neuron", list(sample_spike[k].nonzero()[0]))
                        print("    cpu firing neuron", list(log[k].nonzero()[0]))
                    print("cuda syn current", sample_syn_current[k])
                    print("cpu syn current", syn_current[k])
                bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                bug_all += bug_idx.shape[0]
        fig = plt.figure(figsize=(8, 3))
        ax = fig.gca()
        ax.scatter(*sample_spike[:, :100].nonzero(), marker=',', c='k', s=1.)
        ax.set_xlabel("0.1 ms")
        fig.savefig("spike.png")
        self.assertEqual(bug_all, 0, msg="Error in spike!")

if __name__ == '__main__':
    unittest.main()

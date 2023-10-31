import os
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np


class TestBlock(unittest.TestCase):

    def _test_compare_ou(self):
        from cuda.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block
        from generation.read_block import connect_for_block
        import torch

        path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/jianfeng_region/dti_distribution_0.10m_d100_with_debug/module'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper('11.10.10.42:50051', os.path.join(path, 'uint8'), delta_t, route_path=route_path,
                                  allow_metric=False)

        sample_path = torch.from_numpy(np.load(os.path.join(path, "debug_selection_idx.npy")).astype(np.int64)).cuda()
        print("dist_block._neurons_thrush", dist_block._neurons_thrush)
        print("dist_block.neurons_per_block", dist_block.neurons_per_block)
        sample_idx = dist_block._neurons_thrush[sample_path[:, 0]] + sample_path[:, 1]
        assert (sample_idx < dist_block._neurons_thrush[sample_path[:, 0] + 1]).all()

        assert (dist_block._neurons_thrush[sample_path[:, 0]] <= sample_idx).all()
        run_number = 1000

        with open(os.path.join(path, "debug_dir.txt"), "r") as f:
            cpu_path = f.readline()
            cpu_path = cpu_path.strip("\n")
        cpu_path = cpu_path.strip("block_0.npz")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        print("wuij.dtype", w_uij.dtype)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)

        dist_block.set_samples(sample_idx)
        print('sample_idx.size()', sample_idx.size())

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
        print('mean firing rate (transition time))',
              torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                        dist_block.neurons_per_subblk.float()))
        print("sample_syn_current:", sample_syn_current[0, :10])
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()

        print(dist_block.last_time_stat())
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

            for k in range(run_number):
                cpu_block.run(debug_mode=False)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()
                syn_current[k] = cpu_block.I_syn.numpy()
                iou[k] = cpu_block.i_background.numpy()

                V_i_error = np.nanmax(np.abs(sample_vi[k] - V_i[k]) / np.abs(V_i[k]))
                ou_error = np.nanmax(np.abs(sample_ou.numpy()[k] - iou[k]) / np.abs(iou[k]))
                print("\niteration:", k, "======>>>")
                print(f"syn current comparison, now fire {log[k].sum()} at neuron {log[k].nonzero()[0]}")
                index = np.nonzero(syn_current[k])[0]
                index2 = np.nonzero(sample_syn_current[k])[0]
                if len(index)==len(index2) and (index==index2).all():
                    print("  post-synaptic current are aligned, differ at")
                    index_star = np.where(syn_current[k]!=sample_syn_current[k])[0]
                    print("    in cuda:", sample_syn_current[k][index_star])
                    print("    in python:", syn_current[k][index_star])
                else:
                    print("  syn current is not aligned.")
                syn_i_error = np.nanmax(np.abs(sample_syn_current[k] - syn_current[k]) / np.abs(syn_current[k]))
                print("syn related error", syn_i_error, "ou_error", ou_error, "v related error:", V_i_error)

                bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                if bug_idx.shape[0] > 0:
                    print("")
                    print("error in: ", k, " cpu log: ", log[k, bug_idx], " cuda log: ", sample_spike[k, bug_idx])

        fig2 = plt.figure(figsize=(8, 4))
        ax = fig2.gca()
        ax.scatter(*sample_spike[:, :100].nonzero(), marker=",", s=1., alpha=0.8)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("neurons")
        fig2.savefig("./raster_fig.png", dpi=100)

        self.assertEqual(k, run_number - 1)

    def test_compare_break_ou(self):
        from cuda_develop.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block
        from generation.read_block import connect_for_block
        import torch

        path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks_debug/d4_ou'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper('11.5.4.3:50051', os.path.join(path, 'uint8'), delta_t, route_path=route_path)

        N_1 = np.load(os.path.join(path, "uint8/block_0.npz"))['property'].shape[0]
        N_2 = np.load(os.path.join(path, "uint8/block_1.npz"))['property'].shape[0]
        N_3 = np.load(os.path.join(path, "uint8/block_2.npz"))['property'].shape[0]
        N_4 = np.load(os.path.join(path, "uint8/block_3.npz"))['property'].shape[0]
        N = N_1 + N_2 + N_3 + N_4
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 200

        cpu_path = os.path.join(path, "uint8")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        print('w_uij.dtype', w_uij.dtype)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)
        dist_block.set_samples(sample_idx)

        # dist_block.set_state_rule(file_path="debug_breakpoint", observe_counter=0, observe_interval=100)
        dist_block.load_state_from_file(file_path="debug_breakpoint", observe_counter=200)

        print('sample_idx.size()', sample_idx.size())
        print("total subblks", dist_block.subblk_id.shape[0])
        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        for j in range(run_number):
            for idx, (freqs, spike, vi, i_sy, iouu) in enumerate(
                    dist_block.run(1, freqs=True, vmean=False, sample_for_show=True, iou=True, checked=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_ou[j, :] = iouu.cpu()
                sample_syn_current[j, :] = i_sy.cpu()
                sample_freqs[j, :] = freqs
        print('mean firing rate (transition time))',
              torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                        dist_block.neurons_per_subblk.float()))
        print("sample_syn_current:", sample_syn_current[0, :10])
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()

        # np.save("break_point_result/vi_new_part_0-200.npy", sample_vi)
        # print("save sample ou type", type(sample_ou))
        # np.save("break_point_result/ou_new_part_0-200.npy", sample_ou.numpy())

        print(dist_block.last_time_stat())
        dist_block.shutdown()

        run_past = True
        if run_past:
            past_ou = np.load("./break_point_result/ou_new_part_0-200.npy")
            past_ou = torch.from_numpy(past_ou)
            iteration = 0
            def _ou_past(self, other):
                nonlocal iteration
                iteration += 1
                return past_ou[iteration - 1]

            with mock.patch('models.block_new.block.update_i_background', _ou_past) as n:
                cpu_block = block(node_property=property, w_uij=w_uij, delta_t=1.)
                for k in range(200):
                    cpu_block.run(debug_mode=False)

        run_python=True
        if run_python:
            iteration = 0
            def _ou(self, other):
                nonlocal iteration
                print("use _ou", iteration)
                iteration += 1
                return sample_ou[iteration - 1]
            with mock.patch('models.block_new.block.update_i_background', _ou) as n:
                if not run_past:
                    cpu_block = block(node_property=property, w_uij=w_uij, delta_t=1.)
                cpu_block2 = block(node_property=property, w_uij=w_uij, delta_t=1.)
                v = []
                t = None
                t_last = []
                i_ext_stimuli = []
                i_ou_background_stimuli = []
                j_ex_presynaptics = []
                j_in_presynaptics = []
                for card in range(4):
                    info = np.load("debug_breakpoint/state_%d_200.npz" % card)
                    v.append(torch.from_numpy(info["v_membranes"]))
                    t = info["t"]
                    t_last.append(torch.from_numpy(info["t_actives"]))
                    i_ext_stimuli.append(torch.from_numpy(info["i_ext_stimuli"]))
                    i_ou_background_stimuli.append(torch.from_numpy(info["i_ou_background_stimuli"]))
                    j_ex_presynaptics.append(torch.from_numpy(info['j_ex_presynaptics']))
                    j_in_presynaptics.append(torch.from_numpy(info["j_in_presynaptics"]))
                v = torch.cat(v)
                print("v", v.shape)
                t_last = torch.cat(t_last)
                i_ext_stimuli = torch.cat(i_ext_stimuli)
                i_ou_background_stimuli = torch.cat(i_ou_background_stimuli)
                j_ex_presynaptics = torch.cat(j_ex_presynaptics)
                j_in_presynaptics = torch.cat(j_in_presynaptics)
                jui = torch.cat([j_ex_presynaptics.reshape((-1, 2)), j_in_presynaptics.reshape((-1, 2))], dim=1)
                jui = jui.T
                cpu_block2.t = torch.from_numpy(t)
                cpu_block2.t_ik_last = t_last
                cpu_block2.V_i = v
                cpu_block2.J_ui = jui
                cpu_block2.I_extern_Input = i_ext_stimuli
                cpu_block2.i_background = i_ou_background_stimuli
                cpu_block2.update_I_syn()
                print("check".center(30, "*"))
                print("")
                print("contiuous block, t", cpu_block.t)
                print("break block, t", cpu_block2.t)
                print("")
                print("contiuous block, t_last", cpu_block.t_ik_last)
                print("break block, t_last", cpu_block2.t_ik_last)
                print("")
                print("contiuous block, i_background", cpu_block.i_background)
                print("break block, i_background", cpu_block2.i_background)
                print("")
                print("contiuous block, I_ui", cpu_block.I_ui)
                print("break block, I_ui", cpu_block2.I_ui)
                print("")
                print("contiuous block, J_ui", cpu_block.J_ui)
                print("break block, J_ui", cpu_block2.J_ui)
                print("")
                print("contiuous block, I_extern_Input", cpu_block.I_extern_Input)
                print("break block, I_extern_Input", cpu_block2.I_extern_Input)
                print("")
                print("contiuous block, I_syn", cpu_block.I_syn)
                print("break block, I_syn", cpu_block2.I_syn)
                print("check".center(30, "*"))

                log = np.zeros_like(sample_spike)
                V_i = np.zeros_like(sample_vi)
                syn_current = np.zeros_like(sample_vi)
                iou = np.zeros_like(sample_vi)

                for k in range(run_number):
                    cpu_block.run(debug_mode=False)
                    log[k] = cpu_block.active.numpy()
                    V_i[k] = cpu_block.V_i.numpy()
                    syn_current[k] = cpu_block.I_syn.numpy()
                    iou[k] = cpu_block.i_background.numpy()

                    V_i_error = np.nanmax(np.abs(sample_vi[k] - V_i[k]) / np.abs(V_i[k]))
                    # if k==0:
                    #     load_ou = np.nanmax(sample_ou.numpy()[k]-i_ou_background_stimuli.numpy())
                    #     print("load_ou diff", load_ou)
                    #     print("load ou", i_ou_background_stimuli.numpy())
                    #     print("sample ou next step", sample_ou.numpy()[k])
                    ou_error = np.nanmax(np.abs(sample_ou.numpy()[k] - iou[k]) / np.abs(iou[k]))
                    print("\niteration:", k, "======>>>")
                    print(f"syn current comparison, now fire {log[k].sum()} at neuron {log[k].nonzero()[0]}")
                    if k<run_number-1:
                        print("sample_ou[k]", sample_ou.numpy()[k])
                        print("sample_ou[k+1]", sample_ou.numpy()[k+1])
                    index = np.nonzero(syn_current[k])[0]
                    index2 = np.nonzero(sample_syn_current[k])[0]
                    if len(index) == len(index2) and (index == index2).all():
                        print("  post-synaptic current are aligned, differ at")
                        index_star = np.where(syn_current[k] != sample_syn_current[k])[0]
                        print("    in cuda:", sample_syn_current[k][index_star])
                        print("    in python:", syn_current[k][index_star])
                    else:
                        print("  syn current is not aligned.")
                    syn_i_error = np.nanmax(np.abs(sample_syn_current[k] - syn_current[k]) / np.abs(syn_current[k]))
                    print("syn related error", syn_i_error, "ou_error", ou_error, "v related error:", V_i_error)

                    bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                    if bug_idx.shape[0] > 0:
                        print("")
                        print("error in: ", k, " cpu log: ", log[k, bug_idx], " cuda log: ", sample_spike[k, bug_idx])

                # np.savez(os.path.join("break_point_result", "temp_state.npz"), t=cpu_block.t, vi=cpu_block.V_i.cpu().numpy(),
                #         jui=cpu_block.J_ui.cpu().numpy(), t_last=cpu_block.t_ik_last.cpu().numpy(), iext=cpu_block.I_extern_Input.cpu().numpy(), iou=cpu_block.i_background.cpu().numpy(),
                #          iui=cpu_block.I_ui.cpu().numpy(), isyn=cpu_block.I_syn.cpu().numpy())

            more_run = True
            if more_run:
                iteration = 0
                def _ou(self, other):
                    nonlocal iteration
                    print("use _ou", iteration)
                    iteration += 1
                    return sample_ou[iteration - 1]

                log2 = np.zeros_like(sample_spike)
                V_i2 = np.zeros_like(sample_vi)
                syn_current2 = np.zeros_like(sample_vi)
                iou2 = np.zeros_like(sample_vi)
                with mock.patch('models.block_new.block.update_i_background', _ou) as n:
                    for k in range(run_number):
                        cpu_block2.run(debug_mode=False)
                        log2[k] = cpu_block2.active.numpy()
                        V_i2[k] = cpu_block2.V_i.numpy()
                        syn_current2[k] = cpu_block2.I_syn.numpy()
                        iou2[k] = cpu_block2.i_background.numpy()
                        # Todo (vertify cpu_block with cpu_block2)

                        print("cpu2_block.t", cpu_block2.t)
                        print("cpu_block ou", iou[k])
                        print("cpu_block2 ou", iou2[k])
                        print("cpu_block vi", V_i[k])
                        print("cpu_block2 vi", V_i2[k])
                        print("cpu_block spike", log[k])
                        print("cpu_block2 spike", log2[k])
                        print("cpu_block syn", syn_current[k])
                        print("cpu_block2 syn", syn_current2[k])

    def _test_fr_char(self):
        from cuda_v1point2.python.dist_blockwrapper_pytorch import BlockWrapper
        import torch

        path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/jianfeng_region/dti_distribution_0.10m_d100_with_debug/module'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper('11.10.10.42:50051', os.path.join(path, 'uint8'), delta_t, route_path=route_path,
                                  allow_metric=False)

        run_number = 1000
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        for j, (freqs, ) in enumerate(
                dist_block.run(1000, freqs=True, vmean=False, sample_for_show=False, iou=True, checked=False)):
            sample_freqs[j, :] = freqs
        print('mean firing rate (transition time))',
              torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                        dist_block.neurons_per_subblk.float()))
        sample_freqs = sample_freqs.cpu().numpy()
        np.save("freqs_uint8.npy", sample_freqs)


    def _test_compare_noise(self):
        from cuda.python.dist_blockwrapper_pytorch import BlockWrapper
        # from models.block import block
        from brain_block.block import block
        from generation.read_block import connect_for_block
        import torch

        # path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/jianfeng_region_test/dti_distribution_5m_d1000_with_debug/module'
        path = '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_16_80m'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper('11.5.13.57:50051', os.path.join(path, 'single'), delta_t, route_path=route_path)

        sample_path = torch.from_numpy(np.load(os.path.join(path, "debug_selection_idx.npy")).astype(np.int64)).cuda()
        print("dist_block._neurons_thrush", dist_block._neurons_thrush)
        print("dist_block.neurons_per_block", dist_block.neurons_per_block)
        sample_idx = dist_block._neurons_thrush[sample_path[:, 0]] + sample_path[:, 1]
        assert (sample_idx < dist_block._neurons_thrush[sample_path[:, 0] + 1]).all()

        assert (dist_block._neurons_thrush[sample_path[:, 0]] <= sample_idx).all()

        run_number = 300

        # cpu_path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks/d1000_ou/uint8'
        # cpu_path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks_debug/d1000_ou/single' 
        cpu_path = '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/single_small/single'

        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        print(w_uij.dtype)
        # w_uij = w_uij / torch.tensor(255., dtype=torch.float32)

        dist_block.set_samples(sample_idx)
        print('sample_idx.size()', sample_idx.size())
        print("total subblks", dist_block.subblk_id.shape[0])
        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        for j, (freqs, spike, vi) in enumerate(
                dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True)):
            sample_spike[j, :] = spike
            sample_vi[j, :] = vi
            sample_freqs[j, :] = freqs
        print('mean firing rate (transition time))',
              torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                        dist_block.neurons_per_subblk.float()))
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()

        dist_block.shutdown()

        sample_log_with_noise = sample_spike.copy()

        iter = 0

        def _or(self, other):
            nonlocal iter
            print(sample_log_with_noise[iter].shape, sample_log_with_noise[iter].astype(np.float32).mean() * 1000)
            sample = torch.from_numpy(sample_log_with_noise[iter]).to(torch.bool)
            iter += 1
            return sample

        with mock.patch('torch.Tensor.__or__', _or) as n:
            cpu_block = block(node_property=property, w_uij=w_uij, delta_t=1.)

            log = np.zeros_like(sample_spike)
            V_i = np.zeros_like(sample_vi)
            syn_current = np.zeros_like(sample_vi)

            for k in range(run_number):
                cpu_block.run(noise_rate=0.01)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()
                syn_current[k] = cpu_block.I_syn.numpy()

                V_i_error = np.max(np.abs(sample_vi[k] - V_i[k]) / np.abs(V_i[k]))
                print("\niteration:", k, "======>>>")

                bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                print(" v related error:", V_i_error, np.mean(V_i[k]), np.mean(sample_vi[k]), "bug_spike num",
                      bug_idx.shape[0])

                if bug_idx.shape[0] > 0:
                    print("")
                    print("V_i[k].dtype:", cpu_block.V_i.dtype)
                    print("error in: ", k, " cpu log: ", log[k, bug_idx], " cuda log: ", sample_spike[k, bug_idx])
                    print("successive cpu vi", V_i[k, bug_idx], V_i[k - 1, bug_idx], V_i[k - 2, bug_idx])
                    print("successive cuda vi", sample_vi[k, bug_idx], sample_vi[k - 1, bug_idx],
                          sample_vi[k - 2, bug_idx])
        self.assertEqual(k, run_number - 1)

    def _test_simulate_debug_block(self):
        from cuda3.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block
        from generation.read_block import connect_for_block
        import torch

        path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks_debug/d4_ou'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper('11.10.14.51:50051', os.path.join(path, 'uint8'), delta_t, route_path=route_path)

        N_1 = np.load(os.path.join(path, "uint8/block_0.npz"))['property'].shape[0]
        N_2 = np.load(os.path.join(path, "uint8/block_1.npz"))['property'].shape[0]
        N_3 = np.load(os.path.join(path, "uint8/block_2.npz"))['property'].shape[0]
        N_4 = np.load(os.path.join(path, "uint8/block_3.npz"))['property'].shape[0]
        N = N_1 + N_2 + N_3 + N_4
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 50

        cpu_path = os.path.join(path, "uint8")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        print('w_uij.dtype', w_uij.dtype)
        w_uij = w_uij / torch.tensor(255., dtype=torch.float32)
        dist_block.set_samples(sample_idx)

        print('sample_idx.size()', sample_idx.size())
        print("total subblks", dist_block.subblk_id.shape[0])
        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        for j, (freqs, spike, vi, i_sy, iouu) in enumerate(
                dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True, iou=True)):
            sample_spike[j, :] = spike
            sample_vi[j, :] = vi
            sample_ou[j, :] = iouu.cpu()
            sample_syn_current[j, :] = i_sy.cpu()
            sample_freqs[j, :] = freqs
        print('mean firing rate (transition time))',
              torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                        dist_block.neurons_per_subblk.float()))
        print("sample_syn_current:", sample_syn_current[0, :10])
        sample_spike = sample_spike.cpu().numpy()
        sample_vi = sample_vi.cpu().numpy()
        sample_syn_current = sample_syn_current.numpy()

        print(dist_block.last_time_stat())
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

            for k in range(run_number):
                cpu_block.run(debug_mode=False)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()
                syn_current[k] = cpu_block.I_syn.numpy()
                iou[k] = cpu_block.i_background.numpy()

                V_i_error = np.nanmax(np.abs(sample_vi[k] - V_i[k]) / np.abs(V_i[k]))
                ou_error = np.nanmax(np.abs(sample_ou.numpy()[k] - iou[k]) / np.abs(iou[k]))
                print("\niteration:", k, "======>>>")
                print(f"syn current comparison, now fire {log[k].sum()} at neuron {log[k].nonzero()[0]}")
                index = np.nonzero(syn_current[k])[0]
                index2 = np.nonzero(sample_syn_current[k])[0]
                if len(index) == len(index2) and (index == index2).all():
                    print("  post-synaptic current are aligned, differ at")
                    index_star = np.where(syn_current[k] != sample_syn_current[k])[0]
                    print("    in cuda:", sample_syn_current[k][index_star])
                    print("    in python:", syn_current[k][index_star])
                else:
                    print("  syn current is not aligned.")
                syn_i_error = np.nanmax(np.abs(sample_syn_current[k] - syn_current[k]) / np.abs(syn_current[k]))
                print("syn related error", syn_i_error, "ou_error", ou_error, "v related error:", V_i_error)

                bug_idx = (log[k] != sample_spike[k]).nonzero()[0]
                if bug_idx.shape[0] > 0:
                    print("")
                    print("error in: ", k, " cpu log: ", log[k, bug_idx], " cuda log: ", sample_spike[k, bug_idx])
                    # print("successive cpu vi", V_i[k, bug_idx], V_i[k - 1, bug_idx], V_i[k - 2, bug_idx])
                    # print("successive cuda vi", sample_vi[k, bug_idx], sample_vi[k - 1, bug_idx],
                    #       sample_vi[k - 2, bug_idx])

    def _test_check_map_in_cuda_with_cards_partition(self):
        from cuda.python.dist_blockwrapper_pytorch import BlockWrapper
        from models.block_new import block as block_ou
        from models.block import block as block_noise
        from generation.read_block import connect_for_block
        import torch
        np.set_printoptions(precision=4)
        ou_driven = False
        single_card = True

        path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks_debug/d4_ou_1card'
        route_path = None
        delta_t = 1.
        dist_block = BlockWrapper('11.5.12.22:50051', os.path.join(path, 'single'), delta_t, route_path=route_path)

        bases = [0]
        totals = []

        if single_card:
            cards = 1
        else:
            cards = 4
        for i in range(cards):
            file = np.load(os.path.join(path, "single/block_%d.npz"%i))
            prop = file['property']
            bases.append(bases[-1] + prop.shape[0])
        bases = np.array(bases)

        for i in range(cards):
            file = np.load(os.path.join(path, "single/block_%d.npz"%i))
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
        print("output_neuron| input_neuron| input_channel")
        print(totals[:, :-2].astype(np.int64))

        N = bases[-1]
        sample_idx = torch.arange(N, dtype=torch.int64).cuda()
        run_number = 200

        dist_block.set_samples(sample_idx)

        print('sample_idx.size()', sample_idx.size())
        print("total subblks", dist_block.subblk_id.shape[0])

        sample_spike = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.uint8).cuda()
        sample_vi = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32).cuda()
        sample_freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        sample_ou = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        sample_syn_current = torch.empty([run_number, sample_idx.shape[0]], dtype=torch.float32)
        if ou_driven:
            for j, (freqs, spike, vi, i_sy, iouu) in enumerate(
                    dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True, iou=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_ou[j, :] = iouu.cpu()
                sample_syn_current[j, :] = i_sy.cpu()
                sample_freqs[j, :] = freqs
            print('mean firing rate (transition time))',
                  torch.div(torch.mean(sample_freqs.float(), dim=0) * 1000 / delta_t,
                            dist_block.neurons_per_subblk.float()))
        else:
            for j, (freqs, spike, vi) in enumerate(
                    dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True)):
                sample_spike[j, :] = spike
                sample_vi[j, :] = vi
                sample_freqs[j, :] = freqs
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

        cpu_path = os.path.join(path, "single")
        print(cpu_path)
        property, w_uij = connect_for_block(cpu_path, return_src=False)
        print('w_uij.dtype', w_uij.dtype)
        # w_uij = w_uij / torch.tensor(255., dtype=torch.float32)

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

                for k in range(run_number):
                    cpu_block.run(debug_mode=False)
                    log[k] = cpu_block.active.numpy()
                    V_i[k] = cpu_block.V_i.numpy()
                    syn_current[k] = cpu_block.I_syn.numpy()
                    iou[k] = cpu_block.i_background.numpy()

                    print("\niteration", k)
                    print("cuda vi", sample_vi[k])
                    print("cpu vi", V_i[k])
                    if np.count_nonzero(sample_spike[k])>0:
                        print("    cuda firing neuron", list(sample_spike[k].nonzero()[0]))
                        print("    cpu firing neuron", list(log[k].nonzero()[0]))
                    print("cuda syn current", sample_syn_current[k])
                    print("cpu syn current", syn_current[k])
        else:
            with mock.patch('torch.Tensor.__or__', _or) as n:
                cpu_block = block_noise(node_property=property, w_uij=w_uij, delta_t=1.)
                log = np.zeros_like(sample_spike)
                V_i = np.zeros_like(sample_vi)
                syn_current = np.zeros_like(sample_vi)

                for k in range(run_number):
                    cpu_block.run(noise_rate=0.01)
                    log[k] = cpu_block.active.numpy()
                    V_i[k] = cpu_block.V_i.numpy()
                    syn_current[k] = cpu_block.I_syn.numpy()

                    print("\niteration", k)
                    print("cuda vi", sample_vi[k])
                    print("cpu vi", V_i[k])
                    if np.count_nonzero(sample_log[k]) > 0:
                        print("    cuda firing neuron", list(sample_log[k].nonzero()[0]))
                        print("    cpu firing neuron", list(log[k].nonzero()[0]))


if __name__ == '__main__':
    unittest.main()

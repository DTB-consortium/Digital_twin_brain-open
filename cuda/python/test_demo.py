import unittest
import matplotlib.pylab as plt
import unittest.mock as mock
import numpy as np
import os
import time

v_th = -50

def load_if_exist(func, *args):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func()
        np.save(path, out)
    return out


class TestBlock(unittest.TestCase):
    def _test_cpu_running(self):
        from brain_block.block import block
        from brain_block.bold_model import BOLD
        import brain_block.random_initialize as ri

        path = "./single_small_test/single"
        property, w_uij = ri.connect_for_block(path)
        N, K, _, _, _ = w_uij.shape
        w_uij = w_uij.permute([4, 0, 1, 2, 3]).reshape([4, N * K, N * K])
        property = property.reshape([N * K, -1])
        B = block(
            node_property=property,
            w_uij=w_uij,
            delta_t=1,
        )
        bold = BOLD(epsilon=10, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)

        seed_idx = 473
        log = []
        trace = []
        V_i = []
        J_ui = []
        I_syn = []
        bold_y = []

        for k in range(1000):
            sum_activate, mean_Vi = B.run(noise_rate=0.007)
            active = B.active
            print(k, int(active.sum()))
            log.append(active.numpy())
            trace.append([1 * k, float(B.V_i[seed_idx])])
            bold_y.append(bold.run(sum_activate))
            V_i.append(B.V_i.numpy())
            J_ui.append(B.J_ui.numpy())
            I_syn.append(B.I_syn.numpy())

        log = np.array(log)
        trace = np.array(trace)
        V_i = np.array(V_i)
        J_ui = np.stack(J_ui, 1)
        I_syn = np.array(I_syn)
        bold_y = np.stack(bold_y)

        idx = (log[-150:, :] > 0).nonzero()[1][0] if (log[-150:, :] > 0).nonzero()[1].shape[0] > 0 else 470
        print("idx:", idx)

        np.save(os.path.join(path, "log.npy"), log)
        np.save(os.path.join(path, "trace.npy"), trace)
        np.save(os.path.join(path, "V_i.npy"), V_i)
        np.save(os.path.join(path, "bold_y.npy"), bold_y)

        fig = plt.figure(figsize=(4, 4), dpi=300)
        fig1 = fig.add_subplot(1, 1, 1)
        fig1.imshow(V_i[-500:, :375].T)
        '''
        fig2 = fig.add_subplot(4, 2, 2)
        fig2.plot(trace[-500:, 0], V_i[-500:, idx])
        fig3 = fig.add_subplot(4, 2, 3)
        fig3.plot(trace[-500:, 0], J_ui[0, -500:, idx])
        fig7 = fig.add_subplot(4, 2, 4)
        fig7.plot(trace[-500:, 0], I_syn[-500:, idx])
        fig5 = fig.add_subplot(4, 2, 5)
        fig5.plot(trace[-500:, 0], J_ui[2, -500:, idx])
        fig8 = fig.add_subplot(4, 2, 6)
        fig8.plot(trace[:, 0], bold_y[:, 0])
        fig4 = fig.add_subplot(4, 2, 7)
        fig4.plot(trace[-500:, 0], J_ui[1, -500:, idx])
        fig6 = fig.add_subplot(4, 2, 8)
        fig6.plot(trace[-500:, 0], J_ui[3, -500:, idx])
        '''
        plt.show()

    def _test_compare_via_gpu_noise(self):
        from brain_block.block import block
        import brain_block.random_initialize as ri
        import torch

        path = "dti_24_50m_cpu_noise"

        iter = 0
        noise = np.load(os.path.join(path, "debug_noise.npy"))

        def rand(shape):
            nonlocal iter
            sample = torch.from_numpy(noise[iter % noise.shape[0], :])
            assert sample.shape == shape
            iter += 1
            return sample

        with mock.patch('torch.rand', rand) as n:
            property, w_uij = ri.connect_for_block(path)
            N, K, _, _, _ = w_uij.shape
            w_uij = w_uij.permute([4, 0, 1, 2, 3]).reshape([4, N * K, N * K])
            property = property.reshape([N * K, -1])
            B = block(
                node_property=property,
                w_uij=w_uij,
                delta_t=1,
            )
            seed_idx = 473
            log = []
            trace = []
            V_i = []

            gpu_log = np.load(os.path.join(path, "spike.npy"))
            gpu_V_i = np.load(os.path.join(path, "voltage.npy"))

            for k in range(4000):
                B.run(noise_rate=0.01)
                log.append(B.active.numpy())
                trace.append([1 * k, float(B.V_i[seed_idx])])
                V_i.append(B.V_i.numpy())
                V_i_error = np.max(np.abs(gpu_V_i[k]- B.V_i.numpy()))/(np.max(B.V_i.numpy()) - np.min(B.V_i.numpy()))
                spliking_error = len((B.active.numpy() - gpu_log[k]).nonzero()[0])
                if spliking_error > 0:
                    bug_idx = np.where(B.active.numpy() != gpu_log[k])
                    print("")
                    print("error in ", k)
                    print(gpu_V_i[k, bug_idx][0], gpu_V_i[k-1, bug_idx][0], gpu_V_i[k-2, bug_idx][0])
                    print(V_i[k][bug_idx], V_i[k-1][bug_idx], V_i[k-2][bug_idx])
                    input()
                print(k, np.uint32(log[k]).sum(), np.uint32(gpu_log[k]).sum(), spliking_error, V_i_error)


            log = np.array(log)
            trace = np.array(trace)
            V_i = np.array(V_i)

            V_i_error = np.max(np.abs(gpu_V_i- V_i))/(np.max(V_i) - np.min(V_i))
            spliking_error = len((log - gpu_log).nonzero()[0])

            print("V_i error: {}; spliking error: {}".format(V_i_error, spliking_error))

            fig = plt.figure(figsize=(8, 6), dpi=500)
            fig1 = fig.add_subplot(2, 2, 1)
            fig1.imshow(log[1200:1300])
            fig2 = fig.add_subplot(2, 2, 2)
            fig2.plot(trace[1200:1300, 0], V_i[1200:1300, seed_idx])

            fig3 = fig.add_subplot(2, 2, 3)
            fig3.imshow(gpu_log[1200:1300])
            fig4 = fig.add_subplot(2, 2, 4)
            fig4.plot(trace[1200:1300, 0], gpu_V_i[1200:1300, seed_idx])
            plt.show()


    def test_compare_via_dist_noise(self):
        from cuda.python.dist_blockwrapper import BlockWrapper
        from brain_block.block import block
        import brain_block.random_initialize as ri
        import torch

        path = '/home1/bychen/spliking_nn_for_brain_simulation/dti_4_2m'
        dist_block = BlockWrapper('192.168.2.108:50051', os.path.join(path, 'single'), 0.01, 1.)
        sample_path = np.load(os.path.join(path, "debug_selection_idx.npy")).astype(np.uint32)
        sample_idx = dist_block._neurons_thrush[sample_path[:, 0]] + sample_path[:, 1]
        assert (sample_idx < dist_block._neurons_thrush[sample_path[:, 0]+1]).all()
        order, resort_order, recover_order = np.unique(sample_idx, return_index=True, return_inverse=True)
        assert (order[recover_order] == sample_idx).all()

        #order = np.arange(dist_block.total_neurons, dtype=np.uint32)
        #recover_order = order

        run_number = 25

        dist_block.set_samples(order)
        _, sample_log_with_noise, sample_vi = dist_block.run(run_number, freqs=True, vmean=False, sample_for_show=True)
        print(sample_log_with_noise.shape)
        sample_log_with_noise = sample_log_with_noise[:, recover_order]
        sample_vi = sample_vi[:, recover_order]
        sample_log = (sample_vi == v_th)
        assert (np.logical_and(sample_log, sample_log_with_noise) == sample_log).all()

        print(np.unique(sample_vi[:100], axis=1).shape[1],
              np.mean(sample_log_with_noise.astype(np.float32))*1000,
              np.mean(sample_log.astype(np.float32))*1000)

        path = "./single_small/single"

        iter = 0
        def _or(self, other):
            nonlocal iter
            print(sample_log_with_noise[iter].shape, sample_log_with_noise[iter].astype(np.float32).mean()*1000)
            sample = torch.from_numpy(sample_log_with_noise[iter]).to(torch.bool)
            iter += 1
            return sample

        with mock.patch('torch.Tensor.__or__',  _or) as n:
            property, w_uij = ri.connect_for_block(path)
            N, K, _, _, _ = w_uij.shape
            w_uij = w_uij.permute([4, 0, 1, 2, 3]).reshape([4, N * K, N * K])
            property = property.reshape([N * K, -1])
            cpu_block = block(
                node_property=property,
                w_uij=w_uij,
                delta_t=1,
            )

            log = np.zeros_like(sample_log)
            V_i = np.zeros_like(sample_vi)

            for k in range(run_number):
                cpu_block.run(noise_rate=0.01)
                log[k] = cpu_block.active.numpy()
                V_i[k] = cpu_block.V_i.numpy()

                assert ((V_i[k] == v_th) == log[k]).all()

                V_i_error = np.max(np.abs(sample_vi[k]- V_i[k]))/(np.max(V_i[k]) - np.min(V_i[k]))
                V_i_error_2 = np.max(np.abs(np.sort(sample_vi[k])- np.sort(V_i[k])))/(np.max(V_i[k]) - np.min(V_i[k]))

                V_i_error_3 = (V_i[k] >sample_vi[k]).nonzero()[0].shape[0]/V_i[k].shape[0]

                bug_idx = (log[k] != sample_log[k]).nonzero()[0]

                if bug_idx.shape[0] > 0:
                    print("")
                    print("error in ", k, log[k, bug_idx], sample_log[k, bug_idx])
                    print(V_i[k, bug_idx], V_i[k-1, bug_idx], V_i[k-2, bug_idx])
                    print(sample_vi[k, bug_idx], sample_vi[k-1, bug_idx], sample_vi[k-2, bug_idx])
                    input()
                print(k, V_i_error, V_i_error_2, V_i_error_3, np.mean(V_i[k]), np.mean(sample_vi[k]), bug_idx.shape[0] / sample_log[k].nonzero()[0].shape[0] if sample_log[k].nonzero()[0].shape[0] > 0 else 'nan')

            print(np.mean(log.astype(np.float32))*1000)

    def _test_gpu_running(self):
        from cuda.python.BrainBlock import BlockWrapper as block_gpu
        path = "./dti_single_10m/single"
        arr = np.load(os.path.join(path, "block_0.npz"))
        properties = arr["property"]
        sample_idx = np.random.choice(properties.shape[0], 2000000, replace=False)
        indices = arr["idx"]
        weights = arr["weight"]
        delta_t = np.float32(1.)
        noise_rate = np.float32(.015)
        activates = np.zeros([4000, sample_idx.shape[0]], dtype=np.uint8)
        Vis = np.zeros([4000, sample_idx.shape[0]], dtype=np.float32)

        B = block_gpu(properties, indices, weights, 0, 0, delta_t)
        #bold = BOLD(epsilon=0.5, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
        #bold_y = []
        for k in range(2000):
            B.run(noise_rate)
            sum_activate, mean_Vi = B.get_freqs(), B.get_vmeans()
            print(sum_activate.sum(), mean_Vi.mean())
            #bold_y.append(bold.run(sum_activate))
            #sum_activates.append(sum_activate)
            #mean_Vis.append(mean_Vi)
            pass

        for k in range(4000):
            B.run(noise_rate)
            activate = np.uint8(B.get_t_actives() == 2000+k+1)
            #assert np.int32(activate).sum() == B.get_freqs().sum()
            activates[k, :] = activate[sample_idx]
            Vis[k, :] = B.get_v_membranes()[sample_idx]

        #bold_y = np.stack(bold_y)
        #np.save(os.path.join(path, "bold_y.npy"), bold_y)
        np.save(os.path.join(path, "spike.npy"), activates)
        np.save(os.path.join(path, "voltage.npy"), Vis)
        np.save(os.path.join(path, "sample_idx.npy"), sample_idx)
        np.save(os.path.join(path, "sample_property.npy"), properties[sample_idx])
        sample_conn = np.zeros([properties.shape[0], 1, 100, 4], dtype=weights.dtype)
        conn = weights.reshape([properties.shape[0], 1, 100, 2])
        conn_idx = indices.T.reshape([properties.shape[0], 1, 100, 2, 4])
        sample_conn[:, :, :, 0][conn_idx[:, :, :, 0, 3] == 0] = conn[:, :, :, 0][conn_idx[:, :, :, 0, 3] == 0]
        sample_conn[:, :, :, 2][conn_idx[:, :, :, 0, 3] == 2] = conn[:, :, :, 0][conn_idx[:, :, :, 0, 3] == 2]
        sample_conn[:, :, :, 1][conn_idx[:, :, :, 0, 3] == 1] = conn[:, :, :, 1][conn_idx[:, :, :, 0, 3] == 1]
        sample_conn[:, :, :, 3][conn_idx[:, :, :, 0, 3] == 3] = conn[:, :, :, 1][conn_idx[:, :, :, 0, 3] == 3]
        np.save(os.path.join(path, "sample_conn.npy"), sample_conn[sample_idx])

    def _test_speed(self):
        from cuda.python.dist_blockwrapper import BlockWrapper
        path = '/home1/bychen/spliking_nn_for_brain_simulation/dti_3_10m_new_form'
        length = 800
        sample_number = 200000

        def run(noise):
            block = BlockWrapper('192.168.2.108:50051', path, noise, 1.,)
            block.run(800, freqs=True, vmean=False, sample_for_show=False)
            start_time = time.time()
            Freqs = block.run(800, freqs=True, vmean=False, sample_for_show=False)
            end_time = time.time()
            #print(block.last_time_stat())
            sample_idx = load_if_exist(lambda:np.sort(np.random.choice(block.total_neurons, sample_number, replace=False)) if sample_number > 0 else np.arange(block.total_neurons, dtype=np.uint32), "sample_idx")
            block.set_samples(sample_idx)
            Freqs, _, _ = block.run(800, freqs=True, vmean=False, sample_for_show=True)
            start_time_2 = time.time()
            Freqs, _, _ = block.run(800, freqs=True, vmean=False, sample_for_show=True)
            end_time_2 = time.time()
            #print(block.last_time_stat())
            return 1000 * np.median(Freqs.astype(np.float64).mean(axis=0)/block.neurons_per_subblk.astype(np.float64)), \
                   1000 * (end_time-start_time)/length, \
                   1000 * (end_time_2 -start_time_2)/length

        freqs = []
        times = []
        max_freqs = []

        for i in np.arange(0, 100, 10):
            f, t , t2= run(i/1000)
            freqs.append(f)
            times.append(t)
            max_freqs.append(min(f + i, 200))
            print(f, t, t2,  i)

        fig_fp = plt.figure(figsize=(4, 4), dpi=500)
        fig_fp.gca().plot(freqs, times, 'r', label='output freqs')
        fig_fp.gca().plot(max_freqs, times, 'g', label='input freqs')
        fig_fp.gca().legend()
        fig_fp.gca().set_ylabel('duration(ms)')
        fig_fp.gca().set_xlabel('freq(Hz)')
        fig_fp.savefig('result.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig_fp)



if __name__ == '__main__':
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    unittest.main()


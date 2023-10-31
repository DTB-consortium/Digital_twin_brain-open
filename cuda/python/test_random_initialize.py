import unittest
from mpi4py import MPI
from brain_block.random_initialize import *
from scipy.io import loadmat
from brain_block.block_unitary import block


class TestBlock(unittest.TestCase):
    '''
    20Hz, max=100Hz, degree=1000
    (0.001617693924345076, 
    0.0001014432345982641, 
    0.003837576135993004, 
    0.000286647496977821)
    
    20Hz, max=100Hz, degree=100
    (0.010177472606301308, 
    0.0006282327813096344, 
    0.03813745081424713, 
    0.0021428221371024847)
    
    20Hz, max=100Hz, degree=137
    (0.011156645603477955, 
    0.0006994128925725818, 
    0.03756600618362427, 
    0.002292768796905875)
    
    20Hz, max=100Hz, degree=175
    (0.008750724606215954, 
    0.0005495150107890368, 
    0.02728760614991188, 
    0.0017100359546020627)
    
    20Hz, max=100Hz, degree=250
    (0.00623534107580781, 
    0.000390015309676528, 
    0.0181050356477499, 
    0.0012118576560169458) 
    
    20Hz, max=100Hz, degree=256
    (0.00615951232612133, 
    0.00038484969991259277, 
    0.017425572499632835, 
    0.0011572173098102212)
    
    20Hz, max=100Hz, degree=256, v2
    (0.006593520753085613, 
    0.0004135779454372823, 
    0.017094451934099197, 
    0.0011611274676397443)
    
    10Hz, max=100Hz, degree=256
    (0.011884157545864582, 
    0.0007408251403830945, 
    0.04682011902332306, 
    0.0026754955761134624)
    
    10Hz, max=30Hz, degree=256
    (0.004142856691032648, 
    0.00025999429635703564, 
    0.004413307178765535, 
    0.00031628430588170886)
    
    30Hz, max=100Hz, degree=256
    (0.00413433788344264, 
    0.0002601199084892869, 
    0.010248271748423576, 
    0.0007171945180743933)

    20Hz, max=100Hz, degree=33
    (0.043160390108823776, 
    0.002674056449905038, 
    0.32136210799217224, 
    0.010734910145401955)
    
    20Hz, max=100Hz, degree=20
    (0.07025660574436188, 
    0.004354883451014757, 
    0.9715675711631775, 
    0.018650120124220848)
    
    20Hz, max=50Hz, degree=20
    (0.03231345862150192, 
    0.002023769076913595, 
    0.08683804422616959, 
    0.004143711179494858)
    
    20Hz, max=50Hz, degree=20, g_li=0.003
    (0.022178268060088158, 
    0.0013867146335542202, 
    0.09227322041988373, 
    0.004128940403461456)
    '''
    @staticmethod
    def _random_initialize_for_dti_distributation_block(path, total_neurons, gui, degree, minmum_neurons_for_block=None, dtype=['single']):
        file = loadmat('./DTI_T1_92ROI')
        block_size = file['t1_roi'][:, 0]
        block_size /= block_size.sum(0)
        dti = torch.from_numpy(np.float32(file['weight_dti']))
        dti /= dti.std(1, keepdim=True)
        dti += dti.logsumexp(1).diag()
        dti /= dti.sum(1, keepdim=True)

        if minmum_neurons_for_block is None:
            minmum_neurons_for_block = degree * 100

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   "E_number": int(0.8 * max(i * total_neurons, minmum_neurons_for_block)),
                   "I_number": int(0.2 * max(i * total_neurons, minmum_neurons_for_block))}
                  for i in block_size]

        return connect_for_multi_sparse_block(dti, kwords,
                                       degree=degree,
                                       init_min=0,
                                       init_max=1,
                                       perfix=path,
                                       dtype=dtype)

    @staticmethod
    def _find_gui_in_1000_block(delta_t=1, default_Hz=30, max_output_Hz=100, T_ref=5, degree=256, g_Li=0.03, V_L=-75, V_rst=-65, V_th=-50, path = "./single_small_test/", need_test=False):
        prob = torch.tensor([[1.]])

        gap = V_th - V_rst

        noise_rate = 1/(1000/delta_t/default_Hz)
        max_delta_raise = gap/(1000/delta_t/max_output_Hz-T_ref)
        default_delta_raise = gap/(1000/delta_t/default_Hz-T_ref)

        leaky_compensation = g_Li * ((V_th+V_rst)/2 - V_L)

        label = torch.tensor([0.5 * (max_delta_raise + leaky_compensation),
                              0.5 * (max_delta_raise + leaky_compensation),
                              0.5 * (max_delta_raise - default_delta_raise),
                              0.5 * (max_delta_raise - default_delta_raise)])
        print(label.tolist())

        gui = label

        def test_gui(max_iter=4000, noise_rate=noise_rate):
            property, w_uij = connect_for_block(os.path.join(path, 'single'))
            N, K, _, _, _ = w_uij.shape
            w_uij = w_uij.permute([4, 0, 1, 2, 3]).reshape([4, N * K, N * K])
            property = property.reshape([N * K, -1])
            B = block(
                node_property=property,
                w_uij=w_uij,
                delta_t=delta_t,
            )
            out_list = []
            Hz_list = []

            for k in range(max_iter):
                B.run(noise_rate=noise_rate, isolated=True)
                out_list.append(B.I_ui.mean(-1).abs())
                Hz_list.append(float(B.active.sum())/property.shape[0])
            out = torch.stack(out_list[-500:]).mean(0)
            Hz = sum(Hz_list[-500:]) * 1000/500
            print('out:', out.tolist(), Hz)
            return out

        for i in range(20):
            connect_for_multi_sparse_block(prob, {'g_Li': g_Li,
                                                  'g_ui': gui,
                                                  "V_reset": -65},
                                           E_number=int(1.6e3), I_number=int(4e2), degree=degree, init_min=0, init_max=1, perfix=path)
            gui = gui * label / test_gui()
            print('gui:', gui.tolist())

        if need_test:
            connect_for_multi_sparse_block(prob, {'g_Li': g_Li,
                                                  'g_ui': gui,
                                                  "V_reset": -65},
                                           E_number=int(1.6e3), I_number=int(4e2), degree=degree, init_min=0, init_max=1, perfix=path)
            for i in range(0, 200, 5):
                print("testing ", i)
                test_gui(noise_rate=i/1000)

        return tuple(gui.tolist())

    def _test_random_initialize_for_single_small_block(self):
        prob = torch.tensor([[1.]])
        connect_for_multi_sparse_block(prob, {'g_Li': 0.003,
                                              'g_ui': (0.022178268060088158,
                                                       0.0013867146335542202,
                                                       0.09227322041988373,
                                                       0.004128940403461456),
                                              "V_reset": -65,
                                              "V_th": -50},
                                       E_number=int(8e2), I_number=int(2e2), degree=int(20), init_min=0, init_max=1, perfix="./single_small/")

    def _test_random_initialize_for_10k_with_multi_degree(self):
        degree_list=[5, 10, 20, 50, 100]
        prob = torch.tensor([[1.]])
        g_Li = 0.03
        path = './single_10k'
        os.makedirs(path, exist_ok=True)

        for d in degree_list:
            print('processing', d)
            gui = self._find_gui_in_1000_block(degree=d, g_Li=g_Li)
            old_form_dir = os.path.join(path, 'degree_{}'.format(d))
            connect_for_multi_sparse_block(prob, {'g_Li': g_Li,
                                                  'g_ui': gui,
                                                  "V_reset": -65,
                                                  "V_th": -50},
                                           E_number=int(8e3), I_number=int(2e3), degree=int(d), init_min=0, init_max=1, perfix=old_form_dir)

    def _test_random_initialize_for_dti_distributation_block_200k(self):
        file = loadmat('./GM_AAL_age50')
        block_size = file['GM_AAL_age50'].sum(0)[:90]
        block_size /= block_size.sum(0)
        file = loadmat('./matrix_hcp')
        dti = torch.from_numpy(np.float32(file['matrix_HCP'])).mean(0)[:90, :90]
        merge_group = [(4, 8),
                       (10, 12),
                       (16, 18),
                       (20, 30),
                       (34, 36),
                       (38, 40),
                       (24, 26),
                       (42, 44),
                       (48, 52),
                       (62, 64),
                       (68, 70, 72),
                       (74, 76),
                       (78, 80),
                       (82, 86)]

        delete_list = set()

        for group in merge_group:
            suffix = [0, 1]
            for s in suffix:
                base = group[0] + s
                for _idx in group[1:]:
                    idx = _idx + s
                    delete_list.add(idx)
                    dti[base, :] += dti[idx, :]
                    dti[:, base] += dti[:, idx]
                    block_size[base] += block_size[idx]

        exist_list = [i for i in range(90) if i not in delete_list]
        dti[exist_list, exist_list] = 0
        dti = dti[exist_list, :][:, exist_list]
        block_size = block_size[exist_list]

        dti /= dti.std(1, keepdim=True)
        dti += dti.logsumexp(1).diag()
        dti /= dti.sum(1, keepdim=True)

        total_neurons = int(2e5)

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.003,
                   'g_ui': (0.022178268060088158,
                            0.0013867146335542202,
                            0.09227322041988373,
                            0.004128940403461456),
                   "E_number": int(0.8 * max(i * total_neurons, 2000)),
                   "I_number": int(0.2 * max(i * total_neurons, 2000))}
                  for i in block_size]

        connect_for_multi_sparse_block(dti, kwords,
                                       degree=int(20),
                                       init_min=0,
                                       init_max=1,
                                       perfix="./dti_distribution_200k/",
                                       dtype=["single"])

    def _test_random_initialize_for_dti_single_block(self):
        path = "./dti_single_500k/"
        os.makedirs(path, exist_ok=True)
        self._random_initialize_for_dti_distributation_block('./dti_distribution_500k', int(5e5), (0.001617693924345076,
                                                                                                   0.0001014432345982641,
                                                                                                   0.003837576135993004,
                                                                                                   0.000286647496977821),
                                                             1000)
        merge_dti_distributation_block("./dti_distribution_500k/single", path, dtype=["single", "half"])

    def _test_random_initialize_for_dti_single_block_50m(self):
        path = "./dti_24_50m/"
        os.makedirs(path, exist_ok=True)

        self._random_initialize_for_dti_distributation_block('./dti_distribution_50m', int(5e7), (0.00615951232612133,
                                                                                                  0.00038484969991259277,
                                                                                                  0.017425572499632835,
                                                                                                  0.0011572173098102212),
                                                             256)

        block_threshhold = merge_dti_distributation_block("./dti_distribution_50m/single",
                                             path,
                                             number=24,
                                             dtype=["single"],
                                             debug_block_path="./single_small/single")
        size = block_threshhold[-1]

        sample_rate = 0.02

        debug_selection_idx = np.load(os.path.join(path, "debug_selection_idx.npy"))
        debug_selection_idx = block_threshhold[debug_selection_idx[:, 0]] + debug_selection_idx[:, 1]

        sample_selection_idx = np.random.choice(size, int(sample_rate*2*size), replace=False)
        sample_selection_idx = np.array(list(set(sample_selection_idx) - set(debug_selection_idx)))
        sample_selection_idx = np.random.permutation(sample_selection_idx)[:int(sample_rate*size)]

        assert sample_selection_idx.shape[0] == int(sample_rate*size)

        sample_block_idx, sample_neuron_idx = turn_to_block_idx(sample_selection_idx, block_threshhold)

        np.save(os.path.join(path, "sample_selection_idx"),
                np.ascontiguousarray(
                    np.stack([sample_block_idx, sample_neuron_idx], axis=1)))

    def _test_random_initialize_for_dti_30_600m(self):
        path = "./dti_30_600m/"
        total_blocks = 30
        os.makedirs(path, exist_ok=True)

        #gui = self._find_gui_in_1000_block(degree=100)
        gui = tuple(np.load('./dti_distribution_600m/single/block_75.npz')['property'][0, 10:14].tolist())
        conn = self._random_initialize_for_dti_distributation_block(None, int(6e8), gui, 100)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, total_blocks, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           number=30,
                                           dtype=["single"],
                                           debug_block_path="./single_small/single",
                                           only_load=(i != 0))

    def test_random_initialize_for_dti_16_80m(self):
        path = "./dti_16_80m/"
        total_blocks = 16
        os.makedirs(path, exist_ok=True)
 
        #gui = self._find_gui_in_1000_block(degree=100)
        gui = tuple(np.load('./dti_new_10m/dti_n1_d100/single/block_0.npz')['property'][0, 10:14].tolist())
        conn = self._random_initialize_for_dti_distributation_block(None, int(8e7), gui, 100)
 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
 
        for i in range(rank, total_blocks, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           number=total_blocks,
                                           dtype=["single"],
                                           debug_block_path="./single_small/single",
                                           only_load=(i != 0))

    def _test_random_initialize_for_dti_60_800m(self):
        path = "./dti_60_800m/"
        blocks_number = [2] * 30 + [2/3] * 30
        os.makedirs(path, exist_ok=True)

        #gui = self._find_gui_in_1000_block(degree=100)
        gui = tuple(np.load('./dti_distribution_600m/single/block_75.npz')['property'][0, 10:14].tolist())
        conn = self._random_initialize_for_dti_distributation_block(None, int(8e8), gui, 100)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        for i in range(rank, len(blocks_number), size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           number=blocks_number,
                                           dtype=["single"],
                                           debug_block_path="./single_small/single",
                                           only_load=(i != 0))

    def _test_random_initialize_for_dti_single_block_5m(self):
        path = "./dti_single_5m/"
        os.makedirs(path, exist_ok=True)
        self._random_initialize_for_dti_distributation_block("./dti_distribution_5m", int(5e6), (0.010177472606301308,
                                                                                                 0.0006282327813096344,
                                                                                                 0.03813745081424713,
                                                                                                 0.0021428221371024847),
                                                             100)
        merge_dti_distributation_block("./dti_distribution_5m/single", path, dtype=["single", "half"])

    def _test_random_initialize_for_dti_10m_multi(self):
        degree = [100]
        block_num = [1, 3]

        path = './dti_new_10m'
        os.makedirs(path, exist_ok=True)

        for d in degree:
            gui = self._find_gui_in_1000_block(degree=d)
            #gui = tuple(np.load('/home1/bychen/spliking_nn_for_brain_simulation/dti_new_10m/dti_distribution_d100/single/block_0.npz')['property'][0, 10:14].tolist())
            dti_distribution_path = os.path.join(path, 'dti_distribution_d{}'.format(d))
            self._random_initialize_for_dti_distributation_block(dti_distribution_path, int(1e7), gui, d)
            for n in block_num:
                dti_block_path = os.path.join(path, 'dti_n{}_d{}'.format(n, d))
                os.makedirs(dti_block_path, exist_ok=True)
                merge_dti_distributation_block(os.path.join(dti_distribution_path, 'single'),
                                                     dti_block_path,
                                                     dtype=["single"],
                                                     number=n,
                                                     debug_block_path="./single_small/single")

    def _test_random_initialize_for_dti_single_block_200k(self):
        path = "./dti_single_200k/"
        os.makedirs(path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_200k/single", path, dtype=["single"])

    def _test_random_initialize_for_dti_single_block_1_6m_and_half(self):
        double_path = "./dti_double_1_6m/"
        os.makedirs(double_path, exist_ok=True)


        self._random_initialize_for_dti_distributation_block('./dti_distribution_1_6m', int(1.6e6), (0.043160390108823776,
                                                                                                     0.002674056449905038,
                                                                                                     0.32136210799217224,
                                                                                                     0.010734910145401955),
                                                             33)
        merge_dti_distributation_block("./dti_distribution_1_6m/single", double_path, dtype=["single"], number=2)

        single_path = "./dti_single_1_6m/"
        os.makedirs(single_path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_1_6m/single", single_path, dtype=["single"], number=1)

        recover_idx_name = "recover_idx.npy"
        single_recover_idx = np.load(os.path.join(single_path, recover_idx_name))
        double_recover_idx = np.load(os.path.join(double_path, recover_idx_name))

        np.save(os.path.join(double_path, "resort_idx"), np.argsort(single_recover_idx)[double_recover_idx])

        data = np.load("dti_single_1_6m/single/block_0.npz")
        single_property = data["property"]
        data = np.load("dti_double_1_6m/single/block_0.npz")
        double_A_property = data["property"]
        data = np.load("dti_double_1_6m/single/block_1.npz")
        double_B_property = data["property"]
        resort_idx = np.load("dti_double_1_6m/resort_idx.npy")

        single_noise = np.random.rand(500, single_property.shape[0]).astype(np.float32)
        np.save("dti_single_1_6m/single/sample.npy", np.ascontiguousarray(single_noise))
        np.save("dti_double_1_6m/single/sample_1.npy", np.ascontiguousarray(single_noise[:, resort_idx][:, double_B_property.shape[0]:]))
        np.save("dti_double_1_6m/single/sample_0.npy", np.ascontiguousarray(single_noise[:, resort_idx][:, :double_B_property.shape[0]]))

    def _test_random_initialize_for_dti_with_4block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': (0.07025660574436188,
                            0.004354883451014757,
                            0.9715675711631775,
                            0.018650120124220848),
                   "E_number": int(0.8 * max(4000/90, 0)),
                   "I_number": int(0.2 * max(4000/90, 0))}] * 90

        connect_for_multi_sparse_block(prob, kwords,
                                       degree=20,
                                       init_min=0,
                                       init_max=1,
                                       perfix="./dti_distribution_4k/",
                                       dtype=["single"])

        double_path = "./dti_4_4k/"
        os.makedirs(double_path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_4k/single", double_path, dtype=["single"], number=4)

        single_path = "./dti_single_4k/"
        os.makedirs(single_path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_4k/single", single_path, dtype=["single"], number=1)

        recover_idx_name = "recover_idx.npy"
        single_recover_idx = np.load(os.path.join(single_path, recover_idx_name))
        double_recover_idx = np.load(os.path.join(double_path, recover_idx_name))

        resort_idx = np.argsort(single_recover_idx)[double_recover_idx]
        np.save(os.path.join(double_path, "resort_idx"), np.ascontiguousarray(resort_idx))

        data = np.load(os.path.join(single_path, "single/block_0.npz"))
        single_property = data["property"]

        single_noise = np.random.rand(500, single_property.shape[0]).astype(np.float32)
        np.save(os.path.join(single_path, "single/sample.npy"), np.ascontiguousarray(single_noise))

        single_noise_after_resort = single_noise[:, resort_idx]

        base = 0
        for i in range(4):
            data = np.load(os.path.join(double_path, "single/block_{}.npz".format(i)))
            property = data["property"]
            np.save(os.path.join(double_path, "single/sample_{}.npy".format(i)),
                    np.ascontiguousarray(single_noise_after_resort[:, base:base + property.shape[0]]))
            base += property.shape[0]

        assert base == single_property.shape[0]

    def _test_random_initialize_for_dti_with_4_small_block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.003,
                   'g_ui': (0.022178268060088158,
                            0.0013867146335542202,
                            0.09227322041988373,
                            0.004128940403461456),
                   "E_number": int(0.8 * max(4e4 / 90, 0)),
                   "I_number": int(0.2 * max(4e4 / 90, 0))}] * 90

        out = connect_for_multi_sparse_block(prob, kwords,
                                             degree=20,
                                             init_min=0,
                                             init_max=1,
                                             perfix=None,
                                             dtype=["single"])

        double_path = "./dti_4_4k/"
        os.makedirs(double_path, exist_ok=True)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, 4, size):
            merge_dti_distributation_block(out,
                                           double_path,
                                           dtype=["single"],
                                           number=4,
                                           output_degree=False,
                                           MPI_rank=i)

    def _test_random_initialize_for_dti_with_4_big_block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': (0.001617693924345076,
                            0.0001014432345982641,
                            0.003837576135993004,
                            0.000286647496977821),
                   "E_number": int(0.8 * max(2e6 / 90, 0)),
                   "I_number": int(0.2 * max(2e6 / 90, 0))}] * 90

        out = connect_for_multi_sparse_block(prob, kwords,
                                       degree=1000,
                                       init_min=0,
                                       init_max=1,
                                       perfix=None,
                                       dtype=["single"])

        double_path = "./dti_4_2m/"
        os.makedirs(double_path, exist_ok=True)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, 4, size):
            merge_dti_distributation_block(out,
                                            double_path,
                                            dtype=["single"],
                                            number=4,
                                            debug_block_path="./single_small/single",
                                            output_degree=False,
                                            MPI_rank=i)
        '''
        size = block_threshold[-1]

        single_noise = np.random.rand(500, size).astype(np.float32)
        base = 0
        for i in range(4):
            data = np.load(os.path.join(double_path, "single/block_{}.npz".format(i)))
            property = data["property"]
            np.save(os.path.join(double_path, "single/sample_{}.npy".format(i)),
                    np.ascontiguousarray(single_noise[:, base:base + property.shape[0]]))
            base += property.shape[0]

        assert base == size
        '''

    def _test_random_initialize_for_dti_with_1_big_block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': (0.001617693924345076,
                            0.0001014432345982641,
                            0.003837576135993004,
                            0.000286647496977821),
                   "E_number": int(0.8 * max(5e5 / 90, 0)),
                   "I_number": int(0.2 * max(5e5 / 90, 0))}] * 90

        connect_for_multi_sparse_block(prob, kwords,
                                       degree=1000,
                                       init_min=0,
                                       init_max=1,
                                       perfix="./dti_distribution_500k_randoem/",
                                       dtype=["single"])

        def run(double_path, number):
            os.makedirs(double_path, exist_ok=True)
            block_threshold = merge_dti_distributation_block("./dti_distribution_500k_random/single",
                                                        double_path,
                                                        dtype=["single"],
                                                        number=number,
                                                        debug_block_path="./single_small/single",
                                                        output_degree=True)

            size = block_threshold[-1]

            single_noise = np.random.rand(500, size).astype(np.float32)
            base = 0
            for i in range(number):
                data = np.load(os.path.join(double_path, "single/block_{}.npz".format(i)))
                property = data["property"]
                np.save(os.path.join(double_path, "single/sample_{}.npy".format(i)),
                        np.ascontiguousarray(single_noise[:, base:base + property.shape[0]]))
                base += property.shape[0]

            assert base == size

        run("./dti_single_500k/", 1)
        run("./dti_3_500k/", 3)

    def _test_random_initialize_for_dti_single_block_1_6m_for_verify(self):
        data = np.load("dti_single_1_6m/single/block_0.npz")
        single_idx = data["idx"]
        single_weight = data["weight"]
        data = np.load("dti_double_1_6m/single/block_0.npz")
        double_idx_A = data["idx"]
        double_weight_A = data["weight"]
        data = np.load("dti_double_1_6m/single/block_1.npz")
        double_idx_B = data["idx"]
        double_weight_B = data["weight"]
        resort_idx = np.argsort(np.load("dti_double_1_6m/resort_idx.npy"))

        size = data["property"].shape[0]
        del data

        for ii in np.random.choice(single_idx.shape[1], 10000):
            idx_0, idx_1, idx_2, idx_3 = single_idx[:, ii]
            v = single_weight[ii]
            assert idx_1 == 0
            block_idx = resort_idx[idx_0] // size
            new_idx_0 = resort_idx[idx_0] % size
            new_idx_1 = resort_idx[idx_2] // size
            new_idx_2 = resort_idx[idx_2] % size
            if block_idx == 0:
                select_idx = double_idx_A
                select_weight = double_weight_A
            else:
                select_idx = double_idx_B
                select_weight = double_weight_B
            idx = ((select_idx[0] == new_idx_0)
                  & (select_idx[1] == new_idx_1)
                  & (select_idx[2] == new_idx_2)
                  & (select_idx[3] == idx_3)).nonzero()[0]
            assert idx[0].shape[0] == 1
            idx = idx[0][0]
            print(ii, idx)
            assert select_weight[idx] == v

if __name__ == "__main__":
    unittest.main()

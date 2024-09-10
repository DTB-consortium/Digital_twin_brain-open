import os.path
import pickle
import unittest

import h5py
import numpy as np
import sparse
from mpi4py import MPI
from scipy.io import loadmat
from make_block import *

class TestBlock(unittest.TestCase):
    @staticmethod
    def warning_info():
        title = " " * 8 + "=" * 5 + " " * 2 + "SOME INFO IN GENERATION" + " " * 2 + "=" * 10
        info = "1. Modify each test case to generate corresponding block npz" + \
               "\n2. Default gui is in $PROJECTPATH/default_params.py" + \
               "\n3. debug block should have the same degree with original model if add debug" + \
               "\n4. MPI=CARDS in slurm setting\n\ta). 5m/card in single weight and d100\n\tb). 2m/card in uint8 weight and d1000\n\tc). 18m/card in uint8 weight and d100"
        waring_info = title + "\n" + info + "\n\n"
        print(waring_info)

    @staticmethod
    def _make_directory_tree(root_path, scale, degree, extra_info, dtype="single"):
        """
        make directory tree for each subject.

        Parameters
        ----------
        root_path: str
            each subject has a root path.

        scale: int
            number of neurons of whole brain.
        degree:
            in-degree of each neuron.

        init_min: float
            the lower bound of uniform distribution where w is sampled from.

        init_max: float
            the upper bound of uniform distribution where w is sampled from.

        extra_info: str
            supplementary information.

        Returns
        ----------
        second_path: str
            second path to save connection table

        """
        os.makedirs(root_path, exist_ok=True)
        os.makedirs(os.path.join(root_path, "raw_data"), exist_ok=True)
        second_path = os.path.join(root_path,
                                   f"dti_distribution_{int(scale // 1e6)}m_d{degree}_{extra_info}")
        os.makedirs(second_path, exist_ok=True)
        os.makedirs(os.path.join(second_path, "module"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "multi_module", dtype), exist_ok=True)  # 'single' means the precision.
        os.makedirs(os.path.join(second_path, "supplementary_info"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "DA"), exist_ok=True)

        return second_path, os.path.join(second_path, "module")

    @staticmethod
    def _add_laminar_cortex_model(conn_prob, gm, canonical_voxel=False):
        """
        Process the connection probability matrix, grey matter and degree scale for DTB with pure voxel and micro-column
        structure.  Each voxel is split into 2 populations (E and I). Each micro-column is spilt into 10 populations
        (L1E, L1I, L2/3E, L2/3I, L4E, L4I, L5E, L5I, L6E, L6I).

        Parameters
        ----------
        conn_prob: numpy.ndarray, shape [N, N]
            the connectivity probability matrix between N voxels/micro-columns.

        gm: numpy.ndarray, shape [N]
            the normalized grey matter in each voxel/micro-column.

        canonical_voxel: bool
            Ture for voxel structure; False for micro-column structure.

        Returns
        -------
        out_conn_prob: numpy.ndarray
            connectivity probability matrix between populations (shape [2*N, 2*N] for voxel; shape[10*N, 10*N] for micro
            -column) in the sparse matrix form.

        out_gm: numpy.ndarray
            grey matter for populations in DTB (shape [2*N] for voxel; shape[10*N] for micro-column).

        out_degree_scale: numpy.ndarray
            scale of degree for populations in DTB (shape [2*N] for voxel; shape[10*N] for micro-column).

        """
        if not canonical_voxel:
            lcm_connect_prob = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 3554, 804, 881, 45, 431, 0, 136, 0, 1020],
                                         [0, 0, 1778, 532, 456, 29, 217, 0, 69, 0, 396],
                                         [0, 0, 417, 84, 1070, 690, 79, 93, 1686, 0, 1489],
                                         [0, 0, 168, 41, 628, 538, 36, 0, 1028, 0, 790],
                                         [0, 0, 2550, 176, 765, 99, 621, 596, 363, 7, 1591],
                                         [0, 0, 1357, 76, 380, 32, 375, 403, 129, 0, 214],
                                         [0, 0, 643, 46, 549, 196, 327, 126, 925, 597, 2609],
                                         [0, 0, 80, 8, 92, 3, 159, 11, 76, 499, 1794]], dtype=np.float64
                                        )

            lcm_gm = np.array([0, 0,
                               33.8 * 78, 33.8 * 22,
                               34.9 * 80, 34.9 * 20,
                               7.6 * 82, 7.6 * 18,
                               22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
        else:

            # like micro-circuit, E:I:outer \approx 4:1:2 and most mass center in inner part.
            lcm_connect_prob = np.array([[4/7, 1/7, 2/7],
                                         [4/7, 1/7, 2/7]], dtype=np.float64)
            lcm_gm = np.array([0.8, 0.2], dtype=np.float64)

        lcm_gm /= lcm_gm.sum()

        weight = lcm_gm[::2]
        weight = weight / weight.sum(axis=0)

        syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
        lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
        lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
        lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)

        if conn_prob.shape[0] == 1:
            conn_prob[:, :] = 1
        else:
            conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
            conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)

        conn_prob[np.isnan(conn_prob)] = 0
        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape([-1])
        out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape([-1])
        conn_prob = sparse.COO(conn_prob)
        weight = np.broadcast_to(weight, (conn_prob.data.shape[0], 5))
        if not canonical_voxel:
            corrds1 = np.empty(
                [4, conn_prob.coords.shape[1] * lcm_connect_prob.shape[0] * int(lcm_connect_prob.shape[0] / 2)],
                dtype=np.int64)
            corrds1[3, :] = np.tile(np.repeat(np.arange(0, lcm_connect_prob.shape[0], 2), lcm_connect_prob.shape[0]),
                                conn_prob.coords.shape[1]).reshape([1, -1])
            corrds1[(0, 2), :] = np.broadcast_to(conn_prob.coords[:, :, None],
                                                 [2, conn_prob.coords.shape[1], lcm_connect_prob.shape[0] * int(
                                                     lcm_connect_prob.shape[0] / 2)]).reshape([2, -1])
            corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                              [conn_prob.coords.shape[1] * int(lcm_connect_prob.shape[0] / 2),
                                               lcm_connect_prob.shape[0]]).reshape([1, -1])
            data1 = conn_prob.data[:, None] * lcm_connect_prob[:, -1]
            data1 = (data1[:, None, :] * weight[:, :, None]).reshape([-1])
        else:
            corrds1 = np.empty(
                [4, conn_prob.coords.shape[1] * lcm_connect_prob.shape[0]],
                dtype=np.int64)
            corrds1[3, :] = 0
            corrds1[(0, 2), :] = np.broadcast_to(conn_prob.coords[:, :, None],
                                                 [2, conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([2, -1])
            corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                              [conn_prob.coords.shape[1],
                                               lcm_connect_prob.shape[0]]).reshape([1, -1])
            data1 = (conn_prob.data[:, None] * lcm_connect_prob[:, -1]).reshape(-1)

        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, conn_prob.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(np.arange(conn_prob.shape[0], dtype=np.int64)[:, None],
                                        [conn_prob.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                                [conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

        coords = np.concatenate([corrds1, corrds2], axis=1)
        data = np.concatenate([data1, data2], axis=0)
        shape = [conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                 lcm_connect_prob.shape[1] - 1]

        index = np.where(data)[0]
        print(f"process zero value in conn_prob {len(data)}-->{len(index)}")
        coords = coords[:, index]
        data = data[index]

        out_conn_prob = sparse.COO(coords=coords, data=data, shape=shape)
        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        if conn_prob.shape[0] == 1:
            out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
        return out_conn_prob, out_gm, out_degree_scale

    def test_generate_regional_brain(self, dtype='uint8'):
        path, scale, blocks, degree = './generation/reg_n10_g1_d100', int(5e6), 16, 100
        first_path, second_path = self._make_directory_tree(path, scale, degree, "regional", dtype=dtype)
        dti = loadmat("./DTI_voxel_networks_sum_1204.mat")
        conn_prob = np.float32(dti["net_sum"])
        conn_prob[np.diag_indices_from(conn_prob)] = 0.
        conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
        grey_matter = dti["roi_grey"].squeeze()
        grey_matter /= grey_matter.sum()
        conn_prob, grey_matter, degree_scale = self._add_laminar_cortex_model(conn_prob, grey_matter,
                                                                             canonical_voxel=True)
        degree_ = np.maximum((degree * degree_scale).astype(np.uint16), 1)
        size_scale =  np.array([np.int64(max(b * scale, 0)) for i, b in enumerate(grey_matter)])
        conn_prob, size_scale, degree_scale, order, partition = apply_map(conn_prob, blocks, size_scale, degree_scale, 
                                                                                                     max_rate=1.1, max_iter=8000, use_map=True)
        kwords = [{"C": 0.5,
                   "T_ref": 2,
                   'g_Li': 0.025,
                   "V_L": -70,
                   "V_th": -50,
                   "V_reset": -55,
                   'g_ui': (0.002, 0, 0.01, 0),
                   'tao_ui': (2, 40, 20, 50),
                   'sub_block_idx': order[i],
                   "size": b}
                  for i, b in enumerate(size_scale)]
        conn = connect_for_multi_sparse_block(conn_prob, kwords, degree=degree_, dtype=dtype, multi_conn2single_conn=True)
        # In Rich dynamics
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            if rank == 0:
                self.warning_info()
                population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
                population_base = np.add.accumulate(population_base)
                population_base = np.insert(population_base, 0, 0)
                os.makedirs(os.path.join(first_path, "supplementary_info"), exist_ok=True)
                np.save(os.path.join(first_path, "supplementary_info", "population_base.npy"), population_base)
            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           dtype=dtype,
                                           block_partition=partition)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*- 
# @Time : 2022/8/10 14:31 
# @Author : lepold
# @File : test_generation.py
import os.path
import pickle
import unittest

import h5py
import numpy as np
import sparse
from mpi4py import MPI
from scipy.io import loadmat

from default_params import gui_info
from generation.make_block import *
import psutil


class TestBlock(unittest.TestCase):
    @staticmethod
    def print_system_info(card: int):
        mem = psutil.virtual_memory()
        # 系统总计内存
        zj = float(mem.total) / 1024 / 1024 / 1024
        # 系统已经使用内存
        ysy = float(mem.used) / 1024 / 1024 / 1024

        # 系统空闲内存
        kx = float(mem.free) / 1024 / 1024 / 1024

        # print('%4d卡 系统总计内存:%d.3GB' % (card, zj))
        # print('%4d卡 系统已经使用内存:%d.3GB' % (card, ysy))
        print('%4d卡 系统空闲内存:%d.3GB' % (card, kx))

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

        coords = np.concatenate([corrds1, corrds2,], axis=1)
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

    @staticmethod
    def _add_laminar_cortex_include_subcortical_in_whole_brain(conn_prob, gm, divide_point=22703):
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

        # cortical column \approx (4/7, 1/7, 2/7)
        lcm_connect_prob_subcortical = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 4 / 7, 1 / 7, 0, 0, 2 / 7],
                                                 [0, 0, 0, 0, 0, 0, 4 / 7, 1 / 7, 0, 0, 2 / 7],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 ], dtype=np.float64)

        lcm_gm = np.array([0, 0,
                           33.8 * 78, 33.8 * 22,
                           34.9 * 80, 34.9 * 20,
                           7.6 * 82, 7.6 * 18,
                           22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
        lcm_gm /= lcm_gm.sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
            lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
            lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
            lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)
            lcm_connect_prob = np.where(np.isnan(lcm_connect_prob), 0, lcm_connect_prob)

        if conn_prob.shape[0] == 1:
            conn_prob[:, :] = 1
        else:
            conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
            conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)

        conn_prob[np.isnan(conn_prob)] = 0
        N = len(conn_prob)
        cortical = np.arange(divide_point, dtype=np.int32)
        sub_cortical = np.arange(divide_point, N, dtype=np.int32)
        conn_prob_cortical = conn_prob[cortical]
        conn_prob_subcortical = conn_prob[sub_cortical]

        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape(
            [-1])  # shape[cortical_voxel, 10] reshape to [10 * cortical_voxel]
        for i in sub_cortical:
            out_gm[10 * i:10 * (i + 1)] = 0.
            out_gm[10 * i + 6] = gm[i] * 0.8
            out_gm[10 * i + 7] = gm[i] * 0.2

        out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape(
            [-1])  # shape[cortical_voxel, 10] reshape to [10 * cortical_voxel]
        for i in sub_cortical:
            out_degree_scale[10 * i:10 * (i + 1)] = 0.
            out_degree_scale[10 * i + 6] = 1.
            out_degree_scale[10 * i + 7] = 1.

        """
        deal with outer_connection of cortical voxel 
        """
        conn_prob = sparse.COO(conn_prob)
        conn_prob_cortical = sparse.COO(conn_prob_cortical)
        index_cortical = np.in1d(conn_prob.coords[0], cortical)
        coords_cortical = conn_prob.coords[:, index_cortical]
        # only e5 is allowed to output.
        corrds1 = np.empty([4, coords_cortical.shape[1] * lcm_connect_prob.shape[0]], dtype=np.int64)

        corrds1[3, :] = 6
        corrds1[(0, 2), :] = np.broadcast_to(coords_cortical[:, :, None],
                                             [2, coords_cortical.shape[1], lcm_connect_prob.shape[0]]).reshape([2, -1])
        corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                          [coords_cortical.shape[1], lcm_connect_prob.shape[0]]).reshape([1, -1])
        data1 = (conn_prob_cortical.data[:, None] * lcm_connect_prob[:, -1]).reshape([-1])

        """
        deal with outer_connection of subcortical voxel
        """
        conn_prob_subcortical = sparse.COO(conn_prob_subcortical)
        index_subcortical = np.in1d(conn_prob.coords[0], sub_cortical)
        coords_subcortical = conn_prob.coords[:, index_subcortical]
        coords3 = np.empty([4, coords_subcortical.shape[1] * 2], dtype=np.int64)
        coords3[3, :] = 6
        coords3[(0, 2), :] = np.broadcast_to(coords_subcortical[:, :, None],
                                             [2, coords_subcortical.shape[1], 2]).reshape([2, -1])
        coords3[(1), :] = np.broadcast_to(np.arange(6, 8, dtype=np.int64)[None, :],
                                          [coords_subcortical.shape[1], 2]).reshape([1, -1])
        data3 = (conn_prob_subcortical.data[:, None] * lcm_connect_prob_subcortical[6:8, -1]).reshape([-1])

        """
        deal with inner_connection of cortical voxel
        """
        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, cortical.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(cortical[:, None],
                                        [cortical.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, cortical.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                                [cortical.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

        """
        deal with inner_connection of subcortical voxel
        """
        lcm_connect_prob_inner_subcortical = sparse.COO(lcm_connect_prob_subcortical[:, :-1])
        corrds4 = np.empty([4, sub_cortical.shape[0] * lcm_connect_prob_inner_subcortical.data.shape[0]],
                           dtype=np.int64)
        corrds4[0, :] = np.broadcast_to(sub_cortical[:, None],
                                        [sub_cortical.shape[0],
                                         lcm_connect_prob_inner_subcortical.data.shape[0]]).reshape(
            [-1])
        corrds4[2, :] = corrds4[0, :]
        corrds4[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner_subcortical.coords[:, None, :],
                                             [2, sub_cortical.shape[0],
                                              lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
            [2, -1])
        data4 = np.broadcast_to(lcm_connect_prob_inner_subcortical.data[None, :],
                                [sub_cortical.shape[0], lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
            [-1])

        out_conn_prob = sparse.COO(coords=np.concatenate([corrds1, corrds2, coords3, corrds4], axis=1),
                                   data=np.concatenate([data1, data2, data3, data4], axis=0),
                                   shape=[conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                                          lcm_connect_prob.shape[1] - 1])

        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        if conn_prob.shape[0] == 1:
            out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
        return out_conn_prob, out_gm, out_degree_scale

    @staticmethod
    def _add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob, gm, divide_point=22703,
                                                                   brain_parts=None, degree_partition=None):
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

        # wenyong setting (0.3, 0.2, 0.5), cortical column is (4/7, 1/7, 2/7), xiangshitong (14/25, 5/25, 6/25)
        lcm_connect_prob_subcortical = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 14 / 25, 5 / 25, 0, 0, 6 / 25],
                                                 [0, 0, 0, 0, 0, 0, 14 / 25, 5 / 25, 0, 0, 6 / 25],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 ], dtype=np.float64)
        lcm_connect_prob_brainstem = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 6 / 25, 5 / 25, 0, 0, 14 / 25],
                                               [0, 0, 0, 0, 0, 0, 6 / 25, 5 / 25, 0, 0, 14 / 25],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               ], dtype=np.float64)
        lcm_connect_prob_cerebellum = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0.672, 0.2, 0, 0, 0.128],
                                                [0, 0, 0, 0, 0, 0, 0.672, 0.2, 0, 0, 0.128],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                ], dtype=np.float64)

        lcm_gm = np.array([0, 0,
                           33.8 * 78, 33.8 * 22,
                           34.9 * 80, 34.9 * 20,
                           7.6 * 82, 7.6 * 18,
                           22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
        lcm_gm /= lcm_gm.sum()

        weight = lcm_gm[::2]
        weight = weight / weight.sum(axis=0)
        weight = np.broadcast_to(weight, (divide_point, 5))

        with np.errstate(divide='ignore', invalid='ignore'):
            syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
            lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
            lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
            lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)
            lcm_connect_prob = np.where(np.isnan(lcm_connect_prob), 0, lcm_connect_prob)

        if isinstance(conn_prob, np.ndarray):
            with np.errstate(divide='ignore', invalid='ignore'):
                conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
                conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
            conn_prob[np.isnan(conn_prob)] = 0
            N = len(conn_prob)
            conn_prob = sparse.COO(conn_prob)
        else:
            N = conn_prob.shape[0]
        cortical = np.arange(divide_point, dtype=np.int32)
        sub_cortical = np.arange(divide_point, N, dtype=np.int32)

        brain_stem = np.where(brain_parts == "brainstem")[0]
        cerebellum = np.where(brain_parts == "cerebellum")[0]
        sub = np.where(brain_parts == "subcortex")[0]
        total_here = np.concatenate([sub, brain_stem, cerebellum])
        assert (total_here == sub_cortical).all()

        if brain_parts is not None:
            names = ['cortex', 'cerebellum', 'brainstem', 'subcortex']
            ratios = [16.34 / (16.34 + 69.03 + 0.69), 69.03 / (16.34 + 69.03 + 0.69), 0.69 / (16.34 + 69.03 + 0.69),
                      0.69 / (16.34 + 69.03 + 0.69)]
            for name, ratio in zip(names[:-1], ratios[:-1]):
                if name == 'brainstem':
                    index = np.isin(brain_parts, np.array(['brainstem', 'subcortex']))
                else:
                    index = np.isin(brain_parts, np.array([name]))
                gm[index] = gm[index] / gm[index].sum() * ratio
        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape(
            [-1])  # shape[cortical_voxel, 10] reshape to [10 * cortical_voxel]
        for i in sub_cortical:
            out_gm[10 * i:10 * (i + 1)] = 0.
            out_gm[10 * i + 6] = gm[i] * 0.8
            out_gm[10 * i + 7] = gm[i] * 0.2
        weight_part = np.array([0., 0., 0., 1., 0.])
        weight_part = np.broadcast_to(weight_part, (N - divide_point, 5))
        weight = np.concatenate([weight, weight_part], axis=0)

        out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape(
            [-1])  # shape[cortical_voxel, 10] reshape to [10 * cortical_voxel]
        for i in sub_cortical:
            out_degree_scale[10 * i:10 * (i + 1)] = 0.
            out_degree_scale[10 * i + 6] = 1.
            out_degree_scale[10 * i + 7] = 1.
        if degree_partition is not None:
            brain_steam_cerebellum = np.arange(degree_partition, N, dtype=np.int32)
            assert degree_partition > divide_point
            for i in brain_steam_cerebellum:
                out_degree_scale[10 * i:10 * (i + 1)] = 0.
                out_degree_scale[10 * i + 6] = 0.1
                out_degree_scale[10 * i + 7] = 0.1

        """
        deal with outer_connection of cortical voxel
        """

        index_cortical = np.in1d(conn_prob.coords[0], cortical)
        coords_cortical = conn_prob.coords[:, index_cortical]
        index_exclude = coords_cortical[0] != coords_cortical[1]
        coords_cortical = coords_cortical[:, index_exclude]
        index_cortical = np.where(index_cortical)[0][index_exclude]
        corrds1 = np.empty(
            [4, coords_cortical.shape[1] * lcm_connect_prob.shape[0] * int(lcm_connect_prob.shape[0] / 2)],
            dtype=np.int64)  # 5 denotes:L1E, L2/3E, L4E, L5E, L6E

        corrds1[3, :] = np.tile(np.repeat(np.arange(0, lcm_connect_prob.shape[0], 2), lcm_connect_prob.shape[0]),
                                coords_cortical.shape[1]).reshape([1, -1])
        corrds1[(0, 2), :] = np.broadcast_to(coords_cortical[:, :, None],
                                             [2, coords_cortical.shape[1],
                                              lcm_connect_prob.shape[0] * int(lcm_connect_prob.shape[0] / 2)]).reshape(
            [2, -1])
        corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                          [coords_cortical.shape[1] * int(lcm_connect_prob.shape[0] / 2),
                                           lcm_connect_prob.shape[0]]).reshape([1, -1])
        data1 = conn_prob.data[index_cortical, None] * lcm_connect_prob[:, -1]
        data1 = (data1[:, None, :] * weight[(coords_cortical[1]), :, None]).reshape([-1])

        """
        deal with outer_connection of subcortical voxel
        """
        index_subcortical = np.in1d(conn_prob.coords[0], sub_cortical)
        coords_subcortical = conn_prob.coords[:, index_subcortical]
        index_exclude = coords_subcortical[0] != coords_subcortical[
            1]  # empty [], because diagonal matrix is zero in conn_prob
        coords_subcortical = coords_subcortical[:, index_exclude]
        index_subcortical = np.where(index_subcortical)[0][index_exclude]
        coords3 = np.empty([4, coords_subcortical.shape[1] * 2 * int(lcm_connect_prob.shape[0] / 2)], dtype=np.int64)
        coords3[3, :] = np.tile(np.repeat(np.arange(0, lcm_connect_prob.shape[0], 2), 2),
                                coords_subcortical.shape[1]).reshape([1, -1])
        coords3[(0, 2), :] = np.broadcast_to(coords_subcortical[:, :, None],
                                             [2, coords_subcortical.shape[1],
                                              2 * int(lcm_connect_prob.shape[0] / 2)]).reshape([2, -1])
        coords3[(1), :] = np.broadcast_to(np.arange(6, 8, dtype=np.int64)[None, :],
                                          [coords_subcortical.shape[1] * int(lcm_connect_prob.shape[0] / 2),
                                           2]).reshape(
            [1, -1])
        index_sub = np.in1d(conn_prob.coords[0], sub)
        index_brain_stem = np.in1d(conn_prob.coords[0], brain_stem)
        index_cerebellum = np.in1d(conn_prob.coords[0], cerebellum)

        data3 = np.concatenate([conn_prob.data[index_sub, None] * lcm_connect_prob_subcortical[6:8, -1],
                                conn_prob.data[index_brain_stem, None] * lcm_connect_prob_brainstem[6:8, -1],
                                conn_prob.data[index_cerebellum, None] * lcm_connect_prob_cerebellum[6:8, -1]], axis=0)
        data3 = (data3[:, None, :] * weight[(coords_subcortical[1]), :, None]).reshape([-1])

        """
        deal with inner_connection of cortical voxel
        """
        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, cortical.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(cortical[:, None],
                                        [cortical.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, cortical.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                                [cortical.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

        """
        deal with inner_connection of subcortical voxel (including sub, brainstem, cerebellum)
        """
        corrds4 = []
        data4 = []
        for sub_index, prob in zip([sub, brain_stem, cerebellum],
                                   [lcm_connect_prob_subcortical, lcm_connect_prob_brainstem,
                                    lcm_connect_prob_cerebellum]):
            lcm_connect_prob_inner_subcortical = sparse.COO(prob[:, :-1])
            corrds_here = np.empty([4, sub_index.shape[0] * lcm_connect_prob_inner_subcortical.data.shape[0]],
                                   dtype=np.int64)
            corrds_here[0, :] = np.broadcast_to(sub_index[:, None],
                                                [sub_index.shape[0],
                                                 lcm_connect_prob_inner_subcortical.data.shape[0]]).reshape(
                [-1])
            corrds_here[2, :] = corrds_here[0, :]
            corrds_here[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner_subcortical.coords[:, None, :],
                                                     [2, sub_index.shape[0],
                                                      lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
                [2, -1])
            data_here = np.broadcast_to(lcm_connect_prob_inner_subcortical.data[None, :],
                                        [sub_index.shape[0],
                                         lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
                [-1])
            corrds4.append(corrds_here)
            data4.append(data_here)
        corrds4 = np.concatenate(corrds4, axis=1)
        data4 = np.concatenate(data4, axis=0)

        coords = np.concatenate([corrds1, corrds2, coords3, corrds4], axis=1)
        data = np.concatenate([data1, data2, data3, data4], axis=0)
        index = np.where(data)[0]
        # print(f"process zero value in conn_prob {len(data)}-->{len(index)}")
        coords = coords[:, index]
        data = data[index]
        shape = [conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                 lcm_connect_prob.shape[1] - 1]
        out_conn_prob = sparse.COO(coords=coords, data=data, shape=shape)

        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        if conn_prob.shape[0] == 1:
            out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
        return out_conn_prob, out_gm, out_degree_scale

    @staticmethod
    def _collect_info(conn_prob, block_size, degree_scale, gui_cortical, gui_subcortical, path="./",
                      total_scale=int(1e7), degree=100):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'conn_prob_22703_version.pickle'), 'wb') as f:
            pickle.dump(conn_prob, f)
        np.savez(os.path.join(path, "block_size_22703_version.npz"), block_size=block_size, degree_scale=degree_scale,
                 gui_cortical=gui_cortical, gui_subcortical=gui_subcortical)
        index = conn_prob.sum(axis=1).nonzero()[0]
        print(f"Num of valid population: {len(index)}")
        out_connection_info = np.empty((len(index), 5), dtype=np.float64)
        threshold = 1e-4
        for i in range(len(index)):
            part_conn_prob = conn_prob[index[i]]
            out_connection_info[i, 0] = index[i]
            out_connection_info[i, 1] = part_conn_prob[part_conn_prob.nonzero()].min()
            out_connection_info[i, 2] = part_conn_prob.max()
            out_connection_info[i, 3] = threshold
            out_connection_info[i, 4] = (part_conn_prob > threshold).sum()
        np.savetxt(os.path.join(path, "out_connection_info.txt"), out_connection_info,
                   header="population index | conn_prob_min | conn_prob_max | threshold | connections")
        print("Done all")

    @staticmethod
    def _verify_feasibility(conn_prob, block_size, degree, kwords):
        extern_input_k_sizes = [b["E_number"] + b["I_number"] for b in kwords]
        _extern_input_k_sizes = np.array(extern_input_k_sizes, dtype=np.int64)
        for idx in range(len(block_size)):
            if block_size[idx] == 0.:
                continue
            degree_here = degree[idx]
            extern_input_rate_sparse = conn_prob[idx, :]
            extern_input_idx = extern_input_rate_sparse.coords[0, :]
            extern_input_rate = extern_input_rate_sparse.data
            # extern_input_rate = np.add.accumulate(extern_input_rate.data)
            degree_max = (_extern_input_k_sizes[extern_input_idx] / extern_input_rate).astype(
                np.int64)  # restrict degree to avoid max_K < num
            print(
                f"src: {idx}|  {_extern_input_k_sizes[idx + 1]}, need degree {extern_input_rate_sparse[idx + 1] * degree_here} ")
            condition = (_extern_input_k_sizes[idx + 1] / extern_input_rate_sparse[idx + 1]).astype(
                np.int64)
            print(f"{condition} vs {degree_here}")
            if any(degree_max <= degree_here):
                index = np.where(degree_max < degree_here)[0]
                require_number = degree_here * extern_input_rate[index]
                allowed_number = _extern_input_k_sizes[extern_input_idx][index]
                print(
                    f"Evoke degree warning! population {idx}\nfrom source {extern_input_idx[index]}, require {require_number}, real {allowed_number}")
                degree_max_max = np.min(degree_max)
                degree_chanege = min(degree_max_max, degree_here)
                print(f"change {idx} indegree from {degree_here} to {degree_chanege}\n")

    def _test_make_small_block(self, write_path="/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/d1000_ou_iter0.1",
                               initial_parameter=None):
        self.warning_info()
        prob = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        tau_ui = (2, 40, 10, 50)
        if initial_parameter is None:
            initial_parameter = gui_info["d1000_ou"]['0.1ms']["mu_0.6_s_0.2_tau_10"]["voxel"][0]
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': initial_parameter[i],
                              'T_ref': 5,
                              "V_reset": -65,
                              "noise_rate": 0.01,
                              'tao_ui': tau_ui,
                              'size': num} for i, num in enumerate([8000, 2000])]
        conn = connect_for_multi_sparse_block(prob, population_kwards, degree=1000, dtype="uint8", prefix=None)
        merge_dti_distributation_block(conn, write_path,
                                       MPI_rank=None,
                                       number=1,
                                       avg_degree=(1000,),
                                       dtype="uint8",
                                       debug_block_dir=None,
                                       )
        print("Done")

    def _test_make_a_single_column(self, write_path="/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/isolated_column_d1000_iter0.1", initial_parameter=None,
                                   scale=int(5e5), degree=1000):
        self.warning_info()
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
        lcm_gm /= lcm_gm.sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
            lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
            lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
            lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)
            lcm_connect_prob = np.where(np.isnan(lcm_connect_prob), 0, lcm_connect_prob)

        conn_prob = lcm_connect_prob[:, :-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            conn_prob /= conn_prob.sum(axis=1, keepdims=True)
        conn_prob = np.where(np.isnan(conn_prob), 0, conn_prob)
        block_size = lcm_gm.copy()
        degree_ = (lcm_degree_scale * degree).astype(np.uint16)
        minmum_neurons_for_block = 1  # at least 1
        if initial_parameter is None:
            initial_parameter = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["column"] / 10.
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': initial_parameter[i],
                              "V_reset": -65,
                              "noise_rate": 0.01,
                              'tao_ui': (2, 40, 10, 50),
                              "size": int(max(b * scale, minmum_neurons_for_block)) if b != 0 else 0} for i, b in
                             enumerate(block_size)]
        conn = connect_for_multi_sparse_block(conn_prob, population_kwards, degree=degree_, prefix=None, dtype="uint8",
                                              multi_conn2single_conn=False)
        merge_dti_distributation_block(conn, write_path,
                                       MPI_rank=None,
                                       avg_degree=(degree,),
                                       number=1,
                                       dtype="uint8",
                                       debug_block_dir=None)

    def _test_make_two_connected_columns(self, write_path="/public/home/ssct004t/project/Digital_twin_brain/data/debug_block/two_columns_d1000_iter0.1", initial_parameter=None,
                                   scale=int(5e5), degree=1000):
        conn_prob = np.array([[0., 1.], [1., 0.]])
        gm = np.array([0.5, 0.5])
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_model(conn_prob, gm)
        degree_ = np.maximum((degree * degree_scale).astype(np.uint16),
                             1)
        if initial_parameter is None:
            initial_parameter = gui_info["d1000_ou"]['0.1ms']["mu_0.6_s_0.2_tau_10"]["voxel"][0]
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': initial_parameter,
                              "V_reset": -65,
                              "noise_rate": 0.01,
                              'tao_ui': (2, 40, 10, 50),
                              "size": int(max(b * scale, 10)) if b != 0 else 0} for i, b in
                             enumerate(block_size)]
        conn = connect_for_multi_sparse_block(conn_prob, population_kwards, degree=degree_, prefix=None, dtype="uint8",
                                              multi_conn2single_conn=False)
        merge_dti_distributation_block(conn, write_path,
                                       MPI_rank=None,
                                       avg_degree=(degree,),
                                       number=1,
                                       dtype="uint8",
                                       debug_block_dir=None)

    def _test_generate_normal_voxel_whole_brain(self, root_path="../data/jianfeng_voxel", degree=1000,
                                                minimum_neurons_for_block=(2000, 500),
                                                scale=int(5e7), dtype="single"):
        first_path, second_path = self._make_directory_tree(root_path, scale, degree, "d1000_blocks20", dtype=dtype)
        blocks = 20
        print(f"Total {scale} neurons for DTB, merge to {blocks} blocks")
        file = h5py.File(
            '/public/home/ssct004t/project/yeleijun/spiking_nn_for_brain_simulation/data/jianfeng_normal/A1_1_DTI_voxel_structure_data_jianfeng.mat',
            'r')
        block_size = file['dti_grey_matter'][0]
        dti = np.float32(file['dti_net_full'])
        nonzero_gm = (block_size > 0).nonzero()[0]
        nonzero_dti = (dti.sum(axis=1) > 0).nonzero()[0]
        nonzero_all = np.intersect1d(nonzero_gm, nonzero_dti)
        print(f"valid voxel index length {len(nonzero_all)}")
        print(f"valid voxel index {nonzero_all}")
        block_size = block_size[nonzero_all]
        block_size /= block_size.sum()
        conn_prob = dti[np.ix_(nonzero_all, nonzero_all)]
        conn_prob[np.diag_indices_from(dti)] = 0
        # conn_prob[np.diag_indices_from(dti)] = conn_prob.sum(axis=1) * 5 / 3  # intra_E:inter_E:I=5:3:2
        conn_prob /= conn_prob.sum(axis=1, keepdims=True)
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_model(conn_prob, block_size,
                                                                             canonical_voxel=True)
        gui = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["voxel"][0] / 10.
        degree_ = np.maximum((degree * degree_scale).astype(np.uint16), 1)

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui[i % 2],
                   'tao_ui': (2, 40, 10, 50),
                   'noise_rate': 0.003,
                   "size": int(max(b * scale, minimum_neurons_for_block[i % 2]))}
                  for i, b in enumerate(block_size)]

        conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree_, dtype=dtype,
                                              )
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            if rank == 0:
                self.warning_info()
                population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
                population_base = np.add.accumulate(population_base)
                population_base = np.insert(population_base, 0, 0)
                np.save(os.path.join(first_path, "supplementary_info", "population_base.npy"), population_base)
            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           avg_degree=tuple([degree] * blocks),
                                           dtype=dtype,
                                           debug_block_dir="/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks/d1000_ou/uint8",
                                           # None
                                           only_load=(i != 0))

    def _test_make_whole_brain_include_cortical_laminar_and_subcortical_voxel_model(self,
                                                                                    path="../data/jianfeng_laminar",
                                                                                    degree=100,
                                                                                    minmum_neurons_for_block=200,
                                                                                    scale=int(2e8),
                                                                                    blocks=20, dtype="uint8"
                                                                                    ):
        """
        generate the whole brian connection table at the cortical-column version, and generate index file of populations.
        In simulation, we can use the population_base.npy to sample which neurons we need to track.

        Parameters
        ----------
        path: str
            the path to save information.

        degree: int
            default is 100.

        minmum_neurons_for_block: int
            In cortical-column version, it must be zero.

        scale: int
            simualation size.

        blocks : int
            equals number of gpu cards.


        """

        first_path, second_path = self._make_directory_tree(path, scale, degree, "ou", dtype=dtype)
        whole_brain = np.load(
            '/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz')
        conn_prob = whole_brain["conn_prob"]
        block_size = whole_brain["block_size"]
        divide_point = int(whole_brain['divide_point'])
        cortical = np.arange(divide_point, dtype=np.int32)
        conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
        conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
        block_size = block_size / block_size.sum()
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_include_subcortical_in_whole_brain(conn_prob,
                                                                                                          block_size,
                                                                                                          divide_point)
        # for 100 degree
        gui_laminar = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["column"]
        gui_voxel = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["voxel"][0]
        # self._collect_info(conn_prob, block_size, degree_scale, gui_laminar, gui_voxel, path=path)

        degree_ = np.maximum((degree * degree_scale).astype(np.uint16),
                             1)
        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui_laminar[i % 10] if np.isin(i // 10, cortical) else gui_voxel,
                   "size": int(max(b * scale, minmum_neurons_for_block)) if b != 0 else 0
                   }
                  for i, b in enumerate(block_size)]

        conn = connect_for_multi_sparse_block(conn_prob, kwords, degree=degree_, dtype=dtype)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            if rank == 0:
                self.warning_info()
                population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
                population_base = np.add.accumulate(population_base)
                population_base = np.insert(population_base, 0, 0)
                np.save(os.path.join(first_path, 'supplementary_info', "population_base.npy"), population_base)
                cortical_or_not = np.concatenate(
                    [np.ones(divide_point, dtype=np.int64),
                     np.zeros(block_size.shape[0] - divide_point, dtype=np.int64)])
                np.save(os.path.join(first_path, 'supplementary_info', "cortical_or_not.npy"), cortical_or_not)
            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           avg_degree=tuple([degree] * blocks),
                                           dtype=dtype,
                                           debug_block_dir=None)

    def _test_generate_regional_brain(self, root_path="../data/jianfeng_region", degree=1000,
                                      minimum_neurons_for_block=(2000, 500),
                                      scale=int(8e6), dtype="uint8"):

        first_path, second_storge_path = self._make_directory_tree(root_path, scale, degree, "with_debug", dtype=dtype)
        blocks = 4
        file = loadmat("../data/DTI_voxel_networks_sum_1204.mat")
        conn_prob = np.float32(file["net_sum"])
        conn_prob[np.diag_indices_from(conn_prob)] = 0.
        conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
        block_size = file["roi_grey"].squeeze()
        block_size /= block_size.sum()
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_model(conn_prob, block_size,
                                                                             canonical_voxel=True)
        param = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["voxel"][0] / 10.
        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': param,
                   'tao_ui': (2, 40, 10, 50),
                   'noise_rate': 0.01,  # old setting: 0.01 Hz
                   "size": int(max(b * scale, minimum_neurons_for_block[j % 2]))}
                  for j, b in enumerate(block_size)]
        degree_ = np.maximum((degree * degree_scale).astype(np.uint16), 1)

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
            conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                                  degree=degree_, dtype=dtype)

            merge_dti_distributation_block(conn, second_storge_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           dtype=dtype,
                                           avg_degree=tuple([degree] * blocks),
                                           debug_block_dir="/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks_debug/d1000_ou/uint8",
                                           only_load=(i != 0))

    def _test_make_whole_brain_include_brainstem_laminar_model_86billion(self,
                                                                         path="/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/mutli_d1k_inluding_brainstem",
                                                                         degree=500,
                                                                         minmum_neurons_for_block=1,
                                                                         scale=int(5e9),
                                                                         blocks=None, dtype="uint8",
                                                                         large_model_scaling=True
                                                                         ):
        """
        generate the whole brian connection table at the cortical-column version, and generate index file of populations.
        In simulation, we can use the population_base.npy to sample which neurons we need to track.

        Parameters
        ----------
        path: str
            the path to save information.

        degree: int
            default is 100.

        minmum_neurons_for_block: int

        scale: int
            simualation size.

        blocks : int
            equals number of gpu cards.


        """
        first_path, second_path = self._make_directory_tree(path, scale, degree, "1thJune", dtype=dtype)
        with open("../data/newnewdata_with_brainsteam/raw_data/brain_file_1April.pickle", "rb") as f:
            brain_file = pickle.load(f)
        conn_prob = brain_file['conn_prob']
        divide_point = brain_file['divide_point']
        gm = brain_file['gm']
        N = gm.shape[0]
        brain_parts = brain_file['brain_parts']
        degree_partition = brain_file['degree_partition']
        column_structure = np.arange(divide_point, dtype=np.int64)
        d1000_scale = np.arange(degree_partition, dtype=np.int64)
        d100_scale_popu = np.arange(degree_partition*10, N*10, dtype=np.int64)
        ratio = 5

        # for 100 degree
        gui_laminar = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["column"]
        gui_voxel = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["voxel"][0]

        # in d100, not to scale brainstem degree, but in d1000 of size 86b，we scale brainstem to d100.
        if large_model_scaling:
            out_prob, out_gm, out_degree = self._add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob,
                                                                                                           gm,
                                                                                                           divide_point,
                                                                                                           brain_parts,
                                                                                                           degree_partition=degree_partition)

            def _gui_func(i):
                if np.isin(i // 10, column_structure):
                    return gui_laminar[i % 10] / ratio
                elif np.isin(i // 10, d1000_scale):
                    return gui_voxel / ratio
                else:
                    return gui_voxel

        else:
            out_prob, out_gm, out_degree = self._add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob,
                                                                                                           gm,
                                                                                                           divide_point,
                                                                                                           brain_parts,
                                                                                                           degree_partition=None)
            _gui_func = lambda i: gui_laminar[i % 10] if np.isin(i // 10, column_structure) else gui_voxel
            blocks = int(scale / 1e7)

        degree_ = np.maximum((degree * out_degree).astype(np.uint16), 1)
        # nonzero population must have one neuron at least in generation, so minmum_neurons_for_block>=1.
        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': _gui_func(i),
                   "size": int(max(b * scale, minmum_neurons_for_block)) if b != 0 else 0
                   }
                  for i, b in enumerate(out_gm)]
        degree_[d100_scale_popu] = np.maximum((1000 * out_degree[d100_scale_popu]).astype(np.uint16), 1)
        conn = connect_for_multi_sparse_block(out_prob, kwords, dtype=dtype,
                                              degree=degree_, multi_conn2single_conn=False)

        def _divide_integer(n, k):
            quotient = n // k
            remainder = n % k
            result = [quotient] * k
            for i in range(remainder):
                result[i] += 1
            return result

        population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
        population_base = np.add.accumulate(population_base)
        population_base = np.insert(population_base, 0, 0)
        d1000_cards = int(0.19788519637462235 * scale / 2e6) + 1 # 1.8e6 d1000
        d100_cards = int((1 - 0.19788519637462235) * scale / 1.5e7) + 1  # 1.5e7 d100
        d100_begin_neurons = population_base[degree_partition * 10 + 6]
        block_partition = _divide_integer(d100_begin_neurons, d1000_cards) + _divide_integer(
            population_base[-1] - d100_begin_neurons, d100_cards)
        block_partition = np.array(block_partition, dtype=np.int64)
        assert len(block_partition) == d1000_cards + d100_cards
        blocks = d1000_cards + d100_cards

        avg_degree_each_card = [1020] * d1000_cards + [102] * d100_cards
        avg_degree_each_card = tuple(avg_degree_each_card)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            if i == 0:
                print("last voxel degree", degree_[-10:])
                self.warning_info()
                print("total cards", blocks)
                print("degree_100_popu", d100_scale_popu[0], "-->", d100_scale_popu[-1])
                np.save(os.path.join(first_path, 'supplementary_info', "population_base.npy"), population_base)
                cortical_or_not = np.concatenate(
                    [np.ones(divide_point, dtype=np.int64), np.zeros(out_gm.shape[0] - divide_point, dtype=np.int64)])
                np.save(os.path.join(first_path, 'supplementary_info', "cortical_or_not.npy"), cortical_or_not)
                print("block_partition", block_partition[0], block_partition[d1000_cards - 1],
                      block_partition[d1000_cards], block_partition[-1])
                np.save(os.path.join(first_path, 'supplementary_info', "block_partition.npy"), block_partition)

            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           block_partition=block_partition,
                                           dtype=dtype,
                                           avg_degree=avg_degree_each_card,
                                           debug_block_dir=None)

    def _test_make_whole_brain_include_brainstem_laminar_model(self,
                                                               path="/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam",
                                                               degree=100,
                                                               minmum_neurons_for_block=1,
                                                               scale=int(2e8),
                                                               blocks=20, dtype="uint8", large_model_scaling=False
                                                               ):
        """
        generate the whole brian connection table at the cortical-column version, and generate index file of populations.
        In simulation, we can use the population_base.npy to sample which neurons we need to track.

        Parameters
        ----------
        path: str
            the path to save information.

        degree: int
            default is 100.

        minmum_neurons_for_block: int

        scale: int
            simualation size.

        blocks : int
            equals number of gpu cards.


        """

        first_path, second_path = self._make_directory_tree(path, scale, degree, "1April", dtype=dtype)
        with open("../data/newnewdata_with_brainsteam/raw_data/brain_file_1April.pickle", "rb") as f:
            brain_file = pickle.load(f)
        conn_prob = brain_file['conn_prob']
        divide_point = brain_file['divide_point']
        gm = brain_file['gm']
        brain_parts = brain_file['brain_parts']
        degree_partition = brain_file['degree_partition']

        # in d100, not to scale brainstem degree, but in d1000 of size 86b，we scale brainstem to d100.
        if large_model_scaling:
            out_prob, out_gm, out_degree = self._add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob,
                                                                                                           gm,
                                                                                                           divide_point,
                                                                                                           brain_parts,
                                                                                                           degree_partition=degree_partition)
        else:
            out_prob, out_gm, out_degree = self._add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob,
                                                                                                           gm,
                                                                                                           divide_point,
                                                                                                           brain_parts,
                                                                                                           degree_partition=None)
        # for 100 degree
        gui_laminar = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["column"]
        gui_voxel = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["voxel"][0]

        cortical = np.arange(divide_point, dtype=np.int64)
        degree_ = np.maximum((degree * out_degree).astype(np.uint16), 1)
        # nonzero population must have one neuron at least in generation, so minmum_neurons_for_block>=1.
        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui_laminar[i % 10] if np.isin(i // 10, cortical) else gui_voxel,
                   "size": int(max(b * scale, minmum_neurons_for_block)) if b != 0 else 0
                   }
                  for i, b in enumerate(out_gm)]

        avg_degree_each_card = [100] * blocks
        avg_degree_each_card = tuple(avg_degree_each_card)

        conn = connect_for_multi_sparse_block(out_prob, kwords, dtype=dtype,
                                              degree=degree_, multi_conn2single_conn=True)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            if rank == 0:
                self.warning_info()
                population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
                population_base = np.add.accumulate(population_base)
                population_base = np.insert(population_base, 0, 0)
                np.save(os.path.join(first_path, 'supplementary_info', "population_base.npy"), population_base)
                cortical_or_not = np.concatenate(
                    [np.ones(divide_point, dtype=np.int64), np.zeros(out_gm.shape[0] - divide_point, dtype=np.int64)])
                np.save(os.path.join(first_path, 'supplementary_info', "cortical_or_not.npy"), cortical_or_not)

            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           dtype=dtype,
                                           avg_degree=avg_degree_each_card,
                                           debug_block_dir=None)

    def _test_generate_with_map(self,
                               population_info_path="/work/home/bujingde/project/jsj_xa/graph_file_1April.pickle",
                               scale=int(1e11), blocks=14084, degree=1000, dtype="uint8",
                               write_path="/work/home/bujingde/project//zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam"):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        first_path, second_path = self._make_directory_tree(write_path, scale, degree, "map_15April", dtype=dtype)
        # step 1, load population level info: include conn_prob, size_scale, degree_scale, all are float dtype.
        with open(population_info_path, "rb") as f:
            file = pickle.load(f)
        # conn_prob_dict = np.load("/work/home/bujingde/project/jsj_xa/data_process/newdata41/conn_dict_1_April.npy",
        #                          allow_pickle=True)
        # conn_prob_dict = conn_prob_dict.item()
        conn_prob = file['conn_prob']
        size_scale = file["size"]
        degree_scale = file["degree"]
        divide_point = file["divide_point"]
        d1000_scale = file["degree_partition"]
        size_scale[:d1000_scale] = size_scale[:d1000_scale] * 17 / 20
        size_scale[d1000_scale:] = size_scale[d1000_scale:] * 83 / 80
        degree_scale[d1000_scale:] = degree_scale[d1000_scale:] * 4 / 5
        if rank == 0:
            print("sum size scale", size_scale.sum())
            print("degree", degree_scale[:10], degree_scale[-10:])
        d1000_scale = np.arange(d1000_scale)
        column_structure = np.arange(divide_point)
        N = len(size_scale)

        # step 2, apply map, derive new population level info.
        map_path = "/work/home/bujingde/project/jsj_xa/1000E/map_14084_v1.pkl"
        average_degree_path = "/work/home/bujingde/project/jsj_xa/1000E/map_average_degree_14084_v1_new.npy"
        average_degree = np.load(average_degree_path)
        avg_degree_each_card = tuple(np.ceil(average_degree + 1).tolist())
        conn_prob, size_scale, degree_scale, order, partition = apply_map(conn_prob, size_scale, degree_scale,
                                                                          map=map_path)
        if rank == 0:
            print("apply map")
            print("size scale max", size_scale.max() * scale)
            print("degree min, max", degree_scale.min() * degree, degree_scale.max() * degree)
            self.print_system_info(rank)
        assert len(partition) == blocks
        gui_laminar = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["column"]
        gui_voxel = gui_info["d100_ou"]['1ms']["mu_0.66_s_0.12_tau_10"]["voxel"][0]

        def _gui_func(i):
            if np.isin(i, column_structure):
                return gui_laminar[i % 10] / 10
            elif np.isin(i, d1000_scale):
                return gui_voxel / 10
            else:
                return gui_voxel

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   "size": int(max(b * scale, 1)) if b != 0 else 0
                   }
                  for i, b in enumerate(size_scale)]
        gui_reorder = np.stack([_gui_func(i) for i in order], axis=0)
        assert gui_reorder.shape[0] == len(kwords)
        for i, kword in enumerate(kwords):
            kword.update({"g_ui": gui_reorder[i]})
            kword.update({"sub_block_idx": order[i]})

        # step 3, generate
        degree_ = np.maximum((degree * degree_scale).astype(np.uint16), 1)
        # avg_degree_each_card = [degree + 10] * blocks  # make it larger than degree
        # avg_degree_each_card = tuple(avg_degree_each_card)

        conn = connect_for_multi_sparse_block(conn_prob, kwords, dtype=dtype,
                                              degree=degree_, multi_conn2single_conn=False)

        for i in range(rank, blocks, size):
            if i == 0:
                self.warning_info()
                population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
                population_base = np.add.accumulate(population_base)
                population_base = np.insert(population_base, 0, 0)
                np.save(os.path.join(first_path, 'supplementary_info', "population_base.npy"), population_base)
                cortical_or_not = np.concatenate(
                    [np.ones(divide_point // 10, dtype=np.int64),
                     np.zeros(N // 10 - divide_point // 10, dtype=np.int64)])
                np.save(os.path.join(first_path, 'supplementary_info', "cortical_or_not.npy"), cortical_or_not)

            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=1,
                                           block_partition=partition,
                                           dtype=dtype,
                                           avg_degree=avg_degree_each_card,
                                           debug_block_dir=None)


if __name__ == "__main__":
    unittest.main()

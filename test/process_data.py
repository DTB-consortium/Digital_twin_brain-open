import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import seaborn as sns
import sparse
from scipy.io import loadmat, savemat


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
    print(len(brain_parts), len(brain_stem), len(cerebellum), len(sub), len(brain_stem) + len(cerebellum) + len(sub),
          N - divide_point)
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
                               [lcm_connect_prob_subcortical, lcm_connect_prob_brainstem, lcm_connect_prob_cerebellum]):
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
                                    [sub_index.shape[0], lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
            [-1])
        corrds4.append(corrds_here)
        data4.append(data_here)
    corrds4 = np.concatenate(corrds4, axis=1)
    data4 = np.concatenate(data4, axis=0)

    coords = np.concatenate([corrds1, corrds2, coords3, corrds4], axis=1)
    data = np.concatenate([data1, data2, data3, data4], axis=0)
    index = np.where(data)[0]
    print(f"process zero value in conn_prob {len(data)}-->{len(index)}")
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

    # wenyong setting (0.3, 0.2, 0.5), cortical column is (4/7, 1/7, 2/7)
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

    coords = np.concatenate([corrds1, corrds2, coords3, corrds4], axis=1)
    data = np.concatenate([data1, data2, data3, data4], axis=0)
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


def process_data():
    file = h5py.File(
        "/public/home/ssct004t/project/chao_data/fmri_prep/A1_fMRIPrep_DTI_voxel_structure_data_jianfeng_Connectome_ye_info2.mat",
        "r")
    file.keys()

    aal_label = file['dti_aal_label'][:].squeeze()
    N = len(aal_label)
    aal_label = aal_label.astype(np.uint8)
    brain_part = file['dti_brainPart_label'][:].squeeze()
    gm = file["dti_grey_matter"][:].squeeze()
    brain_part[brain_part == 230] = 205
    nii_label = file["dti_label_num"][:].squeeze()
    xyz = file["dti_xyz"][:]
    rest_bold = file["dti_rest_state"][:].squeeze()
    task_bold_visual = loadmat(
        "/public/home/ssct004t/project/chao_data/fmri_prep/A2_1_DTI_voxel_structure_data_jianfeng_Connectome_v2_task.mat")[
        'evaluation_run1'].T
    task_bold_auditory = loadmat(
        "/public/home/ssct004t/project/chao_data/fmri_prep/A2_1_DTI_voxel_structure_data_jianfeng_Connectome_v2_task.mat")[
        'evaluation_run1_auditory'].T
    assert task_bold_visual.shape[0] == len(gm)
    assert task_bold_auditory.shape[0] == len(gm)

    reader = pd.read_csv(
        "/public/home/ssct004t/project/chao_data/fmri_prep/whole_connectome_dti_ye.csv",
        sep="\t", chunksize=1000, header=None)
    conn_prob = np.zeros((N, N), dtype=np.int64)
    for i, chunk in enumerate(reader):
        chunk_source = chunk.values.astype(np.int32)
        print(i, chunk_source.shape)
        conn_prob[i * 1000:(i + 1) * 1000, :] = chunk_source
    type(conn_prob)

    index = np.argsort(brain_part)
    conn_prob = conn_prob[index]
    conn_prob = conn_prob[:, index]
    aal_label = aal_label[index]
    brain_part = brain_part[index]
    gm = gm[index]
    rest_bold = rest_bold[index]
    task_bold_visual = task_bold_visual[index]
    task_bold_auditory = task_bold_auditory[index]
    nii_label = nii_label[index]
    xyz = xyz[index]

    exclude_index = np.where(gm > 0.4)[0]
    gm = gm[exclude_index]
    brain_part = brain_part[exclude_index]
    aal_label = aal_label[exclude_index]
    rest_bold = rest_bold[exclude_index]
    task_bold_visual = task_bold_visual[exclude_index]
    task_bold_auditory = task_bold_auditory[exclude_index]
    nii_label = nii_label[exclude_index]
    xyz = xyz[exclude_index]
    gm /= gm.sum()
    conn_prob = conn_prob[exclude_index]
    conn_prob = conn_prob[:, exclude_index]

    conn_prob[conn_prob <= 1] = 0  # for sparsity
    exclude_index = np.where(np.sum(conn_prob, axis=1) != 0)[0]
    print(len(exclude_index))
    conn_prob = conn_prob[exclude_index]
    conn_prob = conn_prob[:, exclude_index]
    assert np.sum(conn_prob.sum(axis=1) > 0) == conn_prob.shape[0]
    conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
    gm = gm[exclude_index]
    aal_label = aal_label[exclude_index]
    brain_part = brain_part[exclude_index]
    task_bold_visual = task_bold_visual[exclude_index]
    task_bold_auditory = task_bold_auditory[exclude_index]
    nii_label = nii_label[exclude_index]
    xyz = xyz[exclude_index]
    rest_bold = rest_bold[exclude_index]

    divide_point = (brain_part < 205).sum()
    degree_partition = (brain_part <= 205).sum()

    brain_parts = []
    for id in brain_part:
        if id == 200.:
            brain_parts.append("cortex")
        elif id == 210.:
            brain_parts.append("brainstem")
        elif id == 220.:
            brain_parts.append("cerebellum")
        elif id == 205:
            brain_parts.append("subcortex")
        else:
            raise NotImplementedError

    conn_prob = sparse.COO(conn_prob)
    print("density", conn_prob.density)
    N = conn_prob.shape[0]
    assert gm.shape[0] == nii_label.shape[0] == N
    brain_file = {"conn_prob": conn_prob, "gm": gm, "aal_label": aal_label, "brain_parts": brain_parts,
                  "rest_bold": rest_bold, "divide_point": divide_point, "degree_partition": degree_partition,
                  "task_bold_visual": task_bold_visual, "task_bold_auditory": task_bold_auditory, "nii": nii_label,
                  "xyz": xyz}
    print("total voxels", gm.shape[0], "divide_point", divide_point, "degree partition", degree_partition)
    with open(
            "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam/raw_data/brain_file_23march_degree_partition.pickle",
            "wb") as f:
        pickle.dump(brain_file, f)
    savemat(
        "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam/raw_data/brain_file_23march_degree_partition.mat",
        mdict={"conn_prob": conn_prob, "gm": gm, "aal_label": aal_label, "brain_parts": brain_parts,
               "rest_bold": rest_bold, "task_bold_visual": task_bold_visual, "task_bold_auditory": task_bold_auditory,
               "nii": nii_label, "xyz": xyz})


def process_data2(n: int, prop: float = 0.5):
    global conn_prob
    global gm
    global aal_label
    global brain_parts
    global nii
    global xyz

    shapes = conn_prob.shape
    index_cerebellum = np.where(brain_parts == "cerebellum")[0]
    assert index_cerebellum == shapes[0] - 1
    original_voxel_num = shapes[0] - len(index_cerebellum)
    assert original_voxel_num == index_cerebellum[0]
    original_index = np.arange(original_voxel_num)
    new_conn_prob = np.zeros(
        (original_voxel_num + len(index_cerebellum) * n, original_voxel_num + len(index_cerebellum) * n))
    new_conn_prob[np.ix_(original_index, original_index)] = conn_prob[np.ix_(original_index, original_index)]
    adj_mat = np.zeros((n , n), dtype=np.float32)
    for i in range(n):
        coords = [(i, (i + 1) % n), (i, (i - 1) % n)]
        for x, y in coords:
            adj_mat[x, y] = (1 - prop) / 2
    for idx in index_cerebellum:
        i = idx - original_voxel_num
        row_start = original_voxel_num + i * n
        row_end = original_voxel_num + (i + 1) * n
        new_conn_prob[np.ix_(np.arange(row_start, row_end), original_index)] = conn_prob[idx,
                                                                               :original_voxel_num] * prop
        new_conn_prob[np.ix_(np.arange(row_start, row_end), np.arange(row_start, row_end))] = adj_mat
    for idx in range(original_voxel_num + len(index_cerebellum) * n):
        pass

def whole_brain():
    path = "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam/raw_data/New_Sparse_All_Feng_Data_with_New_Conn_Prob_1April.mat"
    file = h5py.File(path, "r")

    conn_prob = file['new_dti_net_full'][:]
    conn_prob /= conn_prob.sum(axis=1, keepdims=True)

    gm = file["new_dti_grey_matter"][:].squeeze()
    gm /= gm.sum()
    aal_label = file["new_dti_aal_label"][:].squeeze()
    nii = file["dti_label_num"][:].squeeze()
    xyz = file["dti_xyz"][:]
    task_bold_visual = file['Visual_Task_run1'][:]
    rest_bold = file["Resting_state"][:]
    task_bold_auditory = file["Auditory_Task_run1"][:]
    _brain_parts = file["new_dti_brainPart_label"][:].squeeze()

    degree_partition = np.where(_brain_parts == 3)[0][0]
    divide_point = np.where(_brain_parts == 2)[0][0]
    cerebellum_partition = np.where(_brain_parts == 4)[0][0]

    brain_parts = []
    for id in _brain_parts:
        if id == 1:
            brain_parts.append("cortex")
        elif id == 3:
            brain_parts.append("brainstem")
        elif id == 4:
            brain_parts.append("cerebellum")
        elif id == 2:
            brain_parts.append("subcortex")
        else:
            raise NotImplementedError
    brain_parts = np.array(brain_parts)
    assert np.where(brain_parts == "cerebellum")[0][0] == cerebellum_partition

    conn_prob = sparse.COO(conn_prob)
    print("density", conn_prob.density)

    brain_file = {"conn_prob": conn_prob, "gm": gm, "aal_label": aal_label, "brain_parts": brain_parts, "nii": nii,
                  "xyz": xyz, "rest_bold": rest_bold, "task_bold_visual": task_bold_visual,
                  "task_bold_auditory": task_bold_auditory,
                  "degree_partition": degree_partition, "divide_point": divide_point,
                  "cerebellum_partition": cerebellum_partition}

    with open(
            "/public/home/ssct004t/project/Digital_twin_brain/data/raw_data/brain_file_1April.pickle",
            "wb") as f:
        pickle.dump(brain_file, f)

    print("total voxels", gm.shape[0], "divide_point", divide_point, "degree partition", degree_partition)
    out_prob, out_gm, out_degree = _add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob, gm,
                                                                                              divide_point,
                                                                                              brain_parts,
                                                                                              degree_partition=None)
    # size test
    names = ['cortex', 'cerebellum', 'brainstem', 'subcortex']
    ratios = [16.34 / (16.34 + 69.03 + 0.69), 69.03 / (16.34 + 69.03 + 0.69), 0.69 / (16.34 + 69.03 + 0.69),
              0.69 / (16.34 + 69.03 + 0.69)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    print("gm.sum()", gm.sum())
    for name, ratio in zip(names[:-1], ratios[:-1]):
        if name == 'brainstem':
            index = np.isin(brain_parts, np.array(['brainstem', 'subcortex']))
        else:
            index = np.isin(brain_parts, np.array([name]))
        gm[index] = gm[index] / gm[index].sum() * ratio
    for i, name in enumerate(names):
        index = np.isin(brain_parts, np.array([name]))
        sizes = (gm[index] * int(2e9)).astype(np.int64)
        max_size, min_size, mean_size = sizes.max(), sizes.min(), sizes.mean()
        print(name, min_size, max_size, mean_size)
        sns.histplot(sizes, bins=20, ax=axes[i])
        axes[i].set_title(name)
        axes[i].text(0.1, 0.8, f"min:{min_size}, max:{max_size}, mean:{mean_size}", transform=axes[i].transAxes)
    fig.tight_layout()
    fig.savefig("./voxel_size.png", dpi=100)

    _extern_input_k_sizes = (out_gm * 2e9).astype(np.int64)
    print(
        f"popu size {min(_extern_input_k_sizes[np.flatnonzero(_extern_input_k_sizes)])} ===> {max(_extern_input_k_sizes[np.flatnonzero(_extern_input_k_sizes)])}")
    out_degree_real = (out_degree * 1e3).astype(np.int64)
    print(
        f"popu degree {min(out_degree_real[np.flatnonzero(out_degree_real)])} ===> {max(out_degree_real[np.flatnonzero(out_degree_real)])}")

    brain_parts = np.repeat(brain_parts, 10)
    degree_partition = degree_partition * 10
    divide_point = divide_point * 10
    cerebellum_partition = cerebellum_partition * 10
    graph_file = {"conn_prob": out_prob, "size": out_gm, "degree": out_degree,
                  "brain_parts": brain_parts, "divide_point": divide_point, "degree_partition": degree_partition,
                  "cerebellum_partition": cerebellum_partition}
    with open(
            "/public/home/ssct004t/project/Digital_twin_brain/data/raw_data/graph_file_1April_alld100.pickle",
            "wb") as f:
        pickle.dump(graph_file, f)
    print("done!")

def only_big_brain():
    # process_data()
    only_big_brain = True
    path = "../data/New_Sparse_Brain_Feng_Data_with_New_Conn_Prob_3April.mat"
    file = h5py.File(path, "r")

    conn_prob = file['dti_net_full'][:]
    conn_prob /= conn_prob.sum(axis=1, keepdims=True)

    gm = file["dti_grey_matter"][:].squeeze()
    gm /= gm.sum()
    aal_label = file["dti_aal_label"][:].squeeze()
    nii = file["dti_label_num"][:].squeeze()
    xyz = file["dti_xyz"][:]
    task_bold_visual = file['Visual_Task_run1'][:]
    rest_bold = file["Resting_state"][:]
    task_bold_auditory = file["Auditory_Task_run1"][:]
    _brain_parts = file["dti_brainPart_label"][:].squeeze()
    divide_point = np.where(_brain_parts == 2)[0][0]
    if not only_big_brain:
        degree_partition = np.where(_brain_parts == 3)[0][0]
        cerebellum_partition = np.where(_brain_parts == 4)[0][0]

    brain_parts = []
    for id in _brain_parts:
        if id == 1:
            brain_parts.append("cortex")
        elif id == 3:
            brain_parts.append("brainstem")
        elif id == 4:
            brain_parts.append("cerebellum")
        elif id == 2:
            brain_parts.append("subcortex")
        else:
            raise NotImplementedError
    brain_parts = np.array(brain_parts)
    if not only_big_brain:
        assert np.where(brain_parts == "cerebellum")[0][0] == cerebellum_partition
    else:
        assert np.where(brain_parts == "subcortex")[0][0] == divide_point

    # conn_prob = sparse.COO(conn_prob)
    # print("density", conn_prob.density)
    if not only_big_brain:
        brain_file = {"conn_prob": conn_prob, "gm": gm, "aal_label": aal_label, "brain_parts": brain_parts, "nii": nii,
                      "xyz": xyz, "rest_bold": rest_bold, "task_bold_visual": task_bold_visual,
                      "task_bold_auditory": task_bold_auditory,
                      "degree_partition": degree_partition, "divide_point": divide_point,
                      "cerebellum_partition": cerebellum_partition}
    else:
        brain_file = {"conn_prob": conn_prob, "gm": gm, "aal_label": aal_label, "brain_parts": brain_parts, "nii": nii,
                      "xyz": xyz, "rest_bold": rest_bold, "task_bold_visual": task_bold_visual,
                      "task_bold_auditory": task_bold_auditory, "divide_point": divide_point}

    os.makedirs("../data/big_brain_file/raw_data", exist_ok=True)
    with open(
            "../data/big_brain_file/raw_data/only_big_brain_file_3April.pickle",
            "wb") as f:
        pickle.dump(brain_file, f)

    print("total voxels", gm.shape[0], "divide_point", divide_point)
    if not only_big_brain:
        out_prob, out_gm, out_degree = _add_laminar_cortex_include_subcortical_in_whole_brain_new(conn_prob, gm,
                                                                                                  divide_point,
                                                                                                  brain_parts,
                                                                                                  degree_partition=degree_partition)
    else:
        out_prob, out_gm, out_degree = _add_laminar_cortex_include_subcortical_in_whole_brain(conn_prob, gm,
                                                                                              divide_point=divide_point)

    # size test
    # names = ['cortex', 'cerebellum', 'brainstem', 'subcortex']
    # ratios = [16.34 / (16.34 + 69.03 + 0.69), 69.03 / (16.34 + 69.03 + 0.69), 0.69 / (16.34 + 69.03 + 0.69),
    #           0.69 / (16.34 + 69.03 + 0.69)]
    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # axes = axes.flatten()
    # print("gm.sum()", gm.sum())
    # for name, ratio in zip(names[:-1], ratios[:-1]):
    #     if name == 'brainstem':
    #         index = np.isin(brain_parts, np.array(['brainstem', 'subcortex']))
    #     else:
    #         index = np.isin(brain_parts, np.array([name]))
    #     gm[index] = gm[index] / gm[index].sum() * ratio
    # for i, name in enumerate(names):
    #     index = np.isin(brain_parts, np.array([name]))
    #     sizes = (gm[index] * int(2e9)).astype(np.int64)
    #     max_size, min_size, mean_size = sizes.max(), sizes.min(), sizes.mean()
    #     print(name, min_size, max_size, mean_size)
    #     sns.histplot(sizes, bins=20, ax=axes[i])
    #     axes[i].set_title(name)
    #     axes[i].text(0.1, 0.8, f"min:{min_size}, max:{max_size}, mean:{mean_size}", transform=axes[i].transAxes)
    # fig.tight_layout()
    # fig.savefig("./voxel_size.png", dpi=100)

    _extern_input_k_sizes = (out_gm * 2e9).astype(np.int64)
    print(
        f"popu size {min(_extern_input_k_sizes[np.flatnonzero(_extern_input_k_sizes)])} ===> {max(_extern_input_k_sizes[np.flatnonzero(_extern_input_k_sizes)])}")
    out_degree_real = (out_degree * 1e3).astype(np.int64)
    print(
        f"popu degree {min(out_degree_real[np.flatnonzero(out_degree_real)])} ===> {max(out_degree_real[np.flatnonzero(out_degree_real)])}")

    brain_parts = np.repeat(brain_parts, 10)
    divide_point = divide_point * 10
    if not only_big_brain:
        degree_partition = degree_partition * 10
        cerebellum_partition = cerebellum_partition * 10
        graph_file = {"conn_prob": out_prob, "size": out_gm, "degree": out_degree,
                      "brain_parts": brain_parts, "divide_point": divide_point, "degree_partition": degree_partition,
                      "cerebellum_partition": cerebellum_partition}
    else:
        graph_file = {"conn_prob": out_prob, "size": out_gm, "degree": out_degree,
                      "brain_parts": brain_parts, "divide_point": divide_point}
    with open("../data/big_brain_file/raw_data/only_big_brain_graph_file_3April.pickle", "wb") as f:
        pickle.dump(graph_file, f)

    # # scale test
    # total_scale = [2e8]
    # degree = 100
    # for scale in total_scale:
    #     print("\nIn scale %e" % scale)
    #     _extern_input_k_sizes = (out_gm * scale).astype(np.int64)
    #     print("sum(_extern)", sum(_extern_input_k_sizes))
    #     count_invalid = 0
    #     for i in range(out_prob.shape[0]):
    #         extern_input_rate = out_prob[i, :]
    #         extern_input_idx = extern_input_rate.coords[0, :]
    #         if np.abs(np.sum(extern_input_rate.data) - 1) < 1e-5 and _extern_input_k_sizes[i] > 0:
    #             invalid_pop = (degree * extern_input_rate.data > _extern_input_k_sizes[extern_input_idx])
    #             if invalid_pop.sum() > 0:
    #                 count_invalid += invalid_pop.sum()
    #                 print("invalid_pop.sum", invalid_pop.sum())
    #                 index = np.where(invalid_pop)[0][0]
    #                 print(degree * extern_input_rate.data[index], ">", _extern_input_k_sizes[extern_input_idx][index])
    #                 print("scale %e failed in processing destination population %d and source population %d, " % (
    #                     scale, i, extern_input_rate.coords[0, index]))
    #         else:
    #             continue
    #     print("%e: invalid popus: %d" % (scale, count_invalid))

    # verify zero value in conn_prob
    # count = 0
    # for i in range(out_prob.shape[0]):
    #     extern_input_rate = out_prob[i, :]
    #     coords = extern_input_rate.coords[0, :]
    #     if len(coords) > 0 and count < 10:
    #         rate = extern_input_rate.data
    #         if (out_gm[coords] > 0).all():
    #             continue
    #         else:
    #             print(f"dest {i}|zero neurons but have out connection to popus")
    #             ind = np.where(out_gm[coords] == 0)[0]
    #             print("coords", coords[ind])
    #             print("rate", rate[ind])
    #             print()
    #
    #             count += 1
    #     else:
    #         break

    print("Done")


if __name__ == '__main__':
    whole_brain()
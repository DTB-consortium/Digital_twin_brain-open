# -*- coding: utf-8 -*- 
# @Time : 2022/8/27 20:03 
# @Author : lepold
# @File : verify_block.py


import os
import pickle
import re

import numpy as np
import sparse
# from mpi4py import MPI


def reverse(block_dir, store_path):
    block_name = re.compile('block_[0-9]*.npz')
    blocks = [name for name in os.listdir(block_dir) if block_name.fullmatch(name)]
    total = len(blocks)
    os.makedirs(store_path, exist_ok=True)

    cache_property = dict()

    def _get_subblk_id(block_id, neuron_id):
        if block_id not in cache_property:
            cache_property[block_id] = np.load(os.path.join(block_dir, "block_{}.npz".format(block_id)))['property'][:,
                                       3].astype(np.uint32)
        return cache_property[block_id][neuron_id]

    for idx in range(total):
        print(idx)
        conn_count = dict()
        cache_property = dict()
        block_path = os.path.join(block_dir, "block_{}.npz".format(idx))
        file = np.load(block_path)
        dst_neuron_idx = file['output_neuron_idx']
        block_list = file['property'][:, 1]
        subblk_id = file['property'][:, 3].astype(np.uint32)
        size_count = {id: count for id, count in zip(*np.unique(subblk_id[block_list == 0], return_counts=True))}
        src_block_idx = file['input_block_idx'] + idx
        src_neuron_idx = file['input_neuron_idx']
        for dn, sb, sn in zip(dst_neuron_idx, src_block_idx, src_neuron_idx):
            if block_list[dn] == 0:
                dst_subblk_id = _get_subblk_id(idx, dn)
                src_subblk_id = _get_subblk_id(sb, sn)
                if (dst_subblk_id, src_subblk_id) not in conn_count:
                    conn_count[(dst_subblk_id, src_subblk_id)] = 0
                conn_count[(dst_subblk_id, src_subblk_id)] += 1
        with open(os.path.join(store_path, "out_{}.pkl".format(idx)), "wb") as f:
            pickle.dump((size_count, conn_count), f)


def verify(store_path: str, num_population: int):
    total_size_dict = dict()
    total_conn_dict = dict()
    out_name = re.compile('out_[0-9]*.pkl')
    outs = [name for name in os.listdir(store_path) if out_name.fullmatch(name)]
    number_out_file = len(outs)
    for i in range(number_out_file):
        with open(os.path.join(store_path, "out_{}.pkl".format(i)), "rb") as f:
            size_dict, conn_dict = pickle.load(f)
        for key, val in size_dict.items():
            if key not in total_size_dict:
                total_size_dict[key] = 0
            total_size_dict[key] += val
        for key, val in conn_dict.items():
            if key not in total_conn_dict:
                total_conn_dict[key] = 0
            total_conn_dict[key] += val
    conn_prob = sparse.COO(coords=np.array(list(total_conn_dict.keys()), dtype=np.uint32).transpose(),
                           data=np.array(list(total_conn_dict.values()), dtype=np.float32),
                           shape=(num_population, num_population))
    conn_prob = conn_prob.todense()
    conn_prob /= conn_prob.sum(axis=1, keepdims=True)
    population_size = np.zeros(shape=num_population)
    for idx, val in total_size_dict.items():
        population_size[idx] = val
    population_size = population_size / population_size.sum()
    return conn_prob, population_size


def reverse_mpi(block_dir, store_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    total = 40
    # path = "/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_comp/dti_2k_10G/single"
    path = "/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/test/laminar_structure_whole_brain_include_subcortical/200m_structure_d100/single"
    store_path = "./recover_dti_2/"

    os.makedirs(store_path, exist_ok=True)

    i = rank

    cache_property = dict()

    def get_subblk_id(block_id, neuron_id):
        if block_id not in cache_property:
            cache_property[block_id] = np.load(os.path.join(path, "block_{}.npz".format(block_id)))['property'][:,
                                       3].astype(np.uint32)
        return cache_property[block_id][neuron_id]

    for idx in range(rank, total, size):
        print(idx)
        conn_count = dict()
        cache_property = dict()
        block_path = os.path.join(path, "block_{}.npz".format(idx))
        file = np.load(block_path)
        dst_neuron_idx = file['output_neuron_idx']
        block_list = file['property'][:, 1]
        subblk_id = file['property'][:, 3].astype(np.uint32)
        size_count = {id: count for id, count in zip(*np.unique(subblk_id[block_list == 0], return_counts=True))}
        src_block_idx = file['input_block_idx'] + idx
        src_neuron_idx = file['input_neuron_idx']
        for dn, sb, sn in zip(dst_neuron_idx, src_block_idx, src_neuron_idx):
            if block_list[dn] == 0:
                dst_subblk_id = get_subblk_id(idx, dn)
                src_subblk_id = get_subblk_id(sb, sn)
                if (dst_subblk_id, src_subblk_id) not in conn_count:
                    conn_count[(dst_subblk_id, src_subblk_id)] = 0
                conn_count[(dst_subblk_id, src_subblk_id)] += 1
        with open(os.path.join(store_path, "out_{}.pkl".format(idx)), "wb") as f:
            pickle.dump((size_count, conn_count), f)


if __name__ == '__main__':
    reverse(block_dir=r"/Users/lepold/Projects/Digital_twin_brain/data/small_blocks_d1000/uint8",
            store_path="../data/reverse_res")
    conn_prob, population_size = verify(store_path="../data/reverse_res", num_population=2)
    print("conn_prob\n", conn_prob)
    print("population_size\n", population_size)

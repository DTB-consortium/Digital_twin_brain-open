# -*- coding: utf-8 -*- 
# @Time : 2023/2/16 18:37 
# @Author : lepold
# @File : vertify_debug_weight.py

import numpy as np
from mpi4py import MPI
from generation.read_block import connect_for_block
import os
import re

mpi = 4
big_block_path = "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/jianfeng_region/dti_distribution_0m_d10_with_debug/module/uint8"
debug_block_path = "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/one_block/uint8"
debug_idx_path = "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/jianfeng_region/dti_distribution_0m_d10_with_debug/module/debug_selection_idx.npy"
debug_selection_idx_original_path = "/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/jianfeng_region/dti_distribution_0m_d10_with_debug/module/debug_selection_idx_original.npy"
#
# big_block_path = r"E:\PycarmProjects2\Digital_twin_brain\data\small_blocks\uint8"
# debug_idx_path = r"E:\PycarmProjects2\Digital_twin_brain\data\small_blocks\debug_selection_idx.npy"
# debug_block_path = r"E:\PycarmProjects2\Digital_twin_brain\data\debug_blocks\uint8"
# debug_selection_idx_original_path = r"E:\PycarmProjects2\Digital_twin_brain\data\small_blocks\debug_selection_idx_original.npy"

prop_debug, wu_ij = connect_for_block(debug_block_path)
prop_debug = prop_debug.numpy()
degree = 10 # 1000
cards = 4

debug_idx = np.load(debug_idx_path)
debug_selection_idx_original = np.load(debug_selection_idx_original_path)
# prop_debug = np.load(os.path.join(debug_block_path, "block_0.npz"))['property']

# debug_block = np.load(debug_block_path)
# debug_block_w = debug_block['weight']
# debug_block_channels = debug_block['input_channel_offset']
# debug_block_output_neuron = debug_block['output_neuron_idx']
# debug_block_input_neuron = debug_block['input_neuron_idx']
# prop_debug = debug_block['property']


block_name = re.compile('block_[0-9]*.npz')
block_length = len([name for name in os.listdir(big_block_path) if block_name.fullmatch(name)])
bases = [0]
props = []
for i in range(block_length):
    pkl_path = os.path.join(big_block_path, "block_{}.npz".format(i))
    file = np.load(pkl_path)
    props.append(file['property'])
    assert file['property'].dtype == np.float32
    bases.append(bases[-1] + props[i].shape[0])

bases = np.array(bases, dtype=np.int64)
print(bases)
assert (bases[debug_idx[:, 0]] + debug_idx[:, 1] == debug_selection_idx_original).all()

if mpi is None:
    rank = 0
    size = 1
else:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

for idx in range(rank, cards, size):
    big_block = np.load(os.path.join(big_block_path, "block_%d.npz"%idx))
    big_block_w = big_block['weight']
    assert big_block_w.dtype == np.uint8
    big_block_channels = big_block['input_channel_offset']
    assert big_block_channels.dtype == np.uint8
    big_block_output_neuron = big_block['output_neuron_idx']
    assert big_block_output_neuron.dtype == np.uint32
    big_block_input_neuron = big_block['input_neuron_idx']
    assert big_block_input_neuron.dtype == np.uint32
    big_block_input_blocks = big_block['input_block_idx']
    assert big_block_input_blocks.dtype == np.int16

    index = np.where(debug_idx[:, 0] == idx)[0]
    if len(index) == 0:
        continue
    real_index = debug_idx[index, 1] + bases[idx]
    assert (real_index == debug_selection_idx_original[index]).all()

    related_index = debug_idx[index, 1]
    prop = props[idx][related_index]
    prop_diff = prop - prop_debug[index]
    ind = np.where(prop_diff>1e-4)
    if len(ind[0])>0:
        print("prop diff, part of that are")
        print(ind[0][:20], ind[1][:20])

    assert big_block_output_neuron[related_index[0] * degree] == big_block_output_neuron[(related_index[0]+1)*degree-1] == related_index[0]

    # conn_index = np.where(np.logical_and(np.isin(big_block_input_neuron, related_index), big_block_input_blocks==0))
    conn_index = np.where(np.isin(big_block_output_neuron, related_index))
    input_neuron = big_block_input_neuron[conn_index] + bases[idx + big_block_input_blocks[conn_index]]
    output_neuron = big_block_output_neuron[conn_index] + bases[idx]
    assert np.sum(np.isin(input_neuron, debug_selection_idx_original)) == len(input_neuron), f"{np.sum(np.isin(input_neuron, debug_selection_idx_original))} vs {len(input_neuron)}"
    assert np.sum(np.isin(output_neuron, debug_selection_idx_original)) == len(output_neuron)
    assert len(output_neuron) == len(np.unique(output_neuron)) * degree

    channel = big_block_channels[conn_index]
    w = big_block_w[conn_index]
    sort_selection = np.lexsort((input_neuron, output_neuron))
    w = w[sort_selection]
    channel = channel[sort_selection]
    output_neuron = output_neuron[sort_selection]
    input_neuron = input_neuron[sort_selection]

    # using connect for block
    w_2_coords = wu_ij._indices().data.numpy()
    w_2_value = wu_ij._values().data.numpy()
    output_neuron_2 = w_2_coords[1]
    input_neuron_2 = w_2_coords[2]
    u = w_2_coords[0]
    indices = np.isin(output_neuron_2, index)
    output_neuron_2 = output_neuron_2[indices]
    input_neuron_2 = input_neuron_2[indices]
    u = u[indices]
    w_2_value = w_2_value[indices]
    index = np.lexsort((u, input_neuron_2, output_neuron_2))
    output_neuron_2 = output_neuron_2[index]
    input_neuron_2 = input_neuron_2[index]
    u = u[index]
    output_neuron_2 = output_neuron_2[::2]
    input_neuron_2 = input_neuron_2[::2]
    channel_2 = np.array([0 if u[i]<2 else 2 for i in range(len(u))])
    channel_2 = channel_2[::2]
    w_2_value = w_2_value[index]
    w_2 = w_2_value.reshape((-1, 2))

    output_neuron_2 = debug_selection_idx_original[output_neuron_2]
    input_neuron_2 = debug_selection_idx_original[input_neuron_2]
    sort_selection = np.lexsort((input_neuron_2, output_neuron_2))
    w_2 = w_2[sort_selection]
    channel_2 = channel_2[sort_selection]
    output_neuron_2= output_neuron_2[sort_selection]
    input_neuron_2 = input_neuron_2[sort_selection]


    # using np.load
    # conn_index2 = np.where(np.isin(debug_block_output_neuron, index))
    # w_2 = debug_block_w[conn_index2]
    # input_neuron_2 = debug_block_input_neuron[conn_index2]
    # output_neuron_2 = debug_block_output_neuron[conn_index2]
    # channel_2 = debug_block_channels[conn_index2]
    # assert len(conn_index2) == len(conn_index)
    # input_neuron_2 = debug_selection_idx_original[input_neuron_2]
    # output_neuron_2 = debug_selection_idx_original[output_neuron_2]

    # assert (np.unique(input_neuron) == np.unique(input_neuron_2)).all()
    # assert (np.unique(output_neuron) == np.unique(output_neuron_2)).all()
    #
    # sort_selection2 = np.lexsort((input_neuron_2, output_neuron_2))
    # w_2 = w_2[sort_selection2]
    # channel_2 = channel_2[sort_selection2]
    # input_neuron_2 = input_neuron_2[sort_selection2]
    # output_neuron_2 = output_neuron_2[sort_selection2]

    assert (input_neuron_2 - input_neuron).sum() == 0
    assert (output_neuron_2 - output_neuron).sum() == 0
    if (w_2 == w).all() and (channel==channel_2).all():
        print(f"card {idx} pass")
    else:
        print(f"card {idx} fail")











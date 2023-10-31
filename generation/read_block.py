# -*- coding: utf-8 -*- 
# @Time : 2022/8/10 13:48 
# @Author : lepold
# @File : read_block.py

import re
import os
import numpy as np
import torch


def connect_for_block(block_dir, dense=True, bases=None, return_src=False):
    """
    Read npz file , and convert it into node attribute and adjacency matrix id dense is True,
    else convert to generator.

    Parameters
    ----------
    block_dir: str
        the directory of these block npz files.
    dense: bool
        whether convert to dense case.
    bases: optional, None
        the bases of neuronal number.
    return_src: bool
        `old version, may be eliminated`
    Returns
    -------

    """
    block_name = re.compile('block_[0-9]*.npz')
    block_length = len([name for name in os.listdir(block_dir) if block_name.fullmatch(name)])
    if bases is None:
        bases = [0]
        for i in range(block_length):
            pkl_path = os.path.join(block_dir, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            bases.append(bases[-1] + file["property"].shape[0])

        bases = np.array(bases, dtype=np.int64)

    if dense:
        weights = []
        indices = []
        sizes = []
        properties = []
        src = []
        for i in range(block_length):
            pkl_path = os.path.join(block_dir, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            property = file["property"]
            properties.append(property)
            if 'src_timestamp' in file or 'src_neuron_idx' in file:
                assert 'src_timestamp' in file and 'src_neuron_idx' in file
                weight = np.array([], dtype=np.float32)
                idx = np.array([[], [], []], dtype=np.uint32)
                size = [4, file["property"].shape[0], 0]
                _src_timestamp = file["src_timestamp"]
                _src_neuron_idx = file['src_neuron_idx']
                _src = np.zeros([property.shape[0], np.max(_src_timestamp) + 1], dtype=np.uint8)
                _src[_src_neuron_idx, _src_timestamp] = 1
                src.append(_src)
            else:
                output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, weight = \
                    tuple(file[name] for name in
                          ["output_neuron_idx", "input_block_idx", "input_neuron_idx", "input_channel_offset",
                           "weight"])
                idx = np.stack([input_channel_offset.astype(np.uint32),
                                output_neuron_idx.astype(np.uint32),
                                (bases[(input_block_idx + i)] + input_neuron_idx).astype(np.uint32)])
                weight = weight.reshape([-1])
                idx = np.stack([idx, idx], axis=-1)
                idx[0, :, 1] = idx[0, :, 0] + 1
                idx = idx.reshape([3, -1])
                if idx.shape[1] > 0:
                    size = [4, np.max(idx[1]) + 1, np.max(idx[2]) + 1]
                else:
                    size = None

            if size is not None:
                indices.append(idx)
                weights.append(weight)
                sizes.append(size)
        # print(sizes)
        size = tuple(np.max(np.array(sizes), axis=0)[1:].tolist())
        property = torch.cat([torch.from_numpy(property) for property in properties])
        weight = torch.cat([torch.sparse.FloatTensor(
            torch.from_numpy(indices[i].astype(np.int64)),
            torch.from_numpy(weights[i]),
            torch.Size([4, bases[i + 1] - bases[i], bases[-1]])) for i in range(block_length)], dim=1)
        assert property.shape[0] == weight.shape[1]
        assert weight.shape[2] == weight.shape[1]
        print(property.shape, weight.shape)
        if return_src:
            return property, weight, torch.from_numpy(np.concatenate(src).astype(np.bool)) if len(src) > 0 else None
        else:
            return property, weight
    else:
        def conn(i, s, e):
            pkl_path = os.path.join(block_dir, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            if 'src' not in file:
                output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, weight = \
                    tuple(file[name] for name in
                          ["output_neuron_idx", "input_block_idx", "input_neuron_idx", "input_channel_offset", "weight"])
                selection_idx = np.logical_and(output_neuron_idx >= s, output_neuron_idx < e).nonzero()[0]

                output_neuron_idx = np.take(output_neuron_idx, selection_idx, axis=0)
                input_block_idx = np.take(input_block_idx, selection_idx, axis=0)
                input_channel_offset = np.take(input_channel_offset, selection_idx, axis=0)
                input_neuron_idx = np.take(input_neuron_idx, selection_idx, axis=0)
                weight = np.take(weight, selection_idx, axis=0)
                output_neuron_idx = output_neuron_idx.astype(np.uint32)
                input_neuron_idx = input_neuron_idx.astype(np.uint32)

                output_neuron_idx += bases[i].astype(output_neuron_idx.dtype)
                input_neuron_idx += bases[i + input_block_idx].astype(input_neuron_idx.dtype)

                yield output_neuron_idx, input_neuron_idx, input_channel_offset, weight

        def prop(i, s, e):
            pkl_path = os.path.join(block_dir, "block_{}.npz".format(i))
            file = np.load(pkl_path)
            property = file["property"]
            assert 0 <= s <= e <= property.shape[0]
            return property[s:e].copy()

        return prop, conn, bases

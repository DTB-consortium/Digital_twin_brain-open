"""
The WBNN model presents the computational basis of the Digital twin brain and
is composed of two components: the basic computing unites and the network structure.

Generating logic can be basically broken down into the following two steps:

1) provide constructing information of population (minimum specific unit)

    a. Weighted directed graph of connections between groups

    b. The average degree of neurons in each population

    c. The size scale of each population

    d. Parameters of neuron model

2) Construct connections for each neuron based on the above information
"""

import os
import time
from multiprocessing.pool import ThreadPool as Thpool

import numpy as np
import torch
import pickle
import sparse
from generate_map import generate_map_split_only_size

def get_mpi_info():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        rank = 0
        size = 1
    return rank, size, comm

def record_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.5f} seconds to execute.")
        return result

    return wrapper


# @record_runtime
def get_k_idx(max_k, num, except_idx):  # , destination, orign
    """
    Fast implementation of random sampling with Numba.

    Parameters
    ----------
    max_k: int
        allowed range to sample.
    num: int
        the number of samples required.
    except_idx: int
        whether consider self idx.

    Returns
    -------
    sample idx: ndarray

    """
    if except_idx < 0:
        assert num <= max_k  # return np.random.choice(max_k, num, replace=True)
        # if num > max_k:
        #     print("destination: %s ; source: %s, num: %s, max_K: %s"%(destination, orign, num, max_k))
        #     raise ValueError
        if num == max_k:
            return np.arange(0, max_k)
    elif except_idx is not None:
        assert num < max_k
        if num == max_k - 1:
            return np.concatenate((np.arange(0, except_idx), np.arange(except_idx + 1, max_k)))

    j = 2
    while True:
        k_idx = np.unique(np.random.randint(0, max_k, num * j))
        k_idx = k_idx[np.random.permutation(k_idx.shape[0])]
        if except_idx is not None:
            k_idx = k_idx[k_idx != except_idx]
        k_idx = k_idx[:num]
        if k_idx.shape[0] == num:
            break
        j += 1
    return k_idx


@record_runtime
def random_sample(max_k, num, except_idx):
    if except_idx < 0:
        assert num <= max_k
        if num == max_k:
            return np.arange(0, max_k)
    else:
        assert num < max_k
        if num == max_k - 1:
            return np.delete(np.arange(max_k), except_idx)

    sample = np.arange(num, dtype=np.int64)
    sample[sample == except_idx] = max_k - 1
    if except_idx < 0:
        step_range = np.arange(num, max_k - 1)
    elif except_idx >= num:
        step_range = np.delete(np.arange(num, max_k - 1), except_idx - num)
    for i in step_range:
        j = np.random.randint(0, i + 1)
        if j < num:
            sample[j] = i
    return sample


def examine(input_block_idx, extern_input_size):
    unique_input_block_idx = np.unique(input_block_idx)
    index_all = np.ones_like(input_block_idx, dtype=np.bool_)
    for i in unique_input_block_idx:
        index = np.where(input_block_idx == i)[0]
        if np.shape(index)[0] >= extern_input_size[i]:
            index = index[extern_input_size[i] - 1:]  # assert num<max_k in except idx is not -1.
            index_all[index] = np.False_
    out = input_block_idx[index_all]
    return out


def connect_for_multi_sparse_block(population_connect_prob, population_node_init_kwards=None, degree=int(1e3),
                                   multi_conn2single_conn=False, dtype="single"):
    """
    Main api to generate connection table and save in npz file, if given information contain connection portability of populations,
    the average degree of neurons in each population, and The size scale of each population.
    
    Parameters
    ----------
    population_connect_prob: Tensor or sparse.COO
        the connection probability of populations.
    population_node_init_kwards: dict or list
        the information includes size information of each population.
        if dict, it's information of one population and will broadcast to each population.
        if list, it mush be [dict, dict, ..] and contains each population information.
    degree: ndarray or int
        specified degree of each population.

    Returns
    -------

    """
    if isinstance(population_connect_prob, torch.Tensor):
        population_connect_prob = population_connect_prob.numpy()
    assert len(population_connect_prob.shape) == 2 and \
           population_connect_prob.shape[0] == population_connect_prob.shape[1]
    # population_connect_prob should be a [N, N] tensor

    N = population_connect_prob.shape[0]
    population_node_init_kwards = {} if population_node_init_kwards is None else population_node_init_kwards

    if isinstance(population_node_init_kwards, dict):
        extern_input_k_sizes = [population_node_init_kwards["size"]] * N

    elif isinstance(population_node_init_kwards, list):
        extern_input_k_sizes = [b["size"] for b in population_node_init_kwards]
    else:
        raise ValueError

    # print('total {} populations'.format(N))
    def _out():
        if isinstance(population_node_init_kwards, dict):
            number = [population_node_init_kwards['size']] * N
            sub_block_idx = None
        elif isinstance(population_node_init_kwards, list):
            number = [b['size'] for b in population_node_init_kwards]
            if 'sub_block_idx' in population_node_init_kwards[0]:
                sub_block_idx = np.array([population_node_init_kwards[i]["sub_block_idx"] for i in
                                          range(len(population_node_init_kwards))], dtype=np.int64)
            else:
                sub_block_idx = None
        else:
            raise ValueError

        size_table = np.array(number, dtype=np.int64)
        if not isinstance(degree, np.ndarray):
            degree_table = np.array([degree] * N)
        else:
            degree_table = degree
        bases = np.add.accumulate(size_table)
        bases = np.concatenate([np.array([0], dtype=np.int64), bases])

        def prop(i, s, e):
            block_node_init_kward = population_node_init_kwards[i] if isinstance(population_node_init_kwards,
                                                                                 list) else population_node_init_kwards
            if 'sub_block_idx' not in block_node_init_kward:
                return generate_block_node_property(sub_block_idx=i, s=s, e=e, **block_node_init_kward)
            else:
                return generate_block_node_property(s=s, e=e, **block_node_init_kward)

        def conn(i, s, e):
            prob = population_connect_prob[i, :]
            step = int(1e6)
            for _s in range(s, e, step):
                _e = min(_s + step, e)
                output_neuron_idx, input_block_idx, input_neuron_idx, input_neuron_offset, weight = \
                    connect_for_single_sparse_block(i, bases[i + 1] - bases[i],
                                                    prob,
                                                    s=_s,
                                                    e=_e,
                                                    extern_input_k_sizes=extern_input_k_sizes,
                                                    degree=degree if not isinstance(degree, np.ndarray) else degree[i],
                                                    multi_conn2single_conn=multi_conn2single_conn,
                                                    dtype=dtype,
                                                    sub_block_idx=sub_block_idx)

                output_neuron_idx = output_neuron_idx.astype(np.int64)
                input_neuron_idx = input_neuron_idx.astype(np.int64)

                output_neuron_idx += bases[i].astype(output_neuron_idx.dtype)
                input_neuron_idx += bases[i + input_block_idx].astype(input_neuron_idx.dtype)
                # assert len(input_neuron_idx) == len(input_block_idx) == len(output_neuron_idx)
                yield output_neuron_idx, input_neuron_idx, input_neuron_offset, weight

        return prop, conn, bases, size_table, degree_table

    return _out

def connect_for_single_sparse_block(population_idx, k, extern_input_rate, extern_input_k_sizes, degree=int(1e3),
                                    s=0, e=-1, multi_conn2single_conn=False, dtype="single", sub_block_idx=None):
    """
    For each population, we implement the detailed construction of connection table.

    Parameters
    ----------
    population_idx: int
        the idx of processing population.

    k: int
        the number of neurons in this population.

    extern_input_rate: ndarray or sprse.COO
        the connection probability from others to itself.

    extern_input_k_sizes: ndarray
        for those populations that need to connect here, the total number of neurons that can be sampled,
        that is their maximum total number of neurons.

    degree: int
        number of in-degree for this population.

    s: int
        start idx of neurons in this population.

    e: int
        end idx of neurons in this population.

    Returns
    -------

    Notes
    -------
    In the processing, we need to ensure that each source population can meet the sampling requirements of the target populaiton.
    If it is not met, we pop up the error and interrupt the processing. However, although the above requirements are met in terms of probability,
    a few samples may fail. In this case, we use some tricks to slightly adjust the degree requirements of this population.
    -------

    """

    if e == -1:
        e = k
    assert 0 <= s <= e <= k
    _extern_input_k_sizes = np.array(extern_input_k_sizes, dtype=np.int64)

    if s < e:
        if isinstance(extern_input_rate, np.ndarray):
            extern_input_rate = np.add.accumulate(extern_input_rate)
            extern_input_idx = None
        else:
            extern_input_idx = extern_input_rate.coords[0, :]
            # ensure the degree requirement of this target population is reasonable
            if multi_conn2single_conn:
                degree_max = (_extern_input_k_sizes[extern_input_idx] / extern_input_rate.data).astype(
                    np.int64)
                degree_max = np.min(degree_max)
                if degree_max < degree:
                    print(f"{degree_max} ?< {degree}, {extern_input_idx}")
            extern_input_rate = np.add.accumulate(extern_input_rate.data)

        assert np.abs(1 - extern_input_rate[-1]) < 1e-4, f"{s}-{e}, {np.abs(1 - extern_input_rate[-1])}"
        extern_input_rate = extern_input_rate[:-1]
        
        def _run(i):
            r = np.random.rand(degree)
            input_block_idx = np.searchsorted(extern_input_rate, r, 'right').astype(np.int16)
            if extern_input_idx is not None:
                input_block_idx = extern_input_idx[input_block_idx]

            if multi_conn2single_conn:
                input_block_idx = examine(input_block_idx, _extern_input_k_sizes)

            input_channel_offset = np.zeros_like(input_block_idx, dtype=np.uint8)
            output_neuron_idx = np.ones_like(input_block_idx, dtype=np.uint32) * i
            if sub_block_idx is None:
                input_channel_offset[input_block_idx % 2 == 0] = 0
                input_channel_offset[input_block_idx % 2 == 1] = 2
            else:
                conn_EI = sub_block_idx[input_block_idx]
                input_channel_offset[conn_EI % 2 == 0] = 0
                input_channel_offset[conn_EI % 2 == 1] = 2

            if dtype == "single":
                weight = np.random.rand(input_block_idx.shape[0], 2).astype(np.float32)
            elif dtype == "uint8":
                weight = (np.random.randint(0, 256, dtype=np.uint8, size=(input_block_idx.shape[0], 2)))
                # trick process
                # if population_idx < 195700:
                #     index = np.where(np.abs(population_idx - input_block_idx) > 10)[0]
                #     weight[index, :] = weight[index, :] * 1.5
            else:
                raise NotImplementedError

            input_neuron_idx = np.zeros_like(input_block_idx, dtype=np.uint32)
            for _idx in np.unique(input_block_idx):
                extern_incomming_idx = (input_block_idx == _idx).nonzero()[0]
                if _idx != population_idx:
                    extern_outcomming_idx = get_k_idx(_extern_input_k_sizes[_idx], extern_incomming_idx.shape[0],
                                                      -1)  # population_idx, _idx
                else:
                    extern_outcomming_idx = get_k_idx(_extern_input_k_sizes[_idx], extern_incomming_idx.shape[0],
                                                      i)  # population_idx, _idx
                input_neuron_idx[extern_incomming_idx] = extern_outcomming_idx

            input_block_idx -= population_idx
            return input_block_idx, input_neuron_idx, input_channel_offset, output_neuron_idx, weight

        # time1 = time.time()
        with Thpool() as p:
            input_block_idx, input_neuron_idx, input_channel_offset, output_neuron_idx, weight = tuple(
                zip(*p.map(_run, range(s, e))))
        # time2 = time.time()
        # print("done", e - s, time2 - time1)
        input_block_idx = np.concatenate(input_block_idx)
        input_neuron_idx = np.concatenate(input_neuron_idx)
        input_channel_offset = np.concatenate(input_channel_offset)
        output_neuron_idx = np.concatenate(output_neuron_idx)
        weight = np.concatenate(weight, axis=0)
    else:
        input_block_idx = np.zeros([0], dtype=np.int16)
        input_neuron_idx = np.zeros([0], dtype=np.uint32)
        input_channel_offset = np.zeros([0], dtype=np.uint8)
        output_neuron_idx = np.zeros([0], dtype=np.uint32)
        weight = np.random.rand(e - s, degree, 2).astype(np.float32)

    return output_neuron_idx, input_block_idx, input_neuron_idx, input_channel_offset, weight


def generate_block_node_property(size=1000,
                                 noise_rate=0.01,
                                 I_extern_Input=0,
                                 sub_block_idx=0,
                                 C=1,
                                 T_ref=5,
                                 g_Li=0.03,
                                 V_L=-75,
                                 V_th=-50,
                                 V_reset=-65,
                                 g_ui=(5 / 275, 5 / 4000, 3 / 30, 3 / 730),
                                 V_ui=(0, 0, -70, -100),
                                 tao_ui=(2, 40, 10, 50),
                                 s=0, e=-1):
    """
    Generate neuronal property for each population.

    Parameters
    ----------
    noise_rate: float
        different neuron have a background noise, its output spike is calculated as spike | noise.

    I_extern_Input: float
        external current to each neuron

    sub_block_idx: bool
        This is a comparison neurons used to mark those who are used to do accurate verification.

    C: float
        capacitance

    T_ref: float
        refractory time

    g_Li: float
        conductance of leaky channel

    V_L: float
        leaky potential

    V_th: float
        threshold potential

    V_reset: float
        reset potential

    g_ui: tuple[float, ]
        conductance of 4 synaptic channels

    V_ui: tuple[float, ]
        reverse potential of 4 synaptic channels

    tao_ui: tuple[float, ]
        timescale of exponential synaptic filter.

    s: int
        start index
    e: int
        end indext

    Returns
    -------
    property: ndarray
        property of LIF neurons, shape=(e-s, 23)


    Notes
    -------
    each node contain such property::

           noise_rate, blocked_in_stat, I_extern_Input, sub_block_idx, C, T_ref, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tao_ui
      size:  1,   1,               1,                1,           1, 1,     1,    1,   1,    1,       4     4,    4
      dtype: f,   b,               f,                i,           f, f,     f,    f,   f,    f,       f,    f,    f

    b means bool(although storage as float), f means float.

    this function support broadcast, e.g, C can be a scalar for a total block or a [E_number, I_number] tensor for total nodes.

    """
    if e == -1:
        e = size
    assert 0 <= s <= e <= size

    property = np.zeros([e - s, 22], dtype=np.float32)
    property[:, 0] = noise_rate

    property[:, 1] = 0

    property[:, 2] = I_extern_Input

    property[:, 3] = sub_block_idx
    property[:, 4] = C
    property[:, 5] = T_ref
    property[:, 6] = g_Li
    property[:, 7] = V_L
    property[:, 8] = V_th
    property[:, 9] = V_reset

    g_ui = g_ui if isinstance(g_ui, np.ndarray) else np.array(g_ui)
    property[:, 10:14] = g_ui

    V_ui = V_ui if isinstance(V_ui, np.ndarray) else np.array(V_ui)
    property[:, 14:18] = V_ui

    tao_ui = tao_ui if isinstance(tao_ui, np.ndarray) else np.array(tao_ui)

    property[:, 18:22] = tao_ui

    return property


def merge_dti_distributation_block(func, new_path, dtype="single", number=1,
                                   block_partition=None, output_direction=False, MPI_rank=None):
    """
    merge to block that is corresponding to gpu card and then save in a npz file.

    Parameters
    ----------
    orig_path: closure or str
        Call this closure and return three generators.

    new_path: str
        directory to save

    dtype: str
        "single" indicate it's a single ensemble.
        in the old version, it represents the precision of storage data.

    number: int
        number of npz files, corresponding to cpu cards.

    block_partition: optional, None
        customized block partition in each gpu cards.

    output_direction: bool
        whether the output neurons as priorities.

    MPI_rank: int
        mpi rank , assert mpi rank < total gpus.

    Returns
    -------

    """
    assert callable(func)
    prop, conn, dti_block_thresh, size_table, degree_table = func()

    if block_partition is None:
        block_threshold = get_block_threshold(number, dti_block_thresh)
    else:
        if number == 1:
            # old partition way
            assert isinstance(block_partition, list)
            assert sum(block_partition) == dti_block_thresh.shape[0] - 1
            idx_threshold = np.add.accumulate(np.array(block_partition, dtype=np.int64))
            idx_threshold = np.concatenate([np.array([0], dtype=np.int64),
                                            idx_threshold])
            block_threshold = np.ascontiguousarray(dti_block_thresh[idx_threshold])
        else:
            # new block partition
            block_threshold = np.add.accumulate(block_partition)
            block_threshold = np.insert(block_threshold, 0, 0)
            assert block_threshold[-1] == dti_block_thresh[-1]

    def _process(block_i):
        _new_path = os.path.join(new_path, dtype)
        os.makedirs(_new_path, exist_ok=True)
        storage_path = os.path.join(_new_path, "block_{}".format(block_i))
        if os.path.exists(storage_path + '.npz'):
            print("passing processing", block_i)
            return
        # print("in processing", block_i)
        block_start = block_threshold[block_i]
        block_end = block_threshold[block_i + 1]

        dti_block_selection = []
        for j, (s, e) in enumerate(zip(dti_block_thresh[:-1], dti_block_thresh[1:])):
            if s == e:
                continue
            elif s >= block_start and e <= block_end:
                s1 = 0
                e1 = e - s
            elif s <= block_start and e >= block_end:
                s1 = block_start - s
                e1 = block_end - s
            elif s >= block_start and s < block_end:
                s1 = 0
                e1 = block_end - s
            elif e > block_start and e <= block_end:
                s1 = block_start - s
                e1 = e - s
            else:
                continue
            assert s1 >= 0 and e1 >= s1 and e1 <= e - s
            dti_block_selection.append((j, s1, e1))
            # print("property finished", j)

        _property = []
        connect_sum = 0;
        for dti_i, s, e in dti_block_selection:
            _property.append(prop(dti_i, s, e))
            connect_sum += ((e - s) * degree_table[dti_i])
        _property = np.concatenate(_property)
        assert _property.shape[0] == block_end - block_start, f"{_property.shape[0]} vs {block_end - block_start}"
        
        # print("connect_sum", connect_sum)
        _output_neuron_idx = np.empty(connect_sum, dtype=np.int64)
        global _input_neuron_idx
        _input_neuron_idx = np.empty(connect_sum, dtype=np.int64)
        _input_channel_offset = np.empty(connect_sum, dtype=np.uint8)
        _weight = np.empty((connect_sum, 2), dtype=dtype)
        base_cnt = 0
        for dti_i, s, e in dti_block_selection:
            for output_neuron_idx, input_neuron_idx, input_channel_offset, weight in conn(dti_i, s, e):
                # assert len(output_neuron_idx) == len(input_neuron_idx) == len(input_channel_offset), f"{dti_i}, {s}-->{e}"
                temp = np.size(output_neuron_idx)
                _input_neuron_idx[base_cnt: base_cnt + temp] = input_neuron_idx
                _output_neuron_idx[base_cnt: base_cnt + temp] = output_neuron_idx
                _input_channel_offset[base_cnt: base_cnt + temp] = input_channel_offset
                _weight[base_cnt:base_cnt + temp, :] = weight
                base_cnt = base_cnt + temp
        if base_cnt != connect_sum:
            _output_neuron_idx = _output_neuron_idx[:base_cnt]
            _input_channel_offset = _input_channel_offset[:base_cnt]
            _input_neuron_idx = _input_neuron_idx[:base_cnt]
            _weight = _weight[:base_cnt]

        # assert (np.unique(_output_neuron_idx) == np.arange(block_start, block_end,
        #                                                    dtype=_output_neuron_idx.dtype)).all()

        _output_neuron_idx = (_output_neuron_idx - block_start).astype(np.uint32)
        _input_block_idx, _input_neuron_idx, outer_sum = turn_to_block_idx(block_threshold, turn_format=True)
        if not output_direction:
            new_weight_idx = np.lexsort(
                (_input_block_idx, _output_neuron_idx))
        else:
            new_weight_idx = np.lexsort(
                (_input_neuron_idx, _input_block_idx))

        _output_neuron_idx = np.take(_output_neuron_idx, new_weight_idx, axis=0)
        _input_block_idx = np.take(_input_block_idx, new_weight_idx, axis=0)
        _input_block_idx -= block_i
        _input_neuron_idx = np.take(_input_neuron_idx, new_weight_idx, axis=0)
        _input_channel_offset = np.take(_input_channel_offset, new_weight_idx, axis=0)
        _weight = np.take(_weight, new_weight_idx, axis=0)
        print("done in ", block_i, "card and size is ", block_end - block_start, "out conn", outer_sum)

        _new_path = os.path.join(new_path, dtype)
        storage_path = os.path.join(_new_path, "block_{}".format(block_i))

        np.savez(storage_path,
                 property=_property,
                 output_neuron_idx=_output_neuron_idx,
                 input_block_idx=_input_block_idx,
                 input_neuron_idx=_input_neuron_idx,
                 input_channel_offset=_input_channel_offset,
                 weight=_weight)

    block_numbers = block_threshold[1:] - block_threshold[:-1]
    assert (block_numbers > 0).all()

    if MPI_rank is None:
        # because the global _input_neuron_idx is a nonlocal variable, multiprocess will use a common variable, inducing error.
        # or just delete global in function turn_to_block_idx
        # with Thpool() as p:
        #     p.map(_process, range(0, block_numbers.shape[0]))
        for i in range(block_numbers.shape[0]):
            _process(i)

    else:
        assert 0 <= MPI_rank and MPI_rank < block_numbers.shape[0]
        _process(MPI_rank)
    return block_threshold

def get_block_threshold(number, dti_block_thresh):
    if isinstance(number, int):
        _block_number = (dti_block_thresh[-1] - 1) // number + 1
        block_threshold = np.concatenate([np.arange(0, dti_block_thresh[-1], _block_number, dtype=np.int64),
                                          np.array([dti_block_thresh[-1]], dtype=np.int64)])
    elif isinstance(number, list):
        weight = np.array(number)
        _block_number = dti_block_thresh[-1] / np.sum(weight) * weight
        block_threshold = np.add.accumulate(_block_number).astype(np.int64)
        block_threshold = np.concatenate([np.array([0], dtype=np.int64),
                                          block_threshold])
        block_threshold[-1] = dti_block_thresh[-1]
    else:
        raise ValueError
    return block_threshold

def turn_to_block_idx(block_threshold, turn_format=False):
    global _input_neuron_idx
    block_idx = np.searchsorted(block_threshold, _input_neuron_idx, side='right') - 1
    outer_conn = len(block_idx) - (block_idx == 0).sum()
    if turn_format:
        block_idx = block_idx.astype(np.int16)
    neuron_idx = _input_neuron_idx - block_threshold[block_idx]
    if turn_format:
        neuron_idx = neuron_idx.astype(np.uint32)
    return block_idx, neuron_idx, outer_conn


def apply_map(population_connect_prob, number, size_scale, degree_scale, max_rate=None, max_iter=None, only_size=False, use_map=False):
    assert isinstance(size_scale, np.ndarray)
    assert isinstance(degree_scale, np.ndarray)
    assert size_scale.shape[0] == degree_scale.shape[0]
    merge_map = None
    if use_map:
         assert max_rate is not None
         assert max_iter is not None
         rank, _, comm = get_mpi_info()
         if rank == 0:
             merge_map = generate_map_split_only_size(size_scale.shape[0], number, size_scale, degree_scale, max_rate, max_iter, only_size)
         merge_map = comm.bcast(merge_map, root=0)
    
    order = np.concatenate(
        [np.array(merge_map[str(i)], dtype=np.int64) for i in range(len(merge_map))])
    assert np.unique(order).size == size_scale.shape[0], f"{np.unique(order).size} vs. {size_scale.shape[0]}"
    assert max(order) == (size_scale.shape[0] - 1) and min(order) == 0

    size_scale = np.ascontiguousarray(size_scale[order])
    degree_scale = np.ascontiguousarray(degree_scale[order])
    
    if isinstance(population_connect_prob, np.ndarray):
        conn_prob = np.ascontiguousarray(population_connect_prob[order, :][:, order])
    else:
        nonzeros = np.logical_and(np.isfinite(population_connect_prob.data), population_connect_prob.data > 0).nonzero()[0]
        mapping = {o: (order == o).nonzero()[0] for o in np.unique(order)}
        nnz = len(population_connect_prob.data[nonzeros])
        coords1 = np.zeros(nnz, dtype=np.int64)
        coords2 = np.zeros(nnz, dtype=np.int64)
        values = np.zeros(nnz, dtype=np.float32)
        coords_cnt = 0
        for c1, c2, data in zip(population_connect_prob.coords[0][nonzeros],
                                population_connect_prob.coords[1][nonzeros],
                                population_connect_prob.data[nonzeros]):
            idx1 = mapping[c1]
            idx2 = mapping[c2]
            out_idx_1, out_idx_2 = np.meshgrid(idx1, idx2, indexing='ij')
            out_idx_1 = out_idx_1.reshape(-1)
            out_idx_2 = out_idx_2.reshape(-1)
            count = len(out_idx_1)
            coords1[coords_cnt:coords_cnt + count] = out_idx_1
            coords2[coords_cnt:coords_cnt + count] = out_idx_2
            values[coords_cnt:coords_cnt + count] = data
            coords_cnt += count
        coords1 = coords1[:coords_cnt]
        coords2 = coords2[:coords_cnt]
        values = values[:coords_cnt]
        conn_prob = sparse.COO(coords=np.stack([coords1, coords2]),
                               data=values,
                               shape=[len(order), len(order)])
    partition = []
    idx = 0
    for i in range(len(merge_map)):
        count = 0
        for j in range(len(merge_map[str(i)])):
            count += size_scale[idx]
            idx += 1
        partition.append(count)

    return conn_prob, size_scale, degree_scale, order, partition


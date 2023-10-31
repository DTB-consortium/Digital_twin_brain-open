# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 13:36 
# @Author : lepold
# @File : sample.py

import numpy as np


def sample(aal_region, neurons_per_population_base, populations_id, specified_info=None, num_sample_voxel_per_region=1,
                                       num_neurons_per_voxel=300):
    """
    sample neurons from the ``simulation`` object according requirement.

    In simulation object :std:ref:`simulation`, it's required to set neuron sample before ``simulation.run``.


    Parameters
    ----------

    aal_region: ndarrau
        indicate the brain regions label of each voxel.

    neurons_per_population_base: ndarray
        The accumulated number of neurons for each population , corresponding to the population_id.

    populations_id: ndarray
        The population id. Due to the routing algorithm, the population id may be disordered and repeated.
        The population id of neurons distributed between consecutive cards may be duplicated,
        because the neurons of this population may not be completely divided into one card.

    num_sample_voxel_per_region: int, default=1
        the sample number of voxels in each region.

    num_neurons_per_voxel: int, default=300
        the sample number of neurons in each voxel .

    specified_info: ndarray
        according the specified_info info , we can randomly sample neurons which are from given voxel id.

    Returns
    -------

    ndarray which contain sample information

    ----------------|------------------
    0 th column     |     neuron id
    1 th column     |    voxel id
    2 th column     |    population id
    3 th column     |    region id
    ----------------|------------------

    """
    subcortical = np.array([37, 38, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78], dtype=np.int64) - 1  # region index from 0
    subblk_base = [0]
    tmp = 0
    for i in range(len(aal_region)):
        if aal_region[i] in subcortical:
            subblk_base.append(tmp + 2)
            tmp = tmp + 2
        else:
            subblk_base.append(tmp + 8)
            tmp = tmp + 8
    subblk_base = np.array(subblk_base)
    uni_region = np.arange(90)
    num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
    sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)

    s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)
    # lcm_gm = np.array([
    #     33.8 * 78, 33.8 * 22,
    #     34.9 * 80, 34.9 * 20,
    #     7.6 * 82, 7.6 * 18,
    #     22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
    # lcm_gm /= lcm_gm.sum()
    # cn = (lcm_gm * num_neurons_per_voxel).astype(np.int)
    # c1, c2, c3, c4, c5, c6, c7, c8 = cn
    # c8 += num_neurons_per_voxel - np.sum(cn)
    c1, c2, c3, c4, c5, c6, c7, c8 = 80, 20, 80, 20, 20, 10, 60, 10


    count_voxel = 0
    for i in uni_region:
        # print("sampling for region: ", i)
        if specified_info is None:
            choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
        else:
            specified_info_index = np.where(specified_info[:, 3 == i])
            choices = np.unique(specified_info[specified_info_index, 1])
        for choice in choices:
            if i in subcortical:
                index = np.where(np.logical_and(populations_id < (choice+1) * 10, populations_id>=choice * 10))[0]
                sub_populations = populations_id[index]
                assert len(np.unique(sub_populations)) ==2
                popu1 = index[np.where(sub_populations % 10 == 6)]
                neurons = np.concatenate([np.arange(neurons_per_population_base[id], neurons_per_population_base[id+1]) for id in popu1])
                sample1 = np.random.choice(neurons, size=s1, replace=False)
                popu2 = index[np.where(sub_populations % 10 == 7)]
                neurons = np.concatenate(
                    [np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1]) for id in popu2])
                sample2 = np.random.choice(neurons, size=s2, replace=False)
                sample = np.concatenate([sample1, sample2])
                sub_blk = np.concatenate(
                    [np.ones_like(sample1) * (subblk_base[choice]), np.ones_like(sample2) * (subblk_base[choice] +1)])[:,
                          None]
                sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
                sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
                sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
            else:
                index = np.where(np.logical_and(populations_id < (choice + 1) * 10, populations_id >= choice * 10))[0]
                sub_populations = populations_id[index]
                assert len(np.unique(sub_populations)) == 8
                sample_this_region = []
                sub_blk = []
                for yushu, size in zip(np.arange(2, 10), np.array([c1, c2, c3, c4, c5, c6, c7, c8])):
                    popu1 = index[np.where(sub_populations % 10 == yushu)]
                    neurons = np.concatenate(
                        [np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1]) for id in popu1])
                    sample1 = np.random.choice(neurons, size=size, replace=False)
                    sample_this_region.append(sample1)
                    sub_blk.append(np.ones(size) * (subblk_base[choice] + yushu))
                sample_this_region = np.concatenate(sample_this_region)
                sub_blk = np.concatenate(sub_blk)
                sample = np.stack([sample_this_region, np.ones(num_neurons_per_voxel) * choice, sub_blk, np.ones(num_neurons_per_voxel) * i], axis=-1)
                sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
            count_voxel += 1
    return sample_idx.astype(np.int64)


def specified_sample_column(aal_region, neurons_per_population_base, specified_info=None, num_sample_voxel_per_region=1,
                                       num_neurons_per_voxel=300):
    """
    more convenient version to sample neurons for ``simulation`` object.

    neurons_per_population_base is read from :class:`population_base.npy <.TestBlock>` which is generated during generation of connection table.


    Parameters
    ----------

    aal_region: ndarrau
        indicate the brain regions label of each voxel.

    neurons_per_population_base: ndarray
        The accumulated number of neurons for each population , corresponding to the population_id.
        corresponding to [0, 1, 2, 3, 4,... 227029]

    num_sample_voxel_per_region: int, default=1
        the sample number of voxels in each region.

    num_neurons_per_voxel: int, default=300
        the sample number of neurons in each voxel .

    specified_info: ndarray
        according the specified_info info , we can randomly sample neurons which are from given voxel id.

    Returns
    -------

    ndarray which contain sample information

    ----------------|------------------
    0 th column     |     neuron id
    1 th column     |    voxel id
    2 th column     |    population id
    3 th column     |    region id
    ----------------|------------------

    """
    subcortical = np.array([37, 38, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78], dtype=np.int64) - 1  # region index from 0
    subblk_base = [0]
    tmp = 0
    for i in range(len(aal_region)):
        if aal_region[i] in subcortical:
            subblk_base.append(tmp + 2)
            tmp = tmp + 2
        else:
            subblk_base.append(tmp + 8)
            tmp = tmp + 8
    subblk_base = np.array(subblk_base)
    uni_region = np.arange(90)
    num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
    sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)

    s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)
    # lcm_gm = np.array([
    #     33.8 * 78, 33.8 * 22,
    #     34.9 * 80, 34.9 * 20,
    #     7.6 * 82, 7.6 * 18,
    #     22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
    # lcm_gm /= lcm_gm.sum()
    # cn = (lcm_gm * num_neurons_per_voxel).astype(np.int)
    # c1, c2, c3, c4, c5, c6, c7, c8 = cn
    # c8 += num_neurons_per_voxel - np.sum(cn)
    c1, c2, c3, c4, c5, c6, c7, c8 = 80, 20, 80, 20, 20, 10, 60, 10

    # count_voxel = 0
    for i in uni_region:
        while True:
            print("sampling for region: ", i)
            try:
                if specified_info is None:
                    choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
                else:
                    specified_info_index = np.where(specified_info[:, 3 == i])
                    choices = np.unique(specified_info[specified_info_index, 1])
                count_voxel = i * num_sample_voxel_per_region
                for choice in choices:
                    if i in subcortical:
                        id = choice *10 + 6
                        neurons = np.arange(neurons_per_population_base[id], neurons_per_population_base[id+1])
                        sample1 = np.random.choice(neurons, size=s1, replace=False)
                        id = choice *10 + 7
                        neurons = np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1])
                        sample2 = np.random.choice(neurons, size=s2, replace=False)
                        sample = np.concatenate([sample1, sample2])
                        sub_blk = np.concatenate(
                            [np.ones_like(sample1) * (subblk_base[choice]), np.ones_like(sample2) * (subblk_base[choice] +1)])[:,
                                  None]
                        sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
                        sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
                        sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
                    else:
                        sample_this_region = []
                        sub_blk = []
                        for yushu, size in zip(np.arange(2, 10), np.array([c1, c2, c3, c4, c5, c6, c7, c8])):
                            id = choice * 10 + yushu
                            neurons = np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1])
                            sample1 = np.random.choice(neurons, size=size, replace=False)
                            sample_this_region.append(sample1)
                            sub_blk.append(np.ones(size) * (subblk_base[choice] + yushu))
                        sample_this_region = np.concatenate(sample_this_region)
                        sub_blk = np.concatenate(sub_blk)
                        sample = np.stack([sample_this_region, np.ones(num_neurons_per_voxel) * choice, sub_blk, np.ones(num_neurons_per_voxel) * i], axis=-1)
                        sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
                    count_voxel += 1
                break
            except:
                continue

    return sample_idx.astype(np.int64)


def specified_sample_voxel(aal_region, neurons_per_population_base, specified_info=None, num_sample_voxel_per_region=1,
                                       num_neurons_per_voxel=300):
    uni_region = np.arange(90)
    num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
    sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)

    s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)

    for i in uni_region:
        while True:
            print("sampling for region: ", i)
            try:
                if specified_info is None:
                    choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
                else:
                    specified_info_index = np.where(specified_info[:, 3 == i])
                    choices = np.unique(specified_info[specified_info_index, 1])
                count_voxel = i * num_sample_voxel_per_region
                for choice in choices:
                    id1 = choice * 2 + 0
                    neurons = np.arange(neurons_per_population_base[id1], neurons_per_population_base[id1+1])
                    sample1 = np.random.choice(neurons, size=s1, replace=False)
                    id2 = choice * 2 + 1
                    neurons = np.arange(neurons_per_population_base[id2], neurons_per_population_base[id2 + 1])
                    sample2 = np.random.choice(neurons, size=s2, replace=False)
                    sample = np.concatenate([sample1, sample2])
                    sub_blk = np.concatenate(
                        [np.ones_like(sample1) * id1, np.ones_like(sample2) * id2])[:,
                              None]
                    sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
                    sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
                    sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
                    count_voxel += 1
                break
            except:
                continue

    return sample_idx.astype(np.int64)
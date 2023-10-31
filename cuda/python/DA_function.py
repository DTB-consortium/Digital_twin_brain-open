import sys
sys.path.append("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/")
sys.path.append("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/cuda")
sys.path.append("/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/cuda/python")
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import torch
from brain_block.bold_model_pytorch import BOLD
# from brain_block.bold_model import BOLD
import numpy as np
import copy
import time
import os
from cuda.python.dist_blockwrapper import BlockWrapper as block_gpu
from multiprocessing.pool import ThreadPool as pool
from scipy.io import loadmat


def ge_property(p_properties, ensemble_sum, brain_num, path):
    f_properties = copy.deepcopy(p_properties)
    for i in range(ensemble_sum-1):
        f_properties[:, 3] = f_properties[:, 3] + brain_num
        p_properties = np.concatenate((p_properties, f_properties), axis=0)
    print(p_properties.shape)
    p_properties = np.ascontiguousarray(p_properties)
    np.save(os.path.join(path, "block/properties"+str(ensemble_sum)+".npy"), p_properties)
    print('properties')
    return p_properties


def ge_index(p_indices, ensemble_num, k, path):
    f_indices = copy.deepcopy(p_indices)
    for i in range(ensemble_num-1):
        f_indices[0] = f_indices[0] + k
        f_indices[2] = f_indices[2] + k
        p_indices = np.concatenate((p_indices, f_indices), axis=1)
    p_indices = np.ascontiguousarray(p_indices)
    np.save(os.path.join(path, "block/indices"+str(ensemble_num)+".npy"), p_indices)
    print('indices')
    return p_indices


def ge_weight(p_weights, ensemble_num, path):
    f_weights = copy.deepcopy(p_weights)
    for i in range(ensemble_num-1):
        p_weights = np.concatenate((p_weights, f_weights), axis=0)
    p_weights = np.ascontiguousarray(p_weights.astype(np.float32))
    np.save(os.path.join(path, "block/weights"+str(ensemble_num)+".npy"), p_weights)
    print('weights')
    return p_weights


def ge_sub_block(path, number):
    arr = np.load(os.path.join(path, "block_0.npz"))
    properties = arr["property"]
    indices = arr["idx"]
    weights = arr["weight"]
    ge_property(properties, number, 90, path)
    ge_index(indices, number, 5016868, path)
    ge_weight(weights, number, path)


def ge_block(path):
    """"""
    p = np.load(os.path.join(path, "block/properties4.npy"))
    i = np.load(os.path.join(path, "block/indices4.npy"))
    w = np.load(os.path.join(path, "block/weights4.npy"))
    s = np.array([len(p[0]), 60, len(p[0]), 4], dtype=int)
    np.savez(os.path.join(path, 'block_v100/block_' + str(0) + '.npz'), property=p, idx=i, weight=w, size=s)
    for j in range(31):
        print(j+1)
        p[:, 3] += 4*90
        i[1, :] = j+1
        np.savez(os.path.join(path, 'block_v100/block_'+str(j+1)+'.npz'), property=p, idx=i, weight=w, size=s)
    """
    p = np.load(os.path.join(path, "block/properties2.npy"))
    i = np.load(os.path.join(path, "block/indices2.npy"))
    w = np.load(os.path.join(path, "block/weights2.npy"))
    s = np.array([len(p[0]), 60, len(p[0]), 4], dtype=int)
    p[:, 3] += 4 * 90 * 15
    i[1, :] = 15
    np.savez(os.path.join(path, 'block/block_' + str(15) + '.npz'), property=p, idx=i, weight=w, size=s)
    for j in range(15):
        print(j+16)
        p[:, 3] += 2 * 90
        i[1, :] = j + 16
        np.savez(os.path.join(path, 'block/block_' + str(j + 16) + '.npz'), property=p, idx=i, weight=w, size=s)
    """"""
    p = np.load(os.path.join(path, "block/properties1.npy"))
    i = np.load(os.path.join(path, "block/indices1.npy"))
    w = np.load(os.path.join(path, "block/weights1.npy"))
    s = np.array([len(p[0]), 60, len(p[0]), 4], dtype=int)
    p[:, 3] += 4 * 90 * 15 + 2 * 90 * 16
    i[1, :] = 31
    np.savez(os.path.join(path, 'block/block_' + str(31) + '.npz'), property=p, idx=i, weight=w, size=s)
    for j in range(29):
        print(j+32)
        p[:, 3] += 1 * 90
        i[1, :] = j + 32
        np.savez(os.path.join(path, 'block/block_' + str(j + 32) + '.npz'), property=p, idx=i, weight=w, size=s)    
    """"""
    p = np.load(os.path.join(path, "block/properties1.npy"))
    i = np.load(os.path.join(path, "block/indices1.npy"))
    w = np.load(os.path.join(path, "block/weights1.npy"))
    s = np.array([len(p[0]), 60, len(p[0]), 4], dtype=int)
    np.savez(os.path.join(path, 'block1080/block_' + str(0) + '.npz'), property=p, idx=i, weight=w, size=s)
    for j in range(23):
        print(j+1)
        p[:, 3] += 1 * 90
        i[1, :] = j + 1
        np.savez(os.path.join(path, 'block1080/block_' + str(j + 1) + '.npz'), property=p, idx=i, weight=w, size=s)
    """


def ge_brain_index(properties):
    brain_index = copy.deepcopy(properties[:, 3])
    brain_index = np.array(brain_index).astype(np.int)
    nod_name, nod_sum = np.unique(brain_index, return_counts=True)
    return brain_index, nod_name, nod_sum


def ge_parameter(hp_range, para_ind, ensembles, brain_num, brain_index):
    # hp_range = np.array([[1,2],[3,4],[5,6],[7,8]])
    # para_ind = np.array([10,11,12,13], dtype=int)
    hp_num = len(para_ind)
    hp_low = np.tile(hp_range[para_ind - 10, 0], (brain_num, 1))  # shape = brain_num*hp_num
    hp_high = np.tile(hp_range[para_ind - 10, 1], (brain_num, 1))
    hp = np.linspace(hp_low, hp_high, 3*ensembles)[ensembles:-1*ensembles]  # shape = ensembles*brain_num*hp_num
    # para = np.random.exponential(hp[:, brain_index, :])  # shape = ensembles*k*hp_num
    para = np.random.exponential(np.ones([ensembles, len(brain_index), hp_num]))
    para_in_property = np.zeros([len(brain_index)*ensembles*hp_num, 2])
    para_in_property[:, 0] = np.repeat(np.arange(len(brain_index)*ensembles), hp_num)
    para_in_property[:, 1] = np.tile(para_ind, len(brain_index)*ensembles)
    return hp_num, log_abs(hp, hp_low, hp_high), para, hp_low, hp_high, para_in_property.astype(int)


def log_abs(val, lower, upper, scale=10):
    if (val <= upper).all() and (val >= lower).all():
        return scale * (np.log(val - lower) - np.log(upper - val))
    else:
        return None


def sigmoid_abs(val, lower, upper, scale=10):
    assert np.isfinite(val).all()
    return lower + (upper - lower) / (1 + np.exp(-val / scale))


def ensemble_system(Block, Bold, steps, w, para, hp_num, brain_n, ensembles, k, hp_sigma, hp_low, hp_high, brain_index, nod_sum, Bold_sigma, para_in_property, iter):
    start = time.time()
    hp_transf_enkf = w[:, :brain_n*hp_num].reshape(ensembles, brain_n, hp_num) + np.sqrt(hp_sigma) * np.random.randn(ensembles, brain_n, hp_num)
    hp_enkf = sigmoid_abs(hp_transf_enkf, hp_low, hp_high)
    # hp_delta = hp_enkf/sigmoid_abs(hp_fore.reshape(ensembles, brain_n, hp_num), hp_low, hp_high)
    # para = w[:, -k * hp_num:].reshape(ensembles, k, hp_num)*hp_delta[:, brain_index, :]
    print("para_time" + str(time.time() - start))
    # Block.update_property(para_in_property, para.reshape(ensembles*k*hp_num))
    Block.update_property(para_in_property, (para*hp_enkf[:, brain_index, :]).reshape(ensembles * k * hp_num))
    print("update_time" + str(time.time()-start))
    mid_time = time.time()
    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)
    # np.save(os.path.join(path, "./act_save/act"+str(iter)+".npy"), act)
    print("block_time" + str(time.time() - start), time.time()-mid_time)

    error = act/np.tile(nod_sum, ensembles)
    error_idx = (error < 0).nonzero()
    if error_idx[0].shape[0] > 0:
        print(act[error_idx])
        raise ValueError

    for t in range(steps):
        bold = Bold.run(np.maximum(act[t]/np.tile(nod_sum, ensembles), 1e-05))
    bold = bold + np.sqrt(Bold_sigma) * np.random.randn(5, ensembles*brain_n)
    print("bold_time" + str(time.time() - start))
    w_hat = np.concatenate((hp_transf_enkf.reshape(ensembles, brain_n*hp_num), (act[-1]/np.tile(nod_sum, ensembles)).reshape([ensembles, brain_n]),
                            bold[0].reshape([ensembles, brain_n]), bold[1].reshape([ensembles, brain_n]),
                            bold[2].reshape([ensembles, brain_n]), bold[3].reshape([ensembles, brain_n]),
                            bold[4].reshape([ensembles, brain_n]), para.reshape(ensembles, k*hp_num)), axis=1)
    print("w_time" + str(time.time() - start))
    return w_hat


def distributed_kalman(w_hat, brain_n, ensembles, S_sigma, bold_y_t, rate, hp_num):
    w = w_hat[:, :brain_n*(hp_num+6)].copy()
    w_mean = np.mean(w, axis=0, keepdims=True)
    w_diff = w - w_mean
    w_cx = w_diff[:, -brain_n:] * w_diff[:, -brain_n:]
    w_cxx = np.sum(w_cx, axis=0) / (ensembles - 1) + S_sigma[0, 0]
    # print(w_cxx.shape)
    kalman = np.dot(w_diff[:, -brain_n:].T, w_diff) / (w_cxx.reshape([brain_n, 1])) / (ensembles - 1) # (brain_n, w_shape[1])
    ''''''
    w_ensemble = w + (bold_y_t[0, :, None] + np.sqrt(S_sigma[0, 0]) * np.random.randn(brain_n, ensembles)
                      - w[:, -brain_n:].T)[:, :, None] * kalman[:, None, :]  # (brain_n, ensembles, w_shape[1])

    w_hat[:, brain_n*hp_num:brain_n*(hp_num+6)] = rate * w_ensemble[:, :, -6*brain_n:].reshape(
                                        [brain_n, ensembles, 6, brain_n]).diagonal(0, 0, 3).reshape([-1, 6 * brain_n])
    w_hat[:, :brain_n * hp_num] = rate * w_ensemble[:, :, :brain_n * hp_num].reshape(
             [brain_n, ensembles, brain_n, hp_num]).diagonal(0, 0, 2).transpose(0, 2, 1).reshape([-1, brain_n * hp_num])
    '''
    w_ensemble = np.zeros([brain_n, ensembles, w.shape[1]])
    for i in range(brain_n):
        w_ensemble[i] = w + np.dot((bold_y_t[0, i] + np.sqrt(S_sigma[0, 0]) * np.random.randn(
            ensembles, 1) - w[:, -brain_n+i].reshape([ensembles, 1])), kalman[i:i+1, :])
    for i in range(brain_n):
        idx_temp = np.array([0, 1, 2, 3, 4, 5], dtype=int) * brain_n + brain_n * hp_num + i
        w_hat[:, idx_temp] = rate * w_ensemble[i][:, idx_temp]
        idx_temp = i * hp_num + np.arange(hp_num).astype(int)
        w_hat[:, idx_temp] = rate * w_ensemble[i][:, idx_temp]
    '''
    w_hat[:, :brain_n*(hp_num+6)] = w_hat[:, :brain_n*(hp_num+6)] + (1 - rate) * np.mean(w_ensemble, axis=0)
    return w_hat


def ensemble_kalman():
    return None


def da_show(W, data, T, path, brain_num):
    iteration = [i for i in range(T)]
    for i in range(brain_num):
        print("show"+str(i))
        fig = plt.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iteration, data[:T, i], 'r-')
        ax1.plot(iteration, np.mean(W[:T, :, -brain_num+i], axis=1), 'b-')
        plt.fill_between(iteration, np.mean(W[:T, :, -brain_num+i], axis=1) -
                         np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), np.mean(W[:T, :, -brain_num+i], axis=1)
                         + np.sqrt(np.var(W[:T, :, -brain_num+i], axis=1)), color='b', alpha=0.2)
        plt.ylim((0.0, 0.08))
        plt.savefig(os.path.join(path, "show/bold"+str(i)+".png"))
        plt.close(fig)
    return None


def configuration_file(path, name):
    Bold_sigma = 1e-8
    brain_num = 90
    ensembles = 60*2
    # k = 10004232
    k = 5016868
    rate = 0.8
    S_sigma = 0.000001 * np.eye(brain_num)
    # 1.0177473e-02,  6.2823278e-04, 3.8137451e-02,  2.1428221e-03,
    hp_range = np.array([[1e-02/5, 1e-02*5], [6e-04/5, 6e-04*5], [3e-02/5, 3e-02*5], [2e-03/5, 2e-03*5]])
    # para_ind = np.array([10, 11, 12, 13], dtype=int)
    para_ind = np.array([10], dtype=int)
    hp_sigma = 1
    steps = 800
    T = 400
    np.savez(os.path.join(path, name), Bold_sigma=Bold_sigma, brain_num=brain_num, ensembles=ensembles, k=k, rate=rate,
             S_sigma=S_sigma, hp_range=hp_range, para_ind=para_ind, hp_sigma=hp_sigma, steps=steps, T=T)


def transfer(path, output_path):
    file = np.load(path)
    property = file["property"]
    idx = file["idx"][:, ::2]
    weight = file["weight"].reshape([-1, 2])

    output_neuron_idx = np.ascontiguousarray(idx[0, :].astype(np.uint32))
    input_block_idx = np.ascontiguousarray(idx[1, :].astype(np.uint16))
    input_neuron_idx = np.ascontiguousarray(idx[2, :].astype(np.uint32))
    input_channel_offset = np.ascontiguousarray(idx[3, :].astype(np.uint8))

    weight = np.ascontiguousarray(weight).astype(np.float32)
    sort_order = "output_neuron_idx,input_block_idx,input_neuron_idx,input_channel_offset"
    np.savez(output_path, property=property,
             output_neuron_idx=output_neuron_idx,
             input_block_idx=input_block_idx,
             input_neuron_idx=input_neuron_idx,
             input_channel_offset=input_channel_offset,
             weight=weight)


def sub_transfer(path, output_path):
    file = np.load(path)
    property = file["property"]
    output_neuron_idx = file["output_neuron_idx"]
    input_block_idx = file["input_block_idx"]
    input_neuron_idx = file["input_neuron_idx"]
    input_channel_offset = file["input_channel_offset"]
    weight = file["weight"]

    property[:, 3] += 4*90
    input_block_idx += 1

    sort_order = "output_neuron_idx,input_block_idx,input_neuron_idx,input_channel_offset"
    np.savez(output_path, property=property,
             output_neuron_idx=output_neuron_idx,
             input_block_idx=input_block_idx,
             input_neuron_idx=input_neuron_idx,
             input_channel_offset=input_channel_offset,
             weight=weight)


def da_run_block(file_name, path, bold_name, block_path):
    start = time.time()
    print("begin")
    conf = np.load(os.path.join(path, file_name))
    Bold_sigma = conf["Bold_sigma"]  # [膜电位，激活数，(s, q, v, f_in, bold)信号]
    brain_num = conf["brain_num"]
    ensembles = conf["ensembles"]
    k = conf["k"]
    rate = conf["rate"]
    S_sigma = conf["S_sigma"]
    hp_range = conf["hp_range"]
    para_ind = conf["para_ind"]
    hp_sigma = conf["hp_sigma"]
    steps = conf["steps"]
    T = conf["T"]
    # bold_y = np.load(os.path.join(path, bold_name))
    bold_y = loadmat(bold_name)
    bold_y = bold_y['AAL_TC']
    bold_y = np.array(bold_y)[:, :90]
    bold_y = 0.03 + 0.05 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())

    arr = np.load(os.path.join(path, "block_v100/block_0.npz"))
    # arr = np.load(os.path.join(path, "block_all/block_0.npz"))

    brain_index, nod_name, nod_sum = ge_brain_index(arr["property"][:k])
    print(nod_sum)
    hp_num, hp, para, hp_low, hp_high, p_i_p = ge_parameter(hp_range, para_ind, ensembles, brain_num, brain_index)
    print("ge_parameter:"+str(time.time()-start))
    Block = block_gpu('192.168.2.91:50051', block_path, 0.01, 1.)
    print("ge_Block:"+str(time.time()-start))
    # Block.update_property(p_i_p, para.reshape(ensembles * k * hp_num))
    Block.update_property(p_i_p, (para*sigmoid_abs(hp, hp_low, hp_high)[:, brain_index, :]).reshape(ensembles * k * hp_num))
    print("update_Block:"+str(time.time()-start))
    Bold = BOLD(epsilon=120, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)

    act = Block.run(steps, freqs=True, vmean=False, sample_for_show=False)
    print(np.array(act).shape)
    print("run block:"+str(time.time()-start))
    for t in range(steps):
        bold = Bold.run(np.maximum(act[t]/np.tile(nod_sum, ensembles), 1e-05))
    print("run bold:" + str(time.time() - start))
    w = np.concatenate((hp.reshape([ensembles, brain_num*hp_num]), (act[-1]/np.tile(nod_sum, ensembles)).reshape([ensembles, brain_num]),
                        bold[0].reshape([ensembles, brain_num]), bold[1].reshape([ensembles, brain_num]),
                        bold[2].reshape([ensembles, brain_num]), bold[3].reshape([ensembles, brain_num]),
                        bold[4].reshape([ensembles, brain_num]), para.reshape([ensembles, k*hp_num])), axis=1)
    w_save = [w[:, :brain_num*(hp_num+6)]]
    # w_hat = hp.copy().reshape([ensembles, brain_num*hp_num])
    print("begin da:" + str(time.time() - start))
    for t in range(T-1):
        start = time.time()
        bold_y_t = bold_y[t].reshape([1, brain_num])
        w_hat = ensemble_system(Block, Bold, steps, w, para, hp_num, brain_num,
                                ensembles, k, hp_sigma, hp_low, hp_high, brain_index, nod_sum, Bold_sigma, p_i_p, t)
        print("run da_es:" + str(time.time() - start))
        w_save.append(w_hat[:, :brain_num*(hp_num+6)].copy())
        if t <= 8 or t % 50 == 48 or t == (T - 2):
            np.save(os.path.join(path, "W_save_gpu.npy"), w_save)
        w = distributed_kalman(w_hat, brain_num, ensembles, S_sigma, bold_y_t, rate, hp_num)
        print("run da_dk:" + str(time.time() - start))
        ''''''
        Bold.s = torch.from_numpy(w[:, (hp_num+1) * brain_num: (hp_num+2) * brain_num].copy().reshape([ensembles * brain_num])).cuda()
        Bold.q = torch.from_numpy(np.maximum(w[:, (hp_num+2) * brain_num: (hp_num+3) * brain_num], 1e-05).reshape([ensembles * brain_num])).cuda()
        Bold.v = torch.from_numpy(np.maximum(w[:, (hp_num+3) * brain_num: (hp_num+4) * brain_num], 1e-05).reshape([ensembles * brain_num])).cuda()
        Bold.log_f_in = torch.from_numpy(np.maximum(w[:, (hp_num+4) * brain_num: (hp_num+5) * brain_num], -15).reshape([ensembles * brain_num])).cuda()
        '''
        Bold.s = w[:, (hp_num+1) * brain_num: (hp_num+2) * brain_num].reshape([ensembles * brain_num])
        Bold.q = np.maximum(w[:, (hp_num+2) * brain_num: (hp_num+3) * brain_num], 1e-05).reshape([ensembles * brain_num])
        Bold.v = np.maximum(w[:, (hp_num+3) * brain_num: (hp_num+4) * brain_num], 1e-05).reshape([ensembles * brain_num])
        Bold.log_f_in = w[:, (hp_num+4) * brain_num: (hp_num+5) * brain_num].reshape([ensembles * brain_num])
        '''
        print("------------run da"+str(t)+":"+str(time.time()-start))
    da_show(np.array(w_save), bold_y, T, path, brain_num)


if __name__ == "__main__":
    path5m = "./single5m"
    # ge_sub_block(path, 1)
    # ge_sub_block(path, 2)
    # ge_sub_block(path, 4)
    # ge_block(path)
    configuration_file(path5m, 'configuration_5m.npz')
    da_run_block("configuration_5m.npz", path5m, "./single5m/JF_AAL_TC.mat",
                 '/home1/wenyong36/Documents/spliking_nn_for_brain_simulation/single5m/block_v100')

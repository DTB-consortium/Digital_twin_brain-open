"""The Python implementation of the gRPC snn client."""
import time
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool
from queue import Queue

import grpc
import numpy as np
import torch
from scipy import stats

from .snn_pb2 import *
from .snn_pb2_grpc import *
from google.protobuf.empty_pb2 import Empty


class cache_property(object):
    def __init__(self, method):
        # record the unbound-method and the name
        self.method = method
        self.name = method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, inst, cls):
        if inst is None:
            return self
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result

class BlockWrapper:
    property_idx_trans = torch.tensor([19, -1, 0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                                      dtype=torch.int64).cuda()
    MAX_MESSAGE_LENGTH = 2147483647
    buffersize = 1024 ** 3 // 4

    def __init__(self, address, path, delta_t, use_route=False, print_stat=False, force_rebase=False, allow_rebase=True, overlap=2):
        """
        Block api

        Parameters
        ----------
        address: str
            listening ip of server
        path: str
            block path
        delta_t: float
            default is 1.
        route_path: str
            route path
        print_stat: bool
            whether to print detailed info
        force_rebase: bool
            if allow_rebase and the force_rebase option is True, we will forcibly accumulate the population id of each card
        allow_rebase: bool
            default True, if False, it will never accumulate the population id (if you clearly know that the block only contain one ensemble)
        overlap: int
            the population size of each voxel.
        """
        self.print_stat = print_stat
        self._channel = grpc.insecure_channel(address,
                                              options=[('grpc.max_send_message_length', self.MAX_MESSAGE_LENGTH),
                                                       ('grpc.max_receive_message_length', self.MAX_MESSAGE_LENGTH)])
        self._stub = SnnStub(self._channel)
        if use_route:
            mode = InitRequest.CommMode.COMM_ROUTE
        else:
            mode = InitRequest.CommMode.COMM_P2P
        _init_resp = self._stub.Init(InitRequest(file_path=path,
                                                 delta_t=delta_t,
                                                 mode=mode))

        block_id = []
        neurons_per_block = []
        subblk_info = []
        self._subblk_id_per_block = {}
        self.pool = Pool()

        subblk_base = 0
        cortical_subblk_start = 0 if overlap == 2 else 2

        for i, resp in enumerate(_init_resp):
            assert (resp.status == SnnStatus.SNN_OK)
            block_id.append(resp.block_id)
            neurons_per_block.append(resp.neurons_per_block)
            for j, sinfo in enumerate(resp.subblk_info):
                if j == 0 and sinfo.subblk_id == cortical_subblk_start and len(subblk_info) > 0:
                    new_base = max([id for id, _ in subblk_info])
                    if allow_rebase and (force_rebase or new_base != cortical_subblk_start):
                        subblk_base = (new_base + 1 + overlap - 1) // overlap * overlap
                subblk_info.append((sinfo.subblk_id + subblk_base, sinfo.subblk_num))
            self._subblk_id_per_block[resp.block_id] = \
                (subblk_base, torch.unique(torch.tensor([sinfo.subblk_id + subblk_base for sinfo in resp.subblk_info],
                                                        dtype=torch.int64).cuda()))
            assert self._subblk_id_per_block[resp.block_id][1].shape[0] > 0

        self._block_id = torch.tensor(block_id, dtype=torch.int64).cuda()
        self._neurons_per_block = torch.tensor(neurons_per_block, dtype=torch.int64).cuda()
        self._neurons_thrush = torch.cat([torch.tensor([0], dtype=torch.int64).cuda(),
                                          torch.cumsum(self._neurons_per_block, 0)])
        self._subblk_id = torch.tensor([s[0] for s in subblk_info], dtype=torch.int64).cuda()
        self._neurons_per_subblk = torch.tensor([s[1] for s in subblk_info], dtype=torch.int64).cuda()
        self._buff_len = self.buffersize // self._neurons_per_subblk.shape[0]
        self._subblk_id, _subblk_idx, counts = torch.unique(self._subblk_id, return_inverse=True, return_counts=True)
        if (_subblk_idx == torch.arange(_subblk_idx.shape[0], dtype=_subblk_idx.dtype,
                                        device=_subblk_idx.device)).all():
            self._subblk_idx = None
        else:
            self._subblk_idx = _subblk_idx

        # bugfix@2023.12.3
        if self._subblk_idx is None:
            self.neurons_per_subblk = self._neurons_per_subblk
            self.merge_popu_ratio = None
        else:
            self.neurons_per_subblk = self._reduceat(self._neurons_per_subblk)
            self.merge_popu_ratio = self._neurons_per_subblk / torch.repeat_interleave(self.neurons_per_subblk, counts)  # 

        self._reset_hyper_parameter()
        self._sample_order = None
        self._sample_num = 0
        self._iterations = 0

    def _reset_hyper_parameter(self):
        self._last_hyper_parameter = torch.ones([self._subblk_id.shape[0], 20], dtype=torch.float32).cuda()

    @property
    def total_neurons(self):
        return self._neurons_per_block.sum()

    @property
    def block_id(self):
        return self._block_id

    @property
    def subblk_id(self):
        return self._subblk_id

    @property
    def total_subblks(self):
        return self._subblk_id.shape[0]

    @cache_property
    def neurons_per_subblk(self):
        if self._subblk_idx is None:
            return self._neurons_per_subblk
        else:
            return self.neurons_per_subblk

    @property
    def neurons_per_block(self):
        return self._neurons_per_block

    def _reduceat(self, array):  # without mapping scenario is correct, otherwise should be reconsidered.
        assert array.shape[-1] == self._subblk_idx.shape[0]
        out = torch.zeros(array.shape[:-1] + (self._subblk_id.shape[-1],), dtype=array.dtype, device=array.device)
        if len(array.shape) == 2:
            _subblk_idx = self._subblk_idx.unsqueeze(0).expand(array.shape[0], self._subblk_idx.shape[0])
        else:
            assert len(array.shape) == 1
            _subblk_idx = self._subblk_idx
        out.scatter_add_(-1, _subblk_idx, array)
        return out

    def _merge_sbblk(self, array, weight=None):
        assert len(array.shape) in {1, 2} and array.shape[-1] == self._neurons_per_subblk.shape[0], \
            "error, {} vs {}".format(array.shape, self._neurons_per_subblk.shape)
        if self._subblk_idx is None:
            return array
        else:
            if weight is not None:
                array *= weight
            array = self._reduceat(array)
            # if weight is not None:
            #     array /= self._reduceat(weight)
            assert array.shape[-1] == self._subblk_id.shape[0]
            return array

    def run(self, iterations, freqs=True, freq_char=False, vmean=False, output_sample_spike = False,
            output_sample_vmemb = False, output_sample_iou = False,
            output_sample_isynaptic = False, receptor_imean=False, imean=False, iou=False, t_steps=1, equal_sample=False):
        return_list = Queue()
        self.equal_sample = equal_sample

        def _run():
            _recv_time = 0 
            responses = self._stub.Run(RunRequest(iter=iterations,
                                                      iter_offset=self._iterations,
                                                      output_freq=freqs,
                                                      freq_char=freq_char,
                                                      output_vmean=vmean,
                                                      output_sample_spike = output_sample_spike,
                                                      output_sample_vmemb = output_sample_vmemb,
                                                      output_sample_iou = output_sample_iou,
	                                               output_sample_isynaptic = output_sample_isynaptic,
                                                      output_imean=imean,
                                                      use_ou_background=iou,
                                                      t_steps=t_steps,
                                                      output_receptor_imeas=receptor_imean))
            for i in range(iterations):
                time1 = time.time()
                r = next(responses)
                assert (r.status == SnnStatus.SNN_OK)
                _recv_time += time.time() - time1
                j = i % self._buff_len
                if j == 0:
                    return_tuple = []
                    len = min(self._buff_len, iterations - i)
                    if freqs:
                        if not freq_char:
                            _freqs = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.int32)
                            return_tuple.append(_freqs)
                        else:
                            _freqs = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.uint8)
                            return_tuple.append(_freqs)
                    if vmean:
                        _vmeans = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        return_tuple.append(_vmeans)
                    if output_sample_spike:
                        _spike = np.empty([len, self._sample_num], dtype=np.uint8)
                        return_tuple.append(_spike)
                    if output_sample_vmemb:
                        _vi = np.empty([len, self._sample_num], dtype=np.float32)
                        return_tuple.append(_vi)
                    if output_sample_isynaptic:
                        _syn_current = np.empty([len, self._sample_num], dtype=np.float32)
                        return_tuple.append(_syn_current)
                    if output_sample_iou:
                        _iou = np.empty([len, self._sample_num], dtype=np.float32)
                        return_tuple.append(_iou)
                    if imean:
                        _imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        return_tuple.append(_imean)
                    if receptor_imean:
                        _ampa_imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        _nmda_imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        _gabaa_imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        _gabab_imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        return_tuple.append(_ampa_imean)
                        return_tuple.append(_nmda_imean)
                        return_tuple.append(_gabaa_imean)
                        return_tuple.append(_gabab_imean)
                if freqs:
                    if not freq_char:
                        _freqs[j, :] = np.frombuffer(r.freq[0], dtype=np.int32)
                    else:
                        _freqs[j, :] = np.frombuffer(r.freq[0], dtype=np.uint8)
                if vmean:
                    _vmeans[j, :] = np.frombuffer(r.vmean[0], dtype=np.float32)
                if output_sample_spike:
                    _spike[j, :] = np.frombuffer(r.sample_spike[0], dtype=np.uint8)
                if output_sample_vmemb:
                    _vi[j, :] = np.frombuffer(r.sample_vmemb[0], dtype=np.float32)
                if output_sample_isynaptic:
                    _syn_current[j, :] = np.frombuffer(r.sample_isynaptic[0], dtype=np.float32)
                if output_sample_iou:
                    _iou[j, :] = np.frombuffer(r.sample_iou[0], dtype=np.float32)
                if imean:
                    _imean[j, :] = np.frombuffer(r.imean[0], dtype=np.float32)
                if receptor_imean:
                    _ampa_imean[j, :] = np.frombuffer(r.ampa_imean[0], dtype=np.float32)
                    _nmda_imean[j, :] = np.frombuffer(r.nmda_imean[0], dtype=np.float32)
                    _gabaa_imean[j, :] = np.frombuffer(r.gabaa_imean[0], dtype=np.float32)
                    _gabab_imean[j, :] = np.frombuffer(r.gabab_imean[0], dtype=np.float32)
                if j == self._buff_len - 1 or i == iterations - 1:
                    return_list.put([torch.from_numpy(r) for r in return_tuple])
            return _recv_time

        def _error_callback(exc):
            print('error:', exc)
            raise ValueError

        process_thread = self.pool.apply_async(_run, error_callback=_error_callback)

        _run_time = 0
        for i in range(iterations):
            j = i % self._buff_len
            if j == 0:
                out = return_list.get()
                processed_out = list()
                time1 = time.time()
                if freqs:
                    if not freq_char:
                        _freqs = out.pop(0).cuda()
                        _freqs = self._merge_sbblk(_freqs)
                        processed_out.append(_freqs)
                    else:
                        _freqs = out.pop(0).cuda()
                        _freqs = _freqs / torch.tensor(255., dtype=torch.float32,
                                                       device=_freqs.device) * self._neurons_per_subblk
                        _freqs = self._merge_sbblk(_freqs)
                        _freqs = _freqs.to(torch.int32)
                        processed_out.append(_freqs)
                if vmean:
                    _vmeans = out.pop(0).cuda()
                    _vmeans = self._merge_sbblk(_vmeans, self.merge_popu_ratio)
                    processed_out.append(_vmeans)
                if output_sample_spike:
                    _spike = out.pop(0).cuda().reshape((-1, self._sample_num))
                    if self._sample_order is not None:
                        _spike = torch.index_select(_spike, -1, self._sample_order)
                    processed_out.append(_spike)
                if output_sample_vmemb:
                    _vi = out.pop(0).cuda().reshape((-1, self._sample_num))
                    if self._sample_order is not None:
                        _vi = torch.index_select(_vi, -1, self._sample_order)
                    processed_out.append(_vi)
                if output_sample_isynaptic:
                    _syn_current = out.pop(0).cuda().reshape((-1, self._sample_num))
                    if self._sample_order is not None:
                        _syn_current = torch.index_select(_syn_current, -1, self._sample_order)
                    processed_out.append(_syn_current)
                if output_sample_iou:
                    _iou = out.pop(0).cuda().reshape((-1, self._sample_num))
                    if self._sample_order is not None:
                        _iou = torch.index_select(_iou, -1, self._sample_order)
                    processed_out.append(_iou)
                if imean:
                    _imean = out.pop(0).cuda()
                    _imean = self._merge_sbblk(_imean, self.merge_popu_ratio)
                    processed_out.append(_imean)
                if receptor_imean:
                    _ampa_imean = out.pop(0).cuda()
                    _ampa_imean = self._merge_sbblk(_ampa_imean, self.merge_popu_ratio)
                    processed_out.append(_ampa_imean)

                    _nmda_imean = out.pop(0).cuda()
                    _nmda_imean = self._merge_sbblk(_nmda_imean, self.merge_popu_ratio)
                    processed_out.append(_nmda_imean)

                    _gabaa_imean = out.pop(0).cuda()
                    _gabaa_imean = self._merge_sbblk(_gabaa_imean, self.merge_popu_ratio)
                    processed_out.append(_gabaa_imean)

                    _gabab_imean = out.pop(0).cuda()
                    _gabab_imean = self._merge_sbblk(_gabab_imean, self.merge_popu_ratio)
                    processed_out.append(_gabab_imean)
                assert len(out) == 0
                _run_time += time.time() - time1
            if len(processed_out) == 1:
                yield (processed_out[0][j, :])
            else:
                yield tuple(o[j, :] for o in processed_out)

        if self.print_stat:
            _recv_time = process_thread.get()
            print('run merge time: {}, recv time: {}'.format(_run_time, _recv_time))
            # print(self.last_time_stat())
        else:
            process_thread.wait()

    @staticmethod
    def _lexsort(*args, **kwargs):
        for i in range(len(args)):
            idx = torch.argsort(args[i])  # the sort must be stable
            for a in args:
                a[:] = torch.take(a, idx)
            for v in kwargs.values():
                v[:] = v[idx].clone()

    @staticmethod
    def _histogram(number, thresh):
        idx = torch.bucketize(number, thresh, right=True) - 1
        idx, count = torch.unique(idx, return_counts=True)
        out = torch.zeros_like(thresh)[:-1]
        out[idx] = count
        return out
    
    def update_property(self, property_idx, property_weight, bid=None):
        if bid is not None:
            assert 0 <= bid and bid < len(self._neurons_per_block)

        assert isinstance(property_idx, torch.Tensor)
        assert isinstance(property_weight, torch.Tensor)

        assert property_weight.dtype == torch.float32
        assert property_idx.dtype == torch.int64
        assert len(property_idx.shape) == 2 and property_idx.shape[1] == 2
        assert len(property_weight.shape) == 1
        assert property_weight.shape[0] == property_idx.shape[0]

        time_1 = time.time()

        property_idx_0 = property_idx[:, 0].clone()
        property_idx_1 = torch.take(self.property_idx_trans, property_idx[:, 1])
        property_weight = property_weight.clone()
        del property_idx

        self._lexsort(property_idx_1, property_idx_0, value=property_weight)
        time_2 = time.time()
        prepare_time = time_2 - time_1
        process_time = 0

        def message_generator():
            nonlocal bid, prepare_time, process_time
            if bid is not None:
                yield UpdatePropRequest(block_id=bid,
                                        neuron_id=property_idx_0.cpu().numpy().astype(np.uint32).tolist(),
                                        prop_id=property_idx_1.cpu().numpy().astype(np.uint32).tolist(),
                                        prop_val=property_weight.cpu().numpy().astype(np.float32).tolist())
            else:
                time_3 = time.time()
                counts = self._histogram(property_idx_0, self._neurons_thrush)
                counts = torch.cat([torch.tensor([0], dtype=torch.int32).cuda(), torch.cumsum(counts, 0)])
                assert counts[-1] == property_idx_0.shape[0]
                time_4 = time.time()
                prepare_time += time_4 - time_3
                for bid in range(len(self._neurons_per_block)):
                    base = counts[bid]
                    thresh = counts[bid + 1]
                    if base < thresh:
                        time_5 = time.time()
                        _property_idx_0 = (
                                property_idx_0[base:thresh] - self._neurons_thrush[bid]).cpu().numpy().astype(
                            np.uint32).tolist()
                        _property_idx_1 = property_idx_1[base:thresh].cpu().numpy().astype(np.uint32).tolist()
                        _property_weight = property_weight[base:thresh].cpu().numpy().astype(np.float32).tolist()
                        time_6 = time.time()
                        prepare_time += time_6 - time_5
                        out = UpdatePropRequest(block_id=bid,
                                                neuron_id=_property_idx_0,
                                                prop_id=_property_idx_1,
                                                prop_val=_property_weight)
                        yield out
                        time_7 = time.time()
                        process_time += time_7 - time_6

        response = self._stub.Updateprop(message_generator())
        assert response.success == True
        if self.print_stat:
            print("Update Properties {}, prepare_time: {}, process_time: {}".format(response.success, prepare_time,
                                                                                    process_time))
        self._reset_hyper_parameter()

    def _update_property_by_subblk(self, property_idx, property_hyper_parameter, process_hp, generate_request,
                                   grpc_method):
        assert isinstance(property_idx, torch.Tensor)
        assert isinstance(property_hyper_parameter, torch.Tensor)

        assert len(property_idx.shape) == 2 and property_idx.shape[1] == 2
        assert property_hyper_parameter.shape[0] == property_idx.shape[0]
        assert property_idx.dtype == torch.int64
        assert property_hyper_parameter.dtype == torch.float32
        time_1 = time.time()

        property_idx_0 = property_idx[:, 0].clone()
        property_idx_1 = torch.take(self.property_idx_trans, property_idx[:, 1])
        property_hyper_parameter = property_hyper_parameter.clone()
        del property_idx
        self._lexsort(property_idx_1, property_idx_0, value=property_hyper_parameter)

        sub_blk_idx_0 = torch.bucketize(property_idx_0, self._subblk_id, right=True) - 1
        assert (self._subblk_id[sub_blk_idx_0] == property_idx_0).all()

        process_hp(property_hyper_parameter, sub_blk_idx_0, property_idx_1)

        if self.print_stat:
            print("property_hyper_parameter", float(torch.max(property_hyper_parameter)),
                  float(torch.min(property_hyper_parameter)),
                  float(torch.mean(property_hyper_parameter)))

        time_2 = time.time()
        prepare_time = time_2 - time_1
        process_time = 0

        def message_generator():
            nonlocal prepare_time, process_time
            for bid in range(len(self._neurons_per_block)):
                time_3 = time.time()

                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (property_idx_0 == subblk_id[torch.bucketize(property_idx_0, subblk_id, right=True) - 1]).nonzero(
                    as_tuple=True)[0]
                time_4 = time.time()
                prepare_time += time_4 - time_3
                if idx.shape[0] != 0:
                    time_5 = time.time()
                    _property_idx_0 = (property_idx_0[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    _property_idx_1 = property_idx_1[idx].cpu().numpy().astype(np.uint32).tolist()
                    Request, hp_dict = generate_request(property_hyper_parameter[idx].cpu().numpy(),
                                                        bid, _property_idx_0, _property_idx_1)

                    out = Request(block_id=bid,
                                  brain_id=_property_idx_0,
                                  prop_id=_property_idx_1,
                                  **hp_dict)
                    time_6 = time.time()
                    prepare_time += time_6 - time_5
                    yield out
                    time_7 = time.time()
                    process_time += time_7 - time_6

        response = grpc_method(message_generator())
        assert response.success == True
        if self.print_stat:
            print("Update Hyperparameter of Properties {}, prepare_time: {}, process_time: {}".format(response.success,
                                                                                                      prepare_time,
                                                                                                      process_time))

        return response

    def mul_property_by_subblk(self, property_idx, property_hyper_parameter, not_divided=False):
        def process_hp(property_hyper_parameter, sub_blk_idx_0, property_idx_1):
            assert len(property_hyper_parameter.shape) == 1
            # assert (property_hyper_parameter > 0).all()
            if not not_divided:
                property_hyper_parameter /= self._last_hyper_parameter[sub_blk_idx_0, property_idx_1]
                self._last_hyper_parameter[sub_blk_idx_0, property_idx_1] *= property_hyper_parameter

        def generate_request(property_hp, _1, _2, _3):
            _property_hp = property_hp.astype(np.float32).tolist()
            return UpdateHyperParaRequest, {"hpara_val": _property_hp, "assigned": False}

        self._update_property_by_subblk(property_idx, property_hyper_parameter, process_hp, generate_request,
                                        self._stub.Updatehyperpara)

    def assign_property_by_subblk(self, property_idx, property_hyper_parameter):
        def process_hp(property_hyper_parameter, sub_blk_idx_0, property_idx_1):
            assert len(property_hyper_parameter.shape) == 1
            # assert (property_hyper_parameter > 0).all()

        def generate_request(property_hp, _1, _2, _3):
            _property_hp = property_hp.astype(np.float32).tolist()
            return UpdateHyperParaRequest, {"hpara_val": _property_hp, "assigned": True}

        self._update_property_by_subblk(property_idx, property_hyper_parameter, process_hp, generate_request,
                                        self._stub.Updatehyperpara)

    def gamma_property_by_subblk(self, property_idx, gamma_concentration, gamma_rate):
        assert len(gamma_rate.shape) == 1

        if gamma_concentration is None:
            gamma_concentration = torch.ones_like(gamma_rate)
        assert len(gamma_concentration.shape) == 1

        gamma_hp = torch.stack([gamma_concentration, gamma_rate], dim=1)

        def process_hp(gamma_hp, sub_blk_idx_0, property_idx_1):
            assert (gamma_hp >= 0).all()
            self._last_hyper_parameter[sub_blk_idx_0, property_idx_1] = 1

        debug_log = dict()

        def generate_request(gamma_hp, bid, _property_idx_0, _property_idx_1):
            _gamma_concentration = gamma_hp[:, 0].astype(np.float32).tolist()
            _gamma_rate = gamma_hp[:, 1].astype(np.float32).tolist()
            if debug:
                for brain_id, prop_id, gamma_1, gamma_2 in zip(_property_idx_0, _property_idx_1, _gamma_concentration,
                                                               _gamma_rate):
                    debug_log[(bid, brain_id, prop_id)] = (gamma_1, gamma_2)
            return UpdateGammaRequest, {"gamma_concentration": _gamma_concentration,
                                        "gamma_rate": _gamma_rate}
        return self._update_property_by_subblk(property_idx, gamma_hp, process_hp, generate_request,
                                                   self._stub.Updategamma)

    def set_samples(self, sample_idx, bid=None):
        assert isinstance(sample_idx, torch.Tensor)
        assert len(sample_idx.shape) == 1
        assert (sample_idx.dtype == torch.int64)
        order, recover_order = torch.unique(sample_idx, return_inverse=True)
        if (order.shape[0] == sample_idx.shape[0]) and (order == sample_idx).all():
            print('sample_order is None')
            self._sample_order = None
        else:
            self._sample_order = recover_order
        self._sample_num = order.shape[0]

        def message_generator():
            if bid is not None:
                yield UpdateSampleRequest(block_id=bid, sample_idx=order.cpu().numpy().astype(np.uint32).tolist())
            else:
                for i, _bid in enumerate(self._block_id):
                    idx = torch.logical_and(self._neurons_thrush[i] <= order, order < self._neurons_thrush[i + 1])
                    _sample_idx = (order[idx] - self._neurons_thrush[i]).cpu().numpy().astype(np.uint32).tolist()
                    if len(_sample_idx) > 0:
                        yield UpdateSampleRequest(block_id=_bid, sample_idx=_sample_idx)

        response = self._stub.Updatesample(message_generator())

        if self.print_stat:
            print("Set Samples %s" % response.success)

    def update_ou_background_stimuli(self, population_id, correlation_time, mean, deviation):
        assert isinstance(population_id, torch.Tensor)
        assert population_id.dtype == torch.int64
        assert torch.max(population_id) <= torch.max(self._subblk_id)
        population_id = torch.unique(population_id, sorted=True)
        assert isinstance(correlation_time, torch.Tensor)
        assert isinstance(mean, torch.Tensor)
        assert isinstance(deviation, torch.Tensor)
        assert len(population_id) == len(correlation_time)
        assert len(population_id) == len(mean)
        assert len(population_id) == len(deviation)

        def message_generator():
            for bid in range(len(self._neurons_per_block)):
                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (population_id == subblk_id[torch.bucketize(population_id, subblk_id, right=True) - 1]).nonzero(
                    as_tuple=True)[0]
                if idx.shape[0] != 0:
                    _population_id = (population_id[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    _correlation_time = correlation_time[idx].cpu().numpy().tolist()
                    _mean = mean[idx].cpu().numpy().tolist()
                    _deviation = deviation[idx].cpu().numpy().tolist()
                    yield UpdateOUBackgroundParamRequest(block_id=bid,
                                                           brain_id=_population_id,
                                                           correlation_time=_correlation_time,
                                                           mean=_mean,
                                                           deviation=_deviation)
        response = self._stub.Updateoubackgroundparam(message_generator())
        if self.print_stat:
            print("update OU background stimuli %s" % response.success)

    def update_dopamine_current_stimuli(self, population_id, v_da, g_da):
        assert isinstance(population_id, torch.Tensor)
        assert population_id.dtype == torch.int64
        assert torch.max(population_id) <= torch.max(self._subblk_id)
        population_id = torch.unique(population_id, sorted=True)
        assert isinstance(v_da, torch.Tensor)
        assert isinstance(g_da, torch.Tensor)
        assert len(population_id) == len(v_da)
        assert len(population_id) == len(g_da)
        def message_generator():
            for bid in range(len(self._neurons_per_block)):
                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (population_id == subblk_id[torch.bucketize(population_id, subblk_id, right=True) - 1]).nonzero(
                    as_tuple=True)[0]
                if idx.shape[0] != 0:
                    _population_id = (population_id[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    yield UpdateDopamineCurrentParamRequest(block_id=bid,
                                                           brain_id=_population_id,
                                                           v_da=v_da[idx].cpu().numpy().tolist(),
                                                           g_da=g_da[idx].cpu().numpy().tolist())

        response = self._stub.Updatedopaminecurrentparam(message_generator())
        if self.print_stat:
            print("update dopamine current stimuli %s" % response.success)

    def update_ttype_ca_stimuli(self, population_id, h_init_val, g_t, tao_h_minus, tao_h_plus, v_h, v_t):
        assert isinstance(population_id, torch.Tensor)
        assert population_id.dtype == torch.int64
        assert torch.max(population_id) <= torch.max(self._subblk_id)
        population_id = torch.unique(population_id, sorted=True)
        assert isinstance(h_init_val, torch.Tensor)
        assert isinstance(g_t, torch.Tensor)
        assert isinstance(tao_h_minus, torch.Tensor)
        assert isinstance(tao_h_plus, torch.Tensor)
        assert isinstance(v_h, torch.Tensor)
        assert isinstance(v_t, torch.Tensor)
        assert len(population_id) == len(h_init_val)
        assert len(population_id) == len(g_t)
        assert len(population_id) == len(tao_h_minus)
        assert len(population_id) == len(tao_h_plus)
        assert len(population_id) == len(v_h)
        assert len(population_id) == len(v_t)

        def message_generator():
            for bid in range(len(self._neurons_per_block)):
                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (population_id == subblk_id[torch.bucketize(population_id, subblk_id, right=True) - 1]).nonzero(
                    as_tuple=True)[0]
                if idx.shape[0] != 0:
                    _population_id = (population_id[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    yield UpdateTTypeCaCurrentParamRequest(block_id=bid,
                                                           brain_id=_population_id,
                                                           h_init=h_init_val[idx].cpu().numpy().tolist(),
                                                           g_t=g_t[idx].cpu().numpy().tolist(),
                                                           tao_h_minus=tao_h_minus[idx].cpu().numpy().tolist(),
                                                           tao_h_plus=tao_h_plus[idx].cpu().numpy().tolist(),
                                                           v_h=v_h[idx].cpu().numpy().tolist(),
                                                           v_t=v_t[idx].cpu().numpy().tolist())

        response = self._stub.Updatettypecacurrentparam(message_generator())
        if self.print_stat:
            print("update T type Ca stimuli %s" % response.success)


    def update_ttype_ca_stimuli(self, population_id, ca_init_val, ca_decay, alpha, v_k, g_ahp):
        assert isinstance(population_id, torch.Tensor)
        assert population_id.dtype == torch.int64
        assert torch.max(population_id) <= torch.max(self._subblk_id)
        population_id = torch.unique(population_id, sorted=True)
        assert isinstance(ca_init_val, torch.Tensor)
        assert isinstance(ca_decay, torch.Tensor)
        assert isinstance(alpha, torch.Tensor)
        assert isinstance(v_k, torch.Tensor)
        assert isinstance(g_ahp, torch.Tensor)
        assert len(population_id) == len(ca_init_val)
        assert len(population_id) == len(ca_decay)
        assert len(population_id) == len(alpha)
        assert len(population_id) == len(v_k)
        assert len(population_id) == len(g_ahp)
        
        def message_generator():
            for bid in range(len(self._neurons_per_block)):
                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (population_id == subblk_id[torch.bucketize(population_id, subblk_id, right=True) - 1]).nonzero(
                    as_tuple=True)[0]
                if idx.shape[0] != 0:
                    _population_id = (population_id[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    yield UpdateAdaptationCurrentParamRequest(block_id=bid,
                                                           brain_id=_population_id,
                                                           ca_init=ca_init_val[idx].cpu().numpy().tolist(),
                                                           ca_decay=ca_decay[idx].cpu().numpy().tolist(),
                                                           alpha_constant=alpha[idx].cpu().numpy().tolist(),
                                                           v_k=v_k[idx].cpu().numpy().tolist(),
                                                           g_ahp=g_ahp[idx].cpu().numpy().tolist())

        response = self._stub.Updateadaptationcurrentparam(message_generator())
        if self.print_stat:
            print("update Adaptation stimuli %s" % response.success)
    
    def set_samples_by_specifying_popu_idx(self, population_id, sample_number=None):
        if population_id is None:
            population_id = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64).cuda()
            sample_number = torch.tensor([80, 20, 80, 20, 20, 10, 60, 10], dtype=torch.int64).cuda() // 2
        assert isinstance(population_id, torch.Tensor)
        assert population_id.dtype == torch.int64
        assert torch.max(population_id) <= torch.max(self._subblk_id)
        uni_population_id = torch.unique(population_id)
        assert uni_population_id.shape[0] == population_id.shape[0], "population_id should be unique!"
        _subblk_id_in_card = self._subblk_id[self._subblk_idx]

        indices = []
        for element in population_id:
            boolean_tensor = torch.eq(_subblk_id_in_card, element)
            matching_indices = torch.nonzero(boolean_tensor, as_tuple=True)
            indices.append(matching_indices[0])
        indices = torch.cat(indices)

        if indices.shape[0] != population_id.shape[0]:
            raise NotImplementedError(
                "one population has been splited into more than one sub-populations in this network structure and this method is invalid in this case.")
        if sample_number is None:
            min_value = self._neurons_per_subblk[indices].min()
            min_value = torch.minimum(min_value // 2, torch.tensor(50))
            sample_number = torch.ones_like(indices) * min_value
        else:
            assert isinstance(sample_number, torch.cuda.LongTensor)
        assert (sample_number < self._neurons_per_subblk[indices]).all(), "sample number invalid, too big to sample"
        populaiton_base = torch.cat([torch.tensor([0], dtype=torch.int64).cuda(),
                                     torch.cumsum(self._neurons_per_subblk, 0)])
        sample_idx = torch.cat(
            [torch.arange(populaiton_base[i], populaiton_base[i] + sample_number[idx]) for idx, i in enumerate(indices)]).cuda()
        self.set_samples(sample_idx)
        print("set sample is successful")
        return torch.sum(sample_number).item()

    def close(self):
        self._channel.close()

    def shutdown(self):
        response = self._stub.Shutdown(ShutdownRequest())
        if self.print_stat:
            print("Shutdown GRPC Server %s" % response.shutdown)
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_version(self):
        resp = self._stub.GetVersion(Empty())
        ver = f"Version: {resp.version // 1000}.{(resp.version % 1000) // 100}.{resp.version % 100}"
        return ver
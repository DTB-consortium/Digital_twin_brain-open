"""The Python implementation of the gRPC snn client."""
import time
from collections import OrderedDict
from multiprocessing.pool import ThreadPool as Pool
from queue import Queue

import grpc
import numpy as np
import pandas as pd
import torch
from scipy import stats

from .snn_pb2 import *
from .snn_pb2_grpc import *


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

    def __init__(self, address, path, delta_t, route_path=None, print_stat=False, force_rebase=False, allow_rebase=True,
                 allow_metric=False, overlap=2):
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
        if route_path is None:
            mode = InitRequest.CommMode.COMM_POINT_TO_POINT
            route_path = ""
        else:
            mode = InitRequest.CommMode.COMM_ROUTE_WITH_MERGE
        _init_resp = self._stub.Init(InitRequest(file_path=path,
                                                 route_path=route_path,
                                                 delta_t=delta_t,
                                                 mode=mode,
                                                 output_metric=allow_metric))

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
        self._subblk_id, _subblk_idx = torch.unique(self._subblk_id, return_inverse=True)
        if (_subblk_idx == torch.arange(_subblk_idx.shape[0], dtype=_subblk_idx.dtype,
                                        device=_subblk_idx.device)).all():
            self._subblk_idx = None
        else:
            self._subblk_idx = _subblk_idx

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
            return self._reduceat(self._neurons_per_subblk)

    @property
    def neurons_per_block(self):
        return self._neurons_per_block

    def _reduceat(self, array):
        assert array.shape[-1] == self._subblk_idx.shape[0]
        out = torch.zeros(array.shape[:-1] + (self._subblk_id.shape[-1],), dtype=array.dtype, device=array.device)
        if len(array.shape) == 2:
            _subblk_idx = self._subblk_idx.unsqueeze(0).expand(array.shape[0], self._subblk_idx.shape[0])
        else:
            assert len(array.shape) == 1
            _subblk_idx = self._subblk_idx
        out.scatter_add_(-1, _subblk_idx, array)
        return out

    def last_time_stat(self):
        responses = self._stub.Measure(MetricRequest())
        rows = ["computing_before_comm_time_point",
                "computing_after_comm_time_point",
                "computing_time_point_end",
                "routing_computation_time_point_start",
                "routing_computation_duration",
                "routing_computation_time_point_end",
                "reporting_duration",
                "sending_copy_time_point_start",
                "sending_copy_time_point_end",
                "recving_copy_time_point_start",
                "recving_copy_time_point_end",
                "sending_time_point_start",
                "sending_time_point_end",
                "recving_time_point_start",
                "recving_time_point_end",
                "routing_time_point_start",
                "routing_time_point_end",
                "sending_duration_inter_node",
                "sending_duration_intra_node",
                "recving_duration_inter_node",
                "recving_duration_intra_node",
                "routing_duration_inter_node",
                "routing_duration_intra_node",
                "sending_byte_size_inter_node",
                "sending_byte_size_intra_node",
                "recving_byte_size_inter_node",
                "recving_byte_size_intra_node",
                "routing_byte_size_inter_node",
                "routing_byte_size_intra_node",
                "flops_update_v_membrane",
                "flops_update_inner_j_presynaptic",
                "flops_update_outer_j_presynaptic",
                "flops_update_j_presynaptic",
                "flops_update_i_synaptic"]

        col = OrderedDict([('total_mean', lambda x: np.mean(x)),
                           ('total_std', lambda x: np.std(x)),
                           ('spatial_max_temporal_mean', lambda x: np.mean(np.max(x, axis=1), axis=0)),
                           ('spatial_argmax_temporal_mode', lambda x: stats.mode(np.argmax(x, axis=1), axis=None)[0]),
                           ('temporal_std_spatial_max', lambda x: np.max(np.std(x, axis=0), axis=0)),
                           ('temporal_std_spatial_argmax', lambda x: np.argmax(np.std(x, axis=0), axis=0))])

        name = []
        data = []
        for i, resps in enumerate(responses):
            d = []

            if resps.status == SnnStatus.SNN_NOT_ENABLE_METRIC:
                break

            for resp in resps.metric:
                if i == 0:
                    name.append(resp.name)
                d.append([getattr(resp, row) for row in rows])
            data.append(d)

        if not data:
            return None, None

        data = np.array(data)
        stat_data = [[f(data[:, :, i]) for f in col.values()] for i in range(len(rows))]
        table = pd.DataFrame(np.array(stat_data), index=pd.Index(rows), columns=list(col.keys()))
        return data, table

    def _merge_sbblk(self, array, weight=None):
        assert len(array.shape) in {1, 2} and array.shape[-1] == self._neurons_per_subblk.shape[0], \
            "error, {} vs {}".format(array.shape, self._neurons_per_subblk.shape)
        if self._subblk_idx is None:
            return array
        else:
            if weight is not None:
                array *= weight
            array = self._reduceat(array)
            if weight is not None:
                array /= self._reduceat(weight)
            assert array.shape[-1] == self._subblk_id.shape[0]
            return array

    def run(self, iterations, freqs=True, freq_char=False, vmean=False, sample_for_show=False, imean=False, iou=False,
            strategy=None, checked=False, t_steps=1, equal_sample=False):
        return_list = Queue()
        if strategy is None:
            strategy = RunRequest.Strategy.STRATEGY_SEND_PAIRWISE
        self.equal_sample = equal_sample

        def _run():
            _recv_time = 0
            if checked:
                if t_steps > 1:
                    assert self.equal_sample, "In t_steps>1 and checked, the sampling size in each card should be equal"
                responses = self._stub.Runwithcheck(RunRequest(iter=iterations,
                                                               iter_offset=self._iterations,
                                                               output_freq=freqs,
                                                               freq_char=freq_char,
                                                               output_vmean=vmean,
                                                               output_sample=sample_for_show,
                                                               output_imean=imean,
                                                               use_ou_background=iou,
                                                               strategy=strategy,
                                                               t_steps=t_steps))
            else:
                responses = self._stub.Run(RunRequest(iter=iterations,
                                                      iter_offset=self._iterations,
                                                      output_freq=freqs,
                                                      freq_char=freq_char,
                                                      output_vmean=vmean,
                                                      output_sample=sample_for_show,
                                                      output_imean=imean,
                                                      use_ou_background=iou,
                                                      strategy=strategy,
                                                      t_steps=t_steps))
            for i in range(iterations):
                time1 = time.time()
                r = next(responses)
                assert (r.status == SnnStatus.SNN_OK)
                _recv_time += time.time() - time1
                j = i % self._buff_len
                # if (i % 800 == 0):
                #    print(i)
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
                    if sample_for_show:
                        if checked:
                            _spike = np.empty([len, self._sample_num * t_steps], dtype=np.uint8)
                            return_tuple.append(_spike)
                            _vi = np.empty([len, self._sample_num * t_steps], dtype=np.float32)
                            return_tuple.append(_vi)
                        else:
                            _spike = np.empty([len, self._sample_num], dtype=np.uint8)
                            return_tuple.append(_spike)
                            _vi = np.empty([len, self._sample_num], dtype=np.float32)
                            return_tuple.append(_vi)
                        if checked:
                            _syn_current = np.empty([len, self._sample_num * t_steps], dtype=np.float32)
                            return_tuple.append(_syn_current)
                            if iou:
                                _iou = np.empty([len, self._sample_num * t_steps], dtype=np.float32)
                                return_tuple.append(_iou)
                    if imean:
                        _imean = np.empty([len, self._neurons_per_subblk.shape[0]], dtype=np.float32)
                        return_tuple.append(_imean)
                if freqs:
                    if not freq_char:
                        _freqs[j, :] = np.frombuffer(r.freq[0], dtype=np.int32)
                    else:
                        _freqs[j, :] = np.frombuffer(r.freq[0], dtype=np.uint8)
                if vmean:
                    _vmeans[j, :] = np.frombuffer(r.vmean[0], dtype=np.float32)
                if sample_for_show:
                    _spike[j, :] = np.frombuffer(r.sample_spike[0], dtype=np.uint8)
                    _vi[j, :] = np.frombuffer(r.sample_vmemb[0], dtype=np.float32)
                    if checked:
                        _syn_current[j, :] = np.frombuffer(r.sample_isynaptic[0], dtype=np.float32)
                        if iou:
                            _iou[j, :] = np.frombuffer(r.sample_iou[0], dtype=np.float32)
                if imean:
                    _imean[j, :] = np.frombuffer(r.imean[0], dtype=np.float32)
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
                    _vmeans = self._merge_sbblk(_vmeans)
                    processed_out.append(_vmeans)
                if sample_for_show:
                    _spike = out.pop(0).cuda()
                    if checked and t_steps>1 and self.equal_sample:
                        _spike = _spike.reshape((-1, self._block_id.shape[0], t_steps, int(self._sample_num/self._block_id.shape[0]) ))
                        _spike = torch.transpose(_spike, 1, 2)
                        _spike = _spike.reshape((-1, t_steps, self._sample_num))
                    else:
                        _spike = _spike.reshape((-1, self._sample_num))
                    if self._sample_order is not None:
                        _spike = torch.index_select(_spike, -1, self._sample_order)
                    processed_out.append(_spike)
                    _vi = out.pop(0).cuda()
                    if checked and t_steps>1 and self.equal_sample:
                        _vi = _vi.reshape((-1, self._block_id.shape[0], t_steps, int(self._sample_num/self._block_id.shape[0]) ))
                        _vi = torch.transpose(_vi, 1, 2)
                        _vi = _vi.reshape((-1, t_steps, self._sample_num))
                    else:
                        _vi = _vi.reshape((-1, self._sample_num))
                    if self._sample_order is not None:
                        _vi = torch.index_select(_vi, -1, self._sample_order)
                    processed_out.append(_vi)
                    if checked:
                        _syn_current = out.pop(0).cuda()
                        if t_steps>1:
                            _syn_current = _syn_current.reshape((-1, self._block_id.shape[0], t_steps, int(self._sample_num/self._block_id.shape[0]) ))
                            _syn_current = torch.transpose(_syn_current, 1, 2)
                            _syn_current = _syn_current.reshape((-1, t_steps, self._sample_num))
                        else:
                            _syn_current = _syn_current.reshape((-1, self._sample_num))
                        if self._sample_order is not None:
                            _syn_current = torch.index_select(_syn_current, -1, self._sample_order)
                        processed_out.append(_syn_current)
                        if iou:
                            _iou = out.pop(0).cuda()
                            if t_steps>1:
                                _iou = _iou.reshape((-1, self._block_id.shape[0], t_steps, int(self._sample_num/self._block_id.shape[0]) ))
                                _iou = torch.transpose(_iou, 1, 2)
                                _iou = _iou.reshape((-1, t_steps, self._sample_num))
                            else:
                                _iou = _iou.reshape((-1, self._sample_num))
                            if self._sample_order is not None:
                                _iou = torch.index_select(_iou, -1, self._sample_order)
                            processed_out.append(_iou)
                if imean:
                    _imean = out.pop(0).cuda()
                    _imean = self._merge_sbblk(_imean)
                    processed_out.append(_imean)
                assert len(out) == 0
                _run_time += time.time() - time1

            if len(processed_out) == 1:
                yield (processed_out[0][j, :],)
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

    def set_state_rule(self, file_path:str, observe_counter:int, observe_interval:int):
        response = self._stub.Setstaterule(SetStateRuleRequest(save_state=True, file_path=file_path, observe_counter=observe_counter, observe_interval=observe_interval))
        assert response.success == True
        if self.print_stat:
            print("Update Properties {}".format(response.success))

    def load_state_from_file(self, file_path:str, observe_counter:int):
        response = self._stub.Loadstatefromfile(LoadStateRequest(file_path=file_path, observe_counter=observe_counter))
        assert response.success == True
        if self.print_stat:
            print("Update Properties {}".format(response.success))

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

    def gamma_property_by_subblk(self, property_idx, gamma_concentration, gamma_rate, debug=False):
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

        if not debug:
            return self._update_property_by_subblk(property_idx, gamma_hp, process_hp, generate_request,
                                                   self._stub.Updategamma)
        else:
            response = self._update_property_by_subblk(property_idx, gamma_hp, process_hp, generate_request,
                                                       self._stub.Updategammawithresult)
            for r in response:
                brain_id = r.brain_id
                prop_id = r.prop_id
                bid = r.block_id
                if len(r.prop_val) > 0:
                    val = np.frombuffer(r.prop_val[0], dtype=np.float32)
                    print("val", val.shape, val.max(), val.min(), val.mean(), val.std())
                else:
                    val = np.array([], dtype=np.float32)
                    print("val", val.shape)
                assert (bid, brain_id, prop_id) in debug_log
                alpha, beta = debug_log[(bid, brain_id, prop_id)]
                mean_error = abs(val.mean() - alpha / beta) / (alpha / beta)
                var_error = abs(val.var() - alpha / beta ** 2) / (alpha / beta ** 2)
                print("bid: {}, brain_id: {}, prop_id: {}, mean_err: {:.2f}, var_err: {:.2f}".format(bid, brain_id,
                                                                                                     prop_id,
                                                                                                     mean_error,
                                                                                                     var_error))
            return True

    def set_samples(self, sample_idx, bid=None, checked=False):
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

        if checked:
            response = self._stub.Updatesamplewithcheck(message_generator())
        else:
            response = self._stub.Updatesample(message_generator())

        if self.print_stat:
            print("Set Samples %s" % response.success)

    def update_ou_background_stimuli(self, correlation_time, mean, deviation):
        response = self._stub.Updateoubackgroundparam(UpdateOUBackgroundParamRequest(correlation_time=correlation_time,
                                                                                     mean=mean,
                                                                                     deviation=deviation))
        if self.print_stat:
            print("update OU background stimuli %s" % response.success)

    def update_ttype_ca_stimuli(self, population_id, h_init_val, g_t, tao_h_minus, tao_h_plus, v_h, v_t):
        assert isinstance(population_id, torch.Tensor)
        assert population_id.dtype == torch.int64
        assert torch.max(population_id) <= torch.max(self._subblk_id)
        population_id = torch.unique(population_id, sorted=True)

        def message_generator():
            for bid in range(len(self._neurons_per_block)):
                subblk_base, subblk_id = self._subblk_id_per_block[bid]
                idx = (population_id == subblk_id[torch.bucketize(population_id, subblk_id, right=True) - 1]).nonzero(
                    as_tuple=True)[0]
                if idx.shape[0] != 0:
                    _population_id = (population_id[idx] - subblk_base).cpu().numpy().astype(np.uint32).tolist()
                    yield UpdateTTypeCaCurrentParamRequest(block_id=bid,
                                                           brain_id=_population_id,
                                                           h_init=h_init_val,
                                                           g_t=g_t,
                                                           tao_h_minus=tao_h_minus,
                                                           tao_h_plus=tao_h_plus,
                                                           v_h=v_h,
                                                           v_t=v_t)

        response = self._stub.Updatettypecacurrentparam(message_generator())
        if self.print_stat:
            print("update T type Ca stimuli %s" % response.success)

    def check_sample_conn_weight(self, bid, nid):
        '''
        for example:
        bid: 0, population_id: 9
        '''
        response = self._stub.Checksampleconnweight(CheckSampleConnWeightRequest(block_id=bid,
                                                                                 neuron_id=nid))

        if self.print_stat:
            print("Check sample conn weight %s" % response.success)
        input_block_idx = torch.tensor(response.block_id, dtype=torch.int32)
        input_block_idx = input_block_idx.to(torch.int64).cuda()
        input_neuron_idx = torch.tensor(response.neuron_id, dtype=torch.int32).cuda()
        if len(response.channel_offset) > 0:
            _input_channel_offset = np.frombuffer(response.channel_offset[0], dtype=np.uint8)
            input_channel_offset = _input_channel_offset.copy()
            input_channel_offset = torch.from_numpy(input_channel_offset).cuda()
        else:
            print('channel', response.channel_offset)
            print("channel len 0")
        input_neuron_idx = input_neuron_idx + self._neurons_thrush[input_block_idx]
        output_neuron_idx = torch.ones_like(input_neuron_idx).cuda() * (nid + self._neurons_thrush[bid])
        if len(response.double_weight) > 0:
            _weight = np.frombuffer(response.double_weight[0], dtype=np.float64)
        elif len(response.float_weight) > 0:
            _weight = np.frombuffer(response.float_weight[0], dtype=np.float32)
        elif len(response.half_weight) > 0:
            _weight = np.frombuffer(response.half_weight[0], dtype=np.float16)
        elif len(response.int8_weight) > 0:
            _weight = np.frombuffer(response.int8_weight[0], dtype=np.uint8)
            print(_weight.shape)
        else:
            _weight = None
            print('weight', response.int8_weight)
        if len(response.channel_offset) > 0:
            return torch.stack([output_neuron_idx, input_neuron_idx, input_channel_offset], dim=1), torch.from_numpy(
                _weight).cuda()
        else:
            return torch.stack([output_neuron_idx, input_neuron_idx], dim=1), torch.from_numpy(_weight).cuda()

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

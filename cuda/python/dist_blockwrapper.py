"""The Python implementation of the gRPC snn client."""
import grpc
from .snn_pb2 import *
from .snn_pb2_grpc import *
import numpy as np
import time
import pandas as pd


class BlockWrapper:
    property_idx_trans = np.array([-1, -1, 0, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.uint32)

    def __init__(self, address, path, noise_rate, delta_t, print_stat=False, force_rebase=False):
        self.print_stat = print_stat
        self._channel = grpc.insecure_channel(address)
        self._stub = SnnStub(self._channel)
        _init_resp = self._stub.Init(InitRequest(file_path=path, noise_rate=noise_rate, delta_t=delta_t))

        block_id = []
        neurons_per_block = []
        subblk_info = []
        self._subblk_id_per_block = {}

        subblk_base = 0

        for i, resp in enumerate(_init_resp):
            block_id.append(resp.block_id)
            neurons_per_block.append(resp.neurons_per_block)
            for j, sinfo in enumerate(resp.subblk_info):
                if j == 0 and sinfo.subblk_id == 0 and len(subblk_info) > 0:
                    new_base = max([id for id, _ in subblk_info])
                    if force_rebase or new_base != 0:
                        subblk_base = new_base + 1
                subblk_info.append((sinfo.subblk_id + subblk_base, sinfo.subblk_num))
            self._subblk_id_per_block[resp.block_id] = \
                (subblk_base, np.array(list({sinfo.subblk_id + subblk_base for sinfo in resp.subblk_info}), dtype=np.uint32))
            assert self._subblk_id_per_block[resp.block_id][1].shape[0] > 0

        self._block_id = np.array(block_id, dtype=np.uint32)
        self._neurons_per_block = np.array(neurons_per_block, dtype=np.uint32)
        self._total_neurons = self._neurons_per_block.sum()
        self._neurons_thrush = np.concatenate([np.array([0], dtype=np.int64), np.add.accumulate(self._neurons_per_block)]).astype(np.int64)
        self._subblk_id = np.array([s[0] for s in subblk_info], dtype=np.uint32)
        self._neurons_per_subblk = np.array([s[1] for s in subblk_info], dtype=np.uint32)

        self._subblk_order = np.argsort(self._subblk_id, kind='stable')
        if (self._subblk_order == np.arange(self._subblk_id.shape[0], dtype=self._subblk_order.dtype)).all():
            self._subblk_order = None
        else:
            self._subblk_id = np.ascontiguousarray(self._subblk_id[self._subblk_order])
            self._neurons_per_subblk = np.ascontiguousarray(self._neurons_per_subblk[self._subblk_order])
        self._subblk_id, _subblk_count = np.unique(self._subblk_id, return_counts=True)
        if self._subblk_id.shape[0] == self._neurons_per_subblk.shape[0]:
            self._subblk_thrush = None
        else:
            self._subblk_thrush = np.add.accumulate(_subblk_count)
            self._subblk_thrush[1:] = self._subblk_thrush[:-1]
            self._subblk_thrush[0] = 0

        self._reset_hyper_parameter()
        self._sample_order = None
        self._sample_num = 0

    def _reset_hyper_parameter(self):
        self._last_hyper_parameter = {id: np.ones([18], dtype=np.float32) for id in self._subblk_id}

    @property
    def total_neurons(self):
        return self._total_neurons

    @property
    def block_id(self):
        return self._block_id

    @property
    def subblk_id(self):
        return self._subblk_id

    @property
    def total_subblks(self):
        return self._subblk_id.shape[0]

    @property
    def neurons_per_subblk(self):
        if self._subblk_thrush is None:
            return self._neurons_per_subblk
        else:
            return np.add.reduceat(self._neurons_per_subblk, self._subblk_thrush)

    @property
    def neurons_per_block(self):
        return self._neurons_per_block

    def last_time_stat(self):
        responses = self._stub.Measure(MetricRequest())
        rows = ["computed", "queued", "communicated", "reported"]
        cols = ["max", "min", "avg"]
        name = []
        data = []
        for resp in responses:
            name.append(resp.name)
            data.append([[getattr(getattr(resp,  name+"_metric"), col+"_duration_per_iteration") for col in cols] for name in rows])

        data = np.array(data)
        idx = pd.MultiIndex.from_product([name, rows])
        table = pd.DataFrame(data.reshape([-1, len(cols)]), index=idx, columns=cols)
        return table

    def _merge_sbblk(self, array, weight=None):
        assert len(array.shape) == 2 and array.shape[1] == self._neurons_per_subblk.shape[0]
        if self._subblk_order is not None:
            array = array[:, self._subblk_order]
        if self._subblk_thrush is None:
            return array
        else:
            if weight is not None:
                array *= weight
            array = np.add.reduceat(array, self._subblk_thrush, axis=1)
            if weight is not None:
                array /= np.add.reduceat(weight, self._subblk_thrush)
            assert array.shape[1] == self._subblk_thrush.shape[0]
            return array

    def run(self, iterations, freqs=True, vmean=False, sample_for_show=False):
        responses = self._stub.Run(RunRequest(iter=iterations,
                                                      output_freq=freqs,
                                                      output_vmean=vmean,
                                                      output_sample=sample_for_show))
        return_list = []
        _freqs = np.empty([iterations, self._neurons_per_subblk.shape[0]], dtype=np.uint32) if freqs else None
        _vmeans = np.empty([iterations, self._neurons_per_subblk.shape[0]], dtype=np.float32) if vmean else None
        _spike = np.empty([iterations, self._sample_num], dtype=np.uint8) if sample_for_show else None
        _vi = np.empty([iterations, self._sample_num], dtype=np.float32) if sample_for_show else None

        start = time.time()
        for i, r in enumerate(responses):
            if freqs:
                _freqs[i, :] = np.frombuffer(r.freq[0], dtype=np.uint32)
            if vmean:
                _vmeans[i, :] = np.frombuffer(r.vmean[0], dtype=np.float32)
            if sample_for_show:
                _spike[i, :] = np.frombuffer(r.sample_spike[0], dtype=np.uint8)
                _vi[i, :] = np.frombuffer(r.sample_vmemb[0], dtype=np.float32)
        else:
            assert i == iterations - 1
        print('message passing time: {}'.format(time.time()-start))

        if self.print_stat:
            print(self.last_time_stat())

        if freqs:
            return_list.append(self._merge_sbblk(_freqs))
        if vmean:
            return_list.append(self._merge_sbblk(_vmeans, weight=self._neurons_per_subblk))
        if sample_for_show:
            if self._sample_order is not None:
                _spike = _spike[:, self._sample_order]
            return_list.append(_spike)
            if self._sample_order is not None:
                _vi = _vi[:, self._sample_order]
            return_list.append(_vi)

        if len(return_list) == 0:
            return
        if len(return_list) == 1:
            return return_list[0]
        else:
            return tuple(return_list)

    def update_property(self, property_idx, property_weight, bid=None):
        if bid is not None:
            assert 0 <= bid and bid < len(self._neurons_per_block)

        assert property_weight.dtype == np.float32
        assert len(property_idx.shape) == 2 and property_idx.shape[1] == 2
        assert len(property_weight.shape) == 1
        assert property_weight.shape[0] == property_idx.shape[0]

        time_1 = time.time()

        resort_idx = np.lexsort([property_idx[:, 1], property_idx[:, 0]])
        property_idx = property_idx[resort_idx, :].copy()
        property_idx[:, 1] = self.property_idx_trans[property_idx[:, 1]]
        property_weight = property_weight[resort_idx]
        time_2 = time.time()

        prepare_time = time_2 - time_1
        process_time = 0

        def message_generator():
            nonlocal bid, prepare_time, process_time
            if bid is not None:
                yield UpdatePropRequest(block_id=bid,
                                        neuron_id=property_idx[:, 0].tolist(),
                                        prop_id=property_idx[:, 1].tolist(),
                                        prop_val=property_weight.tolist())
            else:
                time_3 = time.time()
                counts, _ = np.histogram(property_idx[:, 0], self._neurons_thrush)
                counts = np.concatenate([np.array([0], dtype=np.uint32), np.add.accumulate(counts.astype(np.uint32))])
                assert counts[-1] == property_idx.shape[0]
                time_4 = time.time()
                prepare_time += time_4 - time_3
                for bid in range(len(self._neurons_per_block)):
                    base = counts[bid]
                    thresh = counts[bid+1]
                    if base < thresh:
                        time_5 = time.time()
                        _property_idx = property_idx[base:thresh]
                        _property_idx[:, 0] -= self._neurons_thrush[bid]
                        _property_weight = property_weight[base:thresh]
                        time_6 = time.time()
                        prepare_time += time_6 - time_5
                        out = UpdatePropRequest(block_id=bid,
                                                neuron_id=_property_idx[:, 0].astype(np.uint32).tolist(),
                                                prop_id=_property_idx[:, 1].astype(np.uint32).tolist(),
                                                prop_val=_property_weight.tolist())
                        yield out
                        time_7 = time.time()
                        process_time += time_7 - time_6

        response = self._stub.Updateprop(message_generator())
        if self.print_stat:
            print("Update Properties {}, prepare_time: {}, process_time: {}".format(response.success, prepare_time, process_time))
        self._reset_hyper_parameter()

    def mul_property_by_subblk(self, property_idx, property_hyper_parameter, accumulate=False):
        assert len(property_idx.shape) == 2 and property_idx.shape[1] == 2
        assert len(property_hyper_parameter.shape) == 1
        assert property_hyper_parameter.shape[0] == property_idx.shape[0]
        assert property_idx.dtype == np.uint32
        assert property_hyper_parameter.dtype == np.float32
        assert (property_hyper_parameter > 0).all()
        time_1 = time.time()
        resort_idx = np.lexsort([property_idx[:, 1], property_idx[:, 0]])
        property_idx = property_idx[resort_idx, :].copy()
        property_idx[:, 1] = self.property_idx_trans[property_idx[:, 1]]
        property_hyper_parameter = property_hyper_parameter[resort_idx].copy()

        if not accumulate:
            property_hyper_parameter /= np.array([self._last_hyper_parameter[id][col] for id, col in property_idx], dtype=np.float32)
        for i, (id, col) in enumerate(property_idx):
            self._last_hyper_parameter[id][col] *= property_hyper_parameter[i]

        if self.print_stat:
            print("property_hyper_parameter", np.mean(property_hyper_parameter),
                  np.min(property_hyper_parameter),
                  np.max(property_hyper_parameter))

        time_2 = time.time()
        prepare_time = time_2 - time_1
        process_time = 0

        def message_generator():
            nonlocal prepare_time, process_time
            for bid in range(len(self._neurons_per_block)):
                time_3 = time.time()

                subblk_base, subblk_id = self._subblk_id_per_block[bid]

                idx = np.isin(property_idx[:, 0], subblk_id)
                time_4 = time.time()
                prepare_time += time_4 - time_3
                if idx.shape[0] != 0:
                    time_5 = time.time()
                    _property_idx = property_idx[idx]
                    _property_idx[:, 0] -= subblk_base
                    _property_hp = property_hyper_parameter[idx]
                    time_6 = time.time()
                    prepare_time += time_6 - time_5
                    out = UpdateHyperParaRequest(block_id=bid,
                                                 brain_id=_property_idx[:, 0].tolist(),
                                                 prop_id=_property_idx[:, 1].tolist(),
                                                 hpara_val=_property_hp.tolist())
                    yield out
                    time_7 = time.time()
                    process_time += time_7 - time_6

        response = self._stub.Updatehyperpara(message_generator())
        if self.print_stat:
            print("Update Hyperparameter of Properties {}, prepare_time: {}, process_time: {}".format(response.success, prepare_time, process_time))

    def set_samples(self, sample_idx, bid=None):
        assert len(sample_idx.shape) == 1
        order, recover_order = np.unique(sample_idx, return_inverse=True)
        if (order.shape[0] == sample_idx.shape[0]) and (order == sample_idx).all():
            self._sample_order = None
        else:
            self._sample_order = recover_order
        self._sample_num = order.shape[0]

        def message_generator():
            if bid is not None:
                yield UpdateSampleRequest(block_id=bid, sample_idx=order.tolist())
            else:
                for i, _bid in enumerate(self._block_id):
                    idx = np.logical_and(self._neurons_thrush[i]<= order, order < self._neurons_thrush[i+1])
                    _sample_idx = (order[idx] - self._neurons_thrush[i]).astype(np.uint32)
                    if _sample_idx.shape[0] > 0:
                        yield UpdateSampleRequest(block_id=_bid, sample_idx=_sample_idx.tolist())

        response = self._stub.Updatesample(message_generator())
        if self.print_stat:
            print("Set Samples %s" % response.success)

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

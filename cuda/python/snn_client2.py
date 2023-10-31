"""The Python implementation of the gRPC snn client."""

from __future__ import print_function

import random
import logging

import grpc

import snn_pb2
import snn_pb2_grpc
import numpy as np

def snn_init(stub, path, delta_t):
    return stub.Init(snn_pb2.InitRequest(file_path=path, delta_t=delta_t))

def snn_run(stub, iter, output_freq, output_vmean, output_sample, ouput_isynapse):
    responses = stub.Run(snn_pb2.RunRequest(iter=iter, output_freq=output_freq, output_vmean=output_vmean, output_sample=output_sample, output_synaptic_current=ouput_isynapse))
    for resp in responses:
        print(resp.status)
        if output_freq:
            print(resp.freq)
        if output_vmean:
            print(resp.vmean)
        if output_sample:
           print(resp.sample_spike)
           print(resp.sample_vmemb)
        if ouput_isynapse:
           print(resp.isynapse)
    #return np.array([r.freq for r in responses], dtype=np.uint32), np.array([r.vmean for r in responses], dtype=np.float32)
def snn_measure(stub):
    responses = stub.Measure(snn_pb2.MetricRequest())
    columns = ['send duration', 'recv duration', 'route duration', 'compute duration',
    			'report duration', 'copy duration before send', 'copy duration after recv',
    			'parse merge duration', 'compute duration before route', 'copy duration before report']
    for resp in responses:
        for each_metric in resp.metric:
            print(each_metric.name)
            print("send duration: %f" % (each_metric.sending_duration))
            print("recv duration: %f" % (each_metric.recving_duration))
            print("route duration: %f" % (each_metric.routing_duration))
            print("compute duration: %f" % (each_metric.computing_duration))
            print("report duration: %f" % (each_metric.reporting_duration))
            print("copy duration before send: %f" % (each_metric.copy_before_sending_duration))
            print("copy duration after recv: %f" % (each_metric.copy_after_recving_duration))
            print("parse merge duration: %f" % (each_metric.parse_merge_duration))
            print("compute duration before route: %f" % (each_metric.route_computing_duration))
            print("copy duration before report: %f" % (each_metric.copy_before_reporting_duration))

def generate_prop(neurons_per_block):
    for bid in range(0, len(neurons_per_block)):
        neuron_id = []
        prop_id = []
        prop_val = []
        for _ in range(0, 10):
            neuron_id.append(random.randint(0, neurons_per_block[bid] - 1))
            prop_id.append(random.randint(0, snn_pb2.PropType.TAO_GABAb))
            prop_val.append(1.0)
        yield snn_pb2.UpdatePropRequest(block_id=bid, neuron_id=neuron_id, prop_id=prop_id, prop_val=prop_val)

def snn_update(stub, neurons_per_block):
   prop_iterator = generate_prop(neurons_per_block)
   response = stub.Updateprop(prop_iterator)
   print("Update Properties %s" % response.success)

def generate_hyperpara(neurons_per_block, subblk_ids):
    for bid in range(0, len(neurons_per_block)):
        brain_id = []
        prop_id = []
        hpara_val = []
        subblk_id = subblk_ids[bid][:]
        for _ in range(0, 10):
            brid = random.randint(0, len(subblk_id) - 1)
            brain_id.append(subblk_id[brid])
            prop_id.append(random.randint(0, snn_pb2.PropType.TAO_GABAb))
            hpara_val.append(1.0)
        yield snn_pb2.UpdateHyperParaRequest(block_id=bid, prop_id=prop_id, brain_id=brain_id, hpara_val=hpara_val)

def snn_update_hyperpara(stub, neurons_per_block, subblk_ids):
   hp_iterator = generate_hyperpara(neurons_per_block, subblk_ids)
   response = stub.Updatehyperpara(hp_iterator)
   print("Update Hyperparameters %s" % response.success)

def generate_sample(neurons_per_block):
    for bid in range(0, len(neurons_per_block)):
        sample_idx = []
        for _ in range(0, 10):
            sample_id = random.randint(0, neurons_per_block[bid] - 1)
            sample_idx.append(sample_id)
        yield snn_pb2.UpdateSampleRequest(block_id=bid, sample_idx=sample_idx)

def snn_sample(stub, neurons_per_block):
    sample_iterator = generate_sample(neurons_per_block)
    response = stub.Updatesample(sample_iterator)
    print("Update Samples %s" % response.success)
   
def snn_shutdown(stub):
    response = stub.Shutdown(snn_pb2.ShutdownRequest())
    print("Shutdown GRPC Server %s" % response.shutdown)

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('11.10.10.62:50051') as channel:
        stub = snn_pb2_grpc.SnnStub(channel)
        path = '/public/home/ssct004t/project/wdm/dti_4_4k/single'
        noise_rate = 0.01
        delta_t=1.
        print("-------------- Init --------------")
        init_resp = snn_init(stub, path, delta_t)
        neurons_per_block = []
        subblk_ids = []
        subblk_nums = []
        for resp in init_resp:
            print("=======================")
            print(resp.block_id)
            neurons_per_block.append(resp.neurons_per_block)
            subblk_id = []
            subblk_num = []
            for sinfo in resp.subblk_info:
                subblk_id.append(sinfo.subblk_id)
                subblk_num.append(sinfo.subblk_num)
            subblk_ids.append(subblk_id)
            subblk_nums.append(subblk_num)
            print(subblk_id)
            print(subblk_num)
        print("=======================")
        print(neurons_per_block)
        print("-------------- Run --------------")
        snn_run(stub, 3, True, True, False, True)
        print("-------------- Update property--------------")
        snn_update(stub, neurons_per_block)
        print("-------------- Sample --------------")
        snn_sample(stub, neurons_per_block)
        print("-------------- Run --------------")
        snn_run(stub, 3, True, True, True, True)
        print("-------------- Measure --------------")
        snn_measure(stub)
        print("-------------- Update hyper-parameter--------------")
        snn_update_hyperpara(stub, neurons_per_block, subblk_ids)
        print("-------------- Shutdown --------------")
        snn_shutdown(stub)

if __name__ == '__main__':
    logging.basicConfig()
    run()


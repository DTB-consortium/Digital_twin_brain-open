"""The Python implementation of the gRPC snn client."""

from __future__ import print_function

import random
import logging

import grpc

import snn_pb2
import snn_pb2_grpc
import numpy as np
import pandas as pd
import sys
import getopt

def snn_init(stub, path, route_path, delta_t, comm_mode):
    return stub.Init(snn_pb2.InitRequest(file_path=path, route_path=route_path, delta_t=delta_t, mode=comm_mode))

def snn_run(stub, iter, iter_offset, output_freq, output_vmean, output_sample,  send_strategy, output_imean):
    responses = stub.Run(snn_pb2.RunRequest(iter=iter, iter_offset=iter_offset, output_freq=output_freq, output_vmean=output_vmean, output_sample=output_sample, strategy=send_strategy, output_imean=output_imean))
    for resp in responses:
        print(resp.status)
        if output_freq:
            print(resp.freq)
        if output_vmean:
            print(resp.vmean)
        if output_sample:
           print(resp.sample_spike)
           print(resp.sample_vmemb)
        if output_imean:
           print(resp.imean)
    return np.array([r.freq for r in responses], dtype=np.uint32), np.array([r.vmean for r in responses], dtype=np.float32)
	
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

'''
            data = []
            data.append(each_metric.sending_duration)
            data.append(each_metric.recving_duration)
            data.append(each_metric.routing_duration)
            data.append(each_metric.computing_duration)
            data.append(each_metric.reporting_duration)
            data.append(each_metric.copy_before_sending_duration)
            data.append(each_metric.copy_after_recving_duration)
            data.append(each_metric.parse_merge_duration)
            data.append(each_metric.route_computing_duration)
            data.append(each_metric.copy_before_reporting_duration)
            df = pd.DataFrame(columns=columns, data=data)
            df.to_csv(each_metric.name + '.csv', mode='a', header=False)
'''
 
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

def run(iter, mode, strategy):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('11.5.4.12:50051') as channel:
        stub = snn_pb2_grpc.SnnStub(channel)
        #path = '/public/home/ssct004t/project/wdm/dti_4_4k/single' 
        path = '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_16_5000000_noise_rate_free/single'
        #route_path = '/public/home/wenyong36/project/istbi/dtb/cuda/python/route.json'
        route_path = '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_16_80m/route.json'
        delta_t=1.
        print("-------------- Init --------------")
        init_resp = snn_init(stub, path, route_path, delta_t, mode)
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
        snn_run(stub, iter, 0, True, True, False, strategy, False)
        print("-------------- Measure --------------")
        snn_measure(stub)
        print("-------------- Shutdown --------------")
        snn_shutdown(stub)

def main(argv):
    iter = 10
    mode = snn_pb2.InitRequest.CommMode.COMM_POINT_TO_POINT
    strategy = snn_pb2.RunRequest.Strategy.STRATEGY_SEND_SEQUENTIAL
    try:
        opts, args = getopt.getopt(argv,"hi:m:s:",["help","iter=","mode=","strategy="])
    
    except getopt.GetoptError:
        print('snn_client.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        print(opt)
        print(arg)
        if opt == '-h':
            print('snn_client.py -i<iteration> -m <communication mode> -s <send strategy> \n' + 
	            'Communication Mode\n' +
         		'0: point-to-point communication mode\n' +
         		'1: add route without merging communicationmode\n' +
         		'2: add route with merging communicationmode\n' +
         		 'Send strategy\n' +
         		'0: Sequential sending\n' +
         		'1: add route without merging communicationmode\n' +
         		'2: add route with merging communicationmode\n')	
            sys.exit()
        elif opt in ("-i", "--iter"):
            iter = int(arg)
        elif opt in ("-m", "--mode"):
            if arg == '0':
                mode = snn_pb2.InitRequest.CommMode.COMM_POINT_TO_POINT
            elif arg == '1':
                mode = snn_pb2.InitRequest.CommMode.COMM_ROUTE_WITHOUT_MERGE
            elif arg == '2':
                mode = snn_pb2.InitRequest.CommMode.COMM_ROUTE_WITH_MERGE
        elif opt in ("-s", "--strategy"):
            if arg == '0':
                strategy = snn_pb2.RunRequest.Strategy.STRATEGY_SEND_SEQUENTIAL
            elif arg == '1':
                strategy = snn_pb2.RunRequest.Strategy.STRATEGY_SEND_PAIRWISE
            elif arg == '2':
                strategy = snn_pb2.RunRequest.Strategy.STRATEGY_SEND_RANDOM
    
    run(iter, mode, strategy)

if __name__ == '__main__':
    logging.basicConfig()
    main(sys.argv[1:])


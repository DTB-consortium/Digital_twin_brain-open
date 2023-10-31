import os

import numpy as np
import torch
import time

from cuda4.python.dist_blockwrapper_pytorch import BlockWrapper

single_card = False

path = '/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/small_blocks_debug/d4_ou'
route_path = None
delta_t = 1.
dist_block = BlockWrapper('11.5.4.6:50051', os.path.join(path, 'uint8'), delta_t, route_path=route_path)

bases = [0]
totals = []
start = time.time()
if single_card:
    cards = 1
else:
    cards = 4
for i in range(cards):
    file = np.load(os.path.join(path, "uint8/block_%d.npz" % i))
    prop = file['property']
    bases.append(bases[-1] + prop.shape[0])
bases = np.array(bases)

for i in range(cards):
    file = np.load(os.path.join(path, "uint8/block_%d.npz" % i))
    output_neuron_idx = file['output_neuron_idx']
    input_neuron_idx = file['input_neuron_idx']
    input_block_idx = file['input_block_idx']
    input_channel_offset = file['input_channel_offset']
    w = file['weight']
    output_neuron_idx = output_neuron_idx + bases[i]
    input_neuron_idx = input_neuron_idx + bases[i + input_block_idx]
    total = np.concatenate(
        [output_neuron_idx[:, None], input_neuron_idx[:, None], input_channel_offset[:, None], w], axis=1)
    totals.append(total)

totals = np.concatenate(totals, axis=0)
print("\n===========PYTHON===========\n")
print("output_neuron| input_neuron| input_channel|w")
print(totals)
end1 = time.time()
print(f"python reading cost time {end1 - start:.2f}")

N = bases[-1]
sample_idx = torch.arange(N, dtype=torch.int64).cuda()
dist_block.set_samples(sample_idx, checked=True)
print('set_sample ok')
totals_cuda = []
if single_card:
    for idx in range(N):
        print("check neuron", idx)
        conn, weight = dist_block.check_sample_conn_weight(bid=0, nid=idx)
        conn = conn.cpu().numpy()
        weight = weight.cpu().numpy()
        assert weight.dtype == np.uint8
        weight = weight.reshape((-1, 2))
        info = np.concatenate([conn, weight], axis=1)
        totals_cuda.append(info)
    totals_cuda = np.concatenate(totals_cuda, axis=0)
else:
    for bid in range(cards):
        for idx in range(int(N / cards)):
            conn, weight = dist_block.check_sample_conn_weight(bid=bid, nid=idx)
            conn = conn.cpu().numpy()
            weight = weight.cpu().numpy()
            assert weight.dtype == np.uint8
            weight = weight.reshape((-1, 2))
            info = np.concatenate([conn, weight], axis=1)
            totals_cuda.append(info)
    totals_cuda = np.concatenate(totals_cuda, axis=0)

print("\n===========CUDA===========\n")
print("output_neuron| input_neuron| input_channel|w")
print(f"cuda reading cost time {time.time() - end1:.2f}")
print(totals_cuda)
print("\n")
if totals.shape[0] != totals_cuda.shape[0]:
    print("shape inconsisent")
else:
    index = np.lexsort((totals[:, 1], totals[:, 0]))
    totals = totals[index, :]
    index = np.lexsort((totals_cuda[:, 1], totals_cuda[:, 0]))
    totals_cuda = totals_cuda[index, :]
coords = np.where(totals_cuda != totals)
print("differ |in python")
print(totals[coords])
print("differ |in cuda")
print(totals_cuda[coords])
dist_block.shutdown()

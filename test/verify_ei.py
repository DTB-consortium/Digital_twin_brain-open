import numpy as np
import pickle

population_info_path="/work/home/bujingde/project/jsj_xa/graph_file_1April.pickle"
with open(population_info_path, "rb") as f:
    size_scale = pickle.load(f)["size"]

populations = np.load("../supplementary_info/population_base.npy")
population_base = populations.copy()
assert ((populations[1:] - populations[:-1])>0).all()
assert len(populations) - 1 == 194342

popu_index_exterme = 95942
neurons_range = np.arange(population_base[popu_index_exterme], population_base[popu_index_exterme+1])

print("load map")
map_path = "/work/home/bujingde/project/jsj_xa/tables/realsim_tables/map_14004_no_split_v5.pkl"
with open(map_path, 'rb') as f:
    _merge_map = pickle.load(f)
assert isinstance(_merge_map, dict)

merge_map = dict()
for k, v in _merge_map.items():
    if isinstance(v, dict):
        merge_map[k] = v
    else:
        merge_map[k] = {v_p: 1 for v_p in v}

order = np.concatenate(
    [np.array(list(merge_map[str(i)].keys()), dtype=np.int64) for i in range(len(merge_map))])
part = np.concatenate(
    [np.array(list(merge_map[str(i)].values()), dtype=np.float64) for i in range(len(merge_map))])
size_scale = np.ascontiguousarray(size_scale[order]) * part
scale = int(8.6e10)
populations2 = np.array([int(max(b * scale, 1)) if b != 0 else 0 for b in size_scale])
populations2 = np.add.accumulate(populations2)
populations2 = np.insert(populations2, 0, 0)
assert (populations2==populations).all()
assert len(size_scale)==194342

block_partition = [len(merge_map[str(i)]) for i in range(len(merge_map))]
assert isinstance(block_partition, list)
assert sum(block_partition) == populations2.shape[0] - 1
idx_threshold = np.add.accumulate(np.array(block_partition, dtype=np.int64))
idx_threshold = np.concatenate([np.array([0], dtype=np.int64),
                                idx_threshold])
block_threshold = np.ascontiguousarray(populations2[idx_threshold])
assert len(block_threshold) == 14004 + 1

start_card = np.searchsorted(block_threshold, population_base[popu_index_exterme]) - 1
start = population_base[popu_index_exterme] - block_threshold[start_card]
assert start>=0
if population_base[popu_index_exterme+1] < block_threshold[start_card+1]:
    end = population_base[popu_index_exterme + 1] - block_threshold[start_card]
    print("in 1 card", block_threshold[start_card]+start, block_threshold[start_card]+end)
else:
    end = block_threshold[start_card+1]
    print("in two card", block_threshold[start_card]+start, block_threshold[start_card]+end)
path = "uint8/block_%d.npz"%start_card
file = np.load(path)
input_neuron = file["input_neuron_idx"]
output_neuron = file["output_neuron_idx"]
input_block = file["input_block_idx"]
input_channel = file["input_channel_offset"]
property = file["property"]
N = input_channel.shape[0]

print("\n\n")
print("begin")
print("belong voxel", order[popu_index_exterme] // 10)
for i in range(start, start+10):
    print("gui", property[i, 10:14])
    idx = np.where(output_neuron==i)[0]
    input_neuron_here = input_neuron[idx]
    print("degree", len(input_neuron_here))
    count = {"e":0, "i":0}
    for j in idx:
        index = block_threshold[input_block[j]+start_card] + input_neuron[j]
        channel = input_channel[j]
        population_id = np.searchsorted(populations2, np.array([index])) - 1
        # print("population", order[population_id])
        flag = order[population_id] % 2
        channel_real = 0 if flag==0 else 2
        assert channel_real == channel
        if channel == 0:
            count["e"] += 1
        else:
            count["i"] += 1
    print(count)
print("Done")

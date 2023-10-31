from cuda.python.dist_blockwrapper import BlockWrapper
import numpy as np
import time

def main():
    t1 = time.time()
    block = BlockWrapper('192.168.2.106:50051', '/home1/bychen/spliking_nn_for_brain_simulation/dti_3_10m_new_form', 0.01, 1., print_stat=True)
    print(time.time() - t1)

    t1 = time.time()
    Freqs = block.run(800, freqs=True, vmean=False, sample_for_show=False)
    print(time.time() - t1)

    sample_idx = np.sort(np.random.choice(block.total_neurons, 100, replace=False))
    block.set_samples(sample_idx)
    t1 = time.time()
    Freqs, vemeans, spike, vi = block.run(800, freqs=True, vmean=True, sample_for_show=True)
    print(time.time() - t1)

    property_idx = np.stack(np.meshgrid(np.arange(block.total_neurons, dtype=np.uint32), np.arange(10, 11, dtype=np.uint32), indexing='ij'), axis=-1)
    property_idx = property_idx.reshape([-1, 2])
    block.update_property(property_idx, np.random.exponential(np.ones([property_idx.shape[0]])).astype(np.float32))
    t1 = time.time()
    Freqs = block.run(800, freqs=True, vmean=False, sample_for_show=False)
    print(time.time() - t1)
    print(Freqs.sum(axis=0).astype(np.float32) / block.neurons_per_subblk.astype(np.float32)/ 0.8)

    property_idx = np.stack(np.meshgrid(block.subblk_id, np.arange(10, 11, dtype=np.uint32), indexing='ij'), axis=-1)
    property_idx = property_idx.reshape([-1, 2])
    block.mul_property_by_subblk(property_idx, 0.025 * np.ones(property_idx.shape[0], dtype=np.float32))
    t1 = time.time()
    Freqs = block.run(800, freqs=True, vmean=False, sample_for_show=False)
    print(time.time() - t1)

    print(Freqs.sum(axis=0).astype(np.float32) / block.neurons_per_subblk.astype(np.float32)/ 0.8)
    block.shutdown()


if __name__ == "__main__":
    main()


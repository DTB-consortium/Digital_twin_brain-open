import unittest
import os
import matplotlib.pylab as plt
import numpy as np
import time
from cuda.python.dist_blockwrapper import BlockWrapper


class TestBlock(unittest.TestCase):
    def test_speed(self):
        #path = '/home1/bychen/spliking_nn_for_brain_simulation/dti_3_10m_new_form'
        path = '/home1/bychen/spliking_nn_for_brain_simulation/dti_4_2m/single'
        length = 800

        def run(noise):
            time.sleep(5)
            block = BlockWrapper('192.168.2.91:50051', path, noise, 1.)
            block.run(800, freqs=True, vmean=False, sample_for_show=False)
            start_time = time.time()
            Freqs = block.run(800, freqs=True, vmean=False, sample_for_show=False)
            end_time = time.time()
            print(block.print_last_time_stat())
            block.shutdown()
            return 1000 * np.median(Freqs.astype(np.float64).mean(axis=0)/block.neurons_per_subblk.astype(np.float64)), 1000 * (end_time-start_time)/length

        freqs = []
        times = []
        max_freqs = []

        for i in np.arange(10, 100, 2.5):
            f, t = run(i/1000)
            freqs.append(f)
            times.append(t)
            max_freqs.append(min(f + i, 200))
            print(f, t, i)

        fig_fp = plt.figure(figsize=(4, 4), dpi=500)
        fig_fp.gca().plot(freqs, times, 'r', label='output freqs')
        fig_fp.gca().plot(max_freqs, times, 'g', label='input freqs')
        fig_fp.gca().legend()
        fig_fp.gca().set_ylabel('duration(ms)')
        fig_fp.gca().set_xlabel('freq(Hz)')
        fig_fp.savefig('result.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig_fp)


if __name__ == "__main__":
    unittest.main()

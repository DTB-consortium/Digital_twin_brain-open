import unittest
import matplotlib.pylab as plt
import numpy as np
from BrainBlock import BlockWrapper
import os

class TestBlock(unittest.TestCase):
    def test_static_running(self):
        path = "../../dti_single_500k/"

        arr = np.load(os.path.join(path, "block_0.npz"))
        properties = arr["property"]
        indices = arr["idx"]
        weights = arr["weight"]
        delta_t = np.float32(1.)
        noise_rate = np.float32(.01)
        
        B = BlockWrapper(properties, indices, weights, 0, 0, delta_t)
        
        for i in range(100):
            B.run(noise_rate)
            print(B.get_freqs())
            print(B.get_vmeans())
            print(B.get_v_membranes())
            print(B.get_t_actives())
            print(B.get_j_presynaptics().reshape(4, -1))
            print(B.get_f_actives())

if __name__ == '__main__':
    unittest.main()

import os
import unittest
import numpy as np


class TestBlock(unittest.TestCase):
    def test_run(self): 
         path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n2_g1_d100/dti_distribution_2m_d100_regional/module'
         #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n3_g1_d100/dti_distribution_3m_d100_regional/module'
         #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n4_g1_d100/dti_distribution_4m_d100_regional/module'
         #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n5_g1_d100/dti_distribution_5m_d100_regional/module/uint8'
         file_name = os.path.join(path, "block_0.npz")
         new_name = os.path.join(path, "block_old_0.npz")
         os.rename(file_name, new_name)
         arr = np.load(new_name)
         _property = arr["property"]
         _property[:, 4] = np.float32(0.5)
         _property[:, 5] = np.float32(2)
         _property[:, 6] = np.float32(0.025)
         _property[:, 7] = np.float32(-70)
         _property[:, 9] = np.float32(-55)
         _property[:, 10] = np.float32(0.002)
         _property[:, 11] = np.float32(0)
         _property[:, 12] = np.float32(0.01)
         _property[:, 13] = np.float32(0)
         _property[:, 20] = np.float32(20)

         _output_neuron_idx = arr["output_neuron_idx"]
         _input_block_idx = arr["input_block_idx"]
         _input_neuron_idx = arr["input_neuron_idx"]
         _input_channel_offset = arr["input_channel_offset"]
         _weight = arr["weight"]

         np.savez(file_name,
                 property=_property,
                 output_neuron_idx=_output_neuron_idx,
                 input_block_idx=_input_block_idx,
                 input_neuron_idx=_input_neuron_idx,
                 input_channel_offset=_input_channel_offset,
                 weight=_weight)



if __name__ == '__main__':
    unittest.main()

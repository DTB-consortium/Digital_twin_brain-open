import os
import unittest
import numpy as np
from python.dist_blockwrapper import BlockWrapper
import torch

class TestBlock(unittest.TestCase):
    def test_run(self):        
        #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n2_g1_d100/dti_distribution_2m_d100_regional/module'
        #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n3_g1_d100/dti_distribution_3m_d100_regional/module'
        #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n4_g1_d100/dti_distribution_4m_d100_regional/module'
        path = '/public/home/ssct004t/project/wanghuarui/dtb/python/generation/reg_n10_g1_d100/dti_distribution_5m_d100_regional/module'
        #path = '/public/home/ssct004t/project/chenzhongyu/generation/reg_n10_g1_d100/dti_distribution_10m_d100_regional/module'
        delta_t = 1
        dist_block = BlockWrapper('11.5.9.63:50051', os.path.join(path, 'uint8'), delta_t, use_route=True)

        print(dist_block.get_version())
        
        print("dist_block._neurons_thrush", dist_block._neurons_thrush)
        print("dist_block.neurons_per_block", dist_block.neurons_per_block)
        run_number = 1000

        ou_mean= torch.ones(dist_block.subblk_id.shape[0]) * torch.tensor(0.4, dtype=torch.float32)    # mean of the ou current
        ou_sigma= torch.ones(dist_block.subblk_id.shape[0]) * torch.tensor(0.15, dtype=torch.float32)    # std. dev. of the ou current
        ou_tau = torch.ones(dist_block.subblk_id.shape[0]) * torch.tensor(4, dtype=torch.float32)  # time constant of the ou current
        dist_block.update_ou_background_stimuli(dist_block.subblk_id, ou_tau.cuda(), ou_mean.cuda(), ou_sigma.cuda())
        #for j, (f) in enumerate(dist_block.run(4000, freqs=True, freq_char=False, iou=True, t_steps=10)):
        #  pass
        
        freqs = torch.empty([run_number, dist_block.subblk_id.shape[0]], dtype=torch.int32).cuda()
        for j, (f) in enumerate(dist_block.run(run_number, freqs=True, freq_char=False, iou=True, t_steps=10)):
            freqs[j, :] = f

        print('mean firing rate (transition time)',
              torch.div(torch.mean(freqs.float(), dim=0) * 1000,
                        dist_block.neurons_per_subblk.float()))
        dist_block.shutdown()

if __name__ == '__main__':
    print(get_dtb_version())
    unittest.main()

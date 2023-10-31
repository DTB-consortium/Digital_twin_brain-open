# -*- coding: utf-8 -*- 
# @Time : 2022/9/26 20:50 
# @Author : lepold
# @File : lifnn.py

"""
Three schemes:
    .. 1. 1ms iteration, 1ms spike transfer
       2. 0.1ms iteration, 0.1 ms spike transfer
       3. 0.1 ms iteration, 1ms spike transfer
    All noise is added to neurons itself in each 0.1ms
"""
import torch.random

from models.block import block

seed = 10080


class NN1ms1ms(block):
    def __init__(self, node_property, w_uij, delta_t=1):
        super(NN1ms1ms, self).__init__(node_property, w_uij, delta_t)
        K, N = self.w_uij.shape[0], self.w_uij.shape[2]
        self.C = torch.ones([N], device=self.w_uij.device, dtype=torch.float32) * 0.75

    def noise_test(self):
        torch.manual_seed(seed)
        out = []
        for _ in range(5):
            out.append(torch.rand(self.w_uij.shape[2], device=self.w_uij.device))
            return torch.stack(out)


class NN01ms01ms_asusual(block):
    def __init__(self, node_property, w_uij, delta_t=0.1):
        super(NN01ms01ms_asusual, self).__init__(node_property, w_uij, delta_t)
        K, N = self.w_uij.shape[0], self.w_uij.shape[2]
        self.C = torch.ones([N], device=self.w_uij.device, dtype=torch.float32) * 0.75

    def noise_test(self):
        torch.manual_seed(seed)
        out = []
        for _ in range(5):
            out.append(torch.rand(self.w_uij.shape[2], device=self.w_uij.device))
            return torch.stack(out)

    def run(self, noise_rate=0.01, isolated=False):
        """
        the main method in this class to evolve this spike neuronal network.
        Each neuron in the network is driven by an independent background synaptic noise to maintain
        network activity. Specifically, the background synaptic noise are modelled as uncorrelated Poisson-type spike trains.
        For the generation of background noise, we implement it by replacing the poission train as a simple random train.

        Parameters
        ----------
        noise_rate: float
            the frequency of background noise.
        isolated: bool
            whether to cut off the synaptic communication in this network, but retain the background noise.
        Returns
        -------

        """
        self.t += self.delta_t
        self.active = self.update_Vi(self.delta_t)
        if not isolated:
            new_active = (torch.rand(self.w_uij.shape[2],
                                     device=self.w_uij.device) < noise_rate) + self.active  # “+” operation
        else:
            new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate)

        self.update_J_ui(self.delta_t, new_active)
        self.update_I_syn()
        self.update_t_ik_last(self.active)
        return torch.sum(self.active)


class NN01ms1ms(block):
    def __init__(self, node_property, w_uij, delta_t=0.1, chunk=10):
        super(NN01ms1ms, self).__init__(node_property, w_uij, delta_t)
        K, self.N = self.w_uij.shape[0], self.w_uij.shape[2]
        self.C = torch.ones([self.N], device=self.w_uij.device, dtype=torch.float32) * 0.75
        self.n = 0
        self.chunk = chunk
        self.spike_strength = torch.zeros((4, self.chunk, self.N), dtype=torch.float32, device=self.w_uij.device)

    def noise_test(self):
        torch.manual_seed(seed)
        out = []
        for _ in range(5):
            out.append(torch.rand(self.w_uij.shape[2], device=self.w_uij.device))
            return torch.stack(out)

    def run(self, noise_rate=0.01, isolated=False):
        """
        the main method in this class to evolve this spike neuronal network.
        Each neuron in the network is driven by an independent background synaptic noise to maintain
        network activity. Specifically, the background synaptic noise are modelled as uncorrelated Poisson-type spike trains.
        For the generation of background noise, we implement it by replacing the poission train as a simple random train.

        Parameters
        ----------
        noise_rate: float
            the frequency of background noise.
        isolated: bool
            whether to cut off the synaptic communication in this network, but retain the background noise.
        Returns
        -------

        """
        self.t += self.delta_t
        self.n += 1
        nn = self.n % self.chunk
        self.active = self.update_Vi(self.delta_t)

        if nn == 0:
            noise_and_spike = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate) + self.active
            self.spike_strength[:, nn, :] = noise_and_spike.float()
            new_active = torch.sum(self.spike_strength, dim=1)
            self.spike_strength = torch.zeros((4, self.chunk, self.N), dtype=torch.float32, device=self.w_uij.device)
            # self.update_J_ui(self.delta_t, new_active)
        else:
            xx = torch.where(self.active)[0]
            # if delay and damping
            # value = torch.exp(-(self.chunk - nn) * self.delta_t / self.tau_ui)
            # self.spike_strength[:, nn, xx] = value[:, xx]

            # esle
            self.spike_strength[:, nn, xx] = 1.

            new_active = torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate
            new_active = new_active.float()
            new_active = torch.broadcast_to(new_active, (4, new_active.shape[0]))
        self.update_J_ui(self.delta_t, new_active)
        self.update_I_syn()
        self.update_t_ik_last(self.active)

        return torch.sum(self.active)

    @staticmethod
    def bmm(H, b):
        if isinstance(H, torch.sparse.Tensor):
            return torch.stack([torch.sparse.mm(H[i], b[i].unsqueeze(1)).squeeze(1) for i in range(4)])
        else:
            return torch.matmul(H, b.unsqueeze(2)).squeeze(2)

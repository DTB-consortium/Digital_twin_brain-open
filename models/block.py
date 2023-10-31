import torch
import numpy as np


class block:
    """
    According to the determined network structure and neuron model parameter ``(LIF)``, simulate spike neural network.
     Detail ref :std:ref:`user guide`.

    A cpu code to simulate the spike network with the pytorch framework and gpu accelerating.
    In real ``large-scale`` simulation, We use CUDA accelerated ``DTB`` platform.

    Parameters
    ----------

    node_property: Tensor
        shape=(N, 23), N denotes number of neurons，23 denotes 23 attributes of LIF neuron.

    w_uij:  Tensor
        shape=(4, N, N), 4 denotes 4 different synatpic channels： AMPA, NMDA, GABAa and GABAb.

    delta_t: float
        Iteration time, unit: milliseconds.
    src: Tensor
        The default is none, otherwise, tensor is used to indicate the designated neuron and the designated firing.


    """

    def __init__(self, node_property, w_uij, delta_t=1, src=None):

        assert len(w_uij.shape) == 3
        N = w_uij.shape[1]
        K = w_uij.shape[0]

        self.w_uij = w_uij  # shape [K, N, N]
        self.src = src
        if self.src is None:
            self.src_neuron = None
            self.iter = None
        else:
            if isinstance(self.w_uij, torch.sparse.Tensor):
                non_src_neuron = torch.unique(self.w_uij.coalesce().indices()[1])
            else:
                non_src_neuron = torch.unique(self.w_uij.nonzero()[:, 1])
            idx = torch.arange(node_property.shape[0], dtype=torch.int64)
            idx = idx[torch.from_numpy(~np.isin(idx.numpy(), non_src_neuron.numpy()))].contiguous()
            self.src_neuron = idx
            assert self.src_neuron.shape[0] == self.src.shape[0]
            self.iter = 0
        self.delta_t = delta_t

        self.update_property(node_property)

        self.t_ik_last = - torch.ones([N], device=self.w_uij.device) * self.T_ref # shape [N]
        self.V_i = torch.ones([N], device=self.w_uij.device) * (self.V_th + self.V_reset)/2  # membrane potential, shape: [N]
        self.J_ui = torch.zeros([K, N], device=self.w_uij.device)  # shape [K, N]
        self.t = torch.tensor(0., device=self.w_uij.device)  # scalar

        self.update_I_syn()

    @staticmethod
    def expand(t, size):
        t = torch.tensor(t)
        shape = list(t.shape) + [1] * (len(size) - len(t.shape))
        return t.reshape(shape).expand(size)

    def update_J_ui(self, delta_t, active):
        # active shape: [N], dtype bool
        # t is a scalar
        self.J_ui = self.J_ui * torch.exp(-delta_t / self.tau_ui)
        J_ui_activate_part = self.bmm(self.w_uij, active.float()) # !!! this part can be sparse.
        self.J_ui += J_ui_activate_part
        pass

    @staticmethod
    def bmm(H, b):
        if isinstance(H, torch.sparse.Tensor):
            return torch.stack([torch.sparse.mm(H[i], b.unsqueeze(1)).squeeze(1) for i in range(4)])
        else:
            return torch.matmul(H, b.unsqueeze(0).unsqueeze(2)).squeeze(2)

    def update_I_syn(self):
        self.I_ui = self.g_ui * (self.V_ui - self.V_i) * self.J_ui
        # [K, N]            [K, N] - [K, 1]
        self.I_syn = self.I_ui.sum(dim=0)
        pass

    def update_Vi(self, delta_t):
        main_part = -self.g_Li * (self.V_i - self.V_L)
        C_diff_Vi = main_part + self.I_syn + self.I_extern_Input
        delta_Vi = delta_t / self.C * C_diff_Vi

        Vi_normal = self.V_i + delta_Vi

        # if t < self.t_ik_last + self.T_ref:
        #   V_i = V_reset
        # else:
        #   V_i = Vi_normal
        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        #print(is_not_saturated.sum())
        active = torch.ge(V_i, self.V_th)
        if self.src_neuron is not None:
            active[self.src_neuron] = self.src[:, self.iter]
            self.iter += 1
            self.V_i[self.src_neuron] = torch.where(self.src[:, self.iter],
                                                    self.V_th[self.src_neuron],
                                                    self.V_reset[self.src_neuron])
        self.V_i = torch.min(V_i, self.V_th)
        # self.V_i[active] = - 40
        return active

    def update_t_ik_last(self, active):
        self.t_ik_last = torch.where(active, self.t, self.t_ik_last)

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
            new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate) | self.active
        else:
            new_active = (torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate)
            # new_active = self.active
        self.update_J_ui(self.delta_t, new_active)
        self.update_I_syn()
        self.update_t_ik_last(self.active)

        # mean_Vi = []
        # sum_activate = []
        # for i in range(self.sub_idx.max().int() + 1):
        #     mean_Vi.append(self.V_i[self.sub_idx == i].mean())
        #     sum_activate.append(self.active[self.sub_idx == i].float().sum())
        #
        # return torch.stack(sum_activate), torch.stack(mean_Vi)

    def update_property(self, node_property):
        # update property
        # column of node_property is
        # E/I, blocked_in_stat, has_extern_Input, no_input, C, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui
        E_I, blocked_in_stat, I_extern_Input, sub_idx, C, T_ref, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui = \
            node_property.transpose(0, 1).split([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4])

        self.I_extern_Input = I_extern_Input.squeeze(0) # extern_input index , shape[K]
        self.V_ui = V_ui  # AMPA, NMDA, GABAa and GABAb potential, shape [K, N]
        self.tau_ui = tau_ui  # shape [K, N]
        self.g_ui = g_ui  # shape [K, N]
        self.g_Li = g_Li.squeeze(0)  # shape [N]
        self.V_L = V_L.squeeze(0)  # shape [N]
        self.C = C.squeeze(0)   # shape [N]
        self.sub_idx = sub_idx.squeeze(0) # shape [N]
        self.V_th = V_th.squeeze(0)   # shape [N]
        self.V_reset = V_reset.squeeze(0)  # shape [N]
        self.T_ref = T_ref.squeeze(0) # shape [N]
        return True

    def update_conn_weight(self, conn_idx, conn_weight):
        # update part of conn_weight
        # conn_idx shape is [4, X']
        # conn_weight shape is [X']
        self.w_uij[conn_idx] = conn_weight
        return True


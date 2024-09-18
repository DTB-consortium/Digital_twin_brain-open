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

    def __init__(self, node_property, w_uij, delta_t=1, i_mean=0.6, i_sigma=0.2, tau_i=10., h=0.5, g_T=0.06, V_h=-70,
                 V_T=120, tau_h_minus=20, tau_h_plus=100):

        assert len(w_uij.shape) == 3
        N = w_uij.shape[1]
        K = w_uij.shape[0]

        self.w_uij = w_uij  # shape [K, N, N]
        self.delta_t = delta_t
        self.i_mean = i_mean * torch.ones([N], device=self.w_uij.device)
        self.i_sigma = i_sigma * torch.ones([N], device=self.w_uij.device)
        self.tau_i = tau_i * torch.ones([N], device=self.w_uij.device)
        self.i_background = torch.ones([N], device=self.w_uij.device) * self.i_mean

        self.h = h * torch.ones([N], device=self.w_uij.device)
        if isinstance(g_T, np.ndarray):
            self.g_T = torch.from_numpy(g_T).cuda(self.w_uij.device)
        else:
            self.g_T = g_T * torch.ones([N], device=self.w_uij.device)
        self.I_T = torch.zeros([N], device=self.w_uij.device)
        self.V_h = V_h
        self.V_T = V_T
        self.tau_h_minus = tau_h_minus
        self.tau_h_plus = tau_h_plus

        self.update_property(node_property)

        self.t_ik_last = - torch.ones([N], device=self.w_uij.device) * self.T_ref  # shape [N]
        self.V_i = torch.ones([N], device=self.w_uij.device) * (
                    self.V_th + self.V_reset) / 2  # membrane potential, shape: [N]
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
        J_ui_activate_part = self.bmm(self.w_uij, active.float())  # !!! this part can be sparse.
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

    def update_i_background(self, delta_t):
        self.i_background += (1 - torch.exp(-delta_t / self.tau_i)) * (self.i_mean - self.i_background) + torch.sqrt(
            1 - torch.exp(-2 * delta_t / self.tau_i)) * self.i_sigma * torch.randn(self.V_i.shape[0],
                                                                                   device=self.V_i.device)
        return self.i_background

    def update_calcium_current(self):
        # T-type calcium current
        self.I_T = -self.g_T * self.h * torch.heaviside(self.V_i - self.V_h,
                                                        torch.tensor([1.], device=self.w_uij.device)) * (
                           self.V_i - self.V_T)

    def update_Vi(self, delta_t, i_background):
        main_part = -self.g_Li * (self.V_i - self.V_L)
        C_diff_Vi = main_part + self.I_syn + self.I_extern_Input + i_background  # + self.I_T

        delta_Vi = delta_t / self.C * C_diff_Vi
        Vi_normal = self.V_i + delta_Vi

        is_not_saturated = (self.t >= self.t_ik_last + self.T_ref)
        V_i = torch.where(is_not_saturated, Vi_normal, self.V_reset)
        active = torch.ge(V_i, self.V_th)

        self.V_i = torch.min(V_i, self.V_th)
        # self.V_i[active] = - 40
        return active

    def update_t_ik_last(self, active):
        self.t_ik_last = torch.where(active, self.t, self.t_ik_last)

    def run(self, debug_mode, noise_rate=None):
        """
        the main method in this class to evolve this spike neuronal network.
        Each neuron in the network is driven by an independent background synaptic noise to maintain
        network activity. Specifically, the background synaptic noise are modelled as uncorrelated Poisson-type spike trains.
        For the generation of background noise, we implement it by replacing the poission train as a simple random train.

        Parameters
        ----------
        noise_rate: float or None
            the frequency of background noise.
        debug_mode: bool
            whether to use spike with simulated noise spike.

        Returns
        -------

        """
        self.t += self.delta_t
        self.i_background = self.update_i_background(self.delta_t)
        self.active = self.update_Vi(self.delta_t, self.i_background)
        if debug_mode:
            new_active = torch.rand(self.w_uij.shape[2], device=self.w_uij.device) < noise_rate
        else:
            new_active = self.active
        self.update_J_ui(self.delta_t, new_active)
        self.update_I_syn()
        # self.update_calcium_current()
        self.update_t_ik_last(self.active)

        mean_psc = []
        mean_activate = []
        for i in range(self.sub_idx.max().int() + 1):
            mean_psc.append(self.I_ui[:, self.sub_idx == i].mean(dim=1))
            mean_activate.append(self.active[self.sub_idx == i].float().mean())

        return torch.stack(mean_activate), torch.stack(mean_psc)

    def update_property(self, node_property):
        # update property
        # column of node_property is
        # E/I, blocked_in_stat, has_extern_Input, no_input, C, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui
        E_I, blocked_in_stat, I_extern_Input, sub_idx, C, T_ref, g_Li, V_L, V_th, V_reset, g_ui, V_ui, tau_ui = \
            node_property.transpose(0, 1).split([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4])

        self.I_extern_Input = I_extern_Input.squeeze(0)  # extern_input index , shape[K]
        self.V_ui = V_ui  # AMPA, NMDA, GABAa and GABAb potential, shape [K, N]
        self.tau_ui = tau_ui  # shape [K, N]
        self.g_ui = g_ui  # shape [K, N]
        self.g_Li = g_Li.squeeze(0)  # shape [N]
        self.V_L = V_L.squeeze(0)  # shape [N]
        self.C = C.squeeze(0)  # shape [N]
        self.sub_idx = sub_idx.squeeze(0)  # shape [N]
        self.V_th = V_th.squeeze(0)  # shape [N]
        self.V_reset = V_reset.squeeze(0)  # shape [N]
        self.T_ref = T_ref.squeeze(0)  # shape [N]
        return True

    def update_conn_weight(self, conn_idx, conn_weight):
        # update part of conn_weight
        # conn_idx shape is [4, X']
        # conn_weight shape is [X']
        self.w_uij[conn_idx] = conn_weight
        return True

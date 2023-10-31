import torch
import numpy as np
import time


class BOLD:
    """
    Ballon model, transforms the fire rate (e.g., generated from :std:ref:`block`) to BOLD signal through a
    approximate convolutional dynamic equation.
    More detail, ref `here <https://www.sciencedirect.com/science/article/pii/S105381190090630X>`_.
    """

    def __init__(self, epsilon, tao_s, tao_f, tao_0, alpha, E_0, V_0,
                 delta_t=1e-3, init_f_in=None, init_s=None, init_v=None, init_q=None):
        self.epsilon = epsilon
        self.tao_s = tao_s
        self.tao_f = tao_f
        self.tao_0 = tao_0
        self.E_0 = E_0
        self.V_0 = V_0
        self.delta_t = delta_t
        self.div_alpha = 1 / alpha
        self.f_in = init_f_in
        self.s = init_s
        self.v = init_v
        self.q = init_q

    def update(self, f_str, df):
        f = self.__getattribute__(f_str)
        if f is None:
            self.__setattr__(f_str, df * self.delta_t)
        else:
            f += df * self.delta_t

    def state_update(self, w):
        if isinstance(w, np.ndarray):
            w = torch.from_numpy(w.astype(np.float)).cuda()
        self.s = w[:, :, 0].reshape(-1)
        self.q = torch.max(w[:, :, 1], torch.tensor([1e-05]).type_as(w)).reshape(-1)
        self.v = torch.max(w[:, :, 2], torch.Tensor([1e-05]).type_as(w)).reshape(-1)
        # self.f_in = torch.min(torch.max(w[:, :, 3], torch.Tensor([-15]).type_as(w)), torch.Tensor([10]).type_as(w)).reshape(-1)
        self.f_in = torch.max(w[:, :, 3], torch.Tensor([1e-05]).type_as(w)).reshape(
            -1)  # usualy setting in zhangwenyong code

    def save_temp_state(self, flag: int, write_path: str):
        state = torch.stack([self.s, self.q, self.v, self.f_in], dim=1)
        state = torch.from_numpy(state.cpu().numpy())
        if write_path[-1] == "/":
            write_path = write_path[:-1]
        np.save(write_path + f"/state_{flag}.npy", state)

    def load_temp_state(self, flag: int, load_path: str):
        if load_path[-1] == "/":
            load_path = load_path[:-1]
        state = np.load(load_path + f"/bold_temp_state_{flag}.npy")
        state = torch.from_numpy(state).cuda()
        self.s = state[:, 0]
        self.q = state[:, 1]
        self.v = state[:, 2]
        self.f_in = state[:, 3]
    def run(self, u):
        """
        the main method in this class to evolve this balloon model.

        Parameters
        ----------
        u : ndarray
            spike activity.

        Returns
        -------
        bold: ndarray
            the shape is the same as input.

        """
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u.astype(np.float)).cuda()
        if self.s is None:
            self.s = torch.zeros_like(u)
        if self.q is None:
            self.q = torch.zeros_like(u)
        if self.v is None:
            self.v = torch.ones_like(u)
        if self.f_in is None:
            self.f_in = torch.ones_like(u)

        d_s = self.epsilon * u - self.s / self.tao_s - (self.f_in - 1) / self.tao_f
        q_part = torch.where(self.f_in > 0, 1 - (1 - self.E_0) ** (1 / self.f_in), torch.ones_like(self.f_in))
        self.update('q', (self.f_in * q_part / self.E_0 - self.q * self.v ** (self.div_alpha - 1)) / self.tao_0)
        self.update('v', (self.f_in - self.v ** self.div_alpha) / self.tao_0)
        self.update('f_in', self.s)
        self.f_in = torch.where(self.f_in > 0, self.f_in, torch.zeros_like(self.f_in))
        self.update('s', d_s)

        out = self.V_0 * (7 * self.E_0 * (1 - self.q) + 2 * (1 - self.q / self.v) + (2 * self.E_0 - 0.2) * (1 - self.v))
        return out

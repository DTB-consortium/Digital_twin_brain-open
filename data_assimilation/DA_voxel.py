# -*- coding: utf-8 -*- 
# @author: yiren14mihu
# @file: DA_voxel.py
# @time: 2023/3/15 18:58

import os
import prettytable as pt
from python.dist_blockwrapper import BlockWrapper as block_gpu
from models.bold_model_pytorch import BOLD
import time
import torch
import numpy as np
import matplotlib.pyplot as mp
import argparse
import pickle

mp.switch_backend('Agg')


class DA_MODEL:
    def __init__(self, block_dict: dict, bold_dict: dict, steps=800, ensembles=100, time=400, hp_sigma=0.01,
                 bold_sigma=1e-8):
        """
        Mainly for the whole brain model consisting of cortical functional column structure and canonical E/I=4:1 structure.

       Parameters
       ----------
       block_dict : dict
           contains the parameters of the block model.
       bold_dict : dict
           contains the parameters of the bold model.
        """
        self.block = block_gpu(block_dict['ip'], block_dict['block_path'],
                               block_dict['delta_t'], use_route=False, overlap=10)
        ou_mean= torch.ones(self.block.subblk_id.shape[0]) * torch.tensor(0.66, dtype=torch.float32)    # mean of the ou current
        ou_sigma= torch.ones(self.block.subblk_id.shape[0]) * torch.tensor(0.12, dtype=torch.float32)    # std. dev. of the ou current
        ou_tau = torch.ones(self.block.subblk_id.shape[0]) * torch.tensor(10, dtype=torch.float32)  # time constant of the ou current
        self.block.update_ou_background_stimuli(self.block.subblk_id, ou_tau.cuda(), ou_mean.cuda(), ou_sigma.cuda())

        self.delta_t = block_dict['delta_t']
        self.bold_cortical = BOLD(bold_dict['epsilon'], bold_dict['tao_s'], bold_dict['tao_f'], bold_dict['tao_0'],
                                  bold_dict['alpha'], bold_dict['E_0'], bold_dict['V_0'])
        self.bold_subcortical = BOLD(bold_dict['epsilon'], bold_dict['tao_s'], bold_dict['tao_f'], bold_dict['tao_0'],
                                     bold_dict['alpha'], bold_dict['E_0'], bold_dict['V_0'])
        self.ensembles = ensembles
        self.num_populations = int(self.block.total_subblks)
        self.num_populations_in_one_ensemble = int(self.num_populations / self.ensembles)
        self.num_neurons = int(self.block.total_neurons)
        self.num_neurons_in_one_ensemble = int(self.num_neurons / self.ensembles)
        self.populations_id = self.block.subblk_id.cpu().numpy()
        self.neurons = self.block.neurons_per_subblk
        self.populations_id_per_ensemble = np.split(self.populations_id, self.ensembles)

        self.T = time
        self.steps = steps
        self.hp_sigma = hp_sigma
        self.bold_sigma = bold_sigma

    @staticmethod
    def log(val, low_bound, up_bound, scale=10):
        assert torch.all(torch.le(val, up_bound)) and torch.all(
            torch.ge(val, low_bound)), "In function log, input data error!"
        return scale * (torch.log(val - low_bound) - torch.log(up_bound - val))

    @staticmethod
    def sigmoid(val, low_bound, up_bound, scale=10):
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(val / scale)
        elif isinstance(val, np.ndarray):
            assert np.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(
                torch.from_numpy(val.astype(np.float32)) / scale).numpy()
        else:
            raise ValueError("val type is wrong!")

    @staticmethod
    def torch_2_numpy(u, is_cuda=True):
        assert isinstance(u, torch.Tensor)
        if is_cuda:
            return u.cpu().numpy()
        else:
            return u.numpy()

    @staticmethod
    def show_bold(W, bold, T, path, brain_num):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            print("show_bold" + str(i))
            fig = mp.figure(figsize=(5, 3), dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(iteration, bold[:T, i], 'r-')
            ax1.plot(iteration, np.mean(W[:T, :, i, -1], axis=1), 'b-')
            mp.fill_between(iteration, np.mean(W[:T, :, i, -1], axis=1) -
                            np.std(W[:T, :, i, -1], axis=1), np.mean(W[:T, :, i, -1], axis=1)
                            + np.std(W[:T, :, i, -1], axis=1), color='b', alpha=0.2)
            mp.ylim((0.0, 0.08))
            ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
            mp.savefig(os.path.join(path, "bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
            mp.close(fig)
        return None

    @staticmethod
    def show_hp(hp, T, path, brain_num, hp_num, hp_real=None):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            for j in range(hp_num):
                print("show_hp", i, 'and', j)
                fig = mp.figure(figsize=(5, 3), dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(iteration, np.mean(hp[:T, :, i, j], axis=1), 'b-')
                if hp_real is None:
                    pass
                else:
                    ax1.plot(iteration, np.tile(hp_real[j], T), 'r-')
                mp.fill_between(iteration, np.mean(hp[:T, :, i, j], axis=1) -
                                np.sqrt(np.var(hp[:T, :, i, j], axis=1)), np.mean(hp[:T, :, i, j], axis=1)
                                + np.sqrt(np.var(hp[:T, :, i, j], axis=1)), color='b', alpha=0.2)
                ax1.set(xlabel='observation time/800ms', ylabel='hyper parameter')
                mp.savefig(os.path.join(path, "hp" + str(i) + "_" + str(j) + ".png"), bbox_inches='tight', pad_inches=0)
                mp.close(fig)
        return None

    def initial_model(self, real_parameter, para_ind):
        """
        initialize the block model, and then determine the random walk range of hyper parameter,
        -------
        """
        raise NotImplementedError

    def evolve(self, steps=800):
        """
        evolve the block model and obtain prediction observation,
        here we apply the MC method to evolve samples drawn from initial Gaussian distribution.
        -------

        """
        raise NotImplementedError

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        use kalman filter to filter the latent state.
        -------

        """
        raise NotImplementedError


class DA_Task_V1(DA_MODEL):
    def __init__(self, block_dict: dict, bold_dict: dict, whole_brain_voxel_info: str, steps, ensembles, time, hp_sigma,
                 bold_sigma):
        super().__init__(block_dict, bold_dict, steps, ensembles, time, hp_sigma, bold_sigma)
        with open(whole_brain_voxel_info, "rb") as f:
            brain_file = pickle.load(f)
        self.device = "cuda:0"
        self.aal_cortical_in_one_ensemble = brain_file["aal_label"] - 1
        self.num_voxel_in_one_ensemble = len(self.aal_cortical_in_one_ensemble)
        self.num_cortical_voxel_in_one_ensemble = brain_file["divide_point"]
        self.num_subcortical_voxel_in_one_ensemble = self.num_voxel_in_one_ensemble - self.num_cortical_voxel_in_one_ensemble

        neurons_per_population_in_one_ensemble = self.block.neurons_per_subblk[
                                                 :self.num_populations_in_one_ensemble].cpu().numpy()
        neurons_per_population_base = np.add.accumulate(neurons_per_population_in_one_ensemble)
        self.neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
        self.neurons_per_voxel_cortical = self.block.neurons_per_subblk.float()[
                                          :self.num_cortical_voxel_in_one_ensemble * 8].reshape((-1, 8)).sum(axis=1)

        self.cortical_population_id = self.block.subblk_id.reshape((self.ensembles, self.num_populations_in_one_ensemble))[:, :self.num_cortical_voxel_in_one_ensemble * 8]
        self.cortical_population_id = self.cortical_population_id.reshape(-1)
        self.subcortical_population_id = self.block.subblk_id.reshape((self.ensembles, self.num_populations_in_one_ensemble))[:, self.num_cortical_voxel_in_one_ensemble * 8:self.num_populations_in_one_ensemble]
        self.subcortical_population_id = self.subcortical_population_id.reshape(-1)
        self.type_int = torch.tensor([0]).type_as(self.block.subblk_id).type(torch.int64)
        self.type_float = torch.tensor([0]).type_as(self.block.subblk_id).type(torch.float32)

        self.low_bound = None
        self.up_bound = None
        self.hp = None
        self.hp_log = None
        self.hp_num = None  # 4
        self.extern_current_voxel = None
        self.num_external_current_voxels = None
        self.extern_populations_name = None

    def update_hp(self, property_index=torch.tensor([11, 12, 13])):
        # gui_cortical_ones_else = torch.ones(self.cortical_population_id.shape[0])
        gui_subcortical_ones_else = torch.ones(self.subcortical_population_id.shape[0])
        for i in range(len(property_index)):
            # self.block.mul_property_by_subblk(torch.stack(
            #     torch.meshgrid(self.cortical_population_id, property_index[i].type_as(self.type_int).reshape(-1)),
            #     dim=-1).reshape(-1, 2).type_as(self.type_int), gui_cortical_ones_else.type_as(self.type_float).reshape(-1) * 0.55)
            self.block.mul_property_by_subblk(torch.stack(
                torch.meshgrid(self.subcortical_population_id, property_index[i].type_as(self.type_int).reshape(-1)),
                dim=-1).reshape(-1, 2).type_as(self.type_int), gui_subcortical_ones_else.type_as(self.type_float).reshape(-1) * 1.8)

    def conditional_info_helper(self, region=(42, 43), excited=True):
        region = np.array(region)
        if not excited:
            raise NotImplementedError
        voxels_belong_to_region = np.isin(self.aal_cortical_in_one_ensemble, region).nonzero()[0]
        # they are all cortical voxels, so have 8 populations.
        populations_index_belong_to_region = [np.array([0, 2, 4, 6]) + 8 * i for i in
                                              voxels_belong_to_region]  # only excited
        # populations_index_belong_to_region = [np.arange(8) + 8 * i for i in voxels_belong_to_region]
        populations_index_belong_to_region = np.concatenate(populations_index_belong_to_region).astype(np.int64)

        populations_name_belong_to_region = self.populations_id_per_ensemble[0][populations_index_belong_to_region]

        neurons_idx_belong_to_region = []
        for idx in populations_index_belong_to_region:
            neurons_idx_belong_to_region.append(
                np.arange(self.neurons_per_population_base[idx], self.neurons_per_population_base[idx + 1]))
        neurons_idx_belong_to_region = np.concatenate(neurons_idx_belong_to_region)
        return voxels_belong_to_region.astype(np.int64), populations_name_belong_to_region.astype(
            np.int64), neurons_idx_belong_to_region.astype(np.int64)

    def gamma_initialize(self, hp_index, population_id=None, alpha=5., beta=5.):
        population_id = self.block.subblk_id if population_id is None else population_id
        print(f"gamma_initialize {hp_index}th attribute, to value gamma({alpha}, {beta}) distribution\n")
        population_info = torch.stack(
            torch.meshgrid(population_id, torch.tensor([hp_index], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        alpha = torch.ones(len(population_id), device="cuda:0") * alpha
        beta = torch.ones(len(population_id), device="cuda:0") * beta
        out = self.block.gamma_property_by_subblk(population_info, alpha, beta)
        print(f"gamma_initializing {hp_index}th attribute is {out}")

    def initial_model(self, mean_hp_after_da, para_ind=None):
        """

        Parameters
        ----------
        mean_hp_after_da : numpy.ndarray
            mean ampa after data assimilation.
        para_ind : list
            default [10]

        Returns
        -------

        """
        start = time.time()

        if para_ind is None:
            para_ind = np.array([10], dtype=np.int64)
        # assert mean_hp_after_da.shape[0] == self.num_populations_in_one_ensemble

        if mean_hp_after_da != None:
            print("mean_hp_after_da != None")
            mean_hp_after_da = torch.from_numpy(mean_hp_after_da.astype(np.float32)).cuda()
            self.gamma_initialize(10)
            self.block.mul_property_by_subblk(torch.stack(
                torch.meshgrid(self.block.subblk_id, torch.tensor([10]).type_as(self.type_int).reshape(-1)),
                dim=-1).reshape(-1, 2).type_as(self.type_int), mean_hp_after_da.repeat(self.ensembles).type_as(self.type_float).reshape(-1))

        self.hp_num = 4
        voxel_id_belong_to_region, populations_name_belong_to_region, neurons_idx_belong_to_region = self.conditional_info_helper(
            region=(78, 79, 80, 81), excited=True)
            # region=(10, 11, 78, 79, 80, 81), excited=True)

        self.extern_current_voxel = torch.from_numpy(voxel_id_belong_to_region).cuda()
        self.extern_populations_name = torch.from_numpy(populations_name_belong_to_region).cuda()
        num_extern_neurons = len(neurons_idx_belong_to_region)  # number of total neurons have to be applied current

        record_populations_name_belong_to_region = []
        for i in range(self.ensembles):
            populations_name_belong_to_region = self.extern_populations_name + self.num_voxel_in_one_ensemble * 10 * i
            record_populations_name_belong_to_region.append(populations_name_belong_to_region)
        self.total_populations_name_belong_to_region = torch.cat(record_populations_name_belong_to_region)
        self.hp_info = torch.stack(
            torch.meshgrid(self.total_populations_name_belong_to_region, torch.tensor([2]).type_as(self.type_int).reshape(-1)),
            dim=-1).reshape(-1, 2).type_as(self.type_int)

        self.gamma_initialize(hp_index=2, population_id=self.total_populations_name_belong_to_region, alpha=5, beta=5)

        self.num_external_current_voxels = voxel_id_belong_to_region.shape[0]

        self.low_bound = np.ones((self.num_external_current_voxels, 4), dtype=np.float32) * 0.01
        self.up_bound = np.ones((self.num_external_current_voxels, 4), dtype=np.float32) * 0.3

        self.hp = np.linspace(self.low_bound, self.up_bound, num=3 * self.ensembles, dtype=np.float32)[
                  self.ensembles: - self.ensembles]
        self.hp = torch.from_numpy(self.hp).cuda()  # ensembles, num_voxel, 4
        self.low_bound, self.up_bound = torch.from_numpy(self.low_bound).cuda(), torch.from_numpy(self.up_bound).cuda()

        self.block.mul_property_by_subblk(self.hp_info, self.hp.reshape(-1))

        print(f"=================Initial DA MODEL done! cost time {time.time() - start:.2f}==========================")

    def filter(self, w_hat, bold_y_t, rate=0.5):
        ensembles, voxels, total_state_num = w_hat.shape  # ensemble, brain_n, hp_num+act+hemodynamic(total state num)
        assert total_state_num == self.hp_num + 6
        w = w_hat.clone()
        w_mean = torch.mean(w_hat, dim=0, keepdim=True)
        w_diff = w_hat - w_mean
        w_cx = w_diff[:, :, -1] * w_diff[:, :, -1]
        w_cxx = torch.sum(w_cx, dim=0) / (self.ensembles - 1) + self.bold_sigma
        temp = w_diff[:, :, -1] / (w_cxx.reshape([1, voxels])) / (self.ensembles - 1)  # (ensemble, brain)
        model_noise = self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(self.ensembles, voxels)).type_as(temp)
        w += rate * (bold_y_t + model_noise - w_hat[:, :, -1])[:, :, None] * torch.sum(
            temp[:, :, None] * w_diff.reshape([self.ensembles, voxels, total_state_num]), dim=0, keepdim=True)
        # print("min1:", torch.min(w[:, :, :self.hp_num]).item(), "max1:", torch.max(w[:, :, :self.hp_num]).item())
        w += (1 - rate) * torch.mm(torch.mm(bold_y_t + model_noise - w_hat[:, :, -1], temp.T) / voxels,
                                   w_diff.reshape([self.ensembles, voxels * total_state_num])).reshape(
            [self.ensembles, voxels, total_state_num])
        print("after filter in extern voxel", "hp_log_min:", torch.min(w[:, :, :self.hp_num]).item(), "hp_log_max:",
              torch.max(w[:, :, :self.hp_num]).item())
        return w

    def evolve(self, steps=800):
        print(f"evolve:")
        start = time.time()

        wjx_act_cortical_record = torch.zeros(steps, self.ensembles * len(self.extern_current_voxel))
        i=0

        steps = steps if steps is not None else self.steps
        for return_info in self.block.run(steps, freq_char=True, freqs=True, vmean=False, output_sample_spike=False, output_sample_vmemb=False, iou=True):
            act, *others = return_info

            act_cortical = torch.empty(size=(self.ensembles, self.num_cortical_voxel_in_one_ensemble, 8),
                                       device=self.device)

            for idx in range(self.ensembles):
                s = idx * self.num_populations_in_one_ensemble
                e = s + self.num_cortical_voxel_in_one_ensemble * 8
                act_cortical[idx] = act.float()[s: e].reshape((self.num_cortical_voxel_in_one_ensemble, 8))

            act_external_voxel = (act_cortical.sum(axis=-1) / self.neurons_per_voxel_cortical)[:,
                                 self.extern_current_voxel].reshape(-1)
            out1 = self.bold_cortical.run(torch.max(act_external_voxel, torch.tensor([1e-05]).type_as(act_cortical)))

            wjx_act_cortical_record[i, :] = act_external_voxel
            i = i+1

        print(
            f'\n\n\n\n   cortical active: mean: {wjx_act_cortical_record.mean(axis=0).mean().item()}, \nmin: {torch.min(wjx_act_cortical_record.mean(axis=0))}, \nmax: {torch.max(wjx_act_cortical_record.mean(axis=0))}'
            f', \n[:8]: {wjx_act_cortical_record.mean(axis=0)[:8]}, \n[8:16]: {wjx_act_cortical_record.mean(axis=0)[8:16]}, \n[16:24]: {wjx_act_cortical_record.mean(axis=0)[16:24]}')
        # print(
        #     f'   cortical active: {act_external_voxel.mean().item():.3f},  {act_external_voxel.min().item():.3f} ------> {act_external_voxel.max().item():.3f}')

        bold1 = torch.stack(
            [self.bold_cortical.s, self.bold_cortical.q, self.bold_cortical.v, self.bold_cortical.f_in, out1])
        print("   cortical bold range:", bold1[-1].mean().data, " | ", bold1[-1].min().data, "------>>",
              bold1[-1].max().data)
        print(f'cortical bold: \n[:8]: {bold1[-1][:8]}, \n[8:16]: {bold1[-1][8:16]}, \n[16:24]: {bold1[-1][16:24]}')

        w = torch.cat(
            (
                self.hp_log, act_external_voxel.reshape([self.ensembles, -1, 1]),
                bold1.T.reshape([self.ensembles, -1, 5])),
            dim=2)
        print(f'End evolving, totally cost time: {time.time() - start:.2f}')
        return w

    def run(self, hp_after_da_path, da_para_ind, bold_path, write_path):
        total_start = time.time()

        if hp_after_da_path == None:
            mean_hp_after_da = None
        else:
            mean_hp_after_da = np.load(hp_after_da_path)[50:, ].mean(axis=0)
            print("mean_hp_after_da.shape", mean_hp_after_da.shape)
            print("self.num_populations_in_one_ensemble", self.num_populations_in_one_ensemble)
            assert mean_hp_after_da.shape[0] == self.num_populations_in_one_ensemble

        self.initial_model(mean_hp_after_da, para_ind=da_para_ind)

        tb = pt.PrettyTable()
        tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
        tb.add_row([1, "name", "evaluation_brain", "ensembles", self.ensembles])
        tb.add_row([2, "total_populations", self.num_populations, "populations_per_ensemble",
                    self.num_populations_in_one_ensemble])
        tb.add_row([3, "total_neurons", self.num_neurons, "neurons_per_ensemble", self.num_neurons_in_one_ensemble])
        tb.add_row([4, "voxels_per_ensemble", self.num_voxel_in_one_ensemble, "populations_per_voxel", "4 & 1"])
        tb.add_row([5, "total_time", self.T, "steps", self.steps])
        tb.add_row([6, "hp_sigma", self.hp_sigma, "bold_sigma", self.bold_sigma])
        tb.add_row([7, "", "", "extern_number", self.num_external_current_voxels])
        tb.add_row([8, "initial_steps", 800, "bold_range", ""])
        tb.add_row([9, "walk_upbound", "", "walk_low_bound", ""])
        print(tb)

        # self.update_hp()

        print("self.hp", self.hp)
        print("self.low_bound", self.low_bound)
        print("self.up_bound", self.up_bound)
        print("self.hp_log", self.hp_log)

        self.hp_log = self.log(self.hp, self.low_bound, self.up_bound)
        print("self.hp_log", self.hp_log)

        # w = self.evolve(steps=800)
        for k in range(5):
            print("k", k)
            w = self.evolve(steps=800)
        bold_y = torch.from_numpy(np.load(bold_path).astype(np.float32)).cuda()[:, self.extern_current_voxel]
        # bold_y = 0.055 + 0.027 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
        bold_y = 0.05 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())

        path = write_path + '/show/'
        os.makedirs(path, exist_ok=True)

        w_save = [self.torch_2_numpy(w, is_cuda=True)]
        print("\n                 BEGIN DA               \n")
        for t in range(self.T - 1):
            step_start = time.time()
            print("\nPROCESSING || %d" % t)
            bold_y_t = bold_y[t].reshape([1, self.num_external_current_voxels])
            self.hp_log = w[:, :, :self.hp_num] + (self.hp_sigma ** 0.5 * torch.normal(0, 1, size=(
            self.ensembles, self.num_external_current_voxels, 1))).type_as(w)
            # self.ensembles, self.num_external_current_voxels, self.hp_num))).type_as(w)
            print("self.hp_log", self.hp_log.min().item(), self.hp_log.max().item())
            self.hp = self.sigmoid(self.hp_log, self.low_bound, self.up_bound)
            print("Hp_Cortical, eg: ", self.hp[0, 0, :4].data)
            print(f"\n\n {t}:debug1 cost time:\t", time.time() - step_start)
            debug2_start = time.time()

            self.block.mul_property_by_subblk(self.hp_info, self.hp.reshape(-1))

            print(f"\n\n {t}:debug2 cost time:\t", time.time() - debug2_start)

            w_hat = self.evolve()
            # w_hat[:, :, -5:] += (self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(
            #     self.ensembles, self.num_external_current_voxels, 5))).type_as(w_hat)
            debug3_start = time.time()

            w = self.filter(w_hat, bold_y_t, rate=0.5)
            self.bold_cortical.state_update(w[:, :, (self.hp_num + 1):(self.hp_num + 5)])

            w_save.append(self.torch_2_numpy(w_hat, is_cuda=True))
            print(f"\n\n {t}:debug3 cost time:\t", time.time() - debug3_start)
            print(f"\n\n {t}:step cost time:\t", time.time() - step_start)

            if t <= 3 or t % 50 == 49:
                np_w_save = np.array(w_save)
                self.show_bold(np_w_save, self.torch_2_numpy(bold_y, is_cuda=True), t+2, path, 10)
                hp_save = self.sigmoid(
                    np_w_save[:, :, :, :self.hp_num].reshape((t+2) * self.ensembles, self.num_external_current_voxels,
                                                          self.hp_num), self.torch_2_numpy(self.low_bound),
                    self.torch_2_numpy(self.up_bound))
                hp_save = hp_save.reshape((t+2, self.ensembles, self.num_external_current_voxels, self.hp_num))
                np.save(os.path.join(write_path, "hp.npy"), hp_save.mean(axis=1))
                self.show_hp(hp_save.reshape(t+2, self.ensembles, self.num_external_current_voxels, self.hp_num),
                             t+2,
                             path, 10, self.hp_num)

        print("\n                 END DA               \n")
        np.save(os.path.join(write_path, "W.npy"), w_save)
        del w_hat, w

        w_save = np.array(w_save)
        self.show_bold(w_save, self.torch_2_numpy(bold_y, is_cuda=True), self.T, path, 10)
        hp_save = self.sigmoid(
            w_save[:, :, :, :self.hp_num].reshape(self.T * self.ensembles, self.num_external_current_voxels,
                                                  self.hp_num), self.torch_2_numpy(self.low_bound),
            self.torch_2_numpy(self.up_bound))
        hp_save = hp_save.reshape((self.T, self.ensembles, self.num_external_current_voxels, self.hp_num))
        np.save(os.path.join(write_path, "hp.npy"), hp_save.mean(axis=1))
        self.show_hp(hp_save.reshape(self.T, self.ensembles, self.num_external_current_voxels, self.hp_num), self.T,
                     path, 10, self.hp_num)
        self.block.shutdown()
        print("\n\n Totally cost time:\t", time.time() - total_start)
        print("=================================================\n")


if __name__ == '__main__':
    block_dict = {"ip": "10.5.4.1:50051",
                  "block_path": "./",
                  "delta_t": 1}
    bold_dict = {"epsilon": 200,
                 "tao_s": 0.8,
                 "tao_f": 0.4,
                 "tao_0": 1,
                 "alpha": 0.2,
                 "E_0": 0.8,
                 "V_0": 0.02}

    parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--force_rebase", type=bool, default=True)
    parser.add_argument("--block_path", type=str,
                        default="/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain-newest/data/newdata/newnewdata_with_brainsteam/dti_distribution_200m_d100_with_stem/multi_module/uint8")
    parser.add_argument("--write_path", type=str,
                        default="/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain_generation/result/200md100column")
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hp_sigma", type=float, default=0.01)
    parser.add_argument("--bold_sigma", type=float, default=1e-8)
    parser.add_argument("--ensembles", type=int, default=80)

    args = parser.parse_args()
    block_dict.update(
        {"ip": args.ip, "block_path": args.block_path, })

    whole_brain_voxel_info_path ="/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam/raw_data/brain_file_1April.pickle"

    da_task = DA_Task_V1(block_dict, bold_dict, whole_brain_voxel_info_path, steps=args.steps,
                                  ensembles=args.ensembles, time=args.T, hp_sigma=args.hp_sigma,
                                  bold_sigma=args.bold_sigma)
    para_ind = torch.tensor([10])
    os.makedirs(args.write_path, exist_ok=True)
    da_task.run(hp_after_da_path=None
                , da_para_ind=para_ind,
                bold_path="/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain_generation/data_process/20220402_new_task_bold_auditory_after_zscore.npy", write_path=args.write_path)


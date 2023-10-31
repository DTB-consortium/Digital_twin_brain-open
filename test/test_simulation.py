# -*- coding: utf-8 -*- 
# @Time : 2022/8/20 21:04 
# @Author : lepold
# @File : test_simulation.py


import argparse
import os

import numpy as np
import torch

from simulation.simulation import simulation
from utils.helpers import load_if_exist, torch_2_numpy
from utils.sample import specified_sample_voxel
from utils.pretty_print import pretty_print
import time


class simulation_critical(simulation):
    def __init__(self, ip: str, block_path: str, **kwargs):
        super().__init__(ip, block_path, **kwargs)

    def sample(self, aal_region, population_base, num_sample_voxel_per_region=1, num_neurons_per_voxel=300,
               specified_info=None):
        sample_idx = load_if_exist(specified_sample_voxel, os.path.join(self.write_path, "sample_idx"),
                                   aal_region=aal_region,
                                   neurons_per_population_base=population_base,
                                   num_sample_voxel_per_region=num_sample_voxel_per_region,
                                   num_neurons_per_voxel=num_neurons_per_voxel, specified_info=specified_info)
        sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
        num_sample = sample_idx.shape[0]
        assert sample_idx.max() < self.num_neurons
        self.block_model.set_samples(sample_idx)
        load_if_exist(lambda: self.block_model.neurons_per_subblk.cpu().numpy(),
                      os.path.join(self.write_path, "blk_size"))
        self.num_sample = num_sample

        return num_sample

    def run(self, step=800, observation_time=100, hp_index=None, hp_total=None):
        """
        Run this block and save the returned block information.

        Parameters
        ----------
        step: int, default=800
             iter number in one observation time point.

        observation_time: int, default=100
            total time points, equal to the time points of bold signal.

        """

        if self.sample_option and not hasattr(self, 'num_sample'):
            raise NotImplementedError('Please set the sampling neurons first in simulation case')

        start_time = time.time()
        if hp_total is not None:
            state = "after"
            for k in hp_index:
                self.gamma_initialize(k, self.population_id)
            population_info = torch.stack(
                torch.meshgrid(self.population_id, torch.tensor(hp_index, dtype=torch.int64, device="cuda:0")),
                dim=-1).reshape((-1, 2))
            self.block_model.mul_property_by_subblk(population_info, hp_total[0].reshape(-1))
            hp_total = hp_total[1:, ]
            total_T = hp_total.shape[0]
            assert observation_time <= total_T
        else:
            state = "before"
            total_T = None

        freqs, _ = self.evolve(8000, vmean_option=False, sample_option=False, imean_option=False)
        Init_end = time.time()
        if self.print_info:
            print(
                f"mean fre: {torch.mean(torch.mean(freqs / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}")
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")

        pretty_print("Begin Simulation")
        print(f"Total time is {total_T}, we only simulate {observation_time}\n")

        bolds_out = np.zeros([observation_time, self.num_voxels], dtype=np.float32)
        for ii in range((observation_time - 1) // 50 + 1):
            nj = min(observation_time - ii * 50, 50)
            FFreqs = np.zeros([nj, step, self.num_populations], dtype=np.uint32)

            if self.vmean_option:
                Vmean = np.zeros([nj, step, self.num_populations], dtype=np.float32)
            if self.sample_option:
                Spike = np.zeros([nj, step, self.num_sample], dtype=np.uint8)
                Vi = np.zeros([nj, step, self.num_sample], dtype=np.float32)
            if self.imean_option:
                Imean = np.zeros([nj, step, self.num_populations], dtype=np.float32)

            for j in range(nj):
                i = ii * 50 + j
                t_sim_start = time.time()
                if hp_total is not None:
                    self.block_model.mul_property_by_subblk(population_info, hp_total[i].reshape(-1))
                out = self.evolve(step, vmean_option=self.vmean_option, sample_option=self.sample_option,
                                  imean_option=self.imean_option)
                FFreqs[j] = torch_2_numpy(out[0])
                out_base = 1
                if self.vmean_option:
                    Vmean[j] = torch_2_numpy(out[out_base])
                    out_base += 1
                if self.sample_option:
                    Spike[j] = torch_2_numpy(out[out_base])
                    Vi[j] = torch_2_numpy(out[out_base + 1])
                    out_base += 2
                if self.imean_option:
                    Imean[j] = torch_2_numpy(out[out_base])

                bolds_out[i, :] = torch_2_numpy(out[-1])
                t_sim_end = time.time()
                print(
                    f"{i}th observation_time, mean fre: {torch.mean(torch.mean(out[0] / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t_sim_end - t_sim_start:.1f}")
                if self.print_info:
                    stat_data, stat_table = self.block_model.last_time_stat()
                    np.save(os.path.join(self.write_path, f"stat_{i}.npy"), stat_data)
                    stat_table.to_csv(os.path.join(self.write_path, f"stat_{i}.csv"))
                    print('print_stat_time', time.time() - t_sim_end, stat_table)
            if ii==(observation_time - 1) //50:
                if self.sample_option:
                    np.save(os.path.join(self.write_path, f"spike_{state}_assim_{ii}.npy"), Spike)
                    np.save(os.path.join(self.write_path, f"vi_{state}_assim_{ii}.npy"), Vi)
                if self.vmean_option:
                    np.save(os.path.join(self.write_path, f"vmean_{state}_assim_{ii}.npy"), Vmean)
                if self.imean_option:
                    np.save(os.path.join(self.write_path, f"imean_{state}_assim_{ii}.npy"), Imean)
                np.save(os.path.join(self.write_path, f"freqs_{state}_assim_{ii}.npy"), FFreqs)
                np.save(os.path.join(self.write_path, f"bold_{state}_assim.npy"), bolds_out)

        pretty_print(f"Totally have Done, Cost time {time.time() - start_time:.2f} ")


def get_args():
    parser = argparse.ArgumentParser(description="Model simulation")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--block_dir", type=str, default=None)
    parser.add_argument("--write_path", type=str, default=None)
    parser.add_argument("--aal_info_path", type=str, default=None)
    parser.add_argument("--hp_after_da_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--num", type=str, default=None)
    args = parser.parse_args()
    return args


def rest_column_simulation(args):
    """
    simulation of resting brain at micro-column version.

    Parameters
    ----------
    args: dict
        some needed parameter in simulation object.

    Returns
    -------

    """
    block_path = os.path.join(args.block_dir, args.num, "single")
    model = simulation_critical(args.ip, block_path, dt=0.1, route_path=None, column=False, print_info=False,
                                vmean_option=False,
                                sample_option=True, name=args.name, write_path=args.write_path)
    # aal_region = np.load(args.aal_info_path)['aal_region']
    aal_region = np.arange(90)
    population_base = np.load(os.path.join(args.block_dir, "supplementary_info", "population_base.npy"))
    model.sample(aal_region, population_base)
    if args.hp_after_da_path == "None":
        hp_path = None
    else:
        hp_path = args.hp_after_da_path
    model(step=8000, observation_time=200, hp_index=[10], hp_path=hp_path)


if __name__ == "__main__":
    args = get_args()
    rest_column_simulation(args)

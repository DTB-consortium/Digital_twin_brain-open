import os
import argparse
import time
import numpy as np
import torch
from DataAssimilation import *
from simulation.simulation import simulation


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
    parser.add_argument("--task", type=str, default="rest")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--is_cuda", type=bool, default=True)
    parser.add_argument("--block_path", type=str, default=None)
    parser.add_argument("--bold_path", type=str, default=None)
    parser.add_argument("--path_out", type=str, default=None)
    parser.add_argument("--bold_idx_path", type=str, default=None)
    parser.add_argument("--para_ind", type=str, default="10 12")
    parser.add_argument("--hp_range_rate", type=str, default="4 2")
    parser.add_argument("--solo_rate", type=float, default=0.5)
    parser.add_argument("--noise_rate", type=float, default=0.01)
    parser.add_argument("--gui_alpha", type=int, default=5)
    parser.add_argument("--I_alpha", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--hp_sigma", type=float, default=1)
    parser.add_argument("--bold_sigma", type=float, default=1e-8)
    parser.add_argument("--ensembles", type=int, default=100)
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--gui_real", type=str, default="6.1644921e-03 8.9986715e-04 2.9690875e-02 1.9053727e-03")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--I_ext", type=float, default=1.)
    parser.add_argument("--task_bold_path", type=str, default=None)
    parser.add_argument("--w_path", type=str, default=None)
    parser.add_argument("--gui_label", type=str, default=None)
    parser.add_argument("--gui_path", type=str, default=None)
    parser.add_argument("--I_path", type=str, default=None)
    return parser.parse_args()


def regular_dict(**kwargs):
    return kwargs


def rest_voxel(args):
    # make dirs
    property_index = np.array([int(s) for s in args.para_ind.split()]).reshape(-1)
    text_para_ind = '_'.join([str(i) for i in property_index.flatten()])
    hp_range_rate = np.array([float(s) for s in args.hp_range_rate.split()]).reshape(2, -1)
    text_hp_range_rate = '_'.join([str(i) for i in hp_range_rate.flatten()])
    path_out = args.path_out + args.label + text_para_ind + text_hp_range_rate + str(args.solo_rate) + str(
        args.ensembles) + '/assimilation/'
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out + 'figure/', exist_ok=True)
    # get real bold signal
    bold_real = get_bold_signal(args.bold_path, b_min=0.02, b_max=0.05, lag=0)
    print('=============================prepare DA configuration=========================================')
    # da_rest
    da_rest = DataAssimilation(args.block_path, args.ip, column=False, ensemble=args.ensembles)
    da_rest.clear_mode()

    # gui boundary initialize
    gui_real = np.array([float(s) for s in args.gui_real.split()]).reshape(-1, 1)
    gui_number = len(property_index)
    gui_low = gui_real[property_index - 10] / hp_range_rate[0]  # shape = gui_number
    gui_high = gui_real[property_index - 10] * hp_range_rate[1]
    gui_low = torch.tensor(gui_low, dtype=torch.float32).reshape(1, gui_number)
    gui_high = torch.tensor(gui_high, dtype=torch.float32).reshape(1, gui_number)
    # da_rest initialize
    da_gui = da_rest.hp_random_initialize(gui_low, gui_high, gui_pblk=True)
    da_rest.hp_index2hidden_state()
    da_rest.da_property_initialize(property_index, args.gui_alpha, da_gui)
    for _ in range(5):
        da_rest.get_hidden_state()
    # da_rest run
    bold_real = torch.cat((bold_real, bold_real[:, :296]), dim=1)
    da_rest.da_rest_run(bold_real, path_out)


def rest_laminar(args):
    # make dirs
    property_index = np.array([int(s) for s in args.para_ind.split()]).reshape(-1)
    text_para_ind = '_'.join([str(i) for i in property_index.flatten()])
    hp_range_rate = np.array([float(s) for s in args.hp_range_rate.split()]).reshape(2, -1)
    text_hp_range_rate = '_'.join([str(i) for i in hp_range_rate.flatten()])
    path_out = args.path_out + args.label + text_para_ind + text_hp_range_rate + str(args.solo_rate) + str(
        args.ensembles) + '/assimilation/'
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out + 'figure/', exist_ok=True)
    # get real bold signal
    bold_real = get_bold_signal(args.bold_path, b_min=0.02, b_max=0.05, lag=0)
    print('=============================prepare DA configuration=========================================')
    # da_rest
    da_rest = DataAssimilation(args.block_path, args.ip, column=True, ensemble=args.ensembles)
    da_rest.clear_mode()

    # gui boundary initialize
    gui_real = np.array([[0.00659512, 0.00093751, 0.1019024, 0.00458985],
                         [0.01381911, 0.00196363, 0.18183651, 0.00727698],
                         [0.00754673, 0.00106148, 0.09852575, 0.00431849],
                         [0.0134587, 0.00189199, 0.15924831, 0.00651926],
                         [0.00643689, 0.00091055, 0.10209763, 0.00444712],
                         [0.01647443, 0.00234132, 0.21505809, 0.00796669],
                         [0.00680198, 0.00095797, 0.06918744, 0.00324963],
                         [0.01438906, 0.00202573, 0.14674303, 0.00587307],
                         [0.00618016, 0.00086915, 0.07027743, 0.00253291]], dtype=np.float64).T
    gui_number = len(property_index)
    gui_low = gui_real[property_index - 10] / hp_range_rate[0]  # shape = gui_number
    gui_high = gui_real[property_index - 10] * hp_range_rate[1]
    gui_low = torch.tensor(gui_low, dtype=torch.float32).reshape(1, gui_number)
    gui_high = torch.tensor(gui_high, dtype=torch.float32).reshape(1, gui_number)
    # da_rest initialize
    da_gui = da_rest.hp_random_initialize(gui_low, gui_high, gui_pblk=True)
    path_cortical_or_not='/public/home/ssct004t/project/zenglb/Digital_twin_brain/data_assimilation/path_cortical_or_not.npy'
    da_rest.hp_index2hidden_state(path_cortical_or_not)
    da_rest.da_property_initialize(property_index, args.gui_alpha, da_gui)
    da_rest.get_hidden_state()
    # da_rest run
    bold_real = torch.cat((bold_real, bold_real[:, :296]), dim=1)
    da_rest.da_rest_run(bold_real, path_out)


def task_demo(args):
    pass


def rest_simulation(args):
    property_index = np.array([int(s) for s in args.para_ind.split()]).reshape(-1)
    path_out = args.path_out + args.label
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out + '/figure/', exist_ok=True)
    bold_real = get_bold_signal(args.bold_path, b_min=0.02, b_max=0.05, lag=0)
    da_simulation = simulation(args.ip, args.block_path, dt=1, column=False, print_info=True, write_path=path_out)
    da_simulation.clear_mode()
    da_simulation.block_model.print_stat = True
    da_simulation.print_info = True
    hp_after_da = np.load(args.path_out + args.gui_path)
    #hp_after_da = np.array([float(s) for s in args.gui_real.split()]).reshape(-1)
    #hp_after_da = np.tile(hp_after_da, [50, da_simulation.num_populations, 1])[:, :, property_index-10]
    observation_time, num_da_population_pblk, hp_num = hp_after_da.shape
    print(observation_time, num_da_population_pblk, hp_num)
    hp_after_da = torch.from_numpy(hp_after_da.astype(np.float32)).cuda()
    da_simulation.run(step=800, observation_time=observation_time-1, hp_index=property_index, hp_total=hp_after_da)
    da_simulation.block_model.shutdown()


if __name__ == "__main__":
    args = get_args()
    if args.task == 'rest':
        rest_voxel(args)
    if args.task == 'task':
        task_demo(args)
    if args.task == 'rest simulation':
        rest_simulation(args)

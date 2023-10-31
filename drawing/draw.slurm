#!/bin/bash
#SBATCH -J simulation_zlb
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
#SBATCH --exclude=e04r1n03
##SBATCH --ntasks-per-node=1


date
export PYTHONPATH=/public/home/ssct004t/project/zenglb/Digital_twin_brain
module switch compiler/rocm/4.0.1
mpirun python draw.py \
  --write_path="../data/subject1/rest_da_ampa/simulation_aug_13th/fig" \
  --log_path="../data/subject1/rest_da_ampa/simulation_aug_13th/spike_after_assim_1.npy" \
  --freqs="../data/subject1/rest_da_ampa/simulation_aug_13th/freqs_after_assim_1.npy" \
  --block_size="../data/subject1/rest_da_ampa/simulation_aug_13th/blk_size.npy" \
  --sample_idx="../data/subject1/rest_da_ampa/simulation_aug_13th/sample_idx.npy" \
  --name_path="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/aal_names.mat" \
  --bold_path="../data/subject1/rest_da_ampa/simulation_aug_13th/bold_after_assim.npy" \
  --real_bold_path="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz" \
  --vmean_path="../data/subject1/rest_da_ampa/simulation_aug_13th/vmean_after_assim_1.npy" \
  --vsample="../data/subject1/rest_da_ampa/simulation_aug_13th/vi_after_assim_1.npy"
date
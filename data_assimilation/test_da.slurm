#!/bin/bash
#SBATCH -J wy36da
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
#SBATCH -x j10r2n07,e13r4n02,b01r2n17,e13r1n02

date
export PYTHONPATH=$PYTHONPATH:/public/home/ssct004t/project/wenyong36/Digital_twin_brain
export PYTHONPATH=$PYTHONPATH:/public/home/ssct004t/project/wenyong36/Digital_twin_brain/cuda/python
module switch compiler/rocm/4.0.1
mpirun python3 DataAssimilation_demo.py \
  --task='rest' \
  --ip=11.5.4.15:50051 \
  --bold_path='/public/home/ssct004t/project/wenyong36/dti_voxel_subfeng_1e8_22703/single/rest_state_afterzscoreV2.npy' \
  --block_path='/public/home/ssct004t/project/yeleijun/Digital_twin_brain/PD-sub401/dti_distribution_100m_d100_w0_1/ensembles' \
  --path_out='/public/home/ssct004t/project/wenyong36/Digital_twin_brain/toymodel/' \
  --para_ind='10' \
  --gui_real='0.00618016 8.9986715e-04 0.07027743 1.9053727e-03' \
  --bold_sigma=1e-8 \
  --hp_sigma=1 \
  --hp_range_rate='4 2' \
  --noise_rate=0.01\
  --ensembles=30 \
  --gui_alpha=5 \
  --label='DTB_restfengvoxel_b25_b8_h1_a5'

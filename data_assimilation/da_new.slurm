#!/bin/bash
#SBATCH -J wjx36_da
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
#SBATCH --exclude=e04r1n03
##SBATCH --ntasks-per-node=1

date
export PYTHONPATH=$PYTHONPATH:/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain_generation
export PYTHONPATH=$PYTHONPATH:/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain_generation/cuda_v1point2_april1/python
module rm compiler/rocm/2.9
module switch compiler/rocm/4.0.1
mpirun python DA_voxel.py \
  --ip=11.5.4.1:50051 \
  --block_path='/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/newnewdata_with_brainsteam/dti_distribution_200m_d100_1April/multi_module/uint8' \
  --write_path='/public/home/ssct004t/project/wangjiexiang/Digital_twin_brain_generation/result_april20230403/DEBUG/200md100column_20_rest_new/78_79_80_81_auditory_570_1e-8_1.21_45_0.05_0.08_0.01_0.3_0.5/'\
  --T=570 \
  --bold_sigma=1e-8 \
  --hp_sigma=1.21 \
  --ensembles=45 \

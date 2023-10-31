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
write_path="/public/home/ssct004t/project/zenglb/Digital_twin_brain/data/subject1/rest_da_ampa/simulation_oct13th"
mkdir -p ${write_path}
now=$(date +"%y%m%d-%H%M")
log_path="$write_path/$now.out"

export PYTHONPATH=/public/home/ssct004t/project/zenglb/Digital_twin_brain:/public/home/ssct004t/project/zenglb/Digital_twin_brain/cuda/python
module switch compiler/rocm/4.0.1
mpirun python test_simulation.py \
  --ip=11.5.4.2:50051 \
  --block_dir="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/test/laminar_structure_whole_brain_include_subcortical/200m_structure_d100" \
  --write_path=${write_path} \
  --name="simulation_after_da" \
  --hp_after_da_path='/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/da_simulation/da_rest_brain_release/hp.npy' \
  --aal_info_path='/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz' \
> ${log_path}


#!/bin/bash
#SBATCH -J wy_server
#SBATCH -p kshdexclu04

#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=dcu:4

#SBATCH -N 151
#SBATCH -x j10r2n07,e13r4n02,b01r2n17
##SBATCH --exclude=./black_list
##SBATCH --exclude=./black_list_jmd
##SBATCH --nodelist=./white_list

#SBATCH -o log/%j.o
#SBATCH -e log/%j.e

#export OMP_NUM_THREADS=1
#echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

#export LOOPMAX=100000
#echo "use mpirun, loop=$LOOPMAX"
#export UCX_LOG_LEVEL=debug

mkdir -p log/$SLURM_JOB_ID
mkdir -p log/$SLURM_JOB_ID/dmesg
mkdir -p log/$SLURM_JOB_ID/debug
mkdir -p log/$SLURM_JOB_ID/output

dmesg_log=log/$SLURM_JOB_ID/dmesg
debug_log=log/$SLURM_JOB_ID/debug
output_log=log/$SLURM_JOB_ID/output
hostfile_path=log/$SLURM_JOB_ID/hostfile

srun hostname |sort |uniq -c |awk '{if(NR==1){printf "%s slots=1\n",$2}else{printf "%s slots=4\n",$2}}' > ${hostfile_path}

module rm compiler/rocm/3.3
module rm compiler/rocm/2.9
module rm compiler/rocm/3.9
module rm compiler/rocm/3.5
module add compiler/cmake/3.15.6
module add compiler/rocm/4.0.1
#conda activate pytorch-1.9

date
mpirun --bind-to none --hostfile ${hostfile_path} --mca pml ucx --mca osc ucx \
       -x DMESG_LOG=${dmesg_log} \
       -x DEBUG_LOG=${debug_log} \
       -x OUTPUT_PATH=${output_log} \
       ../cuda/tools/dist_simulator.sh

date
#done

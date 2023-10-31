#!/bin/bash
#SBATCH -J setup
#SBATCH -p kshdexclu04
##SBATCH -p xahdexclu03
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
##SBATCH --mem-per-cpu=90G

date
module purge
module load compiler/devtoolset/7.3.1
#module load compiler/rocm/4.0.1
module load compiler/rocm/dtk-22.10.1 
module load mpi/hpcx/2.11.0/gcc-7.3.1
#module load mpi/hpcx/2.4.1-gcc-7.3.1
module add compiler/cmake/3.15.6
#unset C_INCLUDE_PATH
#unset CPLUS_INCLUDE_PATH
module list
./gencode.sh
cd 3rdparty/jsoncpp
mkdir -p build
cd build && rm -rf ./*
cmake ../ && make clean && make -j16
cd ../../../
make clean
dbg=0 make -j16 all

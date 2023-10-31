#!/bin/bash
server_file="server_dtb.slurm"
job_id=`sbatch server_dtb.slurm 2>&1 | tr -cd "[0-9]"`
echo $job_id
sleep 25s
cd log
job_out_file="${job_id}.o"
str_row_ip=`cat $job_out_file | grep 'listening' | sed -n 1p`
ip=`echo $str_row_ip | tr -cd "[0-9][.][:]"`
echo $ip
cd ../
client_file="test_simulation.slurm"
sed -in "17c \ \ --ip=${ip} \\\\" $client_file  # modify ip
#sed -in "18c \ \ --block_path=\"\/public\/home\/ssct004t\/project\/zenglb\/CriticalNN\/data\/multi_size_block\/size_${nfiles}\/single\" \\\\" $client_file  # modify write path
#sed -in "19c \ \ --write_path=\"\/public\/home\/ssct004t\/project\/zenglb\/CriticalNN\/data\/size_influence_all\/test6\/size_${nfiles}\" \\\\" $client_file  # modify write path
sbatch $client_file
sleep 5s

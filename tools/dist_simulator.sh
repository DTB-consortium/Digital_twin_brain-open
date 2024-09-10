#!/bin/bash

EXEPATH=`realpath $0`
DIR=`dirname $EXEPATH`
CMD="$DIR/dist_simulator -log $1 -use_double_percision=$2"

# for most of nodes, 4 tasks per node, 4 dcus per node
# first node, local task 0 and task 4 bind to numa-node-0
LOC_RANK=$((OMPI_COMM_WORLD_LOCAL_RANK % 4))

if [ $OMPI_COMM_WORLD_RANK -eq 0 ]; then
	export UCX_NET_DEVICES=mlx5_0:1
	export UCX_IB_PCI_BW=mlx5_0:200Gbs
        echo $CMD
	numactl --cpunodebind=0-3 --membind=0-3 ${CMD}
else
	case ${LOC_RANK} in
	[0])
	  #export HIP_VISIBLE_DEVICES=0
	  export UCX_NET_DEVICES=mlx5_0:1
	  export UCX_IB_PCI_BW=mlx5_0:50Gbs
	  export GOMP_CPU_AFFINITY='0-7'
	  numactl --cpunodebind=0 --membind=0 ${CMD}
	  ;;
	[1])
	  #export HIP_VISIBLE_DEVICES=1
	  export UCX_NET_DEVICES=mlx5_1:1
	  export UCX_IB_PCI_BW=mlx5_1:50Gbs
	  export GOMP_CPU_AFFINITY='8-15'
	  numactl --cpunodebind=1 --membind=1 ${CMD}
	  ;;
	[2])
	  #export HIP_VISIBLE_DEVICES=2
	  export UCX_NET_DEVICES=mlx5_2:1
	  export UCX_IB_PCI_BW=mlx5_2:50Gbs
	  export GOMP_CPU_AFFINITY='16-23'
	  numactl --cpunodebind=2 --membind=2 ${CMD}
	  ;;
	[3])
	  #export HIP_VISIBLE_DEVICES=3
	  export UCX_NET_DEVICES=mlx5_3:1
	  export UCX_IB_PCI_BW=mlx5_3:50Gbs
	  export GOMP_CPU_AFFINITY='24-31'
	  numactl --cpunodebind=3 --membind=3 ${CMD}
	  ;;
	esac
fi

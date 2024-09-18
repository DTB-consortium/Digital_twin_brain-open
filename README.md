# Digital Twin Brain

## Overview

The whole brain neuronal network model presents the computational basis of the Digital Twin Brain (DTB) and is composed of two components: the basic computing units and the network structure.

The basic computing units of the DTB are neurons and synapses, and the spike signal transmitted between neurons by synapses are action potentials, i.e., spikes. Each neuron model receives the postsynaptic currents as the input and describes the generating scheme of the time points of the action potentials as the output. The synapses have different models due to the diverse neurotransmitter receptors.

The computational neuron is an integral unit that receives presynaptic spikes from synapses as the input and generates spike trains as the output postsynaptic currents.

The network model gives the synaptic interactions between neurons by a directed multiplex graph. Structural MRI images (i.e., diffusion-weighted data and T1-weighted data) from biological brains are used to indirectly and partially measure the synaptic connections from neurons to neurons or from sub-regions to sub-regions.

More details can be found in the [Read the Docs](https://readthedocs.org/projects/dtb-open/), and the old code can be found on [GitLab](https://gitlab.com/lu_seminar/spliking_nn_for_brain_simulation).

## Repository Contents

### Python APIs
- **python**: This directory includes two key components: network model generation and the Python API interface for HIP programs.
  - **test_generation.py**: The primary file for network model generation. The main function, **test_generate_regional_brain**, takes user-provided gray matter density and DTI data, along with the specified average neuron in-degree and total neuron count, to generate a network with the desired properties.
    - Process Overview:
      - The function processes the gray matter density and DTI data into a connection probability matrix between voxels, as well as the relative proportions of neurons and average in-degrees per voxel.
      - The **generate_map_split_only_size** function in **generate_map.py** optimizes these matrices on the GPU, minimizing data traffic by reducing GPU connections.
      - The **connect_for_multi_sparse_block** function in **make_block.py** converts the optimized voxel-level matrices into population-level matrices.
      - Finally, the **merge_dti_distribution_block** function in **make_block.py** produces a network with the specified properties, including connectivity structures and neuron parameter attributes.
  - **dist_blockwrapper.py**: Implements the Python gRPC SNN client. It defines the **BlockWrapper** class, which takes the following inputs: block_path (str, directory storing the block.npz), ip (str, server listening address), delta_t (Euler numerical iteration precision), and use_route (whether to use the routing algorithm). The output is a class that provides a C++ interface.
- **test**: This directory contains two sample codes: 
  - **test_demo.py**: Demonstrates numerical simulations using an existing network model. It takes as input the network file path, server address, iteration precision, and routing algorithm choice, and outputs the statistical characteristics (e.g., firing rate) after a period of numerical iteration.
  - **modify_property.py**: Provides an example of modifying neuron parameter attributes in an existing network model. It inputs the network file path, the index of the neuron attribute to be modified, and the new value, and outputs the updated network.
- **data_assimilation**: This directory contains an example implementation of the Voxel-wise Diffusion Hierarchical Mesoscale Data Assimilation (Vw-dHMDA) method.
  - **DA_voxel.py**: This script manages data assimilation for task-based fMRI data. It defines the **DA_Task_V1** class, which takes the network path, server listening address, iteration precision, routing algorithm usage, and assimilation parameters (e.g., hp_sigma, bold_sigma) as inputs, and outputs a class that implements the assimilation algorithm.
  - **da_new.slurm** and **server_dtb.slurm**: These SLURM scripts are used to submit Python-based and HIP-based simulation tasks, respectively.
  - **auto_da.sh**: This script executes the entire assimilation process.
- **models**:This directory contains Python implementations of commonly used models.
  - **block_new.py**: This script provides a Python implementation of the leaky integrate-and-fire (LIF) model. It takes as input a generated network (including the connectivity matrix and neuron attribute matrix), iteration precision, and model parameters (e.g., mean and variance of background OU current). The script outputs a class that performs numerical simulations of the LIF model in Python.
  - **bold_model_pytorch.py**: This script implements the Balloon model, which converts firing rates into BOLD signals using an approximate convolutional dynamic equation. It takes as input the sampling period and Balloon model parameters, and outputs a class that generates BOLD signals.


### C++ Code
- The directories **include**, **protos**, **src**, **tools**, along with the files **gencode.sh**, **Makefile**, and **setup.slurm**, contain the code and scripts necessary for running numerical simulations on the HIP platform.

## Requirements
- **AMD ROCm software** (v4.0+)
  - Required for compiling with ROCm. 
  - Supported only on Linux systems. 
- **NVIDIA HPC-X software toolkit** (v2.4.1+)
- **gRPC** (v1.25.0+)
  - Supports both C++ and Python.
- **Protobuf** (v3.8.0+)
- **PyTorch** (v1.9+)
- **NumPy** (v1.19.2+)
- **PrettyTable** (v2.1.0+)
- **Matplotlib** (v3.4.2+)
- **argparse** (v1.1+)

## Contact
If you have any questions or concerns, feel free to contact us at [dtb.fudan@gmail.com](mailto:dtb.fudan@gmail.com).

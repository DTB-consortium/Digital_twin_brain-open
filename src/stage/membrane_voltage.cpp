#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T>
static __global__ void init_membrane_voltage_kernel(const T* v_ths,
														const T* v_rsts,
														const unsigned int n,
														T* v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		v_membranes[i] = (v_ths[i] + v_rsts[i]) / 2;
	}
}


template<typename T>
void init_membrane_voltage_gpu(const T* v_ths,
									const T* v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(init_membrane_voltage_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_ths,
					v_rsts,
					n,
					v_membranes);
	HIP_POST_KERNEL_CHECK("init_membrane_voltage_kernel");
}

template<typename T>
static __global__ void reset_membrane_voltage_kernel(const T*  v_rsts,
													const unsigned int n,
													T* v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		v_membranes[i] = v_rsts[i];
	}
}

template<typename T>
void reset_membrane_voltage_gpu(const T*  v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(reset_membrane_voltage_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_rsts,
					n,
					v_membranes);
	HIP_POST_KERNEL_CHECK("reset_membrane_voltage_kernel");
}

template<typename T>
static __global__ void update_membrane_voltage_for_input_kernel(const T* __restrict__ v_rsts,
																		const T* __restrict__ v_ths,
																		const unsigned char* __restrict__ f_actives,
																		const unsigned int n,
																		T* __restrict__ v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{
		unsigned char fi = f_actives[i];
		v_membranes[i] = fi ? v_ths[i] : v_rsts[i];
	}
}

template<typename T>
void update_membrane_voltage_for_input_gpu(const T* v_rsts,
										const T* v_ths,
										const unsigned char* f_actives,
										const unsigned int n,
										T* v_membranes,
										hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_membrane_voltage_for_input_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_rsts,
					v_ths,
					f_actives,
					n,
					v_membranes);
	HIP_POST_KERNEL_CHECK("update_membrane_voltage_for_input_kernel");
}

template<typename T>
__global__ void update_membrane_voltage_kernel(const T* __restrict__ i_synaptics,
													const T* __restrict__ i_ext_stimuli,
													const T* __restrict__ v_rsts,
													const T* __restrict__ v_ths,
													const T* __restrict__ c_membrane_reciprocals,
													const T* __restrict__ v_leakages,
													const T* __restrict__ g_leakages,
													const T* __restrict__ t_refs,
													const unsigned int n,
													const T delta_t,
													const int t_steps,
													unsigned char* __restrict__ f_actives,
													int* __restrict__ t_actives,
													T* __restrict__ v_membranes)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{	
  		T vi;
		int ti = t_actives[i];
		unsigned char fi = f_actives[i];
		unsigned char v_changed = 0x00;
		unsigned char f_changed = 0x00;
		unsigned char t_changed = 0x00;

		if(ti < 0)
		{
			if(fi)
			{
				T v_rst = v_rsts[i];
				vi = v_rst;
				fi = 0x00;
				v_changed = 0x01;
				f_changed = 0x01;
			}
			
			ti++;
			t_changed = 0x01;
		}
		else 
		{
			vi = v_membranes[i];
			T g_leak = g_leakages[i];
			T v_leak = v_leakages[i];
			T i_synap = i_synaptics[i];
			T i_ext = i_ext_stimuli[i];
			T c_memb = c_membrane_reciprocals[i];
			T vth = v_ths[i];
			T t_ref = t_refs[i];
			
			T cvi = g_leak * (v_leak - vi);
			cvi += i_synap;
			cvi += i_ext;
			vi += delta_t * c_memb * cvi;
			
			if(vi >= vth)
			{
				vi = vth;
				ti = 1 - (int)(t_ref) * t_steps;
				fi = 0x01;
				f_changed = 0x01;
				t_changed = 0x01;
			}

			v_changed = 0x01;
			
		}

		if(f_changed)
			f_actives[i] = fi;

		if(t_changed)
			t_actives[i] = ti;

		if(v_changed)
			v_membranes[i] = vi;
		
	}
}

template<typename T>
void update_membrane_voltage_gpu(const T* i_synaptics,
										const T* i_ext_stimuli,
										const T* v_rsts,
										const T* v_ths,
										const T* c_membrane_reciprocals,
										const T* v_leakages,
										const T* g_leakages,
										const T* t_refs,
										const unsigned int n,
										const T delta_t,
										const int t_steps,
										unsigned char* f_actives,
										int* t_actives,
										T* v_membranes,
										hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_membrane_voltage_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					i_synaptics,
					i_ext_stimuli,
					v_rsts,
					v_ths,
					c_membrane_reciprocals,
					v_leakages,
					g_leakages,
					t_refs,
					n,
					delta_t,
					t_steps,
					f_actives,
					t_actives,
					v_membranes);
	HIP_POST_KERNEL_CHECK("update_membrane_voltage_kernel");
}


template void init_membrane_voltage_gpu<float>(const float* v_ths,
											const float* v_rsts,
											const unsigned int n,
											float* v_membranes,
											hipStream_t stream);

template void init_membrane_voltage_gpu<double>(const double* v_ths,
											const double* v_rsts,
											const unsigned int n,
											double* v_membranes,
											hipStream_t stream);


template void reset_membrane_voltage_gpu<float>(const float*  v_rsts,
											const unsigned int n,
											float* v_membranes,
											hipStream_t stream);

template void reset_membrane_voltage_gpu<double>(const double*  v_rsts,
											const unsigned int n,
											double* v_membranes,
											hipStream_t stream);

template void update_membrane_voltage_for_input_gpu<float>(const float* v_rsts,
														const float* v_ths,
														const unsigned char* f_actives,
														const unsigned int n,
														float* v_membranes,
														hipStream_t stream);

template void update_membrane_voltage_for_input_gpu<double>(const double* v_rsts,
														const double* v_ths,
														const unsigned char* f_actives,
														const unsigned int n,
														double* v_membranes,
														hipStream_t stream);

template void update_membrane_voltage_gpu<float>(const float* i_synaptics,
												const float* i_ext_stimuli,
												const float* v_rsts,
												const float* v_ths,
												const float* c_membrane_reciprocals,
												const float* v_leakages,
												const float* g_leakages,
												const float* t_refs,
												const unsigned int n,
												const float delta_t,
												const int t_steps,
												unsigned char* f_actives,
												int* t_actives,
												float* v_membranes,
												hipStream_t stream);

template void update_membrane_voltage_gpu<double>(const double* i_synaptics,
												const double* i_ext_stimuli,
												const double* v_rsts,
												const double* v_ths,
												const double* c_membrane_reciprocals,
												const double* v_leakages,
												const double* g_leakages,
												const double* t_refs,
												const unsigned int n,
												const double delta_t,
												const int t_steps,
												unsigned char* f_actives,
												int* t_actives,
												double* v_membranes,
												hipStream_t stream);


}//namespace dtb

#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

#if defined(__HIP_PLATFORM_HCC__)
#define LOG_NUM_OF_WARP_SIZE 6
#elif defined(__HIP_PLATFORM_NVCC__)
#define LOG_NUM_OF_WARP_SIZE 5
#endif


template<typename T2>
static __global__ void init_presynaptic_voltage_kernel(const unsigned int n,
														const T2 val,
														T2* j_u_presynaptics)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		j_u_presynaptics[i] = val;
	}
}

template<typename T2>
void init_presynaptic_voltage_gpu(const unsigned int n,
									const T2 val,
									T2* j_u_presynaptics,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(init_presynaptic_voltage_kernel<T2>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					n,
					val,
					j_u_presynaptics);
	HIP_POST_KERNEL_CHECK("init_presynaptic_voltage_kernel");
}

//a neuron per block for different excitory and inhibitory connections
template<typename T, typename T2> 
static __global__ void update_presynaptic_voltage_kernel(const T2* __restrict__ tao_ex_constants,
															const T2* __restrict__ tao_in_constants,
															const unsigned int n,
															T2* __restrict__ j_ex_presynaptics,
															T2* __restrict__ j_ex_presynaptic_deltas,
															T2* __restrict__ j_in_presynaptics,
															T2* __restrict__ j_in_presynaptic_deltas)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
 	const unsigned int gridSize = blockDim.x * gridDim.x;
		
  	for(unsigned int i = idx; i < n; i += gridSize)
  	{
		T2 ji_ex = j_ex_presynaptics[i];
		T2 tao_ex = tao_ex_constants[i];
		if(NULL != j_ex_presynaptic_deltas)
		{
		 	T2 ji_ex_delta = j_ex_presynaptic_deltas[i];
			ji_ex.x = ji_ex.x * tao_ex.x + ji_ex_delta.x;
			ji_ex.y = ji_ex.y * tao_ex.y + ji_ex_delta.y;
		}
		else
		{
			ji_ex.x = ji_ex.x * tao_ex.x;
			ji_ex.y = ji_ex.y * tao_ex.y;
		}
		
		
		T2 ji_in = j_in_presynaptics[i];
		T2 tao_in = tao_in_constants[i];
		if(NULL != j_in_presynaptic_deltas)
		{
		 	T2 ji_in_delta = j_in_presynaptic_deltas[i];
			ji_in.x = ji_in.x * tao_in.x + ji_in_delta.x;
			ji_in.y = ji_in.y * tao_in.y + ji_in_delta.y;
		}
		else
		{
			ji_in.x = ji_in.x * tao_in.x;
			ji_in.y = ji_in.y * tao_in.y;
		}
		
		j_ex_presynaptics[i] = ji_ex;
		j_in_presynaptics[i] = ji_in;
	}
}

template<typename T, typename T2> 
void update_presynaptic_voltage_gpu(const T2* tao_ex_constants,
										const T2* tao_in_constants,
										const unsigned int n,
										T2* j_ex_presynaptics,
										T2* j_ex_presynaptic_deltas,
										T2* j_in_presynaptics,
										T2* j_in_presynaptic_deltas,
										hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_presynaptic_voltage_kernel<T,T2>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					tao_ex_constants,
					tao_in_constants,
					n,
					j_ex_presynaptics,
					j_ex_presynaptic_deltas,
					j_in_presynaptics,
					j_in_presynaptic_deltas);
	HIP_POST_KERNEL_CHECK("update_presynaptic_voltage_base_kernel");
	
}

template<typename T, typename T2, unsigned int warpsPerBlock>
static __global__ void update_presynaptic_voltage_inner_kernel(const unsigned int* __restrict__ rowptrs,
																	const unsigned int* __restrict__ colinds,
																	const DataType weight_type,
																	const char* __restrict__ w_synaptics,
																	const unsigned char* __restrict__ connkinds,
																	const unsigned int n,
																	const unsigned int* __restrict__ f_indices,
																	const unsigned char* __restrict__ f_actives,
																	T2* __restrict__ j_ex_presynaptics,
																	T2* __restrict__ j_in_presynaptics)
{
	const unsigned int laneId = threadIdx.x & (warpSize - 1);
	const unsigned int idx = blockIdx.x * warpsPerBlock + threadIdx.x / warpSize;
	const unsigned int gridSize = warpsPerBlock * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int fidx = f_indices[i];
		unsigned char fi = f_actives[fidx];
		if(fi)
		{
			unsigned int start = rowptrs[i];
			unsigned int end = rowptrs[i + 1];

			for(unsigned int j = start + laneId; j < end; j += warpSize)
			{
				//unsigned char flag = connkinds[j];
				unsigned char flag = ((reinterpret_cast<const unsigned int*>(connkinds))[j >> 5] >> (j & 31)) & 1;
				unsigned int nidx = colinds[j];
				T2 weight;
				if(is_type<double>(weight_type))
				{
					double2 w = reinterpret_cast<const double2*>(w_synaptics)[j];
					weight.x = static_cast<T>(w.x);
					weight.y = static_cast<T>(w.y);
				}
				else if(is_type<float>(weight_type))
				{
					float2 w = reinterpret_cast<const float2*>(w_synaptics)[j];
					weight.x = static_cast<T>(w.x);
					weight.y = static_cast<T>(w.y);
				}
				else if(is_type<half>(weight_type))
				{
					float2 w = __half22float2(reinterpret_cast<const half2*>(w_synaptics)[j]);
					weight.x = static_cast<T>(w.x);
					weight.y = static_cast<T>(w.y);
				}
				else if(is_type<int8_t>(weight_type))
				{
					uchar2 w = reinterpret_cast<const uchar2*>(w_synaptics)[j];
					weight.x = static_cast<T>(static_cast<float>(w.x) / 255.f);
					weight.y = static_cast<T>(static_cast<float>(w.y) / 255.f);
				}
				
				if(flag)
				{
					atomic_add<T>(&(j_in_presynaptics[nidx].x), weight.x);
					atomic_add<T>(&(j_in_presynaptics[nidx].y), weight.y);
				}
				else
				{
					atomic_add<T>(&(j_ex_presynaptics[nidx].x), weight.x);
					atomic_add<T>(&(j_ex_presynaptics[nidx].y), weight.y);
				}
			}
		}
	}
}

template<typename T, typename T2>
void update_presynaptic_voltage_inner_gpu(const unsigned int* rowptrs,
												const unsigned int* colinds,
												const DataType weight_type,
												const char* w_synaptics,
												const unsigned char* connkinds,
												const unsigned int n,
												const unsigned int* f_indices,
												const unsigned char* f_actives,
												T2* j_ex_presynaptics,
												T2* j_in_presynaptics,
												hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_presynaptic_voltage_inner_kernel<T,T2,(HIP_THREADS_PER_BLOCK >> LOG_NUM_OF_WARP_SIZE)>),
					dim3(divide_up<unsigned int>(n, (HIP_THREADS_PER_BLOCK >> LOG_NUM_OF_WARP_SIZE))),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					rowptrs,
					colinds,
					weight_type,
					w_synaptics,
					connkinds,
					n,
					f_indices,
					f_actives,
					j_ex_presynaptics,
					j_in_presynaptics);

	HIP_POST_KERNEL_CHECK("update_presynaptic_voltage_inner_kernel");
	
}

template<typename T, typename T2, unsigned int warpsPerBlock>
static __global__ void update_presynaptic_voltage_outer_kernel(const unsigned int* __restrict__ rowptrs,
																	const unsigned int* __restrict__ colinds,
																	const DataType weight_type,
																	const char* __restrict__ w_synaptics,
																	const unsigned char* __restrict__ connkinds,
																	const unsigned int* __restrict__ active_colinds,
																	const unsigned int n,
																	T2* __restrict__ j_ex_presynaptics,
																	T2* __restrict__ j_in_presynaptics)
{
	const unsigned int laneId = threadIdx.x & (warpSize - 1);
	const unsigned int idx = blockIdx.x * warpsPerBlock + threadIdx.x / warpSize;
	const unsigned int gridSize = warpsPerBlock * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int fidx = active_colinds[i];
		
		unsigned int start = rowptrs[fidx];
		unsigned int end = rowptrs[fidx + 1];
			
		for(unsigned int j = start + laneId; j < end; j += warpSize)
		{
			//unsigned char flag = connkinds[j];
			unsigned char flag = ((reinterpret_cast<const unsigned int*>(connkinds))[j >> 5] >> (j & 31)) & 1;
			unsigned int nidx = colinds[j];
			T2 weight;
			if(is_type<double>(weight_type))
			{
				double2 w = reinterpret_cast<const double2*>(w_synaptics)[j];
				weight.x = static_cast<T>(w.x);
				weight.y = static_cast<T>(w.y);
			}
			else if(is_type<float>(weight_type))
			{
				float2 w = reinterpret_cast<const float2*>(w_synaptics)[j];
				weight.x = static_cast<T>(w.x);
				weight.y = static_cast<T>(w.y);
			}
			else if(is_type<half>(weight_type))
			{
				float2 w = __half22float2(reinterpret_cast<const half2*>(w_synaptics)[j]);
				weight.x = static_cast<T>(w.x);
				weight.y = static_cast<T>(w.y);
			}
			else if(is_type<int8_t>(weight_type))
			{
				uchar2 w = reinterpret_cast<const uchar2*>(w_synaptics)[j];
				weight.x = static_cast<T>(static_cast<float>(w.x) / 255.f);
				weight.y = static_cast<T>(static_cast<float>(w.y) / 255.f);
			}
			
			if(flag)
			{
				atomic_add<T>(&(j_in_presynaptics[nidx].x), weight.x);
				atomic_add<T>(&(j_in_presynaptics[nidx].y), weight.y);
			}
			else
			{
				atomic_add<T>(&(j_ex_presynaptics[nidx].x), weight.x);
				atomic_add<T>(&(j_ex_presynaptics[nidx].y), weight.y);
			}
		}
	}
}

template<typename T, typename T2>
void update_presynaptic_voltage_outer_gpu(const unsigned int* rowptrs,
												const unsigned int* colinds,
												const DataType weight_type,
												const char* w_synaptics,
												const unsigned char* connkinds,
												const unsigned int* active_colinds,
												const unsigned int n,
												T2* j_ex_presynaptics,
												T2* j_in_presynaptics,
												hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_presynaptic_voltage_outer_kernel<T,T2,(HIP_THREADS_PER_BLOCK >> LOG_NUM_OF_WARP_SIZE)>),
					dim3(divide_up<unsigned int>(n, (HIP_THREADS_PER_BLOCK >> LOG_NUM_OF_WARP_SIZE))),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					rowptrs,
					colinds,
					weight_type,
					w_synaptics,
					connkinds,
					active_colinds,
					n,
					j_ex_presynaptics,
					j_in_presynaptics);
	HIP_POST_KERNEL_CHECK("update_presynaptic_voltage_outer_kernel");
	
}


template void init_presynaptic_voltage_gpu<float2>(const unsigned int n,
												const float2 val,
												float2* j_u_presynaptics,
												hipStream_t stream);

template void init_presynaptic_voltage_gpu<double2>(const unsigned int n,
													const double2 val,
													double2* j_u_presynaptics,
													hipStream_t stream);

template void update_presynaptic_voltage_gpu<float, float2>(const float2* tao_ex_constants,
															const float2* tao_in_constants,
															const unsigned int n,
															float2* j_ex_presynaptics,
															float2* j_ex_presynaptic_deltas,
															float2* j_in_presynaptics,
															float2* j_in_presynaptic_deltas,
															hipStream_t stream);

template void update_presynaptic_voltage_gpu<double, double2>(const double2* tao_ex_constants,
															const double2* tao_in_constants,
															const unsigned int n,
															double2* j_ex_presynaptics,
															double2* j_ex_presynaptic_deltas,
															double2* j_in_presynaptics,
															double2* j_in_presynaptic_deltas,
															hipStream_t stream);


template void update_presynaptic_voltage_inner_gpu<float, float2>(const unsigned int* rowptrs,
																const unsigned int* colinds,
																const DataType weight_type,
																const char* w_synaptics,
																const unsigned char* connkinds,
																const unsigned int n,
																const unsigned int* f_indices,
																const unsigned char* f_actives,
																float2* j_ex_presynaptics,
																float2* j_in_presynaptics,
																hipStream_t stream);

template void update_presynaptic_voltage_inner_gpu<double, double2>(const unsigned int* rowptrs,
																const unsigned int* colinds,
																const DataType weight_type,
																const char* w_synaptics,
																const unsigned char* connkinds,
																const unsigned int n,
																const unsigned int* f_indices,
																const unsigned char* f_actives,
																double2* j_ex_presynaptics,
																double2* j_in_presynaptics,
																hipStream_t stream);



template void update_presynaptic_voltage_outer_gpu<float, float2>(const unsigned int* rowptrs,
																const unsigned int* colinds,
																const DataType weight_type,
																const char* w_synaptics,
																const unsigned char* connkinds,
																const unsigned int* active_colinds,
																const unsigned int n,
																float2* j_ex_presynaptics,
																float2* j_in_presynaptics,
																hipStream_t stream);

template void update_presynaptic_voltage_outer_gpu<double, double2>(const unsigned int* rowptrs,
																const unsigned int* colinds,
																const DataType weight_type,
																const char* w_synaptics,
																const unsigned char* connkinds,
																const unsigned int* active_colinds,
																const unsigned int n,
																double2* j_ex_presynaptics,
																double2* j_in_presynaptics,
																hipStream_t stream);


}//namespace dtb


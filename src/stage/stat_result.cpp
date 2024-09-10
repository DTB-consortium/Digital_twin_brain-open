#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T, unsigned int blockSize>
__device__ void asum_block(const unsigned int pos,
							const unsigned int n,
							const unsigned int exclusive_count,
							const unsigned char* __restrict__ exclusive_flags,
							const T* __restrict__ v_membs,
							const T* __restrict__ i_synapes,
							float& v_mean,
							float& i_mean)
{
	__shared__ T s_sum1[blockSize];
	__shared__ T s_sum2[blockSize];
	
	T sum1 = static_cast<T>(0);
	T sum2 = static_cast<T>(0);
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		T vi = v_membs[idx];
		T ii = i_synapes[idx];
		sum1 += vi * (ei == 0x00 ? (T)1 : (T)0);
		sum2 += ii * (ei == 0x00 ? (T)1 : (T)0);
	}

	asum_shmem<T, T, blockSize>(s_sum1, sum1, s_sum2, sum2, threadIdx.x);

	if (threadIdx.x == 0)
	{
		v_mean = static_cast<float>(s_sum1[0] / (T)(n - exclusive_count));
		i_mean = static_cast<float>(s_sum2[0] / (T)(n - exclusive_count));
	}
}

template<typename T, typename T2, unsigned int blockSize>
__device__ void asum_block(const unsigned int pos,
							const unsigned int n,
							const unsigned int exclusive_count,
							const unsigned char* __restrict__ exclusive_flags,
							const T2* __restrict__ i_synapes,
							float& i_mean1,
							float& i_mean2)
{
	__shared__ T s_sum1[blockSize];
	__shared__ T s_sum2[blockSize];
	
	T sum1 = static_cast<T>(0);
	T sum2 = static_cast<T>(0);
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		T x = i_synapes[idx].x;
		T y = i_synapes[idx].y;
		sum1 += x * (ei == 0x00 ? (T)1 : (T)0);
		sum2 += y * (ei == 0x00 ? (T)1 : (T)0);
	}

	asum_shmem<T, T, blockSize>(s_sum1, sum1, s_sum2, sum2, threadIdx.x);

	if (threadIdx.x == 0)
	{
		i_mean1 = static_cast<float>(s_sum1[0] / (T)(n - exclusive_count));
		i_mean2 = static_cast<float>(s_sum2[0] / (T)(n - exclusive_count));
	}
}


template<unsigned int blockSize>
__device__ void asum_block_freq_int(const unsigned int pos,
							const unsigned int n,
							const unsigned char* __restrict__ exclusive_flags,
							const unsigned char* __restrict__ f_actives,
							unsigned int& f_sum)
{
	__shared__ unsigned int s_sum[blockSize];
	
	unsigned int sum = 0;
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char fi = f_actives[idx];
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		sum += static_cast<unsigned int>(fi) * (ei == 0x00 ? 1 : 0);
	}

	asum_shmem<unsigned int, blockSize>(s_sum, sum, threadIdx.x);

	if (threadIdx.x == 0)
	{
		f_sum = s_sum[0];
	}
}

template<unsigned int blockSize>
__device__ void asum_block_freq_char(const unsigned int pos,
							const unsigned int n,
							const unsigned int exclusive_count,
							const unsigned char* __restrict__ exclusive_flags,
							const unsigned char* __restrict__ f_actives,
							unsigned char& out_sum)
{
	__shared__ unsigned int s_sum[blockSize];
	
	unsigned int sum = 0;
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char fi = f_actives[idx];
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		sum += (static_cast<unsigned int>(fi) * (ei == 0x00 ? 1 : 0));
	}

	asum_shmem<unsigned int, blockSize>(s_sum, sum, threadIdx.x);

	if (threadIdx.x == 0)
	{
		out_sum = static_cast<unsigned char>(lroundf(static_cast<float>(255 * s_sum[0]) / static_cast<float>(n - exclusive_count)));
	}
}


template<typename T, unsigned int blockSize>
__device__ void asum_block(const unsigned int pos,
							const unsigned int n,
							const unsigned int exclusive_count,
							const unsigned char* __restrict__ exclusive_flags,
							const T* __restrict__ in_data,
							float& out_mean)
{
	__shared__ T s_sum[blockSize];
	
	T sum = static_cast<T>(0);
	unsigned int end = n + pos;
	//Cycle through the entire weight array of the neuron per warp.
	for (unsigned int idx = pos + threadIdx.x; idx < end; idx += blockSize)
	{
		unsigned char ei = (exclusive_flags == NULL) ? 0x00 : exclusive_flags[idx];
		T vi = in_data[idx] * (ei == 0x00 ? (T)1 : (T)0);
		sum += vi;
	}

	asum_shmem<T, blockSize>(s_sum, sum, threadIdx.x);

	if (threadIdx.x == 0)
	{
		out_mean = static_cast<float>(s_sum[0] / (T)(n - exclusive_count));
	}
}

template<unsigned int BlockSize>
__global__ void stat_freqs_char_kernel(const unsigned int* __restrict__ sub_bcounts,
									const unsigned int* __restrict__ exclusive_counts,
									const unsigned int n,
									const unsigned char* __restrict__ exclusive_flags,
									const unsigned char* __restrict__ f_actives,
									unsigned char* __restrict__ freqs)
{
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		unsigned int x = sub_bcounts[i];
		unsigned int y = sub_bcounts[i + 1] - x;
		unsigned int exclusive_count = (exclusive_counts == NULL) ? 0 : exclusive_counts[i];
		
		if(y > 0)
		{
			asum_block_freq_char<BlockSize>(x, y, exclusive_count, exclusive_flags, f_actives, freqs[i]);
			__syncthreads();
		}
  	}
}

template<unsigned int BlockSize>
__global__ void stat_freqs_int_kernel(const unsigned int* __restrict__ sub_bcounts,
									const unsigned int* __restrict__ exclusive_counts,
									const unsigned int n,
									const unsigned char* __restrict__ exclusive_flags,
									const unsigned char* __restrict__ f_actives,
									unsigned int* __restrict__ freqs)
{
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		unsigned int x = sub_bcounts[i];
		unsigned int y = sub_bcounts[i + 1] - x;
		
		if(y > 0)
		{
			asum_block_freq_int<BlockSize>(x, y, exclusive_flags, f_actives, freqs[i]);
			__syncthreads();
		}
  	}
}

template<typename T>
void stat_freqs_gpu(const unsigned int* sub_bcounts,
					const unsigned int* exclusive_counts,
					const unsigned int n,
					const unsigned char* exclusive_flags,
					const unsigned char* f_actives,
					T* freqs,
					hipStream_t stream);

template<>
void stat_freqs_gpu<unsigned char>(const unsigned int* sub_bcounts,
					const unsigned int* exclusive_counts,
					const unsigned int n,
					const unsigned char* exclusive_flags,
					const unsigned char* f_actives,
					unsigned char* freqs,
					hipStream_t stream)

{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_freqs_char_kernel<HIP_THREADS_PER_BLOCK>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_bcounts,
					exclusive_counts,
					n,
					exclusive_flags,
					f_actives,
					freqs);
	HIP_POST_KERNEL_CHECK("stat_freqs_char_kernel");	
}

template<>
void stat_freqs_gpu<unsigned int>(const unsigned int* sub_bcounts,
					const unsigned int* exclusive_counts,
					const unsigned int n,
					const unsigned char* exclusive_flags,
					const unsigned char* f_actives,
					unsigned int* freqs,
					hipStream_t stream)

{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_freqs_int_kernel<HIP_THREADS_PER_BLOCK>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_bcounts,
					exclusive_counts,
					n,
					exclusive_flags,
					f_actives,
					freqs);
	HIP_POST_KERNEL_CHECK("stat_freqs_int_kernel");	
}



template<typename T, unsigned int BlockSize>
__global__ void stat_vmeans_and_imeans_kernel(const unsigned int* __restrict__ sub_bcounts,
												const unsigned int* __restrict__ exclusive_counts,
												const unsigned int n,
												const unsigned char* __restrict__ exclusive_flags,
												const T* __restrict__ v_membranes,
												const T* __restrict__ i_synapses,
												float* __restrict__ vmeans,
												float* __restrict__ imeans)
{
	unsigned int exclusive_count;
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		unsigned int x = sub_bcounts[i];
		unsigned int y = sub_bcounts[i + 1] - x;
		exclusive_count = (exclusive_counts == NULL) ? 0 : exclusive_counts[i];
		
		if(y > 0)
		{
			if(NULL != v_membranes && NULL != i_synapses)
				asum_block<T, BlockSize>(x, y, exclusive_count, exclusive_flags, v_membranes, i_synapses, vmeans[i], imeans[i]);
			else if(NULL != v_membranes)
				asum_block<T, BlockSize>(x, y, exclusive_count, exclusive_flags, v_membranes, vmeans[i]);
			else
				asum_block<T, BlockSize>(x, y, exclusive_count, exclusive_flags, i_synapses, imeans[i]);
			__syncthreads();
		}
  	}
}

template<typename T>
void stat_vmeans_and_imeans_gpu(const unsigned int* sub_bcounts,
								const unsigned int* exclusive_counts,
								const unsigned int n,
								const unsigned char* exclusive_flags,
								const T* v_membranes,
								const T* i_synapses,
								float* vmeans,
								float* imeans,
								hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_vmeans_and_imeans_kernel<T, HIP_THREADS_PER_BLOCK>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_bcounts,
					exclusive_counts,
					n,
					exclusive_flags,
					v_membranes,
					i_synapses,
					vmeans,
					imeans);
	HIP_POST_KERNEL_CHECK("stat_vmeans_and_imeans_kernel");	
}

template<typename T, typename T2, unsigned int BlockSize>
__global__ void stat_receptor_imeans_kernel(const unsigned int* __restrict__ sub_bcounts,
												const unsigned int* __restrict__ exclusive_counts,
												const unsigned int n,
												const unsigned char* __restrict__ exclusive_flags,
												const T2* __restrict__ i_ex_synapses,
												const T2* __restrict__ i_in_synapses,
												float* __restrict__ ampa_imeans,
												float* __restrict__ nmda_imeans,
												float* __restrict__ gabaa_imeans,
												float* __restrict__ gabab_imeans)
{
	unsigned int exclusive_count;
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		unsigned int x = sub_bcounts[i];
		unsigned int y = sub_bcounts[i + 1] - x;
		exclusive_count = (exclusive_counts == NULL) ? 0 : exclusive_counts[i];
		
		if(y > 0)
		{
			asum_block<T, T2, BlockSize>(x, y, exclusive_count, exclusive_flags, i_ex_synapses, ampa_imeans[i], nmda_imeans[i]);
			asum_block<T, T2, BlockSize>(x, y, exclusive_count, exclusive_flags, i_in_synapses, gabaa_imeans[i], gabab_imeans[i]);
			__syncthreads();
		}
  	}
}


template<typename T, typename T2>
void stat_receptor_imeans_gpu(const unsigned int* sub_bcounts,
								const unsigned int* exclusive_counts,
								const unsigned int n,
								const unsigned char* exclusive_flags,
								const T2* i_ex_synapses,
								const T2* i_in_synapses,
								float* ampa_imeans,
								float* nmda_imeans,
								float* gabaa_imeans,
								float* gabab_imeans,
								hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_receptor_imeans_kernel<T, T2, HIP_THREADS_PER_BLOCK>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_bcounts,
					exclusive_counts,
					n,
					exclusive_flags,
					i_ex_synapses,
					i_in_synapses,
					ampa_imeans,
					nmda_imeans,
					gabaa_imeans,
					gabab_imeans);
	HIP_POST_KERNEL_CHECK("stat_receptor_imeans_kernel");	
}


template<typename T>
static __global__  void stat_samples_kernel(const unsigned int* __restrict__ samples,
											const unsigned int n,
											const unsigned char* __restrict__ f_actives,
											const T* __restrict__ v_membranes,
											const T* __restrict__ i_synaptics,
											const T* __restrict__ i_ou_background_stimuli,
											char* __restrict__ spikes,
											float* __restrict__ vmembs,
											float* __restrict__ isynaptics,
											float* __restrict__ ious)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = gridDim.x * blockDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int sid = samples[i];
		if(NULL != spikes)
		{
			spikes[i] = f_actives[sid];
		}
		
		if(NULL != vmembs)
		{
			vmembs[i] = static_cast<float>(v_membranes[sid]);
		}

		if(NULL != isynaptics)
		{
			isynaptics[i] = static_cast<float>(i_synaptics[sid]);
		}

		if(NULL != i_ou_background_stimuli &&
			NULL != ious)
		{
			ious[i] = static_cast<float>(i_ou_background_stimuli[sid]);
		}
	}
}

template<typename T>
void stat_samples_gpu(const unsigned int* samples,
							const unsigned int n,
							const unsigned char* f_actives,
							const T* v_membranes,
							const T* i_synaptics,
							const T* i_ou_background_stimuli,
							char* spikes,
							float* vmembs,
							float* isynaptics,
							float* ious,
							hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(stat_samples_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					samples,
					n,
					f_actives,
					v_membranes,
					i_synaptics,
					i_ou_background_stimuli,
					spikes,
					vmembs,
					isynaptics,
					ious);
	HIP_POST_KERNEL_CHECK("stat_samples_kernel");	
}

template void stat_vmeans_and_imeans_gpu<float>(const unsigned int* sub_bcounts,
										const unsigned int* exclusive_counts,
										const unsigned int n,
										const unsigned char* exclusive_flags,
										const float* v_membranes,
										const float* i_synapses,
										float* vmeans,
										float* imeans,
										hipStream_t stream);

template void stat_vmeans_and_imeans_gpu<double>(const unsigned int* sub_bcounts,
										const unsigned int* exclusive_counts,
										const unsigned int n,
										const unsigned char* exclusive_flags,
										const double* v_membranes,
										const double* i_synapses,
										float* vmeans,
										float* imeans,
										hipStream_t stream);

template void stat_receptor_imeans_gpu<float, float2>(const unsigned int* sub_bcounts,
										const unsigned int* exclusive_counts,
										const unsigned int n,
										const unsigned char* exclusive_flags,
										const float2* i_ex_synapses,
										const float2* i_in_synapses,
										float* ampa_imeans,
										float* nmda_imeans,
										float* gabaa_imeans,
										float* gabab_imeans,
										hipStream_t stream);

template void stat_receptor_imeans_gpu<double, double2>(const unsigned int* sub_bcounts,
										const unsigned int* exclusive_counts,
										const unsigned int n,
										const unsigned char* exclusive_flags,
										const double2* i_ex_synapses,
										const double2* i_in_synapses,
										float* ampa_imeans,
										float* nmda_imeans,
										float* gabaa_imeans,
										float* gabab_imeans,
										hipStream_t stream);

template void stat_samples_gpu<float>(const unsigned int* samples,
									const unsigned int n,
									const unsigned char* f_actives,
									const float* v_membranes,
									const float* i_synaptics,
									const float* i_ou_background_stimuli,
									char* spikes,
									float* vmembs,
									float* isynaptics,
									float* ious,
									hipStream_t stream);

template void stat_samples_gpu<double>(const unsigned int* samples,
									const unsigned int n,
									const unsigned char* f_actives,
									const double* v_membranes,
									const double* i_synaptics,
									const double* i_ou_background_stimuli,
									char* spikes,
									float* vmembs,
									float* isynaptics,
									float* ious,
									hipStream_t stream);

}//namespace dtb

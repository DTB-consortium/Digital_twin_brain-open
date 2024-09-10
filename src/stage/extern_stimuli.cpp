#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"
#include "logging.hpp"

namespace dtb {

__global__ void update_input_spike_kernel(const unsigned int* __restrict__ neuron_inds,
											const unsigned int n,
											unsigned char* __restrict__ f_actives)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for (unsigned int i = idx; i < n; i += gridSize)
  	{
	  	int fidx = neuron_inds[i];
		f_actives[fidx] = 0x01;
	}
}

void update_input_spike_gpu(const unsigned int* neuron_inds,
								const unsigned int n,
								unsigned char* f_actives,
								hipStream_t stream)
{
	hipLaunchKernelGGL(
					update_input_spike_kernel,
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					neuron_inds,
					n,
					f_actives);
	HIP_POST_KERNEL_CHECK("update_input_spike_kernel");
}

template<typename T>
static __global__ void update_ou_background_stimuli_kernel(hiprandStatePhilox4_32_10_t* __restrict__ states,
																const unsigned int beg,
																const unsigned int end,
																const T mu,
																const T factor_1,
																const T factor_2,
																const T sigma,
																const T mean,
																const T stdv,
																T* __restrict__ i_backgrounds,
																T* __restrict__ i_synaptics,
																T* __restrict__ samples)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
  	for(unsigned int i = beg + idx; i < end; i += gridSize)
	{
		T x = static_cast<T>(hiprand_normal(&states[idx]));
		T i_ou = i_backgrounds[i];
		T i_synap = i_synaptics[i];
		T y = x * stdv + mean;
		i_ou += (factor_1 * (mu - i_ou) + factor_2 * sigma * y);
		i_synap += i_ou;
		i_backgrounds[i] = i_ou;
		i_synaptics[i] = i_synap; 
		if(NULL != samples)
			samples[i] = y;

	}
}

template<typename T>
void update_ou_background_stimuli_gpu(hiprandStatePhilox4_32_10_t* states,
											const unsigned int* rowptrs,
											const std::unordered_map<unsigned int, OU_Background_Current_Param>& param_map,
											const T delta_t,
											const T mean,
											const T stdv,
											T* i_backgrounds,
											T* i_synaptics,
											T* samples,
											hipStream_t stream)
{
	for(auto& p : param_map)
	{
		unsigned int beg = rowptrs[p.first];
		unsigned int end = rowptrs[p.first + 1];

		const T factor = static_cast<T>(-1) * delta_t / static_cast<T>(p.second.correlation_time_); 
		const T factor_1 = static_cast<T>(1) - texp<T>(factor);
		const T factor_2 = tsqrt<T>(static_cast<T>(1) - texp<T>(static_cast<T>(2) * factor));
		hipLaunchKernelGGL(HIP_KERNEL_NAME(update_ou_background_stimuli_kernel<T>),
						dim3(divide_up<unsigned int>((end - beg), HIP_THREADS_PER_BLOCK)), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						states,
						beg,
						end,
						static_cast<T>(p.second.mean_),
						factor_1,
						factor_2,
						static_cast<T>(p.second.deviation_),
						mean,
						stdv,
						i_backgrounds,
						i_synaptics,
						samples);
		HIP_POST_KERNEL_CHECK("update_ou_background_stimuli_kernel");
	}
	
}

template<typename T>
static __global__ void update_ttype_ca_stimuli_kernel(const T* __restrict__ v_membranes,
														const unsigned int beg,
														const unsigned int end,
														const T_Type_Ca_Current_Param param,
														const T* __restrict__ h_ttype_ca,
														T* __restrict__ i_ttype_ca)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = beg + idx; i < end; i += gridSize)
  	{
		const T vi = v_membranes[i];
		const T hi = h_ttype_ca[i];
		T ii = i_ttype_ca[i];
		if(vi >= param.v_h_)
		{
			ii += (static_cast<T>(-1) * static_cast<T>(param.g_t_) * hi * (vi - static_cast<T>(param.v_t_)));
		}

		i_ttype_ca[i] = ii;
	}
}

template<typename T>
void update_ttype_ca_stimuli_gpu(const T* v_membranes,
									const unsigned int* rowptrs,
									const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
									const T* h_ttype_ca,
									T* i_ttype_ca,
									hipStream_t stream)
{
	for(auto& p : param_map)
	{
		unsigned int beg = rowptrs[p.first];
		unsigned int end = rowptrs[p.first + 1];
		
		hipLaunchKernelGGL(HIP_KERNEL_NAME(update_ttype_ca_stimuli_kernel<T>),
					dim3(divide_up<unsigned int>((end - beg), HIP_THREADS_PER_BLOCK)), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_membranes,
					beg,
					end,
					p.second,
					h_ttype_ca,
					i_ttype_ca);
		HIP_POST_KERNEL_CHECK("update_ttype_ca_stimuli_kernel");
		//hipStreamSynchronize(stream);
	}
}

template<typename T>
__global__ void update_h_ttype_ca_kernel(const T* __restrict__ v_membranes,
													const unsigned int beg,
													const unsigned int end,
													const T_Type_Ca_Current_Param param,
													const T delta_t,
													T* __restrict__ h_ttype_ca)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = beg + idx; i < end; i += gridSize)
  	{
		const T vi = v_membranes[i];
		T hi = h_ttype_ca[i];
		if(vi >= param.v_h_)
		{
			hi += (static_cast<T>(-1) * hi / static_cast<T>(param.tao_h_minus_) * delta_t);
		}
		else
		{
			hi += ((static_cast<T>(1) - hi) / static_cast<T>(param.tao_h_plus_) * delta_t);
		}

		h_ttype_ca[i] = hi;
	}
}

template<typename T>
void update_h_ttype_ca_gpu(const T* v_membranes,
										const unsigned int* rowptrs,
										const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
										const T delta_t,
										T* h_ttype_ca,
										hipStream_t stream)
{
	for(auto& p : param_map)
	{
		
		unsigned int beg = rowptrs[p.first];
		unsigned int end = rowptrs[p.first + 1];
		
		hipLaunchKernelGGL(HIP_KERNEL_NAME(update_h_ttype_ca_kernel<T>),
					dim3(divide_up<unsigned int>((end - beg), HIP_THREADS_PER_BLOCK)), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					v_membranes,
					beg,
					end,
					p.second,
					delta_t,
					h_ttype_ca);
		HIP_POST_KERNEL_CHECK("update_h_ttype_ca_stimuli_kernel");
		//hipStreamSynchronize(stream);
	}
}

template<typename T>
static __global__ void reset_partial_kernel(const T init,
									const unsigned int beg,
									const unsigned int end,
									T* out)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = beg + idx; i < end; i += gridSize)
  	{
		out[i] = init;
	}
}

template<typename T>
void reset_partial_gpu(const T init,
					const unsigned int offset,
					const unsigned int* rowptrs,
					T* out,
					hipStream_t stream)
{
	unsigned int beg = rowptrs[offset];
	unsigned int end = rowptrs[offset + 1];
	const unsigned int n = end - beg;
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(reset_partial_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					init,
					beg,
					end,
					out);
	HIP_POST_KERNEL_CHECK("reset_partial_kernel");
	HIP_CHECK(hipStreamSynchronize(stream));
}


template<typename T>
__global__ void update_dopamine_stimuli_kernel(const T* __restrict__ v_membranes,
													const unsigned int beg,
													const unsigned int end,
													const Dopamine_Current_Param param,
													T* __restrict__ i_synaptics)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = beg + idx; i < end; i += gridSize)
  	{
		const T vmi = v_membranes[i];
		T ii = i_synaptics[i];
		ii += (static_cast<T>(param.g_dopamine_) * (vmi - static_cast<T>(param.v_dopamine_)));
		i_synaptics[i] += ii;
	}
}

template<typename T>
void update_dopamine_stimuli_gpu(const T* v_membranes,
										const unsigned int* rowptrs,
										const std::unordered_map<unsigned int, Dopamine_Current_Param>& param_map,
										T* i_synaptics,
										hipStream_t stream)
{
	for(auto& p : param_map)
	{
		unsigned int beg = rowptrs[p.first];
		unsigned int end = rowptrs[p.first + 1];
		hipLaunchKernelGGL(HIP_KERNEL_NAME(update_dopamine_stimuli_kernel<T>),
						dim3(divide_up<unsigned int>((end - beg), HIP_THREADS_PER_BLOCK)),
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						v_membranes,
						beg,
						end,
						p.second,
						i_synaptics);
		HIP_POST_KERNEL_CHECK("update_dopamine_stimuli_kernel");
	}
}

template<typename T>
__global__ void update_adaptation_stimuli_kernel(const T* __restrict__ v_membranes,
													const unsigned int beg,
													const unsigned int end,
													const Adaptation_Current_Param param,
													const T* ahp_ca_concentrations,
													T* __restrict__ i_synaptics)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = beg + idx; i < end; i += gridSize)
  	{
		const T vmi = v_membranes[i];
		const T ca_concentration = ahp_ca_concentrations[i];
		T ii = i_synaptics[i];
		ii += (static_cast<T>(param.g_ahp_) * ca_concentration * (static_cast<T>(param.v_k_) - vmi));
		i_synaptics[i] = ii;
	}
}

template<typename T>
void update_adaptation_stimuli_gpu(const T* v_membranes,
										const unsigned int* rowptrs,
										const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
										const T* ahp_ca_concentrations,
										T* i_synaptics,
										hipStream_t stream)
{
	for(auto& p : param_map)
	{
		unsigned int beg = rowptrs[p.first];
		unsigned int end = rowptrs[p.first + 1];
		hipLaunchKernelGGL(HIP_KERNEL_NAME(update_adaptation_stimuli_kernel<T>),
						dim3(divide_up<unsigned int>((end - beg), HIP_THREADS_PER_BLOCK)),
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						v_membranes,
						beg,
						end,
						p.second,
						ahp_ca_concentrations,
						i_synaptics);
		HIP_POST_KERNEL_CHECK("update_adaptation_stimuli_kernel");
	}
}

template<typename T>
__global__ void update_ahp_ca_concentration_kernel(const unsigned char* f_spikes,
													const unsigned int beg,
													const unsigned int end,
													const Adaptation_Current_Param param,
													T* __restrict__ ahp_ca_concentrations)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = beg + idx; i < end; i += gridSize)
  	{
  		char f_spike = f_spikes[i];
		T ca_concentration = ahp_ca_concentrations[i];
		if(f_spike)
		{
			ca_concentration += static_cast<T>(param.alpha_constant_);
		}
		else
		{
			ca_concentration *= static_cast<T>(param.ca_decay_);
		}

		ahp_ca_concentrations[i] = ca_concentration;
	}
}

template<typename T>
void update_ahp_ca_concentration_gpu(const unsigned char* f_spikes,
										const unsigned int* rowptrs,
										const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
										T* ahp_ca_concentrations,
										hipStream_t stream)
{
	for(auto& p : param_map)
	{
		
		unsigned int beg = rowptrs[p.first];
		unsigned int end = rowptrs[p.first + 1];
		
		hipLaunchKernelGGL(HIP_KERNEL_NAME(update_ahp_ca_concentration_kernel<T>),
					dim3(divide_up<unsigned int>((end - beg), HIP_THREADS_PER_BLOCK)), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					f_spikes,
					beg,
					end,
					p.second,
					ahp_ca_concentrations);
		HIP_POST_KERNEL_CHECK("update_ahp_ca_concentration_kernel");
		//hipStreamSynchronize(stream);
	}
}


template void update_ou_background_stimuli_gpu<float>(hiprandStatePhilox4_32_10_t* states,
													const unsigned int* rowptrs,
													const std::unordered_map<unsigned int, OU_Background_Current_Param>& param_map,
													const float delta_t,
													const float mean,
													const float stdv,
													float* i_backgrounds,
													float* i_synaptics,
													float* samples,
													hipStream_t stream);

template void update_ou_background_stimuli_gpu<double>(hiprandStatePhilox4_32_10_t* states,
													const unsigned int* rowptrs,
													const std::unordered_map<unsigned int, OU_Background_Current_Param>& param_map,
													const double delta_t,
													const double mean,
													const double stdv,
													double* i_backgrounds,
													double* i_synaptics,
													double* samples,
													hipStream_t stream);

template void update_ttype_ca_stimuli_gpu<float>(const float* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
												const float* h_ttype_ca,
												float* i_ttype_ca,
												hipStream_t stream);

template void update_ttype_ca_stimuli_gpu<double>(const double* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
												const double* h_ttype_ca,
												double* i_ttype_ca,
												hipStream_t stream);

template void update_h_ttype_ca_gpu<float>(const float* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
												const float delta_t,
												float* h_ttype_ca,
												hipStream_t stream);

template void update_h_ttype_ca_gpu<double>(const double* v_membranes,
													const unsigned int* rowptrs,
													const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
													const double delta_t,
													double* h_ttype_ca,
													hipStream_t stream);

template void reset_partial_gpu<float>(const float init,
										const unsigned int offset,
										const unsigned int* rowptrs,
										float* out,
										hipStream_t stream);

template void reset_partial_gpu<double>(const double init,
										const unsigned int offset,
										const unsigned int* rowptrs,
										double* out,
										hipStream_t stream);

template void update_dopamine_stimuli_gpu<float>(const float* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, Dopamine_Current_Param>& param_map,
												float* i_synaptics,
												hipStream_t stream);

template void update_dopamine_stimuli_gpu<double>(const double* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, Dopamine_Current_Param>& param_map,
												double* i_synaptics,
												hipStream_t stream);

template void update_adaptation_stimuli_gpu<float>(const float* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
												const float* ahp_ca_concentrations,
												float* i_synaptics,
												hipStream_t stream);

template void update_adaptation_stimuli_gpu<double>(const double* v_membranes,
												const unsigned int* rowptrs,
												const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
												const double* ahp_ca_concentrations,
												double* i_synaptics,
												hipStream_t stream);

template void update_ahp_ca_concentration_gpu<float>(const unsigned char* f_spikes,
													const unsigned int* rowptrs,
													const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
													float* ahp_ca_concentrations,
													hipStream_t stream);

template void update_ahp_ca_concentration_gpu<double>(const unsigned char* f_spikes,
													const unsigned int* rowptrs,
													const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
													double* ahp_ca_concentrations,
													hipStream_t stream);
}//namespace dtb

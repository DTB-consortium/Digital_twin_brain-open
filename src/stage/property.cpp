#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/find.h>
#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

#if defined(__HIP_PLATFORM_HCC__)
/* include MTGP pre-computed parameter sets */
#include <rocrand_philox4x32_10.h>
#else
/* include MTGP pre-computed parameter sets */
#include <curand_philox4x32_x.h>
#endif

namespace dtb {

template<typename T, typename T2>
struct tao_transformation
{
  __host__ __device__
  T2 operator()(const T2& a, const T& b) const
  {
  	T2 result;
  	result.x = texp<T>(b / a.x);
	result.y = texp<T>(b / a.y);
    return result;
  }
};

template<typename T, typename T2>
void update_tao_constant_gpu(const T delta_t,
							const T2* h_tao_constants,
							const unsigned int n,
							T2* d_tao_constants)
{
	thrust::constant_iterator<T> d_scales(((-1) * delta_t));
	HIP_CHECK(hipMemcpy(d_tao_constants, h_tao_constants, n * sizeof(T2), hipMemcpyHostToDevice));
	thrust::transform(thrust::device_pointer_cast(d_tao_constants),
					thrust::device_pointer_cast(d_tao_constants) + n,
					d_scales, thrust::device_pointer_cast(d_tao_constants), tao_transformation<T, T2>());
}

template<typename T>
void update_refractory_period_gpu(const T delta_t,
									const unsigned int n,
									T* d_t_refs)
{
	thrust::transform(thrust::device_pointer_cast(d_t_refs),
					thrust::device_pointer_cast(d_t_refs) + n,
					thrust::make_constant_iterator<T>(delta_t), thrust::device_pointer_cast(d_t_refs), thrust::divides<T>());
}



template<typename T, typename T2>
__global__  void update_props_kernel(const unsigned int* neuron_indice,
										const unsigned int* prop_indice,
										const float* prop_vals,
										const unsigned int n,
										Properties<T, T2> prop)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = gridDim.x * blockDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int nid = neuron_indice[i];
		unsigned int pid = prop_indice[i];
		T val = static_cast<T>(prop_vals[i]);
		
		if(nid < prop.n)
		{
			switch(pid)
			{
				case EXT_STIMULI_I:
					prop.i_ext_stimuli[nid] = val;
				break;
					
				case MEMBRANE_C:
					prop.c_membrane_reciprocals[nid] = ((T)1 / val);
				break;
				
				case REF_T:
					prop.t_refs[nid] = val;
				break;
				
				case LEAKAGE_G:
					prop.g_leakages[nid] = val;
				break;
				
				case LEAKAGE_V:
					prop.v_leakages[nid] = val;
				break;
				
				case THRESHOLD_V:
					prop.v_thresholds[nid] = val;
				break;
				
				case RESET_V:
					prop.v_resets[nid] = val;
				break;
				
				case CONDUCT_G_AMPA:
					prop.g_ex_conducts[nid].x = val;
				break;
				
				case CONDUCT_G_NMDA:
					prop.g_ex_conducts[nid].y = val;
				break;
				
				case CONDUCT_G_GABAa:
					prop.g_in_conducts[nid].x = val;
				break;
				
				case CONDUCT_G_GABAb:
					prop.g_in_conducts[nid].y = val;
				break;
				
				case MEMBRANE_V_AMPA:
					prop.v_ex_membranes[nid].x = val;
				break;
				
				case MEMBRANE_V_NMDA:
					prop.v_ex_membranes[nid].y = val;
				break;
				
				case MEMBRANE_V_GABAa:
					prop.v_in_membranes[nid].x = val;
				break;
				
				case MEMBRANE_V_GABAb:
					prop.v_in_membranes[nid].y = val;
				break;
				
				case TAO_AMPA:
					//prop.tao_ex_constants[nid].x = val;
				break;
				
				case TAO_NMDA:
					//prop.tao_ex_constants[nid].y = val;
				break;
				
				case TAO_GABAa:
					//prop.tao_in_constants[nid].x = val;
				break;
				
				case TAO_GABAb:
					//prop.tao_in_constants[nid].y = val;
				break;
				case NOISE_RATE:
					prop.noise_rates[nid] = val;
				break;
				default:
				break;
			}
		}
	}
}

template<typename T, typename T2>
void update_props_gpu(const unsigned int* neuron_indice,
						const unsigned int* prop_indice,
						const float* prop_vals,
						const unsigned int n,
						Properties<T, T2>& prop,
						hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_props_kernel<T, T2>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					neuron_indice,
					prop_indice,
					prop_vals,
					n,
					prop);

	HIP_POST_KERNEL_CHECK("update_props_kernel");
}

template<typename T, typename T2>
__global__  void assign_prop_cols_kernel(const unsigned int* sub_bids,
													const unsigned int* sub_bcounts,
													const unsigned int m,
													const unsigned int* prop_indice,
													const unsigned int* brain_indice,
													const float* hp_vals,
													const unsigned int n,
													Properties<T, T2> prop)
{
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		const unsigned int* iter = thrust::find(thrust::device, sub_bids, sub_bids + m, brain_indice[i]);
		const unsigned int bid = iter - sub_bids;
		if(bid < m)
		{
			const unsigned int pid = prop_indice[i];
			const T val = static_cast<T>(hp_vals[i]);
			const unsigned int beg = sub_bcounts[bid];
			const unsigned int end = sub_bcounts[bid + 1];
			for(unsigned int j = beg + threadIdx.x; j < end; j += blockDim.x)
			{
				switch(pid)
				{
					case EXT_STIMULI_I:
						prop.i_ext_stimuli[j] = val;
					break;
						
					case MEMBRANE_C:
						prop.c_membrane_reciprocals[j] = ((T)1 / val);
					break;
					
					case REF_T:
						prop.t_refs[j] = val;
					break;
					
					case LEAKAGE_G:
						prop.g_leakages[j] = val;
					break;
					
					case LEAKAGE_V:
						prop.v_leakages[j] = val;
					break;
					
					case THRESHOLD_V:
						prop.v_thresholds[j] = val;
					break;
					
					case RESET_V:
						prop.v_resets[j] = val;
					break;
					
					case CONDUCT_G_AMPA:
						prop.g_ex_conducts[j].x = val;
					break;
					
					case CONDUCT_G_NMDA:
						prop.g_ex_conducts[j].y = val;
					break;
					
					case CONDUCT_G_GABAa:
						prop.g_in_conducts[j].x = val;
					break;
					
					case CONDUCT_G_GABAb:
						prop.g_in_conducts[j].y = val;
					break;
					
					case MEMBRANE_V_AMPA:
						prop.v_ex_membranes[j].x = val;
					break;
					
					case MEMBRANE_V_NMDA:
						prop.v_ex_membranes[j].y = val;
					break;
					
					case MEMBRANE_V_GABAa:
						prop.v_in_membranes[j].x = val;
					break;
					
					case MEMBRANE_V_GABAb:
						prop.v_in_membranes[j].y = val;
					break;
					
					case TAO_AMPA:
						prop.tao_ex_constants[j].x = val;
					break;
					
					case TAO_NMDA:
						prop.tao_ex_constants[j].y = val;
					break;
					
					case TAO_GABAa:
						prop.tao_in_constants[j].x = val;
					break;
					
					case TAO_GABAb:
						prop.tao_in_constants[j].y = val;
					break;
					
					case NOISE_RATE:
						prop.noise_rates[j] = val;
					break;
					default:
					break;
				}
			}
		}
  	}
}

template<typename T, typename T2>
void assign_prop_cols_gpu(const unsigned int* sub_bids,
							const unsigned int* sub_bcounts,
							const unsigned int m,
							const unsigned int* prop_indice,
							const unsigned int* brain_indice,
							const float* hp_vals,
							const unsigned int n,
							Properties<T, T2>& prop,
							hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(assign_prop_cols_kernel<T, T2>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_bids,
					sub_bcounts,
					m,
					prop_indice,
					brain_indice,
					hp_vals,
					n,
					prop);
	HIP_POST_KERNEL_CHECK("assign_prop_cols_kernel");
}


template<typename T, typename T2>
__global__  void update_prop_cols_kernel(const unsigned int* sub_bids,
													const unsigned int* sub_bcounts,
													const unsigned int m,
													const unsigned int* prop_indice,
													const unsigned int* brain_indice,
													const float* hp_vals,
													const unsigned int n,
													Properties<T, T2> prop)
{
  	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		const unsigned int* iter = thrust::find(thrust::device, sub_bids, sub_bids + m, brain_indice[i]);
		const unsigned int bid = iter - sub_bids;
		if(bid < m)
		{
			const unsigned int pid = prop_indice[i];
			const T val = static_cast<T>(hp_vals[i]);
			const unsigned int beg = sub_bcounts[bid];
			const unsigned int end = sub_bcounts[bid + 1];
			for(unsigned int j = beg + threadIdx.x; j < end; j += blockDim.x)
			{
				switch(pid)
				{
					case EXT_STIMULI_I:
						prop.i_ext_stimuli[j] *= val;
					break;
						
					case MEMBRANE_C:
						prop.c_membrane_reciprocals[j] *= ((T)1 / val);
					break;
					
					case REF_T:
						prop.t_refs[j] *= val;
					break;
					
					case LEAKAGE_G:
						prop.g_leakages[j] *= val;
					break;
					
					case LEAKAGE_V:
						prop.v_leakages[j] *= val;
					break;
					
					case THRESHOLD_V:
						prop.v_thresholds[j] *= val;
					break;
					
					case RESET_V:
						prop.v_resets[j] *= val;
					break;
					
					case CONDUCT_G_AMPA:
						prop.g_ex_conducts[j].x *= val;
					break;
					
					case CONDUCT_G_NMDA:
						prop.g_ex_conducts[j].y *= val;
					break;
					
					case CONDUCT_G_GABAa:
						prop.g_in_conducts[j].x *= val;
					break;
					
					case CONDUCT_G_GABAb:
						prop.g_in_conducts[j].y *= val;
					break;
					
					case MEMBRANE_V_AMPA:
						prop.v_ex_membranes[j].x *= val;
					break;
					
					case MEMBRANE_V_NMDA:
						prop.v_ex_membranes[j].y *= val;
					break;
					
					case MEMBRANE_V_GABAa:
						prop.v_in_membranes[j].x *= val;
					break;
					
					case MEMBRANE_V_GABAb:
						prop.v_in_membranes[j].y *= val;
					break;
					
					case TAO_AMPA:
						prop.tao_ex_constants[j].x *= val;
					break;
					
					case TAO_NMDA:
						prop.tao_ex_constants[j].y *= val;
					break;
					
					case TAO_GABAa:
						prop.tao_in_constants[j].x *= val;
					break;
					
					case TAO_GABAb:
						prop.tao_in_constants[j].y *= val;
					break;
					
					case NOISE_RATE:
						prop.noise_rates[j] *= val;
					break;
					default:
					break;
				}
			}
		}
  	}
}

template<typename T, typename T2>
void update_prop_cols_gpu(const unsigned int* sub_bids,
							const unsigned int* sub_bcounts,
							const unsigned int m,
							const unsigned int* prop_indice,
							const unsigned int* brain_indice,
							const float* hp_vals,
							const unsigned int n,
							Properties<T, T2>& prop,
							hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_prop_cols_kernel<T, T2>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					sub_bids,
					sub_bcounts,
					m,
					prop_indice,
					brain_indice,
					hp_vals,
					n,
					prop);
	HIP_POST_KERNEL_CHECK("update_prop_cols_kernel");
}

// The function `sample_gamma` is
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:

/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

template<typename T>
__device__ T sample_gamma(T alpha, hiprandStatePhilox4_32_10_t* state)
{
	T scale = 1.0f;

	// Boost alpha for higher acceptance probability.
	if (alpha < 1.0f)
	{
		if (alpha == 0.f)
			return 0.f;
		scale *= tpow<T>(1 - hiprand_uniform(state), 1.0f / alpha);
		alpha += 1.0f;
	}

	// This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
	// doi:10.1145/358407.358414
	const T d = alpha - 1.0f / 3.0f;
	const T c = 1.0f / tsqrt<T>(9.0f * d);
	for (;;)
	{
		T x, y;
		do{
		  x = hiprand_normal(state);
		  y = 1.0f + c * x;
		}while (y <= 0);
		const T v = y * y * y;
		const T u = 1 - hiprand_uniform(state);
		const T xx = x * x;
		if (u < 1.0f - 0.0331f * xx * xx)
		  return static_cast<T>(scale * d * v);
		if (tlog<T>(u) < 0.5f * xx + d * (1.0f - v + tlog<T>(v)))
		  return static_cast<T>(scale * d * v);
	}
}

template<typename T, typename T2>
__global__ void gamma_kernel(hiprandStatePhilox4_32_10_t* states,
								const unsigned int* sub_bids,
								const unsigned int* sub_bcounts,
								const unsigned int m,
								const unsigned int* prop_indice,
								const unsigned int* brain_indice,
								const float* alphas,
								const float* betas,
								const unsigned int n,
								Properties<T, T2> prop)
{
	for(unsigned int i = blockIdx.x; i < n; i += gridDim.x)
	{
		//x: start position,
		//y: neuron number
		const unsigned int* iter = thrust::find(thrust::device, sub_bids, sub_bids + m, brain_indice[i]);
		unsigned int bid = iter - sub_bids;
		if(bid < m)
		{
			const unsigned int pid = prop_indice[i];
			const unsigned int beg = sub_bcounts[bid];
			const unsigned int end = sub_bcounts[bid + 1];
			for(unsigned int j = beg + threadIdx.x; j < end; j += blockDim.x)
			{
				hiprandStatePhilox4_32_10_t state = states[j];
				T alpha = static_cast<T>(alphas[i]);
				T beta = static_cast<T>(betas[i]);
				T val = sample_gamma<T>(alpha, &state);
				T min_val = numeric_limits<T>::lowest();
		        val = ((min_val > val) ? min_val : val) / beta;
				
				switch(pid)
				{
					case EXT_STIMULI_I:	
						prop.i_ext_stimuli[j] = val;
					break;
					
					case MEMBRANE_C:
						prop.c_membrane_reciprocals[j] = ((T)1 / val);
					break;
					
					case REF_T:
						prop.t_refs[j] = val;
					break;
					
					case LEAKAGE_G:
						prop.g_leakages[j] = val;
					break;
					
					case LEAKAGE_V:
						prop.v_leakages[j] = val;
					break;
					
					case THRESHOLD_V:
						prop.v_thresholds[j] = val;
					break;
					
					case RESET_V:	
						prop.v_resets[j] = val;
					break;
					
					case CONDUCT_G_AMPA:
						prop.g_ex_conducts[j].x = val;
					break;
					
					case CONDUCT_G_NMDA:
						prop.g_ex_conducts[j].y = val;
					break;
					
					case CONDUCT_G_GABAa:
						prop.g_in_conducts[j].x = val;
					break;
					
					case CONDUCT_G_GABAb:
						prop.g_in_conducts[j].y = val;
					break;
					
					case MEMBRANE_V_AMPA:
						prop.v_ex_membranes[j].x = val;
					break;
					
					case MEMBRANE_V_NMDA:
						prop.v_ex_membranes[j].y = val;
					break;
					
					case MEMBRANE_V_GABAa:
						prop.v_in_membranes[j].x = val;
					break;
					
					case MEMBRANE_V_GABAb:
						prop.v_in_membranes[j].y = val;
					break;
					
					case TAO_AMPA:
						prop.tao_ex_constants[j].x = val;
					break;
					
					case TAO_NMDA:
						prop.tao_ex_constants[j].y = val;
					break;
					
					case TAO_GABAa:
						prop.tao_in_constants[j].x = val;
					break;
					
					case TAO_GABAb:
						prop.tao_in_constants[j].y = val;
					break;
					
					case NOISE_RATE:
						prop.noise_rates[j] = val;
					break;
					
					default:
					break;
				}

				states[j] = state;
			}
		}
	}
}

template<typename T, typename T2>
void gamma_gpu(hiprandStatePhilox4_32_10_t* states,
					const unsigned int* sub_bids,
					const unsigned int* sub_bcounts,
					const unsigned int m,
					const unsigned int* prop_indice,
					const unsigned int* brain_indice,
					const float* alphas,
					const float* betas,
					const unsigned int n,
					Properties<T, T2>& prop,
					hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(gamma_kernel<T, T2>),
					dim3(n),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					states,
					sub_bids,
					sub_bcounts,
					m,
					prop_indice,
					brain_indice,
					alphas,
					betas,
					n,
					prop);

	HIP_POST_KERNEL_CHECK("gamma_kernel");
}

template void update_tao_constant_gpu<float, float2>(const float delta_t,
													const float2* h_tao_constants,
													const unsigned int n,
													float2* d_tao_constants);

template void update_tao_constant_gpu<double, double2>(const double delta_t,
													const double2* h_tao_constants,
													const unsigned int n,
													double2* d_tao_constants);

template void update_refractory_period_gpu<float>(const float delta_t,
												const unsigned int n,
												float* d_t_refs);

template void update_refractory_period_gpu<double>(const double delta_t,
												const unsigned int n,
												double* d_t_refs);

template void update_props_gpu<float, float2>(const unsigned int* neuron_indice,
											const unsigned int* prop_indice,
											const float* prop_vals,
											const unsigned int n,
											Properties<float, float2>& prop,
											hipStream_t stream);

template void update_props_gpu<double, double2>(const unsigned int* neuron_indice,
											const unsigned int* prop_indice,
											const float* prop_vals,
											const unsigned int n,
											Properties<double, double2>& prop,
											hipStream_t stream);

template void update_prop_cols_gpu<float, float2>(const unsigned int* sub_bids,
											const unsigned int* sub_bcounts,
											const unsigned int m,
											const unsigned int* prop_indice,
											const unsigned int* brain_indice,
											const float* hp_vals,
											const unsigned int n,
											Properties<float, float2>& prop,
											hipStream_t stream);

template void update_prop_cols_gpu<double, double2>(const unsigned int* sub_bids,
													const unsigned int* sub_bcounts,
													const unsigned int m,
													const unsigned int* prop_indice,
													const unsigned int* brain_indice,
													const float* hp_vals,
													const unsigned int n,
													Properties<double, double2>& prop,
													hipStream_t stream);


template void assign_prop_cols_gpu<float, float2>(const unsigned int* sub_bids,
											const unsigned int* sub_bcounts,
											const unsigned int m,
											const unsigned int* prop_indice,
											const unsigned int* brain_indice,
											const float* hp_vals,
											const unsigned int n,
											Properties<float, float2>& prop,
											hipStream_t stream);

template void assign_prop_cols_gpu<double, double2>(const unsigned int* sub_bids,
											const unsigned int* sub_bcounts,
											const unsigned int m,
											const unsigned int* prop_indice,
											const unsigned int* brain_indice,
											const float* hp_vals,
											const unsigned int n,
											Properties<double, double2>& prop,
											hipStream_t stream);

template void gamma_gpu<float, float2>(hiprandStatePhilox4_32_10_t* states,
									const unsigned int* sub_bids,
									const unsigned int* sub_bcounts,
									const unsigned int m,
									const unsigned int* prop_indice,
									const unsigned int* brain_indice,
									const float* alphas,
									const float* betas,
									const unsigned int n,
									Properties<float, float2>& prop,
									hipStream_t stream);

template void gamma_gpu<double, double2>(hiprandStatePhilox4_32_10_t* states,
										const unsigned int* sub_bids,
										const unsigned int* sub_bcounts,
										const unsigned int m,
										const unsigned int* prop_indice,
										const unsigned int* brain_indice,
										const float* alphas,
										const float* betas,
										const unsigned int n,
										Properties<double, double2>& prop,
										hipStream_t stream);


}//namespace dtb

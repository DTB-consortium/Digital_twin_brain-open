#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

template<typename T, typename T2>
static __global__ void update_synaptic_current_kernel(const T2* j_ex_presynaptics,
															const T2* j_in_presynaptics,
															const T2* g_ex_conducts,
															const T2* g_in_conducts,
															const T2* v_ex_membranes,
															const T2* v_in_membranes,
															const T* v_membranes,
															const unsigned int n,
															T2* i_ex_synaptics,
															T2* i_in_synaptics,
															T* i_synaptics)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
 	const unsigned int gridSize = blockDim.x * gridDim.x;
	
  	for(unsigned int i = idx; i < n; i += gridSize)
  	{
  		
  		const T vi = v_membranes[i];
		
		const T2 vi_ex = v_ex_membranes[i];
		const T2 ji_ex = j_ex_presynaptics[i];
		const T2 gi_ex = g_ex_conducts[i];

		const T2 vi_in = v_in_membranes[i];
		const T2 ji_in = j_in_presynaptics[i];
		const T2 gi_in = g_in_conducts[i];

		T2 ii_ex;
		T2 ii_in;
#if 1
		ii_ex.x = gi_ex.x * (vi_ex.x - vi) * ji_ex.x;
		ii_ex.y = gi_ex.y * (vi_ex.y - vi) * ji_ex.y;
		ii_in.x = gi_in.x * (vi_in.x - vi) * ji_in.x;
		ii_in.y = gi_in.y * (vi_in.y - vi) * ji_in.y;
		i_synaptics[i] = ii_ex.x + ii_ex.y + ii_in.x + ii_in.y;
		if(NULL != i_ex_synaptics)
			i_ex_synaptics[i] = ii_ex;
		if(NULL != i_in_synaptics)
			i_in_synaptics[i] = ii_in;
#else
		//Kahan summation algorithm
		T comp = static_cast<T>(0);
		T t;
		comp -= gi_ex.x * (vi_ex.x - vi) * ji_ex.x;
		t = sum - comp;
		comp = (t - sum) + comp;
		sum = t;
		
		comp -= gi_ex.y * (vi_ex.y - vi) * ji_ex.y;
		t = sum - comp;
		comp = (t - sum) + comp;
		sum = t;

		comp -= gi_in.x * (vi_in.x - vi) * ji_in.x;
		t = sum - comp;
		comp = (t - sum) + comp;
		sum = t;

		comp -= gi_in.y * (vi_in.y - vi) * ji_in.y;
		i_synaptics[i] = (sum - comp);
#endif	
	}
}

template<typename T, typename T2>
void update_synaptic_current_gpu(const T2* j_ex_presynaptics,
									const T2* j_in_presynaptics,
									const T2* g_ex_conducts,
									const T2* g_in_conducts,
									const T2* v_ex_membranes,
									const T2* v_in_membranes,
									const T* v_membranes,
									const unsigned int n,
									T2* i_ex_synaptics,
									T2* i_in_synaptics,
									T* i_synaptics,
									hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_synaptic_current_kernel<T, T2>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					j_ex_presynaptics,
					j_in_presynaptics,
					g_ex_conducts,
					g_in_conducts,
					v_ex_membranes,
					v_in_membranes,
					v_membranes,
					n,
					i_ex_synaptics,
					i_in_synaptics,
					i_synaptics);
	HIP_POST_KERNEL_CHECK("update_synaptic_current_kernel");
}

template void update_synaptic_current_gpu<float, float2>(const float2* j_ex_presynaptics,
														const float2* j_in_presynaptics,
														const float2* g_ex_conducts,
														const float2* g_in_conducts,
														const float2* v_ex_membranes,
														const float2* v_in_membranes,
														const float* v_membranes,
														const unsigned int n,
														float2* i_ex_synaptics,
														float2* i_in_synaptics,
														float* i_synaptics,
														hipStream_t stream);

template void update_synaptic_current_gpu<double, double2>(const double2* j_ex_presynaptics,
														const double2* j_in_presynaptics,
														const double2* g_ex_conducts,
														const double2* g_in_conducts,
														const double2* v_ex_membranes,
														const double2* v_in_membranes,
														const double* v_membranes,
														const unsigned int n,
														double2* i_ex_synaptics,
														double2* i_in_synaptics,
														double* i_synaptics,
														hipStream_t stream);

}//namespace dtb 

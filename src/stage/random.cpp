#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"

namespace dtb {

__global__ void init_random_kernel(const unsigned long long seed,
								const unsigned long long offset,
								const unsigned int n,
								hiprandStatePhilox4_32_10_t *states)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	/* Each thread gets same seed, a different sequence
	number,  offset */
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		hiprand_init(seed, i, offset, &states[i]);
	}
}

void init_random_gpu(const unsigned long long seed,
					const unsigned long long offset,
					const unsigned int n,
					hiprandStatePhilox4_32_10_t *states,
					hipStream_t stream)
{
	hipLaunchKernelGGL(
					init_random_kernel,
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					seed,
					offset,
					n,
					states);
	HIP_POST_KERNEL_CHECK("init_random_kernel");
}

}//namespace dtb 

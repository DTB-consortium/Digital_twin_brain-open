#include "util/transpose.hpp"
#include "common.hpp"

namespace dtb {

#define TILE_DIM 32
#define BLOCK_ROWS 8

template<typename T>
static __global__ void transpose_kernel(const T* input,
										const unsigned int height,
										const unsigned int width,
		                                T* output)
{
    __shared__ T tile[TILE_DIM][TILE_DIM+1];

    unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned int index_in = xIndex + yIndex * width;

    for (unsigned int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (xIndex < width && (yIndex + i) < height)
        {
            tile[threadIdx.y + i][threadIdx.x] = input[index_in + i * width];
        }
    }
    __syncthreads();
    
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    unsigned int index_out = xIndex + yIndex * height;

    for (unsigned int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (xIndex < height && (yIndex + i) < width)
        {
            output[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

template<typename T>
void transpose_gpu(const T* input,
					const unsigned int height,
					const unsigned int width,
                    T* output,
					hipStream_t stream)
{

	dim3 threads(TILE_DIM, BLOCK_ROWS);
	dim3 blocks((width - 1) / TILE_DIM + 1, (height - 1) / TILE_DIM + 1);
	hipLaunchKernelGGL(transpose_kernel<T>, dim3(blocks), dim3(threads), 0, stream, input,
														height,
														width,
														output);
	HIP_POST_KERNEL_CHECK("transpose_kernel");	
}

template<typename T>
void transpose_cpu(const T* input,
					const unsigned int height,
					const unsigned int width,
					T* output)
{
    for(unsigned int i = 0; i < height; i++)
    {
        for(unsigned int j = 0; j < width; j++)
        {
            output[j * height + i] = input[i * width + j];
        }
    }
}

template void transpose_gpu<unsigned char>(const unsigned char* input,
								const unsigned int height,
								const unsigned int width,
			                    unsigned char* output,
								hipStream_t stream);

template void transpose_gpu<unsigned int>(const unsigned int* input,
								const unsigned int height,
								const unsigned int width,
			                    unsigned int* output,
								hipStream_t stream);

template void transpose_gpu<long long>(const long long* input,
								const unsigned int height,
								const unsigned int width,
			                    long long* output,
								hipStream_t stream);

template void transpose_gpu<float>(const float* input,
								const unsigned int height,
								const unsigned int width,
			                    float* output,
								hipStream_t stream);

template void transpose_cpu<unsigned char>(const unsigned char* input,
								const unsigned int height,
								const unsigned int width,
			                    unsigned char* output);

template void transpose_cpu<unsigned int>(const unsigned int* input,
								const unsigned int height,
								const unsigned int width,
			                    unsigned int* output);

template void transpose_cpu<long long>(const long long* input,
								const unsigned int height,
								const unsigned int width,
			                    long long* output);

template void transpose_cpu<float>(const float* input,
								const unsigned int height,
								const unsigned int width,
			                    float* output);

}//namespace dtb 

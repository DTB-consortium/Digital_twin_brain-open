#pragma once
#include <hip/hip_runtime.h>

namespace dtb {

template<typename T>
void transpose_gpu(const T* input,
					const unsigned int height,
					const unsigned int width,
                    T* output,
					hipStream_t stream = NULL);

template<typename T>
void transpose_cpu(const T* input,
					const unsigned int height,
					const unsigned int width,
					T* output);
}//namespace istbi 
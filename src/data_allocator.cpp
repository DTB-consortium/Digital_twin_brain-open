#include <cassert>
#include <string.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#include "common.hpp"
#include "data_allocator.hpp"

namespace dtb {

template<typename T>
void DataAllocator<T>::malloc_host(void** ptr, size_t size)
{
  if(use_cuda_)
  {
    HIP_CHECK(hipHostMalloc(ptr, size));
  }
  else
  {
    *ptr = malloc(size);
  }
}

template<typename T>
void DataAllocator<T>::free_host(void* ptr)
{
  if (use_cuda_)
  {
    HIP_CHECK(hipHostFree(ptr));
  }
  else
  {
    free(ptr);
  }
}


template<typename T>
DataAllocator<T>::~DataAllocator()
{
	free_cpu_data();
	free_gpu_data();
}

template<typename T>
void DataAllocator<T>::free_cpu_data()
{
	if (cpu_ptr_)
	{
		free_host(cpu_ptr_);
	}
	cpu_ptr_ = nullptr;
}

template<typename T>
void DataAllocator<T>::free_gpu_data()
{
	if (gpu_ptr_)
	{
		int current_device;  // Just to check CUDA status:
		hipError_t status = hipGetDevice(&current_device);
		HIP_CHECK(hipFree(gpu_ptr_));
 	}
	gpu_ptr_ = nullptr;
}

template<typename T>
inline void DataAllocator<T>::to_cpu()
{
	if(NULL == cpu_ptr_)
	{
	  	malloc_host((void**)&cpu_ptr_, size_);
		assert(cpu_ptr_);
	    memset(cpu_ptr_, 0, size_);
	}
}

template<typename T>
inline void DataAllocator<T>::to_gpu()
{
	assert(use_cuda_);
	if(NULL == gpu_ptr_)
	{
	    HIP_CHECK_MALLOC(rank_, hipMalloc((void**)&gpu_ptr_, size_));
	    HIP_CHECK(hipMemset(gpu_ptr_, 0, size_));
	}
}

template<typename T>
const T* DataAllocator<T>::cpu_data()
{
	to_cpu();
	return cpu_ptr_;
}

template<typename T>
const T* DataAllocator<T>::gpu_data()
{
	to_gpu();
	return gpu_ptr_;
}

template<typename T>
T* DataAllocator<T>::mutable_cpu_data()
{
	to_cpu();
	return cpu_ptr_;
}

template<typename T>
T* DataAllocator<T>::mutable_gpu_data()
{
	to_gpu();
	return gpu_ptr_;
}

template class DataAllocator<float>;
template class DataAllocator<float2>;
template class DataAllocator<double>;
template class DataAllocator<double2>;
template class DataAllocator<int>;
template class DataAllocator<unsigned int>;
template class DataAllocator<uint2>;
template class DataAllocator<char>;
template class DataAllocator<unsigned char>;
template class DataAllocator<unsigned long long>;
template class DataAllocator<long long>;
template class DataAllocator<hiprandStatePhilox4_32_10_t>;

}//namespace istbi


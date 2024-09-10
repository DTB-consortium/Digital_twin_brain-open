#pragma once

#include <iostream>
#include <memory>
#include <hip/hip_runtime.h>
#include <hiprand.hpp>
#include <stdio.h>
#include <stdlib.h>

namespace dtb {

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)


template<typename T>
inline T divide_up(T x, T y) {
	return (x + y - 1) / y;
}

template<typename T>
inline bool is_pow2(T x){
    return ((x & (x-1)) == 0);
}

inline unsigned int power_radix2(unsigned int x)
{
	unsigned int log2x = 0;
	--x;
	for(; (x & 1) == 0; x >>= 1, log2x++);
	return log2x;
}


///> the biggest number n which is not greater than val and divisible by 2^power
template<int power, typename T>
inline T align_down(T val)
{
  return val & ~((1 << power) - 1);
}

///> the smallest number n which is not less than val and divisible by 2^power
template<int power, typename T>
inline T align_up(T val)
{
  return !(val & ((1 << power) - 1)) ? val : (val | ((1 << power) - 1)) + 1;
}

unsigned int warp_size();

// random seeding
unsigned long long gen_seed(); 

void report_mem_info(double* used_mem = nullptr);
void report_dev_info(double* used_mem = nullptr, double* total_mem = nullptr);

bool report_gpu_mem(const int rank, const char* hostname, const double limit);
bool report_mem(const int rank, const char* hostname, const double limit);

#define HIP_THREADS_PER_BLOCK 256
#define HIP_ITEMS_PER_THREAD 16


#ifndef DEVICE_RESET
#define DEVICE_RESET hipDeviceReset();
#endif

void check_malloc(hipError_t result, int rank, char const *const func, const char *const file, int const line);
void check_hip(hipError_t result, char const *const func, const char *const file, int const line);
void check_hiprand(hiprandStatus_t result, char const *const func, const char *const file, int const line);

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define HIP_CHECK(val)  check_hip((val), #val, __FILE__, __LINE__)
#define HIP_CHECK_MALLOC(rank, val)  check_malloc((val), (rank), #val, __FILE__, __LINE__)
#define HIPRAND_CHECK(val)  check_hiprand((val), #val, __FILE__, __LINE__)

void check_kernel(const char* err_msg, const char* file, const int line);
// This will output the proper error string when calling cudaGetLastError
#define HIP_POST_KERNEL_CHECK(msg)      check_kernel(msg, __FILE__, __LINE__)

// Shared Hip Stream for correct life cycle management
class HipStream {
	explicit HipStream(bool high_priority);
 
public:
	~HipStream();

	static std::shared_ptr<HipStream> create(bool high_priority = false) {
		std::shared_ptr<HipStream> pstream(new HipStream(high_priority));
		return pstream;
	}

	hipStream_t get() const {
		return stream_;
	}

private:
	hipStream_t stream_;
	DISABLE_COPY_AND_ASSIGN(HipStream);
};

class HipRandGenerator {
	HipRandGenerator(unsigned long long seed, hipStream_t stream);
public:
	~HipRandGenerator();

	static std::shared_ptr<HipRandGenerator> create(unsigned long long seed = gen_seed(), hipStream_t stream = nullptr) {
		std::shared_ptr<HipRandGenerator> phandle(new HipRandGenerator(seed, stream));
		return phandle;
	}

	hiprandGenerator_t get() const {
		return handle_;
	}
 private:
	hiprandGenerator_t handle_;
	DISABLE_COPY_AND_ASSIGN(HipRandGenerator);
};

}//namespace dtb

#pragma once

#include <limits.h>
#include <math.h>
#include <float.h>

namespace dtb {

inline __device__ unsigned int power_radix2(unsigned int x)
{
	unsigned int log2x = 0;
	--x;
	for(; (x & 1) == 0; x >>= 1, log2x++);
	return log2x;
}

template<typename T>
inline __device__ T divide_up(T x, T y)
{
	return (x + y - 1) / y;
}


///> the smallest number n which is not less than val and divisible by 2^power
template<int power, typename T>
inline __device__ T align_up(T val)
{
  return !(val & ((1 << power) - 1)) ? val : (val | ((1 << power) - 1)) + 1;
}

template<typename T>
inline __host__ __device__ T texp(T x);

template<>
inline __host__ __device__ float texp<float>(float x){
	return expf(x);
}

template<>
inline __host__ __device__ double texp<double>(double x){
	return exp(x);
}

template<typename T>
inline __host__ __device__ T tmax(T x, T y);

template<>
inline __host__ __device__ float tmax<float>(float x, float y){
	return fmaxf(x, y);
}

template<>
inline __host__ __device__ double tmax<double>(double x, double y){
	return fmax(x, y);
}

template<typename T>
inline __host__ __device__ T tabs(T x);

template<>
inline __host__ __device__ float tabs<float>(float x){
	return fabsf(x);
}

template<>
inline __host__ __device__ double tabs<double>(double x){
	return fabs(x);
}

//the biggest number n which is not greater than val and divisible by warpSize
__device__ __forceinline__ unsigned int align_warp_down(unsigned int val)
{
  return val & ~(warpSize - 1);
}

//the smallest number n which is not less than val and divisible by warpSize
__device__ __forceinline__ unsigned int align_warp_up(unsigned int val)
{
  return !(val & (warpSize - 1)) ? val : (val | (warpSize - 1)) + 1;
}

template<typename T>
__device__ __forceinline__ T atomic_add(T* addr, T val)
{
	return atomicAdd(addr, val);
}

template<typename T, typename TR>
__device__ __forceinline__ TR tsum(T a, T b)
{
  return TR(a + b);
}

template<typename T, typename TR>
__device__ __forceinline__ void tsum_replace(volatile TR *a, T b)
{
	*a += b;
}

template<typename T, typename TR>
__device__ __forceinline__ void tmax_replace(volatile TR *a, T b)
{
	if(b > *a)
	{
		*a = b;
	}
}

template<typename T>
inline __host__ __device__ T tpow(T x, T y);

template<>
inline __host__ __device__ float tpow<float>(float x, float y){
	return powf(x, y);
}

template<>
inline __host__ __device__ double tpow<double>(double x, double y){
	return pow(x, y);
}

template<typename T>
inline __host__ __device__ T tsqrt(T x);

template<>
inline __host__ __device__ float tsqrt<float>(float x){
	return sqrtf(x);
}

template<>
inline __host__ __device__ double tsqrt<double>(double x){
	return sqrt(x);
}

template<typename T>
inline __host__ __device__ T tlog(T x);

template<>
inline __host__ __device__ float tlog<float>(float x){
	return logf(x);
}

template<>
inline __host__ __device__ double tlog<double>(double x){
	return log(x);
}

template <typename T>
struct numeric_limits {
};

// WARNING: the following numeric_limits definitions are there only to support
//          HIP compilation for the moment. Use std::numeric_limits if you are not
//          compiling for ROCm.
//          from @colesbury: "The functions on numeric_limits aren't marked with
//          __device__ which is why they don't work with ROCm. CUDA allows them
//          because they're constexpr."

namespace {
  // ROCm doesn't like INFINITY too.
  constexpr double inf = INFINITY;
}

template <>
struct numeric_limits<uint8_t> {
  static inline __host__ __device__ uint8_t lowest() { return 0; }
  static inline __host__ __device__ uint8_t max() { return UINT8_MAX; }
  static inline __host__ __device__ uint8_t lower_bound() { return 0; }
  static inline __host__ __device__ uint8_t upper_bound() { return UINT8_MAX; }
};

template <>
struct numeric_limits<int8_t> {
  static inline __host__ __device__ int8_t lowest() { return INT8_MIN; }
  static inline __host__ __device__ int8_t max() { return INT8_MAX; }
  static inline __host__ __device__ int8_t lower_bound() { return INT8_MIN; }
  static inline __host__ __device__ int8_t upper_bound() { return INT8_MAX; }
};

template <>
struct numeric_limits<int16_t> {
  static inline __host__ __device__ int16_t lowest() { return INT16_MIN; }
  static inline __host__ __device__ int16_t max() { return INT16_MAX; }
  static inline __host__ __device__ int16_t lower_bound() { return INT16_MIN; }
  static inline __host__ __device__ int16_t upper_bound() { return INT16_MAX; }
};

template <>
struct numeric_limits<int32_t> {
  static inline __host__ __device__ int32_t lowest() { return INT32_MIN; }
  static inline __host__ __device__ int32_t max() { return INT32_MAX; }
  static inline __host__ __device__ int32_t lower_bound() { return INT32_MIN; }
  static inline __host__ __device__ int32_t upper_bound() { return INT32_MAX; }
};

template <>
struct numeric_limits<int64_t> {
#ifdef _MSC_VER
  static inline __host__ __device__ int64_t lowest() { return _I64_MIN; }
  static inline __host__ __device__ int64_t max() { return _I64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return _I64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return _I64_MAX; }
#else
  static inline __host__ __device__ int64_t lowest() { return INT64_MIN; }
  static inline __host__ __device__ int64_t max() { return INT64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return INT64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return INT64_MAX; }
#endif
};

template <>
struct numeric_limits<float> {
  static inline __host__ __device__ float lowest() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }
  static inline __host__ __device__ float lower_bound() { return -static_cast<float>(inf); }
  static inline __host__ __device__ float upper_bound() { return static_cast<float>(inf); }
};

template <>
struct numeric_limits<double> {
  static inline __host__ __device__ double lowest() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }
  static inline __host__ __device__ double lower_bound() { return -inf; }
  static inline __host__ __device__ double upper_bound() { return inf; }
};

#define BLOCK_REDUCE_ASUM_TWO(TNUM) 			\
if (BlockSize >= (TNUM) * 2) { 					\
  if (tid < (TNUM)) { 							\
    tsum_replace(st1, s_sum1[tid + (TNUM)]); 	\
    tsum_replace(st2, s_sum2[tid + (TNUM)]); 	\
  } 											\
  __syncthreads(); 								\
}

#define REDUCE_ASUM_TOW(TNUM) 					\
if (tid + (TNUM) < thread_count) { 				\
  tsum_replace(st1, s_sum1[tid + (TNUM)]); 		\
  tsum_replace(st2, s_sum2[tid + (TNUM)]); 		\
  __syncthreads(); 								\
}


template<typename T, typename T2, unsigned int BlockSize>
__device__ void asum_shmem(volatile T *s_sum1, const T sum1, volatile T2 *s_sum2,
								const T2 sum2, unsigned int tid)
{
	const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
  	volatile T* st1 = s_sum1 + tid;
	volatile T2* st2 = s_sum2 + tid;
	*st1 = sum1;
	*st2 = sum2;
  	__syncthreads();
    // do reduction in shared mem
  	BLOCK_REDUCE_ASUM_TWO(256)
  	BLOCK_REDUCE_ASUM_TWO(128)
  	BLOCK_REDUCE_ASUM_TWO(64)
  	
    if (tid < 32)
    {
    	REDUCE_ASUM_TOW(32)
    	REDUCE_ASUM_TOW(16)
    	REDUCE_ASUM_TOW(8)
    	REDUCE_ASUM_TOW(4)
    	REDUCE_ASUM_TOW(2)
    	REDUCE_ASUM_TOW(1)
    }
}

#define BLOCK_REDUCE_ASUM(TNUM) 				\
if (BlockSize >= (TNUM) * 2) { 					\
  if (tid < (TNUM)) { 							\
    tsum_replace(st, s_sum[tid + (TNUM)]); 		\
  } 											\
  __syncthreads(); 								\
}

#define REDUCE_ASUM(TNUM) 						\
if (tid + (TNUM) < thread_count) { 				\
  tsum_replace(st, s_sum[tid + (TNUM)]); 		\
  __syncthreads(); 								\
}

template<typename T, unsigned int BlockSize>
__device__ void asum_shmem(volatile T *s_sum, const T sum, unsigned int tid)
{
	const unsigned int thread_count = blockDim.x * blockDim.y * blockDim.z;
  	volatile T* st = s_sum + tid;
	*st = sum;
  	__syncthreads();
    // do reduction in shared mem
  	BLOCK_REDUCE_ASUM(256)
  	BLOCK_REDUCE_ASUM(128)
  	BLOCK_REDUCE_ASUM(64)
  	
    if (tid < 32)
    {
    	REDUCE_ASUM(32)
    	REDUCE_ASUM(16)
    	REDUCE_ASUM(8)
    	REDUCE_ASUM(4)
    	REDUCE_ASUM(2)
    	REDUCE_ASUM(1)
    }
}

// Goes from (0, 1] to [0, 1). Note 1-x is not sufficient since for some floats
// eps near 0, 1-eps will round to 1.
template<typename T>
__device__ inline T reverse_bounds(T value) 
{
  if (value == static_cast<T>(1))
  {
    return static_cast<T>(0);
  }
  
  return value;
}

}

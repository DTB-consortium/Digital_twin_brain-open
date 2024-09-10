#pragma once

#include <memory>
#include <stdint.h>
#include "data_allocator.hpp"
#include <hip/hip_fp16.h>

namespace dtb {

/// \brief Half-precision floating point type
using half = ::__half;
using half2 = ::__half2;

enum DataType
{
	DOUBLE,
	FLOAT,
	FLOAT16,
	INT32,
	INT8
};

size_t data_size(DataType dtype);

struct Weights
{
  DataType type_;
  std::shared_ptr<DataAllocator<char>> data_;

  size_t elem_size() const
  {
  	return 2 * data_size(type_);
  }
  size_t count() const
  {
  	return data_->size() / elem_size();
  }
};

template <typename T>
constexpr DataType TDataType();

template <>
inline constexpr DataType TDataType<double>() {
  return DOUBLE;
}
template <>
inline constexpr DataType TDataType<float>() {
  return FLOAT;
}

template <>
inline constexpr DataType TDataType<half>() {
  return FLOAT16;
}

template <>
inline constexpr DataType TDataType<int32_t>() {
  return INT32;
}

template <>
inline constexpr DataType TDataType<int8_t>() {
  return INT8;
}

template <typename T>
__host__ __device__ bool is_type(DataType dtype);

template <>
__host__ __device__ inline bool is_type<double>(DataType dtype) {
  return dtype == DOUBLE;
}
template <>
__host__ __device__ inline bool is_type<float>(DataType dtype) {
  return dtype == FLOAT;
}

template <>
__host__ __device__ inline bool is_type<int32_t>(DataType dtype) {
  return dtype == INT32;
}

template <>
__host__ __device__ inline bool is_type<half>(DataType dtype) {
  return dtype == FLOAT16;
}

template <>
__host__ __device__ inline bool is_type<int8_t>(DataType dtype) {
  return dtype == INT8;
}

}

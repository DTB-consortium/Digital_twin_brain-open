#pragma once

#include "common.hpp"

//digital twin brain
namespace dtb {

template<typename T>
class DataAllocator
{
 public:
  explicit DataAllocator(int rank, size_t size, bool use_cuda = true)
      : rank_(rank), cpu_ptr_(NULL), gpu_ptr_(NULL), count_(size / sizeof(T)), size_(size), use_cuda_(use_cuda) {}
  ~DataAllocator();
  void free_cpu_data();
  void free_gpu_data();
  const T* cpu_data();
  const T* gpu_data();
  T* mutable_cpu_data();
  T* mutable_gpu_data();
  size_t size() { return size_; }
  size_t count() { return count_; }
  
 private:
  void malloc_host(void** ptr, size_t size);
  void free_host(void* ptr);
  void to_cpu();
  void to_gpu();
  T* cpu_ptr_;
  T* gpu_ptr_;
  size_t size_;
  size_t count_;
  bool use_cuda_;
  int rank_;

  DISABLE_COPY_AND_ASSIGN(DataAllocator);
  
};  // class SyncedMemory

}// namespace istbi

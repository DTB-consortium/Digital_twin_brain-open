#include "stage.hpp"
#include "common.hpp"
#include <rocprim/rocprim.hpp>

namespace dtb {
template<
		unsigned int BlockSize,
		unsigned int ItemsPerThread,
		class InputIterator,
		class OutputIterator,
		class OffsetIterator,
		class InputType,
		class ResultType,
		class CompareFunction = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>
>
__global__ 
__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) 
void segmented_find_kernel(InputIterator input,
						OutputIterator output,
						OffsetIterator begin_offsets,
	                    OffsetIterator end_offsets,
	                    InputType value,
	                    ResultType invaild_value,
	                    CompareFunction compare_op = CompareFunction())
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
    const unsigned int segment_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();

    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset = end_offsets[segment_id];
	__shared__ output_type result;

	 // Empty segment
    if(end_offset <= begin_offset)
    {
        if(flat_id == 0)
        {
            output[segment_id] = invaild_value;
        }
        return;
    }

    unsigned int block_offset = begin_offset;
	input_type values[ItemsPerThread];
	
	if(flat_id == 0)
		result = invaild_value;
	__syncthreads();
	
	// Load next full blocks and continue
	while(block_offset + items_per_block < end_offset)
	{
		::rocprim::block_load_direct_striped<BlockSize>(flat_id, input + block_offset, values);
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			if(result != invaild_value)
				break;
		    bool flag = compare_op(value, values[i]);
			if(flag)
			{
				result = (flat_id + i * BlockSize + block_offset);
			}
			__syncthreads();
		}
		block_offset += items_per_block;
	}
	
    if((result == invaild_value) && (block_offset + items_per_block) > end_offset)
    {
         // Load the last (probably partial) block and continue
        const unsigned int valid_count = end_offset - block_offset;
        ::rocprim::block_load_direct_striped<BlockSize>(flat_id, input + block_offset, values, valid_count);
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
        	if(result != invaild_value)
				break;
        	unsigned int offset = flat_id + i * BlockSize;
            if(offset < valid_count)
            {
            	bool flag = compare_op(value, values[i]);
				if(flag)
				{
	            	result = block_offset + offset;
            	}
            }

			__syncthreads();
        }
    }

	if(flat_id == 0)
		output[segment_id] = result;
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto error = hipPeekAtLastError(); \
        if(error != hipSuccess) return error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto error = hipStreamSynchronize(stream); \
            if(error != hipSuccess) return error; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }


template<
	class InputIterator,
	class OutputIterator,
	class OffsetIterator,
	class InputType,
	class ResultType
>
hipError_t segmented_find_impl(InputIterator input,
								OutputIterator output,
								const size_t segments,
								OffsetIterator begin_offsets,
								OffsetIterator end_offsets,
								InputType value,
								ResultType invalid_value,
								hipStream_t stream,
								bool debug_synchronous)
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using result_type = typename std::iterator_traits<OutputIterator>::value_type;

	if( segments == 0u )
        return hipSuccess;

	std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
	
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(segmented_find_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
					dim3(segments),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					input,
					output,
					begin_offsets,
					end_offsets,
					static_cast<input_type>(value),
					static_cast<result_type>(invalid_value));
	HIP_POST_KERNEL_CHECK("segmented_find_kernel");

	ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_find_kernel", segments, start);

    return hipSuccess;
	
}

void fetch_sample_offset_gpu(const unsigned int* colinds,
								const unsigned int segments,
								const unsigned int* begin_offsets,
								const unsigned int* end_offsets,
								const unsigned int nid,
								unsigned int* results,
								hipStream_t stream,
								bool debug_synchronous)
{
	HIP_CHECK(segmented_find_impl(colinds, 
							results, 
							segments, 
							begin_offsets, 
							end_offsets, 
							nid, 
							static_cast<unsigned int>(-1), 
							stream, 
							debug_synchronous));
	
}

}//namespace dtb


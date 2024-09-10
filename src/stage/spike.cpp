#include "stage.hpp"
#include "common.hpp"
#include "device_function.hpp"
#include <iostream>
#include <rocprim/rocprim.hpp>

namespace dtb {

template<typename T>
struct reset_spike_op
{
	__host__ __device__ inline
	unsigned char operator()(const T& a, const T& b) const
	{
		return (a == b) ? 0x01 : 0x00;
	}
};

template<typename T>
void reset_spike_gpu(const unsigned int n,
						const T* v_membranes,
						const T* v_thresholds,
						unsigned char* f_actives,
						hipStream_t stream)
{

	reset_spike_op<T> op;
	::rocprim::transform(v_membranes, v_thresholds, f_actives, n, op, stream);
}

template<typename T>
void init_spike_time_gpu(const unsigned int n,
							const T* vals,
							const T scale,
							int* t_actives,
							hipStream_t stream)
{
	::rocprim::multiplies<T> op;
	::rocprim::transform(vals, ::rocprim::make_constant_iterator<T>(scale), t_actives, n, op, stream);
}

template<typename T>
static __global__ void update_spike_kernel(hiprandStatePhilox4_32_10_t* states,
											const unsigned int n,
											const T* noise_rates,
											const unsigned char* f_actives,
											const T a,
											const T b,
											unsigned char* f_recorded_actives,
											T* samples)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
  	for(unsigned int i = idx; i < n; i += gridSize)
	{
		T x = static_cast<T>(hiprand_uniform(&states[i]));
		if(i < n)
		{
			unsigned char fi = f_actives[i];
			T y = reverse_bounds<T>(x) * (b - a) + a; 
     	 	fi |= static_cast<unsigned char>(y < noise_rates[i]);
			f_recorded_actives[i] = fi;
			if(NULL != samples)
				samples[i] = y;
		}
	}
}

template<typename T>
void update_spike_gpu(hiprandStatePhilox4_32_10_t* states,
						const unsigned int n,
						const T* noise_rates,
						const unsigned char* f_actives,
						unsigned char* f_recorded_actives,
						const T a,
						const T b,
						T* samples,
						hipStream_t stream)
{
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(update_spike_kernel<T>),
					dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					states,
					n,
					noise_rates,
					f_actives,
					a,
					b,
					f_recorded_actives,
					samples);
	HIP_POST_KERNEL_CHECK("update_spike_kernel");
}

void update_accumulated_spike_gpu(const unsigned char* f_actives,
										const unsigned int n,
										unsigned char* f_recorded_actives,
										hipStream_t stream)
{
	auto transform_op = [] __device__ (const unsigned char a, const unsigned char b) -> unsigned char {
        return a | b;
	};
	
	::rocprim::transform(f_actives, f_recorded_actives, f_recorded_actives, n, transform_op, stream);
}

template<
	unsigned int BlockSize,
	unsigned int ItemsPerThread,
	class InputIterator,
	class OutputIterator
>
__global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU) 
void transform_kernel(
				InputIterator inputs,
				const size_t size,
				OutputIterator outputs)
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	static_assert(::rocprim::is_integral<output_type>::value, "Type IndexIterator must be integral type");
	static_assert(std::is_convertible<input_type, output_type>::value,
                      "The type OutputIterator must be such that an object of type InputIterator"
                      "can be dereferenced and then implicitly converted to OutputIterator.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr int items_per_block = BlockSize * ItemsPerThread;
	auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    unsigned int block_offset = (flat_block_id * items_per_block);

	output_type values[ItemsPerThread];
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values,
            valid_in_last_block
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = flat_id + i * BlockSize;
			if(offset < valid_in_last_block)
			{
				outputs[block_offset + offset] = static_cast<output_type>(inputs[values[i]]);
			}
		}
    }
    else
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = flat_id + i * BlockSize;
			outputs[block_offset + offset] = static_cast<output_type>(inputs[values[i]]);

		}
    }
}


template<
	unsigned int BlockSize,
	unsigned int ItemsPerThread,
	class InputIterator,
	class OutputIterator,
	class Default
>
__global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void fill_kernel(
				InputIterator inputs,
				const size_t size,
				OutputIterator outputs,
				Default default_value)
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	static_assert(::rocprim::is_integral<input_type>::value, "Type IndexIterator must be integral type");
	static_assert(std::is_convertible<Default, output_type>::value,
					  "The type OutputIterator must be such that an object of type Default "
					  "can be dereferenced and then implicitly converted to type OutputIterator.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr int items_per_block = BlockSize * ItemsPerThread;
	auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    unsigned int block_offset = (flat_block_id * items_per_block);

	input_type values[ItemsPerThread];
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            inputs + block_offset,
            values,
            valid_in_last_block
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = i * BlockSize;
			if(flat_id + offset < valid_in_last_block)
			{
				outputs[values[i]] = static_cast<output_type>(default_value);
			}
		}
    }
    else
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            inputs + block_offset,
            values
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			outputs[values[i]] = static_cast<output_type>(default_value);
		}
    }
}


template<
	unsigned int BlockSize,
	class T,
	unsigned int ItemsPerThread,
    class InputIterator,
    class IndexIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<T>
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void expand_and_reduce_kernel(
					InputIterator inputs,
					IndexIterator indice,
					const size_t size,
					OutputIterator outputs,
					T* reductions,
					const T out_of_bound = (T)0,
					const T initial_value = (T)0,
					BinaryFunction reduce_op = BinaryFunction())
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	using index_type = typename std::iterator_traits<IndexIterator>::value_type;

	static_assert(::rocprim::is_integral<index_type>::value, "Type IndexIterator must be integral type");
	static_assert(std::is_convertible<input_type, output_type>::value,
					  "The type OutputIterator must be such that an object of type InputIterator "
					  "can be dereferenced and then implicitly converted to type OutputIterator.");
	static_assert(std::is_convertible<output_type, T>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly converted to T.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr int items_per_block = BlockSize * ItemsPerThread;
	auto valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    unsigned int block_offset = (flat_block_id * items_per_block);

	using block_reduce_type = ::rocprim::block_reduce<
        T, BlockSize,
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
	
	union{
		output_type values[ItemsPerThread];
		index_type indice[ItemsPerThread];
	} storage;

	T reduction = out_of_bound;
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            indice + block_offset,
            storage.indice,
            valid_in_last_block
        );

		#pragma unroll
		for(unsigned int i = 0; i < ItemsPerThread; i++)
		{
			unsigned int offset = i * BlockSize;
			if(flat_id + offset < valid_in_last_block)
			{
				storage.values[i] = inputs[storage.indice[i]];
				reduction = reduce_op(reduction, static_cast<T>(storage.values[i]));
			}
		}

		 block_reduce_type()
            .reduce(
                reduction, // input
                reduction, // output
                reduce_op
            );
		 
		 ::rocprim::block_store_direct_striped<BlockSize>(
			 flat_id,
			 outputs + block_offset,
			 storage.values,
			 valid_in_last_block
		 );
    }
    else
    {
        ::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            indice + block_offset,
            storage.indice
        );

		#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
			storage.values[i] = inputs[storage.indice[i]];
			reduction = reduce_op(reduction, static_cast<T>(storage.values[i]));
        }

		// load input values into values
        block_reduce_type()
            .reduce(
                reduction, // input
                reduction, // output
                reduce_op
            );

		::rocprim::block_store_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            storage.values
        );
    }

	// Save block reduction
    if(flat_id == 0)
    {
       reductions[flat_block_id] = reduce_op(reduction, initial_value);
    }
}

template<
	unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class InputIterator,
    class OffsetIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<OutputIterator>::value_type>,
    class Default = typename std::iterator_traits<OutputIterator>::value_type
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void exclusive_scan_and_select_kernel(
						InputIterator inputs,
						const size_t size,
						OffsetIterator offsets,
						OutputIterator outputs,
						Default out_of_bound = Default(),
						BinaryFunction scan_op = BinaryFunction())
{	
	using offset_type = typename std::iterator_traits<OffsetIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;
	static_assert(std::is_convertible<offset_type, output_type>::value,
                      "The type OutputIterator must be such that an object of type OffsetIterator "
                      "can be dereferenced and then implicitly converted to OutputIterator.");
	
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto flat_block_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();
	const unsigned int number_of_blocks = hipGridDim_x;
	constexpr auto items_per_block = BlockSize * ItemsPerThread;
	unsigned int valid_in_last_block = size - items_per_block * (number_of_blocks - 1);
    auto block_offset = (flat_block_id * items_per_block);

   using block_load_type = ::rocprim::block_load<
        output_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose
    >;
    using block_exchange_type = ::rocprim::block_exchange<
        output_type, BlockSize, ItemsPerThread
    >;

	using block_scan_type = ::rocprim::block_scan<
        output_type, BlockSize,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;

    __shared__ union
    {
        typename block_load_type::storage_type load;
        typename block_exchange_type::storage_type exchange;
        typename block_scan_type::storage_type scan;
    } storage;

    output_type values[ItemsPerThread];
	offset_type offset = offsets[flat_block_id];

	if(offsets[flat_block_id + 1] <= offset)
		return;
	
	if(flat_block_id == (number_of_blocks - 1)) // last block
    {
	    // load input values into values
	    block_load_type()
	        .load(
	            inputs + block_offset,
	            values,
	            valid_in_last_block,
	            out_of_bound,
	            storage.load
	        );
		 
	    ::rocprim::syncthreads(); // sync threads to reuse shared memory

		block_scan_type()
			.exclusive_scan(
	        values, // input
	        values, // output
	        static_cast<output_type>(offset),
	        storage.scan,
	        scan_op
	    );

		::rocprim::syncthreads(); // sync threads to reuse shared memory

		// Save values into output array
		block_exchange_type()
			.blocked_to_striped(
			   values, 
			   values,
			   storage.exchange
			);

		::rocprim::syncthreads(); // sync threads to reuse shared memory

        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            auto _offset = i * BlockSize + flat_id;
            if(_offset < valid_in_last_block && inputs[block_offset + _offset])
            {
                outputs[values[i]] = block_offset + _offset;
            }
        }

    }
    else
    {
    	 // load input values into values
	    block_load_type()
	        .load(
	            inputs + block_offset,
	            values,
	            storage.load
	        );
		 
	    ::rocprim::syncthreads(); // sync threads to reuse shared memory

		block_scan_type()
			.exclusive_scan(
	        values, // input
	        values, // output
	        static_cast<output_type>(offset),
	        storage.scan,
	        scan_op
	    );

		::rocprim::syncthreads(); // sync threads to reuse shared memory

		// Save values into output array
		block_exchange_type()
		.blocked_to_striped(
		   values, 
		   values,
		   storage.exchange
		);

		::rocprim::syncthreads(); // sync threads to reuse shared memory
		
		#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            auto _offset = i * BlockSize + flat_id;
            if(inputs[block_offset + _offset])
            {
                 outputs[values[i]] = block_offset + _offset;
            }
        }
    }
	
}

template<
	unsigned int BlockSize,
	unsigned int ItemsPerThread,
	class InputIterator,
	class OffsetIterator,
    class OutputIterator,
	class BinaryFunction = ::rocprim::minus<typename std::iterator_traits<OutputIterator>::value_type>
>
static __global__

__launch_bounds__(BlockSize, ROCPRIM_DEFAULT_MIN_WARPS_PER_EU)
void segmented_transform_kernel(
						InputIterator inputs,
						OffsetIterator begin_offsets,
						OffsetIterator end_offsets,
						OutputIterator outputs,
						BinaryFunction transform_op = BinaryFunction())
{
	using input_type = typename std::iterator_traits<InputIterator>::value_type;
	using output_type = typename std::iterator_traits<OutputIterator>::value_type;

	static_assert(std::is_convertible<input_type, output_type>::value,
	                  "The type OutputIterator must be such that an object of type OffsetIterator "
	                  "can be dereferenced and then implicitly converted to OutputIterator.");

	constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
	const auto flat_id = ::rocprim::flat_block_thread_id<BlockSize, 1, 1>();
	const auto segment_id = ::rocprim::flat_block_id<BlockSize, 1, 1>();

	const unsigned int begin_offset = begin_offsets[segment_id];
	const unsigned int end_offset = end_offsets[segment_id];

	// Empty segment
	if(end_offset <= begin_offset)
	{
	    return;
	}

	const output_type segment_value = static_cast<output_type>(inputs[segment_id]);
    output_type values[ItemsPerThread];
	
	unsigned int block_offset = begin_offset;

	// Load next full blocks and continue transforming
	for(;block_offset + items_per_block <= end_offset; block_offset += items_per_block)
	{
		::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values
        );
		
		#pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
	        values[i] = transform_op(values[i], segment_value);
        }

        ::rocprim::block_store_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values
        );
	}
		
	if(block_offset < end_offset)
	{
		const unsigned int valid_count = end_offset - block_offset;
		::rocprim::block_load_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values,
            valid_count
        );

        #pragma unroll
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(BlockSize * i + flat_id < valid_count)
            {
                values[i] = transform_op(values[i], segment_value);
            }
        }

        ::rocprim::block_store_direct_striped<BlockSize>(
            flat_id,
            outputs + block_offset,
            values,
            valid_count
        );
	} 
}

void count_sending_spikes_temporary_storage_size(const unsigned int sending_count,
														const unsigned int segments,
														unsigned int* block_rowptrs,
														unsigned int* active_rowptrs,
														size_t& storage_size_bytes,
														hipStream_t stream)
{
	constexpr auto items_per_block = HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD;
	const unsigned int number_of_blocks =
			std::max(1u, divide_up<unsigned int>(sending_count, items_per_block));
	size_t temporary_storage_size_bytes = 0;
	
	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                temporary_storage_size_bytes,
                                active_rowptrs + segments + 1,
                                active_rowptrs,
                                0,
                                segments + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));
	storage_size_bytes = std::max(temporary_storage_size_bytes, sending_count * sizeof(unsigned int));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                temporary_storage_size_bytes,
                                block_rowptrs + number_of_blocks + 1,
                                block_rowptrs,
                                0,
                                number_of_blocks + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	storage_size_bytes = std::max(temporary_storage_size_bytes, storage_size_bytes);
}

void update_sending_spikes_gpu(const unsigned char* f_actives,
									const unsigned int* sending_rowptrs,
									const unsigned int* sending_colinds,
									const unsigned int segments,
									const unsigned int sending_count,
									unsigned char* f_sending_actives,
									unsigned int* block_rowptrs,
									unsigned int* active_rowptrs,
									unsigned int* active_colinds,
									hipStream_t stream)
{

	constexpr auto items_per_block = HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD;
	const unsigned int number_of_blocks =
			std::max(1u, divide_up<unsigned int>(sending_count, items_per_block));
	
	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(expand_and_reduce_kernel<HIP_THREADS_PER_BLOCK, unsigned int, HIP_ITEMS_PER_THREAD>),
					dim3(number_of_blocks), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					f_actives,
					sending_colinds,
					sending_count,
					f_sending_actives,
					block_rowptrs + number_of_blocks + 1);
	HIP_POST_KERNEL_CHECK("expand_and_reduce_kernel");

	size_t  storage_size_bytes = 4; 
	
    HIP_CHECK(
        rocprim::segmented_reduce(
            static_cast<void*>(active_colinds),
            storage_size_bytes,
            f_sending_actives,
            active_rowptrs + segments + 1,
            segments,
            sending_rowptrs,
            sending_rowptrs + 1,
            ::rocprim::plus<unsigned int>(), 
            0,
            stream
        )
    );
	
	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                storage_size_bytes,
                                active_rowptrs + segments + 1,
                                active_rowptrs,
                                0,
                                segments + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(static_cast<void*>(active_colinds),
                                storage_size_bytes,
                                active_rowptrs + segments + 1,
                                active_rowptrs,
                                0,
                                segments + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(nullptr,
                                storage_size_bytes,
                                block_rowptrs + number_of_blocks + 1,
                                block_rowptrs,
                                0,
                                number_of_blocks + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	HIP_CHECK(
	 	::rocprim::exclusive_scan(static_cast<void*>(active_colinds),
                                storage_size_bytes,
                                block_rowptrs + number_of_blocks + 1,
                                block_rowptrs,
                                0,
                                number_of_blocks + 1,
                                ::rocprim::plus<unsigned int>(),
                                stream));

	hipLaunchKernelGGL(
					HIP_KERNEL_NAME(exclusive_scan_and_select_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
					dim3(number_of_blocks), 
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					stream,
					f_sending_actives,
					sending_count,
					block_rowptrs,
					active_colinds,
					0);
	HIP_POST_KERNEL_CHECK("exclusive_scan_and_select_kernel");

	if(segments > 1)
	{
		hipLaunchKernelGGL(
						HIP_KERNEL_NAME(segmented_transform_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
						dim3(segments - 1), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						sending_rowptrs + 1,
						active_rowptrs + 1,
						active_rowptrs + 2,
						active_colinds);
		HIP_POST_KERNEL_CHECK("segmented_transform_kernel");
		
	}
	
}

void update_recving_spikes_gpu(const unsigned int* inputs,
									const unsigned int* rowptrs,
									const unsigned int  segments,
									unsigned int* outputs,	
									hipStream_t stream)
{
	if(segments > 1)
	{
		hipLaunchKernelGGL(
						HIP_KERNEL_NAME(segmented_transform_kernel<HIP_THREADS_PER_BLOCK, HIP_ITEMS_PER_THREAD>),
						dim3(segments - 1), 
						dim3(HIP_THREADS_PER_BLOCK),
						0,
						stream,
						inputs + 1,
						rowptrs + 1,
						rowptrs + 2,
						outputs,
						::rocprim::plus<unsigned int>());
		HIP_POST_KERNEL_CHECK("segmented_transform_kernel");
	}
}


template void reset_spike_gpu<float>(const unsigned int n,
									const float* v_membranes,
									const float* v_thresholds,
									unsigned char* f_actives,
									hipStream_t stream);

template void reset_spike_gpu<double>(const unsigned int n,
									const double* v_membranes,
									const double* v_thresholds,
									unsigned char* f_actives,
									hipStream_t stream);


template void init_spike_time_gpu<float>(const unsigned int n,
										const float* vals,
										const float scale,
										int* t_actives,
										hipStream_t stream);

template void init_spike_time_gpu<double>(const unsigned int n,
										const double* vals,
										const double scale,
										int* t_actives,
										hipStream_t stream);


template void update_spike_gpu<float>(hiprandStatePhilox4_32_10_t* states,
									const unsigned int n,
									const float* noise_rates,
									const unsigned char* f_actives,
									unsigned char* f_recorded_actives,
									const float a,
									const float b,
									float* samples,
									hipStream_t stream);

template void update_spike_gpu<double>(hiprandStatePhilox4_32_10_t* states,
									const unsigned int n,
									const double* noise_rates,
									const unsigned char* f_actives,
									unsigned char* f_recorded_actives,
									const double a,
									const double b,
									double* samples,
									hipStream_t stream);

}//namespace istbi 

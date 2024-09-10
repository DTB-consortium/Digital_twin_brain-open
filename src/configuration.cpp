#include <thrust/detail/type_traits.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/find.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include "configuration.hpp"
#include "common.hpp"
#include "device_function.hpp"
#include "util/transpose.hpp"
#include "util/cnpy.h"
#include "logging.hpp"

namespace dtb {

#define MAX_ELEMS (64 * 1024 * 1024)

template<typename T>
static __global__ void merge_kernel(const T* highs,
									const unsigned int* lows,
									const unsigned int n,
									unsigned long long* vals)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
  		unsigned long long val = (static_cast<unsigned long long>(highs[i]) << 32);
		val |= lows[i];
  		vals[i] = val;
	}
}

template<typename T>
static void merge(const T* highs,
					const unsigned int* lows,
					const unsigned int n,
					thrust::device_vector<unsigned long long>& d_vals)
{
	thrust::device_vector<T> d_highs;
	thrust::device_vector<unsigned int> d_lows;
	unsigned int offset = 0;

	if(n > MAX_ELEMS)
	{
		d_highs.resize(MAX_ELEMS);
		d_lows.resize(MAX_ELEMS);
	}
	else
	{
		d_highs.resize(n);
		d_lows.resize(n);
	}
	
	do{
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_highs.data()), highs + offset, size * sizeof(T), hipMemcpyHostToDevice));
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_lows.data()), lows + offset, size * sizeof(unsigned int), hipMemcpyHostToDevice));
		merge_kernel<T><<<
					dim3(divide_up<unsigned int>(size, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					0>>>(
					thrust::raw_pointer_cast(d_highs.data()),
					thrust::raw_pointer_cast(d_lows.data()),
					size,
					thrust::raw_pointer_cast(d_vals.data()) + offset);
	
		HIP_POST_KERNEL_CHECK("merge_kernel");
		HIP_CHECK(hipDeviceSynchronize());
		offset += size;
	}while(offset < n);
	assert(offset == n);
}


template<typename T>
static __global__ void split_kernel(const unsigned long long* vals,
									const unsigned int n,
									T* highs,
									unsigned int* lows)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
  	{
  		unsigned long long val = vals[i];
		highs[i] = static_cast<T>(val >> 32);
		lows[i] = static_cast<unsigned int>(val & 0xffffffffu);
	}
}

template<typename T>
static void split(const thrust::device_vector<unsigned long long>& d_vals,
				const unsigned int n,
				T* highs,
				unsigned int* lows)

{
	assert(n <= d_vals.size());
	thrust::device_vector<T> d_highs;
	thrust::device_vector<unsigned int> d_lows;

	unsigned int offset = 0;
	if(n > MAX_ELEMS)
	{
		d_highs.resize(MAX_ELEMS);
		d_lows.resize(MAX_ELEMS);
	}
	else
	{
		d_highs.resize(n);
		d_lows.resize(n);
	}
	
	do{
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		split_kernel<<<
					dim3(divide_up<unsigned int>(size, HIP_THREADS_PER_BLOCK)),
					dim3(HIP_THREADS_PER_BLOCK),
					0,
					0>>>(
					thrust::raw_pointer_cast(d_vals.data()) + offset,
					size,
					thrust::raw_pointer_cast(d_highs.data()),
					thrust::raw_pointer_cast(d_lows.data()));
	
		HIP_POST_KERNEL_CHECK("split_kernel");
		HIP_CHECK(hipMemcpy(highs + offset, thrust::raw_pointer_cast(d_highs.data()), size * sizeof(T), hipMemcpyDeviceToHost));
		HIP_CHECK(hipMemcpy(lows + offset, thrust::raw_pointer_cast(d_lows.data()), size * sizeof(unsigned int), hipMemcpyDeviceToHost));
		offset += size;
	}while(offset < n);
	assert(offset == n);
}

static __global__ void char_to_bits_kernel(const unsigned char* chars,
											unsigned int n,
											unsigned char* bits)
{
	const unsigned int laneId = ::__lane_id();
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	unsigned long long val = (idx < n) ? (chars[idx] ? 1 : 0) : 0;
	unsigned long long sum = (val << laneId);
	
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) 
    {
        sum += __shfl_down(sum, offset);
    }

	if(0 == laneId && idx < n)
	{
		*(reinterpret_cast<unsigned long long*>(bits) + (idx >> 6)) = sum;
	}
}

static void char_to_bits(const unsigned char* d_chars,
						const unsigned int n,
						unsigned char* d_bits)
{
	char_to_bits_kernel<<<
				dim3(divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK)),
				dim3(HIP_THREADS_PER_BLOCK),
				0,
				0>>>(
				d_chars,
				n,
				d_bits);
	
	HIP_POST_KERNEL_CHECK("char_to_bits_kernel");
	HIP_CHECK(hipDeviceSynchronize());
}

template<typename T>
static void gather_values(const thrust::host_vector<unsigned int>& h_maps,
						const unsigned int n,
						T* vals)
{
	assert(n <= h_maps.size());
	thrust::device_vector<T> d_vals(n);
	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), vals, n * sizeof(T), hipMemcpyHostToDevice));

	unsigned int offset = 0;
	thrust::device_vector<unsigned int> d_maps;
	thrust::device_vector<T> d_outputs;
	
	if(n > MAX_ELEMS)
	{
		d_maps.resize(MAX_ELEMS);
		d_outputs.resize(MAX_ELEMS);
	}
	else
	{
		d_maps.resize(n);
		d_outputs.resize(n);
	}
	
	do{
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_maps.data()), h_maps.data() + offset, size * sizeof(unsigned int), hipMemcpyHostToDevice));
		thrust::gather(d_maps.begin(), d_maps.begin() + size, d_vals.begin(), d_outputs.begin());
		HIP_CHECK(hipMemcpy(vals + offset, thrust::raw_pointer_cast(d_outputs.data()), size * sizeof(T), hipMemcpyDeviceToHost));
		offset += size;
	}while(offset < n);
}

template<typename T>
static void sort_by_keys(thrust::device_vector<T>& d_keys,
							thrust::device_vector<unsigned int>& d_vals,
							const unsigned int n,
							T* keys = NULL,
							unsigned int* vals = NULL)
{
	if(d_keys.empty() || d_vals.empty())
		return;
	
	assert(n <= d_keys.size() && n <= d_vals.size());
	
	thrust::stable_sort_by_key(d_keys.begin(), d_keys.begin() + n, d_vals.begin());
	if(NULL != keys)
	{
		thrust::copy(d_keys.begin(), d_keys.begin() + n, keys);
	}

	if(NULL != vals)
	{
		thrust::copy(d_vals.begin(), d_vals.begin() + n, vals);
	}
}

template<typename T, bool sequenced>
static void sort_by_keys(const unsigned int n,
							T* keys,
							unsigned int* vals,
							unsigned int init = 0)
{
	if(n <= MAX_ELEMS)
	{
		thrust::device_vector<T> d_keys(n);
		thrust::device_vector<unsigned int> d_vals;

		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys, n * sizeof(T), hipMemcpyHostToDevice));

		if(sequenced)
		{
			d_vals.resize(n);
			thrust::sequence(d_vals.begin(), d_vals.end(), init);
		}
		
		if(!thrust::is_sorted(d_keys.begin(), d_keys.end()))
		{
			if(!sequenced)
			{
				d_vals.resize(n);
				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), vals, n * sizeof(unsigned int), hipMemcpyHostToDevice));
			}
			
			sort_by_keys<T>(d_keys, d_vals, n, keys, vals);
			assert(thrust::is_sorted(d_keys.begin(), d_keys.end()));
		}
		else if(sequenced)
		{
			HIP_CHECK(hipMemcpy(vals, thrust::raw_pointer_cast(d_vals.data()), n * sizeof(unsigned int), hipMemcpyDeviceToHost));
		}
		
		return;
	}

	unsigned int count = divide_up<unsigned int>(n, MAX_ELEMS);
	//LOG_INFO << "conns: " << n << ", segments: " << count << std::endl;
	thrust::host_vector<unsigned int> h_key_rowptrs(count + 1);
	h_key_rowptrs[0] = 0;
	
	for(unsigned int i = 0; i < count; i++)
	{
		unsigned int size = (i < (count - 1)) ? MAX_ELEMS : (n - i * MAX_ELEMS);
		h_key_rowptrs[i + 1] = size + h_key_rowptrs[i];
		sort_by_keys<T, sequenced>(size,
								keys + h_key_rowptrs[i],
								vals + h_key_rowptrs[i],
								i * MAX_ELEMS);
	}

	unsigned int total = 0;
	do{
		unsigned int offset = 0;
		const unsigned int step = MAX_ELEMS / count;
		std::unordered_set<T> i_keys;
		std::vector<std::vector<unsigned int>> i_offsets;
		std::vector<std::vector<unsigned int>> e_offsets;
		unsigned int merge_count = 0;
		unsigned int selected_idx = 0;

		for(unsigned int i = 0; i < count; i++)
		{
			unsigned int size = h_key_rowptrs[i + 1] - h_key_rowptrs[i];
			if(step < size)
			{
				i_keys.insert(keys[h_key_rowptrs[i] + step]);
			}
			else
			{
				i_keys.insert(keys[h_key_rowptrs[i + 1] - 1]);
			}
		}

		i_offsets.resize(i_keys.size());
		e_offsets.resize(i_keys.size());
		for(unsigned int i = 0; i < i_keys.size(); i++)
		{
			i_offsets[i].resize(count);
			e_offsets[i].resize(count);
		}

		//LOG_INFO << "Have " << i_keys.size() << " keys" << std::endl;
	
		for(unsigned int i = 0; i < count; i++)
		{
			unsigned int size = h_key_rowptrs[i + 1] - h_key_rowptrs[i];
			assert(size > 0 && size <= MAX_ELEMS);
			//LOG_INFO << i <<" Size: " << size << std::endl;
			thrust::device_vector<T> d_key(size);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_key.data()), keys + h_key_rowptrs[i], size * sizeof(T), hipMemcpyHostToDevice));
			unsigned int j = 0;
			for(const auto& key : i_keys)
			{
				auto it = thrust::lower_bound(d_key.begin(), d_key.end(), key);
				size = it - d_key.begin();
				assert(size <= d_key.size());
				i_offsets[j][i] = size;

				it = thrust::upper_bound(d_key.begin(), d_key.end(), key);
				size = it - d_key.begin();
				assert(size <= d_key.size());
				e_offsets[j][i] = size;
				j++;
			}
			assert(j == i_keys.size());
		}

		{
			unsigned int min_diff = MAX_ELEMS + 1;
			for(unsigned int i = 0; i < i_offsets.size(); i++)
			{
				unsigned int size = 0;
				for(unsigned int j = 0; j < count; j++)
				{
					size += i_offsets[i][j];
				}
				
				if(size > MAX_ELEMS)
					continue;
				
				unsigned int diff = MAX_ELEMS - size;
				if(diff < min_diff)
				{
					min_diff = diff;
					selected_idx = i;
				}
			}

			//LOG_INFO << "select index: " << selected_idx << std::endl;
		}

		#if 0
		{
			unsigned int j = 0;
			for(const auto& key : i_keys)
			{
				if(j == selected_idx)
				{
					LOG_INFO << "key: " << key << std::endl;
					for(unsigned int i = 0; i < count; i++)
					{
						LOG_INFO << i_offsets[j][i];
					}
					break;
				}
				j++;
			}
		}
		#endif

		for(unsigned int i = 0; i < count; i++)
		{
			if(i_offsets[selected_idx][i] > 0)
			{
				merge_count++;
			}
		}

		if(merge_count > 0)
		{
			if(1 == merge_count)
			{
				if(i_offsets[selected_idx][0] > 0)
				{
					offset += i_offsets[selected_idx][0];
				}
				else
				{
					thrust::host_vector<T> h_keys;
					thrust::host_vector<unsigned int> h_vals;
					for(int i = static_cast<int>(count) - 1; i >= 0; i--)
					{
						unsigned int size = i_offsets[selected_idx][i];
						if(size > 0)
						{
							h_keys.resize(size);
							h_vals.resize(size);
							memcpy(h_keys.data(), keys + h_key_rowptrs[i], size * sizeof(T));
							memcpy(h_vals.data(), vals + h_key_rowptrs[i], size * sizeof(unsigned int));
						}
						else if(!h_keys.empty())
						{
							size = h_key_rowptrs[i + 1] - h_key_rowptrs[i];
							{
								thrust::host_vector<T> h_temps(size);
								memcpy(h_temps.data(), keys + h_key_rowptrs[i], size * sizeof(T));
								memcpy(keys + h_key_rowptrs[i] + h_keys.size(), h_temps.data(), size * sizeof(T));
							}

							{
								thrust::host_vector<unsigned int> h_temps(size);
								memcpy(h_temps.data(), vals + h_key_rowptrs[i], size * sizeof(unsigned int));
								memcpy(vals + h_key_rowptrs[i] + h_vals.size(), h_temps.data(), size * sizeof(unsigned int));
							}
						}
					}
					assert(!h_keys.empty());
					memcpy(keys, h_keys.data(), h_keys.size() * sizeof(T));
					memcpy(vals, h_vals.data(), h_vals.size() * sizeof(unsigned int));
					offset += h_keys.size();
				}
			}
			else
			{
				thrust::device_vector<T> d_merge_key;
				thrust::device_vector<unsigned int> d_merge_val;
				unsigned int merge_n = 0;
				for(unsigned int i = 0; i < count; i++)
				{
					unsigned int size = i_offsets[selected_idx][i];
					if(size > 0)
					{
						if(d_merge_key.empty())
						{
							d_merge_key.resize(MAX_ELEMS);
							d_merge_val.resize(MAX_ELEMS);
							HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_merge_key.data()), keys + h_key_rowptrs[i], size * sizeof(T), hipMemcpyHostToDevice));
							HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_merge_val.data()), vals + h_key_rowptrs[i], size * sizeof(unsigned int), hipMemcpyHostToDevice));
						}
						else
						{
							assert((size + merge_n) <= MAX_ELEMS); 
							thrust::device_vector<T> d_key_result(size + merge_n);
							thrust::device_vector<unsigned int> d_val_result(size + merge_n);
							thrust::device_vector<T> d_key(size);
							thrust::device_vector<unsigned int> d_val(size);
							HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_key.data()), keys + h_key_rowptrs[i], size * sizeof(T), hipMemcpyHostToDevice));
							HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_val.data()), vals + h_key_rowptrs[i], size * sizeof(unsigned int), hipMemcpyHostToDevice));
							thrust::merge_by_key(d_merge_key.begin(),
										d_merge_key.begin() + merge_n,
										d_key.begin(), 
										d_key.end(),
										d_merge_val.begin(),
										d_val.begin(),
										d_key_result.begin(),
										d_val_result.begin());
							HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_merge_key.data()), thrust::raw_pointer_cast(d_key_result.data()), d_key_result.size() * sizeof(T), hipMemcpyDeviceToDevice));
							HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_merge_val.data()), thrust::raw_pointer_cast(d_val_result.data()), d_val_result.size() * sizeof(unsigned int), hipMemcpyDeviceToDevice));
						}
						merge_n += size;
					}
				}

				unsigned int toal_size_to_be_left = 0;
				for(int i = static_cast<int>(count) - 1; i >= 0; i--)
				{
					unsigned int size = i_offsets[selected_idx][i];
					
					if(0 == toal_size_to_be_left)
					{
						toal_size_to_be_left += size;
					}
					else
					{
						toal_size_to_be_left += size;
						unsigned int size_to_be_moved = h_key_rowptrs[i + 1] - h_key_rowptrs[i] - size;
						if(size_to_be_moved > 0)
						{
							{
								thrust::host_vector<T> h_temps(size_to_be_moved);
								memcpy(h_temps.data(), keys + h_key_rowptrs[i] + size, size_to_be_moved * sizeof(T));
								memcpy(keys + h_key_rowptrs[i] + toal_size_to_be_left, h_temps.data(), size_to_be_moved * sizeof(T));
							}
							{
								thrust::host_vector<unsigned int> h_temps(size_to_be_moved);
								memcpy(h_temps.data(), vals + h_key_rowptrs[i] + size, size_to_be_moved * sizeof(unsigned int));
								memcpy(vals + h_key_rowptrs[i] + toal_size_to_be_left, h_temps.data(), size_to_be_moved * sizeof(unsigned int));
							}
						}
					}
				}

				assert(toal_size_to_be_left == merge_n);
				HIP_CHECK(hipMemcpy(keys, thrust::raw_pointer_cast(d_merge_key.data()), merge_n * sizeof(T), hipMemcpyDeviceToHost));
				HIP_CHECK(hipMemcpy(vals, thrust::raw_pointer_cast(d_merge_val.data()), merge_n * sizeof(unsigned int), hipMemcpyDeviceToHost));

				offset += merge_n;
			}
		}
		else
		{
			unsigned int merge_n = 0;
			for(int i = 0; i < static_cast<int>(count); i++)
			{
				//LOG_INFO << e_offsets[selected_idx][i] << std::endl;
				
				unsigned int size = e_offsets[selected_idx][i];
				if(offset == h_key_rowptrs[i])
				{
					offset += size;
				}
				else if(size > 0)
				{
					assert(offset < h_key_rowptrs[i]);
					
					thrust::host_vector<T> h_keys(size);
					thrust::host_vector<unsigned int> h_vals(size);
					memcpy(h_keys.data(), keys + h_key_rowptrs[i], size * sizeof(T));
					memcpy(h_vals.data(), vals + h_key_rowptrs[i], size * sizeof(unsigned int));

					thrust::host_vector<unsigned int> h_temps(MAX_ELEMS);
					unsigned int move_n = 0;
					do{
						unsigned int move_size = merge_n - move_n;
						move_size = (move_size > MAX_ELEMS) ? MAX_ELEMS : move_size;
						unsigned int move_offset = h_key_rowptrs[i] - move_n - move_size;
						memcpy(h_temps.data(), keys + move_offset, move_size * sizeof(T));
						memcpy(keys + move_offset + size, h_temps.data(), move_size * sizeof(T));
						memcpy(h_temps.data(), vals + move_offset, move_size * sizeof(unsigned int));
						memcpy(vals + move_offset + size, h_temps.data(), move_size * sizeof(unsigned int));
						move_n += move_size;
					}while(move_n < merge_n);

					memcpy(keys + offset , h_keys.data(), h_keys.size() * sizeof(T));
					memcpy(vals + offset, h_vals.data(), h_vals.size() * sizeof(unsigned int));
					offset += size;
					//LOG_INFO << "Direct copy: " << h_keys.size() << std::endl;
				}
				
				merge_n += (h_key_rowptrs[i + 1] - h_key_rowptrs[i] - size);
			}
		}

		{
			unsigned int j = 0;
			unsigned int key_offset = h_key_rowptrs[0];
			for(unsigned int i = 0; i < count; i++)
			{
				
				unsigned int size = h_key_rowptrs[i + 1] - key_offset;
				key_offset = h_key_rowptrs[i + 1];
				
				if(0 < merge_count)
				{
					assert(i_offsets[selected_idx][i] <= size);
					size -= i_offsets[selected_idx][i];
					if(size > 0)
					{
						h_key_rowptrs[j + 1] = h_key_rowptrs[j] + size;
						j++;
					}
				}
				else
				{
					assert(e_offsets[selected_idx][i] <= size);
					size -= e_offsets[selected_idx][i];
					if(size > 0)
					{
						h_key_rowptrs[j + 1] = h_key_rowptrs[j] + size;
						j++;
					}
				}
			}
			
			count = j;
		}
		
		if(0 == count)
		{
			assert(offset == (n - total));
			break;
		}
		else if(1 == count)
		{
			assert((offset + h_key_rowptrs[1] + total) == n);
			break;
		}
		else
		{
			keys += offset;
			vals += offset;
			total += offset;
		}
	}while(total < n);
}

template<typename T>
static void sort_by_keys(const unsigned int n,
							T* highs,
							unsigned int* lows,
							unsigned int* vals)
{
	if(nullptr == highs || nullptr == lows || nullptr == vals)
	{
		return;
	}
	
	thrust::device_vector<unsigned long long> d_keys(n);
	merge<T>(highs, lows, n, d_keys);
	
	thrust::device_vector<unsigned int> d_vals(n);
	HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_vals.data()), vals, n * sizeof(unsigned int), hipMemcpyHostToDevice));

	sort_by_keys<unsigned long long>(d_keys, d_vals, n, nullptr, vals);
	split<T>(d_keys, n, highs, lows);
}

template<typename T>
static void sort_by_keys(const thrust::host_vector<unsigned int>& h_rowptrs,
							T* highs,
							unsigned int* lows,
							thrust::host_vector<unsigned int>& h_maps)
{
	
	const unsigned int bins = h_rowptrs.size() - 1;
	for(unsigned int idx = 0; idx < bins;)
	{
		unsigned int n = 0;
		unsigned int num = 0;
		unsigned int offset = h_rowptrs[idx];
		
		for(unsigned int i = idx; i < bins; i++)
		{
			const unsigned int count = h_rowptrs[i + 1] - h_rowptrs[i];
			
			if(n > 0 && (n + count) > MAX_ELEMS)
				break;

			n += count;
			num++;
			idx++;
		}

		//only single block
		if(1 == num)
		{
			sort_by_keys<unsigned int, false>(n,
											lows + offset,
											h_maps.data() + offset);
		}
		else
		{
			sort_by_keys<T>(n,
						highs + offset,
						lows + offset,
						h_maps.data() + offset);
		}
	}
}

template<typename T, bool saving_key>
static unsigned int unique_by_keys(const unsigned int n,
									T* highs,
									unsigned int* lows,
									thrust::host_vector<unsigned int>& h_vals,
									const unsigned int init = 0)
{
	if(0 == n)
		return 0;
	
	unsigned int offset = 0;
	unsigned int total_bins = 0;
	thrust::device_vector<unsigned long long> d_keys;
	h_vals.clear();
	assert(h_vals.empty());
	
	if(n > MAX_ELEMS)
	{
		d_keys.resize(MAX_ELEMS);
	}
	else
	{
		d_keys.resize(n);
	}
	
	do{
		unsigned int bins_offset = 0;
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		
		merge<T>(highs + offset,
			lows + offset,
			size,
			d_keys);

		assert(thrust::is_sorted(d_keys.begin(), d_keys.begin() + size));
		unsigned int bins = thrust::inner_product(d_keys.begin(), d_keys.begin() + size - 1,
					                             d_keys.begin() + 1,
					                             1,
					                             thrust::plus<unsigned int>(),
					                             thrust::not_equal_to<unsigned long long>());
		assert(bins >= 1);

		thrust::device_vector<unsigned long long> d_unique_keys(bins);
		thrust::device_vector<unsigned int> d_vals(bins + 1);

		// compact find the end of each bin of values
		thrust::reduce_by_key(d_keys.begin(), d_keys.begin() + size,
		                    thrust::constant_iterator<unsigned int>(1),
		                    d_unique_keys.begin(),
		                    d_vals.begin());
		assert(thrust::unique(d_unique_keys.begin(), d_unique_keys.end()) == d_unique_keys.end());

		if(!h_vals.empty())
		{
			thrust::exclusive_scan(d_vals.begin(), d_vals.end(), d_vals.begin(), h_vals.back());
			assert((highs[offset] > highs[offset - 1]) || 
				((highs[offset] == highs[offset - 1]) && (lows[offset - 1] <= lows[offset])));
			if((highs[offset] > highs[offset - 1]) || 
				((highs[offset] == highs[offset - 1]) && (lows[offset - 1] < lows[offset])))
			{
				bins_offset++;
			}

			unsigned int last_bins = total_bins;
			total_bins += (bins + bins_offset - 1);
			if(total_bins > last_bins)
			{	
				h_vals.resize(total_bins + 1);
			}
		
			HIP_CHECK(hipMemcpy(h_vals.data() + last_bins + bins_offset, thrust::raw_pointer_cast(d_vals.data()) + 1, bins * sizeof(unsigned int), hipMemcpyDeviceToHost));
		
			if(saving_key && (total_bins > last_bins))
			{
				thrust::host_vector<T> h_highs(bins);
				thrust::host_vector<unsigned int> h_lows(bins);
				split<T>(d_unique_keys, bins, h_highs.data(), h_lows.data());

				if(0 == bins_offset)
				{
					assert(highs[last_bins - 1] == h_highs[0] && lows[last_bins - 1] == h_lows[0]);
				}
				memcpy(highs + last_bins + bins_offset - 1, h_highs.data(), bins * sizeof(T));
				memcpy(lows + last_bins + bins_offset - 1, h_lows.data(), bins * sizeof(unsigned int));
				
			}
		}
		else
		{
			total_bins += bins;
			assert(total_bins > 0);
			h_vals.resize(total_bins + 1);
			thrust::exclusive_scan(d_vals.begin(), d_vals.end(), d_vals.begin(), init);
			HIP_CHECK(hipMemcpy(h_vals.data(), thrust::raw_pointer_cast(d_vals.data()), d_vals.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
			if(saving_key)
			{
				thrust::host_vector<T> h_highs(bins);
				thrust::host_vector<unsigned int> h_lows(bins);
				split<T>(d_unique_keys, bins, h_highs.data(), h_lows.data());
				memcpy(highs, h_highs.data(), bins * sizeof(T));
				memcpy(lows, h_lows.data(), bins * sizeof(unsigned int));
			}
		}

		offset += size;
		assert(h_vals.back() == (offset + init));
	}while(offset < n);

	LOG_INFO << "[unique_by_keys]: offset [" << offset << "]" << "] n/total_bins [" << n << "/" << total_bins << "]" << std::endl;

	assert(offset == n);
	
	if(h_vals.empty())
	{
		LOG_INFO << "[unique_by_keys]: invaild address table" << std::endl;
	}
	else
	{
		if(h_vals.back() != (n + init))
		{
			LOG_INFO << "[unique_by_keys]: h_vals.back()/(n + init) " << h_vals.back() << "/" << (n + init) << std::endl;
		}
		assert(h_vals.back() == (n + init));
	}

	#if 0
	if(saving_key)
	{
		offset = 0;
		do{
			if(offset > 0)
			{
				assert((highs[offset] > highs[offset - 1]) ||
					((highs[offset] == highs[offset - 1]) &&
					(lows[offset] > lows[offset - 1])));
			}
			unsigned int size = total_bins - offset;
			size = (size > d_keys.size()) ? d_keys.size() : size;
			merge<T>(highs + offset,
				lows + offset,
				size,
				d_keys);
			assert(thrust::is_sorted(d_keys.begin(), d_keys.begin() + size));
			assert(thrust::unique(d_keys.begin(), d_keys.begin() + size) == (d_keys.begin() + size));
			offset += size;
		}while(offset < total_bins);
		assert(offset == total_bins);
	}
	#endif
	
	return total_bins;
}

template<typename T, bool saving_key>
static unsigned int unique_by_keys(const unsigned int n,
									T* keys,
									thrust::host_vector<unsigned int>& h_vals,
									const unsigned int init = 0)
{ 
	if(0 == n)
		return 0;
	
	unsigned int offset = 0;
	unsigned int total_bins = 0;
	thrust::device_vector<T> d_keys;
	h_vals.clear();
	assert(h_vals.empty());
	
	if(n > MAX_ELEMS)
	{
		d_keys.resize(MAX_ELEMS);
	}
	else
	{
		d_keys.resize(n);
	}
	
	do{
		unsigned int bins_offset = 0;
		unsigned int size = n - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys + offset, size * sizeof(T), hipMemcpyHostToDevice));
		assert(thrust::is_sorted(d_keys.begin(), d_keys.begin() + size));
		unsigned int bins = thrust::inner_product(d_keys.begin(), d_keys.begin() + size - 1,
					                             d_keys.begin() + 1,
					                             1,
					                             thrust::plus<unsigned int>(),
					                             thrust::not_equal_to<T>());
		assert(bins >= 1);

		thrust::device_vector<T> d_unique_keys(bins);
		thrust::device_vector<unsigned int> d_vals(bins + 1);

		// compact find the end of each bin of values
		thrust::reduce_by_key(d_keys.begin(), d_keys.begin() + size,
		                    thrust::constant_iterator<unsigned int>(1),
		                    d_unique_keys.begin(),
		                    d_vals.begin());
		assert(thrust::unique(d_unique_keys.begin(), d_unique_keys.end()) == d_unique_keys.end());

		if(!h_vals.empty())
		{
			thrust::exclusive_scan(d_vals.begin(), d_vals.end(), d_vals.begin(), h_vals.back());
			assert(keys[offset - 1] <= keys[offset]);
			if(keys[offset - 1] < keys[offset])
			{
				bins_offset++;
			}

			unsigned int last_bins = total_bins;
			total_bins += bins + bins_offset - 1;
			h_vals.resize(total_bins + 1);
		
			HIP_CHECK(hipMemcpy(h_vals.data() + last_bins + bins_offset, thrust::raw_pointer_cast(d_vals.data()) + 1, bins * sizeof(unsigned int), hipMemcpyDeviceToHost));
		
			if(saving_key && (total_bins > last_bins))
			{
				HIP_CHECK(hipMemcpy(keys + last_bins + bins_offset - 1, thrust::raw_pointer_cast(d_unique_keys.data()), bins * sizeof(T), hipMemcpyDeviceToHost));
			}
		}
		else
		{
			total_bins += bins;
			assert(total_bins > 0);
			h_vals.resize(total_bins + 1);
			thrust::exclusive_scan(d_vals.begin(), d_vals.end(), d_vals.begin(), init);
			HIP_CHECK(hipMemcpy(h_vals.data(), thrust::raw_pointer_cast(d_vals.data()), d_vals.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
			if(saving_key)
			{
				HIP_CHECK(hipMemcpy(keys, thrust::raw_pointer_cast(d_unique_keys.data()), d_unique_keys.size() * sizeof(T), hipMemcpyDeviceToHost));
			}
		}

		offset += size;
		assert(h_vals.back() == (offset + init));
	}while(offset < n);

	assert(offset == n && h_vals.back() == (n + init));
	#if 0
	if(saving_key)
	{
		offset = 0;
		do{
			if(offset > 0)
			{
				assert(keys[offset] > keys[offset - 1]);
			}
			unsigned int size = total_bins - offset;
			size = (size > d_keys.size()) ? d_keys.size() : size;
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys + offset, size * sizeof(T), hipMemcpyHostToDevice));
			assert(thrust::is_sorted(d_keys.begin(), d_keys.begin() + size));
			assert(thrust::unique(d_keys.begin(), d_keys.begin() + size) == (d_keys.begin() + size));
			offset += size;
		}while(offset < total_bins);
		assert(offset == total_bins);
	}
	#endif
	
	return total_bins;
}

template<typename BlockID>
static void adjust_relative_conn_bid(const BlockID bid,
										const unsigned int conns,
										BlockID* conn_bids)
{
	unsigned int offset = 0;
	thrust::device_vector<BlockID> d_conn_bids;
	if(conns > MAX_ELEMS)
	{
		d_conn_bids.resize(MAX_ELEMS);
	}
	else
	{
		d_conn_bids.resize(conns);
	}
	
	thrust::constant_iterator<BlockID> constant(bid);
	thrust::plus<BlockID> op;
	
	do{
		unsigned int size = conns - offset;
		size = (size > MAX_ELEMS) ? MAX_ELEMS : size;
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_bids.data()), conn_bids + offset, size * sizeof(BlockID), hipMemcpyHostToDevice));
		thrust::transform(d_conn_bids.begin(), d_conn_bids.begin() + size, constant, d_conn_bids.begin(), op);
		HIP_CHECK(hipMemcpy(conn_bids + offset, thrust::raw_pointer_cast(d_conn_bids.data()), size * sizeof(BlockID), hipMemcpyDeviceToHost));
		offset += size;
	}while(offset < conns);
	assert(offset == conns);
}

static unsigned int parse_single_block(const unsigned int conns,
										unsigned int* conn_neuron_ids,
										thrust::host_vector<unsigned int>& h_maps,
										thrust::host_vector<unsigned int>& h_rowptrs)
{
	sort_by_keys<unsigned int, false>(conns,
									conn_neuron_ids,
									h_maps.data());

	return  unique_by_keys<unsigned int, true>(conns,
											conn_neuron_ids,
											h_rowptrs);
}


template<typename T>
static unsigned int parse_multi_block(const unsigned int conns,
										T* conn_block_ids,
										unsigned int* conn_neuron_ids,
										thrust::host_vector<unsigned int>& h_maps,
										thrust::host_vector<unsigned int>& h_rowptrs)
{	
	sort_by_keys<T>(h_rowptrs,
					conn_block_ids,
					conn_neuron_ids,
					h_maps);

	
	return unique_by_keys<T, true>(conns,
								conn_block_ids,
								conn_neuron_ids,
								h_rowptrs);
}

template<typename T>
static void parse_inputs_from_numpy(const T block_id,
								const unsigned int n,
								unsigned int* timestamps,
								unsigned int* neuron_ids,
								InputSpike& input)
{
	
	thrust::host_vector<unsigned int> h_rowptrs;
	{
		thrust::device_vector<unsigned int> d_keys(n);
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_keys.data()), timestamps, n * sizeof(unsigned int),hipMemcpyHostToDevice));
		if(!thrust::is_sorted(d_keys.begin(), d_keys.end()))
		{
			thrust::host_vector<unsigned int> h_maps(n);
			sort_by_keys<unsigned int, true>(n,
											timestamps,
											h_maps.data());
			
			gather_values<unsigned int>(h_maps,
										n,
										neuron_ids);
		}
	}

	assert(thrust::is_sorted(timestamps, timestamps + n));
	const unsigned int bins = unique_by_keys<unsigned int, true>(n, timestamps, h_rowptrs);
	std::cout << "input have " << bins << " timestamps" << std::endl;
	assert(h_rowptrs.size() == (bins + 1));
	assert(h_rowptrs.back() == n);
	
	input.input_timestamps = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * bins, false);
	input.input_rowptrs = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * h_rowptrs.size(), false);
	input.input_colinds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * n);
	memcpy(input.input_timestamps->mutable_cpu_data(), timestamps, input.input_timestamps->size());
	#if 0
	{
		thrust::host_vector<unsigned int> ts(bins);
		unsigned int i = 0;
		unsigned int val = -1;
		unsigned int offset = 0;
		for(; i < n; i++)
		{
			if(val != timestamps[i])
			{
				val = timestamps[i];
				ts[offset++] = val;
				assert(offset <= bins);
			}
		}
		assert(i == n && offset == bins);
		for(i = 0; i < bins; i++)
			assert(ts[i] == input.input_timestamps->cpu_data()[i]);
	}
	#endif
	memcpy(input.input_rowptrs->mutable_cpu_data(), &h_rowptrs[0], input.input_rowptrs->size());
	HIP_CHECK(hipMemcpy(input.input_colinds->mutable_gpu_data(), neuron_ids, input.input_colinds->size(), hipMemcpyHostToDevice));
}

template<typename BlockID>
static void parse_outdegree_from_numpy(const BlockID block_id,
								const std::string& filename,
								unsigned int& conns,
								unsigned int& conn_bins,
								unsigned int& same_bid_count,
								unsigned int& same_bid_begin,
								unsigned int& same_bid_end,
								thrust::host_vector<unsigned int>& h_rowptrs,
								thrust::host_vector<unsigned int>& h_maps,
								ConnectionTable<BlockID>& tab)
{
	thrust::host_vector<BlockID> h_conn_bids;
	thrust::host_vector<unsigned int> h_conn_nids;
	{
		cnpy::NpyArray arr_conn_bids = cnpy::npz_load(filename, "input_block_idx");
		BlockID* conn_bids = arr_conn_bids.data<BlockID>();
		assert(arr_conn_bids.shape.size() == 1);

		cnpy::NpyArray arr_conn_nids = cnpy::npz_load(filename, "input_neuron_idx");
		unsigned int* conn_nids = arr_conn_nids.data<unsigned int>();
		assert(arr_conn_nids.shape.size() == 1);

		assert(arr_conn_bids.shape[0] == arr_conn_nids.shape[0]);

		same_bid_count = 0;
		same_bid_begin = 0;
		same_bid_end = 0;
		conns = arr_conn_bids.shape[0];
		h_maps.resize(conns);

		if(0 == conns)
			return;
		
		adjust_relative_conn_bid<BlockID>(block_id, conns, conn_bids);
		sort_by_keys<BlockID, true>(conns, conn_bids, h_maps.data());
		assert(thrust::is_sorted(conn_bids, conn_bids + conns));

		#if 0
		{
		 	thrust::device_vector<unsigned int> d_maps(2 * MAX_ELEMS);
			unsigned int count = divide_up<unsigned int>(conns, MAX_ELEMS);
			//LOG_INFO << "conns: " << conns << ", segments: " << count << std::endl;
			for(unsigned int i = 0; i < count; i++)
			{
				unsigned int size = (i < (count - 1)) ? MAX_ELEMS : (conns - i * MAX_ELEMS);
				thrust::copy(h_maps.begin() + i * MAX_ELEMS, h_maps.begin() + i * MAX_ELEMS + size , d_maps.begin() + MAX_ELEMS);
				assert(thrust::unique(d_maps.begin() + MAX_ELEMS, d_maps.begin() + MAX_ELEMS + size) == (d_maps.begin() + MAX_ELEMS + size));
				for(unsigned int j = 0; j < i; j++)
				{
					thrust::copy(h_maps.begin() + j * MAX_ELEMS , h_maps.begin() + (j + 1) * MAX_ELEMS , d_maps.begin());
					assert(thrust::unique(d_maps.begin(), d_maps.begin() + MAX_ELEMS + size) == (d_maps.begin() + MAX_ELEMS + size));
				}
			}
		}
		#endif
		
		unsigned int bins = unique_by_keys<BlockID, false>(conns,
													conn_bids,
													h_rowptrs);
		assert(bins >= 1);

		//LOG_INFO << "[parse_outdegree_from_numpy] sort block id done..." << std::endl;
		
		if(1 == bins)
		{
			conn_bins = parse_single_block(conns,
										conn_nids,
										h_maps,
										h_rowptrs);
			assert(conn_bins >= 1);
			if(conn_bids[0] == block_id)
			{
				same_bid_end = conn_bins;
			}

			h_conn_bids.push_back(conn_bids[0]);
			h_conn_nids.resize(conn_bins);
			thrust::copy(conn_nids, conn_nids + conn_bins, h_conn_nids.begin());
			//LOG_INFO << "after called parse_single_block..." << std::endl;
		}
		else
		{
			gather_values<unsigned int>(h_maps, conns, conn_nids);
			conn_bins = parse_multi_block<BlockID>(conns,
										conn_bids,
										conn_nids,
										h_maps,
										h_rowptrs);

			assert(conn_bins >= 1);

			//LOG_INFO << "after called parse_multi_block..." << std::endl;

			thrust::device_vector<BlockID> d_conn_bids(conn_bins);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_bids.data()), conn_bids, conn_bins * sizeof(BlockID), hipMemcpyHostToDevice));
			auto it = thrust::find(d_conn_bids.begin(), d_conn_bids.end(), block_id);
			if(it != d_conn_bids.end())
			{
				assert(it == thrust::lower_bound(d_conn_bids.begin(), d_conn_bids.end(), block_id));
				same_bid_begin = it - d_conn_bids.begin();
				it = thrust::upper_bound(d_conn_bids.begin(), d_conn_bids.end(), block_id);
				same_bid_end = it - d_conn_bids.begin();
			}

			h_conn_bids.resize(conn_bins);
			thrust::copy(conn_bids, conn_bids + conn_bins, h_conn_bids.begin());
			h_conn_nids.resize(conn_bins);
			thrust::copy(conn_nids, conn_nids + conn_bins, h_conn_nids.begin());
		}
	}

	#if 0
	{
	 	thrust::device_vector<unsigned int> d_maps(2 * MAX_ELEMS);
		unsigned int count = divide_up<unsigned int>(conns, MAX_ELEMS);
		//LOG_INFO << "conns: " << conns << ", segments: " << count << std::endl;
		for(unsigned int i = 0; i < count; i++)
		{
			unsigned int size = (i < (count - 1)) ? MAX_ELEMS : (conns - i * MAX_ELEMS);
			thrust::copy(h_maps.begin() + i * MAX_ELEMS, h_maps.begin() + i * MAX_ELEMS + size , d_maps.begin() + MAX_ELEMS);
			assert(thrust::unique(d_maps.begin() + MAX_ELEMS, d_maps.begin() + MAX_ELEMS + size) == (d_maps.begin() + MAX_ELEMS + size));
			for(unsigned int j = 0; j < i; j++)
			{
				thrust::copy(h_maps.begin() + j * MAX_ELEMS, h_maps.begin() + (j + 1) * MAX_ELEMS , d_maps.begin());
				assert(thrust::unique(d_maps.begin(), d_maps.begin() + MAX_ELEMS + size) == (d_maps.begin() + MAX_ELEMS + size));
			}
		}
	}

	//LOG_INFO << "check map done" << std::endl;
	assert(0 == h_rowptrs[0] && h_rowptrs.back() == conns && h_rowptrs.size() == (conn_bins + 1));

	
	thrust::host_vector<unsigned int> h_block_rowptrs;
	h_block_rowptrs.push_back(0);
	{
		assert(thrust::is_sorted(h_conn_bids.begin(), h_conn_bids.end()));
		cnpy::NpyArray arr_check_conn_bids = cnpy::npz_load(filename, "input_block_idx");
		BlockID* check_conn_bids = arr_check_conn_bids.data<BlockID>();
		adjust_relative_conn_bid<BlockID>(block_id, conns, check_conn_bids);
		//LOG_INFO << "adjust_relative_conn_bid done" << std::endl;
		unsigned int idx = 0;
		unsigned int count = h_rowptrs[idx + 1];
		for(unsigned int i = 0; i < conns; i++)
		{
			if(i == count)
			{
				idx++;
				if(1 < h_conn_bids.size())
				{
					if(h_conn_bids[idx] != h_conn_bids[idx - 1])
					{
						h_block_rowptrs.push_back(idx);
					}
				}
				count = h_rowptrs[idx + 1];
			}

			if(1 < h_conn_bids.size())
			{
				assert(h_conn_bids[idx] == check_conn_bids[h_maps[i]]);
			}
			else
			{
				assert(h_conn_bids[0] == check_conn_bids[h_maps[i]]);
			}
			
		}

		h_block_rowptrs.push_back(conn_bins);
		assert(idx == (conn_bins - 1));
		
	}
	{
		cnpy::NpyArray arr_check_conn_nids = cnpy::npz_load(filename, "input_neuron_idx");
		unsigned int* check_conn_nids = arr_check_conn_nids.data<unsigned int>();
		
		for(unsigned int i = 0; i < h_block_rowptrs.size() - 1; i++)
		{
			assert(thrust::is_sorted(h_conn_nids.begin() + h_block_rowptrs[i], h_conn_nids.begin() + h_block_rowptrs[i + 1]));
		}

		unsigned int idx = 0;
		unsigned int count = h_rowptrs[idx + 1];
		for(unsigned int i = 0; i < conns; i++)
		{
			if(i == count)
			{
				idx++;
				count = h_rowptrs[idx + 1];
			}
			assert(h_conn_nids[idx] == check_conn_nids[h_maps[i]]);
		}
	}
	h_block_rowptrs.clear();
	LOG_INFO << "check block id and neuron id done" << std::endl;
	#endif
		
	assert(0 == h_rowptrs[0] && h_rowptrs[conn_bins] == conns);
	same_bid_count = same_bid_end - same_bid_begin;

	if(conn_bins > same_bid_count)
	{
		unsigned int count = conn_bins - same_bid_count;
		//LOG_INFO << "outer connection neurons count: " << count << std::endl;
		tab.outer_rowptrs = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (count + 1));
		thrust::device_vector<BlockID> d_outer_conn_bids;
		thrust::device_vector<unsigned int> d_outer_conn_nids;
		if(1 < h_conn_bids.size())
		{
			d_outer_conn_bids.resize(count);
			d_outer_conn_nids.resize(count);
		}
		
		if(0 < same_bid_begin)
		{
			assert(!d_outer_conn_bids.empty());
			HIP_CHECK(hipMemcpy((tab.outer_rowptrs)->mutable_gpu_data(), &h_rowptrs[0], (same_bid_begin + 1) * sizeof(unsigned int), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_bids.data()), h_conn_bids.data(), same_bid_begin * sizeof(BlockID), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_nids.data()), h_conn_nids.data(), same_bid_begin * sizeof(unsigned int), hipMemcpyHostToDevice));
		}
		
		if(same_bid_end < conn_bins)
		{
			if(0 < same_bid_end)
			{
				thrust::device_vector<unsigned int> d_rowptrs((conn_bins - same_bid_end) + 1);
				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_rowptrs.data()), &h_rowptrs[same_bid_end], d_rowptrs.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
				thrust::constant_iterator<unsigned int> constant(h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin]);
				thrust::minus<unsigned int> op;
				thrust::transform(d_rowptrs.begin(), d_rowptrs.end(), constant, thrust::device_pointer_cast((tab.outer_rowptrs)->mutable_gpu_data()) + same_bid_begin, op);
			}
			else
			{
				assert(same_bid_begin == 0 && same_bid_end == 0);
				HIP_CHECK(hipMemcpy((tab.outer_rowptrs)->mutable_gpu_data(), &h_rowptrs[0], h_rowptrs.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			}

			if(!d_outer_conn_bids.empty())
			{
				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_bids.data()) + same_bid_begin, h_conn_bids.data() + same_bid_end, (conn_bins - same_bid_end) * sizeof(BlockID), hipMemcpyHostToDevice));
				HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_outer_conn_nids.data()) + same_bid_begin, h_conn_nids.data() + same_bid_end, (conn_bins - same_bid_end) * sizeof(unsigned int), hipMemcpyHostToDevice));
			}
		}

		{
			thrust::device_vector<BlockID> d_keys;
			thrust::device_vector<unsigned int> d_counts;
			unsigned int bins = 1;
			if(!d_outer_conn_bids.empty())
			{
				bins = thrust::inner_product(d_outer_conn_bids.begin(), d_outer_conn_bids.end() - 1,
		                                 d_outer_conn_bids.begin() + 1,
		                                 1,
		                                 thrust::plus<unsigned int>(),
		                                 thrust::not_equal_to<BlockID>());
				// resize histogram storage
				d_keys.resize(bins);
				d_counts.resize(bins + 1);
				// compact find the end of each bin of values
				thrust::reduce_by_key(d_outer_conn_bids.begin(), d_outer_conn_bids.end(),
				                    thrust::constant_iterator<unsigned int>(1),
				                    d_keys.begin(),
				                    d_counts.begin());
				
				thrust::exclusive_scan(d_counts.begin(),
									d_counts.end(),
									d_counts.begin());
			}
			tab.outer_conn_bids.resize(bins);
			tab.outer_conn_inds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (bins + 1));
			tab.outer_conn_nids = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * d_outer_conn_nids.size(), false);

			if(!d_outer_conn_bids.empty())
			{
				HIP_CHECK(hipMemcpy(tab.outer_conn_bids.data(), thrust::raw_pointer_cast(d_keys.data()), d_keys.size() * sizeof(BlockID), hipMemcpyDeviceToHost));
				HIP_CHECK(hipMemcpy((tab.outer_conn_inds)->mutable_cpu_data(), thrust::raw_pointer_cast(d_counts.data()), d_counts.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
				assert(((tab.outer_conn_inds)->cpu_data()[bins] - (tab.outer_conn_inds)->cpu_data()[0]) == d_outer_conn_nids.size());
				HIP_CHECK(hipMemcpy((tab.outer_conn_nids)->mutable_cpu_data(), thrust::raw_pointer_cast(d_outer_conn_nids.data()), d_outer_conn_nids.size() * sizeof(unsigned int), hipMemcpyDeviceToHost));
			}
			else
			{
				tab.outer_conn_bids[0] = h_conn_bids[0];
				tab.outer_conn_inds->mutable_cpu_data()[0] = 0;
				tab.outer_conn_inds->mutable_cpu_data()[1] = h_conn_nids.size();
				memcpy((tab.outer_conn_nids)->mutable_cpu_data(), h_conn_nids.data(), h_conn_nids.size() * sizeof(unsigned int));
			}
		}
	}

	if(0 < same_bid_count)
	{
		//LOG_INFO << "inner connection neurons count: " << same_bid_count << std::endl;
		tab.inner_rowptrs = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (same_bid_count + 1));
		tab.inner_conninds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * same_bid_count);
		
		if(0 < same_bid_begin)
		{
			thrust::device_vector<unsigned int> d_rowptrs(same_bid_count + 1);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_rowptrs.data()), &h_rowptrs[same_bid_begin], d_rowptrs.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			thrust::constant_iterator<unsigned int> constant(h_rowptrs[same_bid_begin]);
			thrust::minus<unsigned int> op;
			thrust::transform(d_rowptrs.begin(), d_rowptrs.end(), constant, thrust::device_pointer_cast((tab.inner_rowptrs)->mutable_gpu_data()), op);
		}
		else
		{
			HIP_CHECK(hipMemcpy((tab.inner_rowptrs)->mutable_gpu_data(), &h_rowptrs[0], (same_bid_count + 1) * sizeof(unsigned int), hipMemcpyHostToDevice));
		}
		HIP_CHECK(hipMemcpy((tab.inner_conninds)->mutable_gpu_data(), h_conn_nids.data() + same_bid_begin, same_bid_count * sizeof(unsigned int), hipMemcpyHostToDevice));
	}
	
	report_dev_info();
}

template<typename BlockID>
CONFIG_BLOCK_TYPE parse_conn_table_from_numpy(const BlockID block_id,
								const std::string& filename,
								InputSpike& input,
								ConnectionTable<BlockID>& tab)
{
	if(cnpy::npz_find(filename, "src_timestamp"))
	{
		cnpy::NpyArray arr_ts = cnpy::npz_load(filename, "src_timestamp");
		cnpy::NpyArray arr_nids = cnpy::npz_load(filename, "src_neuron_idx");
		unsigned int* timestamps = arr_ts.data<unsigned int>();
		unsigned int* nids = arr_nids.data<unsigned int>();
		assert(arr_ts.shape[0] == arr_nids.shape[0]);
		parse_inputs_from_numpy<BlockID>(block_id, arr_nids.shape[0], timestamps, nids, input);
		return CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_INPUT;
	}

	DataType weight_type;
	if(filename.rfind("/double") != string::npos)
    {
            weight_type = DOUBLE;
            LOG_INFO << "[parse_conn_table_from_numpy]: weight type double" << std::endl;
    }
    else if(filename.rfind("/single") != string::npos)
    {
            weight_type = FLOAT;
            LOG_INFO << "[parse_conn_table_from_numpy]: weight type float" << std::endl;
    }
    else if(filename.rfind("/half") != string::npos)
    {
            weight_type = FLOAT16;
            LOG_INFO << "[parse_conn_table_from_numpy]: weight type float16" << std::endl;
    }
    else if(filename.rfind("/int8") != string::npos || filename.rfind("/uint8") != string::npos)
    {
            weight_type = INT8;
			LOG_INFO << "[parse_conn_table_from_numpy]: weight type int8" << std::endl;
    }
    else
    {
            return CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_UNKNOWN;
    }
	#if 0
	std::string fname;
	{
		size_t n = filename.length();
		size_t pos = n;
		while(pos > 0)
		{
			if(filename[pos] == '/')
				pos--;
			else
				break;
		}
		n = pos;
		assert(n > 0);
		pos = filename.substr(0, n).rfind('/');
		if(pos == std::string::npos)
		{
			pos = 0;
		}
		else
		{
			pos++;
		}
		assert(n > pos);
		fname = filename.substr(pos, n - pos);
	}
	
	if(fname == "double")
	{
		weight_type = DOUBLE;
		LOG_INFO << "[parse_conn_table_from_numpy]: weight type double" << std::endl;
	}
	else if(fname == "single")
	{
		weight_type = FLOAT;
		LOG_INFO << "[parse_conn_table_from_numpy]: weight type float" << std::endl;
	}
	else if(fname == "half")
	{
		weight_type = FLOAT16;
		LOG_INFO << "[parse_conn_table_from_numpy]: weight type float16" << std::endl;
	}
	else if(fname == "int8" || fname == "uint8")
	{
		weight_type = INT8;
		LOG_INFO << "[parse_conn_table_from_numpy]: weight type int8" << std::endl; 
	}
	else
	{
		return CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_UNKNOWN;
	}
	#endif

	unsigned int conns;
	unsigned int conn_bins;
	unsigned int same_bid_count = 0;
	unsigned int same_bid_begin = 0;
	unsigned int same_bid_end = 0;
	thrust::host_vector<unsigned int> h_rowptrs;
	thrust::host_vector<unsigned int> h_maps;

	parse_outdegree_from_numpy<BlockID>(block_id,
							filename,
							conns,
							conn_bins,
							same_bid_count,
							same_bid_begin,
							same_bid_end,
							h_rowptrs,
							h_maps,
							tab);

	cnpy::NpyArray arr_nids = cnpy::npz_load(filename, "output_neuron_idx");
	unsigned int* nids = arr_nids.data<unsigned int>();
	assert(arr_nids.shape.size() == 1 && arr_nids.shape[0] == conns);
	 
	cnpy::NpyArray arr_conn_kinds = cnpy::npz_load(filename, "input_channel_offset");
	unsigned char* conn_kinds = arr_conn_kinds.data<unsigned char>();
	assert(arr_conn_kinds.shape.size() == 1 &&
		arr_conn_kinds.shape[0] == conns);

	cnpy::NpyArray arr_weight = cnpy::npz_load(filename, "weight");
	//T2* weights = reinterpret_cast<T2*>(arr_weight.data<T>());
	assert(arr_weight.shape.size() == 2 &&
		arr_weight.shape[0] == conns &&
		arr_weight.shape[1] == 2);

	void* weights;
	if(weight_type == DOUBLE)
	{
		weights = static_cast<void*>(arr_weight.data<double>());
		gather_values<double2>(h_maps, conns, reinterpret_cast<double2*>(weights));
	}
	else if(weight_type == FLOAT)
	{
		weights = static_cast<void*>(arr_weight.data<float>());
		gather_values<float2>(h_maps, conns, reinterpret_cast<float2*>(weights));
	}
	else if(weight_type == FLOAT16)
	{
		weights = static_cast<void*>(arr_weight.data<unsigned short>());
		gather_values<half2>(h_maps, conns, reinterpret_cast<half2*>(weights));
	}
	else
	{
		assert(weight_type == INT8);
		weights = static_cast<void*>(arr_weight.data<unsigned char>());

		#if 0
		std::vector<uchar2> h_weights(conns);
		memcpy(h_weights.data(), weights, conns * 2 * sizeof(unsigned char));
		gather_values<uchar2>(h_maps, conns, reinterpret_cast<uchar2*>(weights));
		uchar2* w = (uchar2*)weights;

		for(unsigned int i = 0; i < conns; i++)
		{
			assert(w[i].x == h_weights[h_maps[i]].x &&
				w[i].y == h_weights[h_maps[i]].y);
		}
		LOG_INFO << "check weight done" << std::endl;
		#else
		gather_values<uchar2>(h_maps, conns, reinterpret_cast<uchar2*>(weights));
		#endif
	}

	#if 0
	{
		std::vector<unsigned int> h_nids(conns);
		memcpy(h_nids.data(), nids, conns * sizeof(unsigned int));
		gather_values<unsigned int>(h_maps, conns, nids);
		for(unsigned int i = 0; i < conns; i++)
		{
			assert(nids[i] == h_nids[h_maps[i]]);
		}

		LOG_INFO << "check output neuron id done" << std::endl;
	}
	#else
	gather_values<unsigned int>(h_maps, conns, nids);
	#endif
	
	#if 0
		std::vector<unsigned char> h_conn_kinds(conns);
		memcpy(h_conn_kinds.data(), conn_kinds, conns * sizeof(unsigned char));
		gather_values<unsigned char>(h_maps, conns, conn_kinds);
		for(unsigned int i = 0; i < conns; i++)
		{
			assert(conn_kinds[i] == h_conn_kinds[h_maps[i]]);
		}

		LOG_INFO << "check input channel offset done" << std::endl;
	#else
	gather_values<unsigned char>(h_maps, conns, conn_kinds);
	#endif
	h_maps.clear();
		
	if(conn_bins > same_bid_count)
	{
		unsigned int count = h_rowptrs[conn_bins] - h_rowptrs[same_bid_end] + h_rowptrs[same_bid_begin];
		tab.outer_colinds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * count);
		//tab.outer_connkinds = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), sizeof(unsigned char) * count);
		tab.outer_connkinds = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), (align_up<6, unsigned int>(count) >> 3));
		tab.outer_vals.type_ = weight_type;
		tab.outer_vals.data_ = make_shared<DataAllocator<char>>(static_cast<int>(block_id + 1), tab.outer_vals.elem_size() * count);
		thrust::device_vector<unsigned char> d_conn_kinds(count);
		
		if(same_bid_begin)
		{
			HIP_CHECK(hipMemcpy((tab.outer_colinds)->mutable_gpu_data(), nids, h_rowptrs[same_bid_begin] * sizeof(unsigned int), hipMemcpyHostToDevice));
			//HIP_CHECK(hipMemcpy((tab.outer_connkinds)->mutable_gpu_data(), conn_kinds, h_rowptrs[same_bid_begin] * sizeof(unsigned char), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_kinds.data()), conn_kinds, h_rowptrs[same_bid_begin] * sizeof(unsigned char), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy((tab.outer_vals.data_)->mutable_gpu_data(), weights, h_rowptrs[same_bid_begin] * tab.outer_vals.elem_size(), hipMemcpyHostToDevice));
		}
		
		if(same_bid_end < conn_bins)
		{
			HIP_CHECK(hipMemcpy((tab.outer_colinds)->mutable_gpu_data() + h_rowptrs[same_bid_begin], nids + h_rowptrs[same_bid_end], (h_rowptrs[conn_bins] - h_rowptrs[same_bid_end]) * sizeof(unsigned int), hipMemcpyHostToDevice));
			//HIP_CHECK(hipMemcpy((tab.outer_connkinds)->mutable_gpu_data() + h_rowptrs[same_bid_begin], conn_kinds + h_rowptrs[same_bid_end], (h_rowptrs[conn_bins] - h_rowptrs[same_bid_end]) * sizeof(unsigned char), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_kinds.data()) + h_rowptrs[same_bid_begin], conn_kinds + h_rowptrs[same_bid_end], (h_rowptrs[conn_bins] - h_rowptrs[same_bid_end]) * sizeof(unsigned char), hipMemcpyHostToDevice));
			HIP_CHECK(hipMemcpy((tab.outer_vals.data_)->mutable_gpu_data() + h_rowptrs[same_bid_begin] * tab.outer_vals.elem_size(), reinterpret_cast<char*>(weights) + h_rowptrs[same_bid_end] * tab.outer_vals.elem_size(), (h_rowptrs[conn_bins] - h_rowptrs[same_bid_end]) * tab.outer_vals.elem_size(), hipMemcpyHostToDevice));
		}

		char_to_bits(thrust::raw_pointer_cast(d_conn_kinds.data()), count, (tab.outer_connkinds)->mutable_gpu_data());
		#if 0
		thrust::host_vector<unsigned char> h_chars(count);
		thrust::copy(d_conn_kinds.begin(), d_conn_kinds.end(), h_chars.begin());
		thrust::host_vector<unsigned char> h_bits((tab.outer_connkinds)->count());
		HIP_CHECK(hipMemcpy(h_bits.data(), (tab.outer_connkinds)->gpu_data(), h_bits.size(), hipMemcpyDeviceToHost));
		for(unsigned int i = 0; i < count; i++)
		{
			unsigned char bit = h_bits[i >> 3];
			bit = (bit >> (i & 7)) & 0x01;
			assert(bit == (h_chars[i] ? 0x01 : 0x00));
		}
		#endif
	}

	if(same_bid_count)
	{
		unsigned int count = h_rowptrs[same_bid_end] - h_rowptrs[same_bid_begin];
		tab.inner_colinds = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * count);
		//tab.inner_connkinds = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), count);
		tab.inner_connkinds = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), (align_up<6, unsigned int>(count) >> 3));
		tab.inner_vals.type_ = weight_type;
		tab.inner_vals.data_ = make_shared<DataAllocator<char>>(static_cast<int>(block_id + 1), tab.inner_vals.elem_size() * count);
		thrust::device_vector<unsigned char> d_conn_kinds(count);
		
		HIP_CHECK(hipMemcpy((tab.inner_colinds)->mutable_gpu_data(), nids + h_rowptrs[same_bid_begin], count * sizeof(unsigned int), hipMemcpyHostToDevice));
		//HIP_CHECK(hipMemcpy((tab.inner_connkinds)->mutable_gpu_data(), conn_kinds + h_rowptrs[same_bid_begin], count * sizeof(unsigned char), hipMemcpyHostToDevice));
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_conn_kinds.data()), conn_kinds + h_rowptrs[same_bid_begin], count * sizeof(unsigned char), hipMemcpyHostToDevice));
		char_to_bits(thrust::raw_pointer_cast(d_conn_kinds.data()), count, (tab.inner_connkinds)->mutable_gpu_data());
		HIP_CHECK(hipMemcpy((tab.inner_vals.data_)->mutable_gpu_data(), reinterpret_cast<char*>(weights) + h_rowptrs[same_bid_begin] * tab.inner_vals.elem_size(), count * tab.inner_vals.elem_size(), hipMemcpyHostToDevice));	

		#if 0
		thrust::host_vector<unsigned char> h_chars(count);
		thrust::copy(d_conn_kinds.begin(), d_conn_kinds.end(), h_chars.begin());
		thrust::host_vector<unsigned char> h_bits((tab.inner_connkinds)->count());
		HIP_CHECK(hipMemcpy(h_bits.data(), (tab.inner_connkinds)->gpu_data(), h_bits.size(), hipMemcpyDeviceToHost));
		for(unsigned int i = 0; i < count; i++)
		{
			unsigned char bit = h_bits[i >> 3];
			bit = (bit >> (i & 7)) & 0x01;
			assert(bit == (h_chars[i] ? 0x01 : 0x00));
		}
		#endif
	}

	LOG_INFO << "after parse_conn_table_from_numpy..." << std::endl;
	report_dev_info();

	return CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_NORMAL;
}

template<typename T>
struct cap_reciprocal

{
  __host__ __device__ inline
  T operator()(const float& a) const
  {
    return (T)1 / a;
  }
};


template<typename T2>
struct type_convertion
{
  __host__ __device__
  T2 operator()(const float& x, const float& y) const
  {
  	T2 result;
  	result.x = x;
  	result.y = y;
    return result;
  }
};

static __global__ void init_index_kernel(const unsigned char* flags,
										const unsigned int n,
					                    unsigned int* indices)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < n; i += gridSize)
	{
		unsigned int index = 0xffffffff;
		unsigned char fi = flags[i];
		if(fi)
		{
			index = i;
		}
		
		indices[i] = index;
	}
}

static void flat_non_zero(thrust::device_vector<unsigned char>& flags,
							thrust::device_vector<unsigned int>& indices,
							unsigned int& n)
{
	hipLaunchKernelGGL(init_index_kernel, 
					dim3(divide_up<unsigned int>(flags.size(), HIP_THREADS_PER_BLOCK)), 
					dim3(HIP_THREADS_PER_BLOCK), 
					0, 
					0,
					thrust::raw_pointer_cast(flags.data()),
					flags.size(),
					thrust::raw_pointer_cast(indices.data()));
																									
	HIP_POST_KERNEL_CHECK("init_index_kernel");
	HIP_CHECK(hipDeviceSynchronize());
	thrust::sort(indices.begin(), indices.end());
	n = (thrust::find(indices.begin(), indices.end(), 0xffffffff) - indices.begin());
}

static __global__ void stat_exclusive_count_kernel(const unsigned int* exclusive_bids,
													const unsigned int* bcounts,
													const unsigned int m,
													const unsigned int* bids,
													const unsigned int n,
								                    unsigned int* exclusive_counts)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int gridSize = blockDim.x * gridDim.x;
	
	for(unsigned int i = idx; i < m; i += gridSize)
	{
		unsigned int bid = exclusive_bids[i];
		const unsigned int* iter = thrust::find(thrust::device, bids, bids + n, bid);
		unsigned int idx = iter - bids;
		if(idx != n)
		{
			exclusive_counts[idx] = bcounts[i];
		}
	}
}

static void stat_exclusive_count(const unsigned int* exclusive_bids,
									const unsigned int* bcounts,
									const unsigned int m,
									const unsigned int* bids,
									const unsigned int n,
				                    unsigned int* exclusive_counts)
{
	hipLaunchKernelGGL(stat_exclusive_count_kernel, 
					dim3(divide_up<unsigned int>(m, HIP_THREADS_PER_BLOCK)), 
					dim3(HIP_THREADS_PER_BLOCK), 
					0, 
					0,
					exclusive_bids,
					bcounts,
					m,
					bids,
					n,
					exclusive_counts);
	HIP_POST_KERNEL_CHECK("stat_exclusive_count_kernel");
	HIP_CHECK(hipDeviceSynchronize());
}


template<typename T, typename T2, typename C>
void parse_params_from_numpy(const C block_id,
									const std::string& filename,
									unsigned int& neurons,
									ConfigParameter<T, T2>& params)
{
	const unsigned int upper_y = 65535 * 32;
	thrust::host_vector<float> h_props;
	unsigned int height;
	unsigned int width;
	{
		cnpy::NpyArray arr_prop = cnpy::npz_load(filename, "property");
		assert(arr_prop.word_size == 4);
		assert(arr_prop.shape.size() == 2 &&  arr_prop.shape[1] == 22);
		float* props = arr_prop.data<float>();
		height = arr_prop.shape[0];
		width = arr_prop.shape[1];
		neurons = height;
		unsigned int total = height * width;
		h_props.resize(total);

		thrust::device_vector<float> d_props;
		unsigned int count = divide_up<unsigned int>(height, upper_y);
		thrust::device_vector<float> d_temps;
		for(unsigned int i = 0; i < count; i++)
		{
			unsigned int h = (i < (count - 1)) ? upper_y : (height - i * upper_y);
			d_props.resize(h * width);
			d_temps.resize(h * width);
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_temps.data()), props + i * upper_y * width, h * width * sizeof(float), hipMemcpyHostToDevice));
			transpose_gpu<float>(thrust::raw_pointer_cast(d_temps.data()),
							h,
							width,
							thrust::raw_pointer_cast(d_props.data()));
			HIP_CHECK(hipDeviceSynchronize());
			for(unsigned int j = 0; j < width; j++)
			{
				HIP_CHECK(hipMemcpy(h_props.data() + j * height + i * upper_y, thrust::raw_pointer_cast(d_props.data()) + j * h, h * sizeof(float), hipMemcpyDeviceToHost));
			}
		}
	}
	
	unsigned int soffset = 0;
	unsigned int eoffset = height;
	{
		params.noise_rates = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
		if(sizeof(T) == 8)
		{
			thrust::device_vector<float> d_props(height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
			thrust::transform(d_props.begin(), 
							d_props.end(), 
							thrust::device_pointer_cast((params.noise_rates)->mutable_gpu_data()),
							thrust::identity<T>());
		}
		else
		{
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.noise_rates)->mutable_gpu_data()));
		}
	}
	
	soffset += height;
	eoffset += height;

	{
		thrust::device_vector<unsigned int> d_exclusive_colinds;
		{
		
			params.exclusive_flags = nullptr;
			thrust::device_vector<unsigned char> d_datum(height);
			
			{
				thrust::device_vector<float> d_props(height);
				thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
				thrust::transform(d_props.begin(), 
								d_props.end(), 
								d_datum.begin(),
								thrust::identity<unsigned char>());
			}
			thrust::device_vector<unsigned int> d_indices(height);
			unsigned int n;
			flat_non_zero(d_datum, d_indices, n);

			if(n > 0)
			{
				params.exclusive_flags = make_shared<DataAllocator<unsigned char>>(static_cast<int>(block_id + 1), sizeof(unsigned char) * height);
				thrust::copy(d_datum.begin(), d_datum.end(), thrust::device_pointer_cast(params.exclusive_flags->mutable_gpu_data()));
				d_exclusive_colinds.resize(n);
				thrust::copy(d_indices.begin(), d_indices.begin() + n, d_exclusive_colinds.begin());
			}
		}

		soffset += height;
		eoffset += height;
		{
			params.i_ext_stimuli = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
			if(sizeof(T) == 8)
			{
				thrust::device_vector<float> d_props(height);
				thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
				thrust::transform(d_props.begin(), 
								d_props.end(), 
								thrust::device_pointer_cast((params.i_ext_stimuli)->mutable_gpu_data()),
								thrust::identity<T>());
			}
			else
			{
				thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.i_ext_stimuli)->mutable_gpu_data()));
			}
		}
	
		soffset += height;
		eoffset += height;
		{
			params.exclusive_counts = nullptr;
			thrust::device_vector<unsigned int> d_datum(height);
			{
				thrust::device_vector<float> d_props(height);
				thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
				thrust::transform(d_props.begin(), 
							d_props.end(), 
							d_datum.begin(),
							thrust::identity<unsigned int>());
			}
			unsigned int bins = thrust::inner_product(d_datum.begin(),
													d_datum.end() - 1,
													d_datum.begin() + 1,
													1,
													thrust::plus<unsigned int>(),
													thrust::not_equal_to<unsigned int>());
			if(bins > 0)
			{
				params.subids = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * bins);
				params.subcounts = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * (bins + 1));
				
				// compact find the end of each bin of values
				thrust::reduce_by_key(d_datum.begin(),
									d_datum.end(),
				                    thrust::constant_iterator<unsigned int>(1),
				                    thrust::device_pointer_cast(params.subids->mutable_gpu_data()),
				                    thrust::device_pointer_cast(params.subcounts->mutable_gpu_data()));

				thrust::exclusive_scan(thrust::device_pointer_cast(params.subcounts->mutable_gpu_data()),
									thrust::device_pointer_cast(params.subcounts->mutable_gpu_data()) + bins + 1,
									thrust::device_pointer_cast(params.subcounts->mutable_gpu_data()));

				thrust::copy(thrust::device_pointer_cast(params.subids->mutable_gpu_data()), 
							thrust::device_pointer_cast(params.subids->mutable_gpu_data()) + bins,
							params.subids->mutable_cpu_data());
				
				thrust::copy(thrust::device_pointer_cast(params.subcounts->mutable_gpu_data()),
							thrust::device_pointer_cast(params.subcounts->mutable_gpu_data()) + bins + 1,
							params.subcounts->mutable_cpu_data());

				if(!d_exclusive_colinds.empty())
				{
					params.exclusive_counts = make_shared<DataAllocator<unsigned int>>(static_cast<int>(block_id + 1), sizeof(unsigned int) * params.subids->count());
					HIP_CHECK(hipMemset(params.exclusive_counts->mutable_gpu_data(),
										0x00,
										params.exclusive_counts->size()));
					thrust::device_vector<unsigned int> d_exclusive_datum(d_exclusive_colinds.size());
					
					thrust::gather(d_exclusive_colinds.begin(),
								d_exclusive_colinds.end(),
								d_datum.begin(),
								d_exclusive_datum.begin());

					bins = thrust::inner_product(d_exclusive_datum.begin(), 
												d_exclusive_datum.end() - 1,
												d_exclusive_datum.begin() + 1,
												1,
												thrust::plus<unsigned int>(),
												thrust::not_equal_to<unsigned int>());
					assert(bins > 0 && bins <= params.subids->count());

					if(bins > 0)
					{
						 // resize histogram storage	
						thrust::device_vector<unsigned int> d_bids(bins);
						thrust::device_vector<unsigned int> d_counts(bins); 

						// compact find the end of each bin of values
						thrust::reduce_by_key(d_exclusive_datum.begin(),
											d_exclusive_datum.end(),
											thrust::constant_iterator<unsigned int>(1),
											d_bids.begin(),
											d_counts.begin());

						stat_exclusive_count(thrust::raw_pointer_cast(d_bids.data()),
											thrust::raw_pointer_cast(d_counts.data()),
											bins,
											params.subids->mutable_gpu_data(),
											params.subids->count(),
											params.exclusive_counts->mutable_gpu_data());
					}
				}
			}
		}
	}

	soffset += height;
	eoffset += height;
	params.c_membrane_reciprocals= make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	{
		thrust::device_vector<float> d_props(height);
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
		thrust::transform(d_props.begin(), 
						d_props.end(), 
						thrust::device_pointer_cast((params.c_membrane_reciprocals)->mutable_gpu_data()),
						cap_reciprocal<T>());
	}

	soffset += height;
	eoffset += height;
	params.t_refs = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	if(sizeof(T) == 8)
	{
		thrust::device_vector<float> d_props(height);
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
		thrust::transform(d_props.begin(), 
						d_props.end(), 
						thrust::device_pointer_cast((params.t_refs)->mutable_gpu_data()),
						thrust::identity<T>());
	}
	else
	{
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.t_refs)->mutable_gpu_data()));
	}

	soffset += height;
	eoffset += height;
	params.g_leakages = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	if(sizeof(T) == 8)
	{
		thrust::device_vector<float> d_props(height);
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
		thrust::transform(d_props.begin(), 
						d_props.end(), 
						thrust::device_pointer_cast((params.g_leakages)->mutable_gpu_data()),
						thrust::identity<T>());
	}
	else
	{
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.g_leakages)->mutable_gpu_data()));
	}

	soffset += height;
	eoffset += height;
	params.v_leakages = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	if(sizeof(T) == 8)
	{
		thrust::device_vector<float> d_props(height);
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
		thrust::transform(d_props.begin(), 
						d_props.end(), 
						thrust::device_pointer_cast((params.v_leakages)->mutable_gpu_data()),
						thrust::identity<T>());
	}
	else
	{
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.v_leakages)->mutable_gpu_data()));
	}

	soffset += height;
	eoffset += height;
	params.v_thresholds = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	if(sizeof(T) == 8)
	{
		thrust::device_vector<float> d_props(height);
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
		thrust::transform(d_props.begin(), 
						d_props.end(), 
						thrust::device_pointer_cast((params.v_thresholds)->mutable_gpu_data()),
						thrust::identity<T>());
	}
	else
	{
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.v_thresholds)->mutable_gpu_data()));
	}

	soffset += height;
	eoffset += height;
	params.v_resets = make_shared<DataAllocator<T>>(static_cast<int>(block_id + 1), sizeof(T) * height);
	if(sizeof(T) == 8)
	{
		thrust::device_vector<float> d_props(height);
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, d_props.begin());
		thrust::transform(d_props.begin(), 
						d_props.end(), 
						thrust::device_pointer_cast((params.v_resets)->mutable_gpu_data()),
						thrust::identity<T>());
	}
	else
	{
		thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset, thrust::device_pointer_cast((params.v_resets)->mutable_gpu_data()));
	}
	
	{
		soffset += height;
		eoffset += height;
		params.g_ex_conducts = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		{
			thrust::device_vector<float> d_props(2 * height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset + height, d_props.begin());
			thrust::transform(d_props.begin(), d_props.begin() + height, d_props.begin() + height, thrust::device_pointer_cast((params.g_ex_conducts)->mutable_gpu_data()), type_convertion<T2>());
		}
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;
		params.g_in_conducts = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		{
			thrust::device_vector<T> d_props(2 * height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset + height, d_props.begin());
			thrust::transform(d_props.begin(), d_props.begin() + height, d_props.begin() + height, thrust::device_pointer_cast((params.g_in_conducts)->mutable_gpu_data()), type_convertion<T2>());
		}
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;
		params.v_ex_membranes = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		{
			thrust::device_vector<float> d_props(2 * height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset + height, d_props.begin());
			thrust::transform(d_props.begin(), d_props.begin() + height, d_props.begin() + height, thrust::device_pointer_cast((params.v_ex_membranes)->mutable_gpu_data()), type_convertion<T2>());
		}
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;
		params.v_in_membranes = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		{
			thrust::device_vector<float> d_props(2 * height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset + height, d_props.begin());
			thrust::transform(d_props.begin(), d_props.begin() + height, d_props.begin() + height, thrust::device_pointer_cast((params.v_in_membranes)->mutable_gpu_data()), type_convertion<T2>());
		}
		soffset += height;
		eoffset += height;
	}

	{
		soffset += height;
		eoffset += height;

		params.tao_ex_constants = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		{
			thrust::device_vector<float> d_props(2 * height);
			thrust::device_vector<T2> d_datum(height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset + height, d_props.begin());
			thrust::transform(d_props.begin(), d_props.begin() + height, d_props.begin() + height, d_datum.begin(), type_convertion<T2>());
			thrust::copy(d_datum.begin(), d_datum.end(), params.tao_ex_constants->mutable_cpu_data());
		}

		soffset += height;
		eoffset += height;
	
		soffset += height;
		eoffset += height;
		params.tao_in_constants = make_shared<DataAllocator<T2>>(static_cast<int>(block_id + 1), sizeof(T2) * height);
		{
			thrust::device_vector<float> d_props(2 * height);
			thrust::device_vector<T2> d_datum(height);
			thrust::copy(h_props.begin() + soffset, h_props.begin() + eoffset + height, d_props.begin());
			thrust::transform(d_props.begin(), d_props.begin() + height, d_props.begin() + height, d_datum.begin(), type_convertion<T2>());
			thrust::copy(d_datum.begin(), d_datum.end(), params.tao_in_constants->mutable_cpu_data());
		}
	}

	report_dev_info();
}


template CONFIG_BLOCK_TYPE parse_conn_table_from_numpy<unsigned short>(const unsigned short block_id,
																const std::string& filename,
																InputSpike& input,
																ConnectionTable<unsigned short>& tab);

template void parse_params_from_numpy<float, float2, unsigned short>(const unsigned short block_id,
																	const std::string& filename,
																	unsigned int& neurons,
																	ConfigParameter<float, float2>& params);

template void parse_params_from_numpy<double, double2, unsigned short>(const unsigned short block_id,
																	const std::string& filename,
																	unsigned int& neurons,
																	ConfigParameter<double, double2>& params);


}//namespace istbi

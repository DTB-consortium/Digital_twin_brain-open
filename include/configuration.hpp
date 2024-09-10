#pragma once
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>
#include <map>
#include "data_allocator.hpp"
#include "weights.hpp"

using namespace std;

namespace dtb {

enum CONFIG_BLOCK_TYPE
{
	CONFIG_BLOCK_TYPE_INPUT = 0,
	CONFIG_BLOCK_TYPE_NORMAL,
	CONFIG_BLOCK_TYPE_UNKNOWN
};

struct InputSpike
{
	shared_ptr<DataAllocator<unsigned int>> input_timestamps;
	shared_ptr<DataAllocator<unsigned int>> input_rowptrs;
	shared_ptr<DataAllocator<unsigned int>> input_colinds;
};

template<typename BockID>
struct ConnectionTable
{
	shared_ptr<DataAllocator<unsigned int>> inner_rowptrs;
	shared_ptr<DataAllocator<unsigned int>> inner_colinds;
	Weights inner_vals;
	shared_ptr<DataAllocator<unsigned int>> inner_conninds;
	shared_ptr<DataAllocator<unsigned char>> inner_connkinds;

	vector<BockID> outer_conn_bids;
	shared_ptr<DataAllocator<unsigned int>> outer_conn_inds;
	shared_ptr<DataAllocator<unsigned int>> outer_conn_nids;
	
	shared_ptr<DataAllocator<unsigned int>> outer_rowptrs;
	shared_ptr<DataAllocator<unsigned int>> outer_colinds;
	Weights outer_vals;
	shared_ptr<DataAllocator<unsigned char>> outer_connkinds;
};

template<typename T, typename T2>
struct ConfigParameter
{
	shared_ptr<DataAllocator<T>> noise_rates;
	shared_ptr<DataAllocator<unsigned char>> exclusive_flags;
	shared_ptr<DataAllocator<unsigned int>> exclusive_counts;
	shared_ptr<DataAllocator<T>> i_ext_stimuli;
	shared_ptr<DataAllocator<unsigned int>> subids;
	shared_ptr<DataAllocator<unsigned int>> subcounts;
	shared_ptr<DataAllocator<T2>> g_ex_conducts;
	shared_ptr<DataAllocator<T2>> g_in_conducts;
	shared_ptr<DataAllocator<T2>> v_ex_membranes;
	shared_ptr<DataAllocator<T2>> v_in_membranes;
	shared_ptr<DataAllocator<T2>> tao_ex_constants;
	shared_ptr<DataAllocator<T2>> tao_in_constants;
	shared_ptr<DataAllocator<T>> v_resets;
	shared_ptr<DataAllocator<T>> v_thresholds;
	shared_ptr<DataAllocator<T>> c_membrane_reciprocals;
	shared_ptr<DataAllocator<T>> v_leakages;
	shared_ptr<DataAllocator<T>> g_leakages;
	shared_ptr<DataAllocator<T>> t_refs;
};

template<typename BlockID>
CONFIG_BLOCK_TYPE parse_conn_table_from_numpy(const BlockID block_id,
										const std::string& filename,
										InputSpike& input,
										ConnectionTable<BlockID>& tab);

template<typename T, typename T2, typename BlockID>
void parse_params_from_numpy(const BlockID block_id,
							const std::string& filename,
							unsigned int& neurons,
							ConfigParameter<T, T2>& params);

}//namespace dtb

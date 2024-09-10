#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include "device_function.hpp"
#include "block.hpp"
#include "configuration.hpp"
#include "logging.hpp"
#include <sstream>
#include <cmath>


namespace dtb {

template<typename T, typename T2>
BrainBlock<T, T2>::BrainBlock(const unsigned short block_id,
								const int gpu_id,
								const T delta_t,
								const unsigned long long seed)
	:block_type_(BLOCK_TYPE_NORMAL),
	bid_(block_id),
	gpu_id_(gpu_id),
	seed_(seed),
	delta_t_(delta_t),
	input_timestamps_(nullptr),
	input_rowptrs_(nullptr),
	input_colinds_(nullptr)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_config_params_gpu(const std::string& filename)
{
	ConfigParameter<T, T2> params;
	parse_params_from_numpy<T, T2, unsigned short>(bid_,
							filename,
							total_neurons_,
							params);

	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		j_ex_presynaptics_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_ex_presynaptic_deltas_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_ex_presynaptic_deltas_->gpu_data();
		j_in_presynaptics_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_in_presynaptic_deltas_ = make_shared<DataAllocator<T2>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T2));
		j_in_presynaptic_deltas_->gpu_data();
		i_synaptics_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
		t_actives_ = make_shared<DataAllocator<int>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(int));
		gen_states_ = make_shared<DataAllocator<hiprandStatePhilox4_32_10_t>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(hiprandStatePhilox4_32_10_t));
		uniform_samples_ = nullptr;
	}
	v_membranes_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
	f_inner_actives_ = make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), total_neurons_);
	f_recorded_inner_actives_ = make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), total_neurons_);
	
	f_exclusive_flags_ = params.exclusive_flags;
	f_exclusive_counts_ = params.exclusive_counts;
	if(nullptr != f_exclusive_counts_)
	{
		HIP_CHECK(hipMemcpy(f_exclusive_counts_->mutable_cpu_data(), f_exclusive_counts_->gpu_data(), f_exclusive_counts_->size(), hipMemcpyDeviceToHost));
	}
	
	sub_bids_ = params.subids;
	sub_bcounts_ = params.subcounts;
	v_resets_ = params.v_resets;
	v_thresholds_ = params.v_thresholds;
		
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		noise_rates_ = params.noise_rates;
		i_ext_stimuli_= params.i_ext_stimuli;
		g_ex_conducts_ = params.g_ex_conducts;
		g_in_conducts_ = params.g_in_conducts;
		v_ex_membranes_ = params.v_ex_membranes;
		v_in_membranes_ = params.v_in_membranes;
		tao_ex_constants_ = params.tao_ex_constants;
		tao_in_constants_ = params.tao_in_constants;
		c_membrane_reciprocals_ = params.c_membrane_reciprocals;
		v_leakages_ = params.v_leakages;
		g_leakages_ = params.g_leakages;
		t_refs_ = params.t_refs;
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_connection_table_gpu(const std::string& filename)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));

	InputSpike input;
	ConnectionTable<unsigned short> tab;
	if(CONFIG_BLOCK_TYPE::CONFIG_BLOCK_TYPE_INPUT == parse_conn_table_from_numpy<unsigned short>(
													bid_,
													filename,
													input,
													tab))
	{
		block_type_ = BLOCK_TYPE_INPUT;
		input_timestamps_ = input.input_timestamps;
		input_rowptrs_ = input.input_rowptrs;
		input_colinds_ = input.input_colinds;
		return;
	}

	block_type_ = BLOCK_TYPE_NORMAL;
	inner_conninds_ = tab.inner_conninds;
	
	inner_rowptrs_ = tab.inner_rowptrs;
	inner_colinds_ = tab.inner_colinds;
	inner_w_synaptics_ = tab.inner_vals;
	inner_connkinds_ = tab.inner_connkinds;
	
	f_receiving_bids_ = tab.outer_conn_bids;
	f_receiving_rowptrs_ = tab.outer_conn_inds;
	f_receiving_colinds_ = tab.outer_conn_nids;

	outer_rowptrs_ = tab.outer_rowptrs;
	outer_colinds_ = tab.outer_colinds;
	outer_w_synaptics_ = tab.outer_vals;
	outer_connkinds_ = tab.outer_connkinds;
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_random_state()
{
	init_random_gpu(gen_seed(), 0, total_neurons_, gen_states_->mutable_gpu_data());
	HIP_CHECK(hipDeviceSynchronize());
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::init_all_stages_gpu()
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		init_membrane_voltage_gpu<T>(v_thresholds_->gpu_data(),
							v_resets_->gpu_data(),
							total_neurons_,
							v_membranes_->mutable_gpu_data());

		//create_generator_state(seed_, gen_states_->mutable_gpu_data(), kernel_params_->mutable_gpu_data());
		HIP_CHECK(hipMemset(t_actives_->mutable_gpu_data(), 0x00, t_actives_->size()));
		//init_spike_time_gpu<T>(total_neurons_, t_refs_->gpu_data(), static_cast<T>(-1), t_actives_->mutable_gpu_data());
		update_refractory_period_gpu<T>(delta_t_, t_refs_->count(), t_refs_->mutable_gpu_data());
		if(!f_receiving_bids_.empty())
		{
			size_t total_bytes = 0;
			LOG_INFO << "total number of recved from other node: " << f_receiving_colinds_->count()  << std::endl; 
			HIP_CHECK(hipMemcpy(f_receiving_rowptrs_->mutable_gpu_data(), f_receiving_rowptrs_->cpu_data(), f_receiving_rowptrs_->size(), hipMemcpyHostToDevice));
			f_receiving_active_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), f_receiving_rowptrs_->size());
			f_receiving_active_rowptrs_->cpu_data();
			f_receiving_active_rowptrs_->gpu_data();
			total_bytes += f_receiving_rowptrs_->size();
			f_receiving_active_colinds_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), f_receiving_colinds_->size(), false);
			f_receiving_active_colinds_->cpu_data();
			total_bytes += f_receiving_colinds_->size();

			LOG_INFO << "need " << total_bytes << " bytes for recving actives." << std::endl;
		}
		
		T2 val = {0.f, 0.f};
		init_presynaptic_voltage_gpu<T2>(total_neurons_,
										val,
										j_ex_presynaptics_->mutable_gpu_data());
		
		init_presynaptic_voltage_gpu<T2>(total_neurons_,
										val,
										j_in_presynaptics_->mutable_gpu_data());
		update_I_synaptic_gpu();
	}
	
	HIP_CHECK(hipMemset(f_inner_actives_->mutable_gpu_data(), 0x00, f_inner_actives_->size()));
}

template<typename T, typename T2>
void BrainBlock<T, T2>::reset_V_membrane_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		reset_membrane_voltage_gpu<T>(v_resets_->gpu_data(),
									total_neurons_,
									v_membranes_->mutable_gpu_data(),
									stream);
	}
}

template<typename T, typename T2>
unsigned int BrainBlock<T, T2>::update_F_input_spike_gpu(const unsigned int timestamp, 
															const unsigned int offset,
															hipStream_t stream)
{
	if(BLOCK_TYPE_INPUT == block_type_)
	{
		HIP_CHECK(hipMemsetAsync(f_inner_actives_->mutable_gpu_data(), 0x00, f_inner_actives_->size(), stream));
		if(offset < input_timestamps_->count())
		{
			//std::cout << "(" << timestamp << ", " << input_timestamps_->cpu_data()[offset] << ")" << std::endl;
			if(input_timestamps_->cpu_data()[offset] == timestamp)
			{
				const unsigned int n = input_rowptrs_->cpu_data()[offset + 1] - input_rowptrs_->cpu_data()[offset];
				if(n > 0)
				{
					update_input_spike_gpu(input_colinds_->gpu_data() + input_rowptrs_->cpu_data()[offset],
											n,
											f_inner_actives_->mutable_gpu_data(),
											stream);
				}
				return offset + 1;
			}
		}
	}
	return offset;
}

template<typename T, typename T2>
void BrainBlock<T, T2>::set_I_ou_current_param(const vector<unsigned int>& brain_indice,
												const vector<float>& means,
												const vector<float>& deviations,
												const vector<float>& correlation_times,
												hipStream_t stream)
{
	for(auto index : brain_indice)
	{
		unsigned int offset = 0;
		if(1 == sub_bids_->count())
		{
			if(index != sub_bids_->cpu_data()[0])
				continue;
		}
		else
		{
			offset = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), index) - sub_bids_->cpu_data();
			if(offset == sub_bids_->count())
			{
				continue;
			}
		}

		std::stringstream ss;
		if(ou_current_params_.find(offset) == ou_current_params_.end())
		{
			auto param = ou_current_params_.emplace(std::make_pair(offset, OU_Background_Current_Param()));
			assert(param.second);
			param.first->second.mean_ = means[offset];
			param.first->second.deviation_ = deviations[offset];
			param.first->second.correlation_time_ = correlation_times[offset];
			if(nullptr == i_ou_background_stimuli_)
			{
				i_ou_background_stimuli_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
			}
			reset_partial_gpu<T>(static_cast<T>(means[offset]),
								offset,
								sub_bcounts_->cpu_data(),
								i_ou_background_stimuli_->mutable_gpu_data(),
								stream);
			ss << "add map: " << "[" << offset << "]" << std::endl;
		}
		else
		{
			assert(nullptr != i_ou_background_stimuli_);
			auto& param = ou_current_params_[offset];
			param.mean_ = means[offset];
			param.deviation_ = deviations[offset];
			param.correlation_time_ = correlation_times[offset];
			ss << "modify map: " << "[" << offset << "]" << std::endl;
		}

		ss << "mean: " << means[offset] << std::endl;
		ss << "deviation: " << deviations[offset] << std::endl;
		ss << "correlation_time: " << correlation_times[offset] << std::endl;

		LOG_INFO << ss.str();
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_ou_background_gpu(const T mean,
													   const T stdv,
													   bool saving_sample,
													   hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(nullptr != i_ou_background_stimuli_)
		{
			if(saving_sample)
			{
				if(nullptr == uniform_samples_)
				{
					uniform_samples_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
				}

				update_ou_background_stimuli_gpu<T>(gen_states_->mutable_gpu_data(),
													sub_bcounts_->cpu_data(),
													ou_current_params_,
													delta_t_ / t_steps_,
													mean,
													stdv,
													i_ou_background_stimuli_->mutable_gpu_data(),
													i_synaptics_->mutable_gpu_data(),
													uniform_samples_->mutable_gpu_data(),
													stream);
			}
			else
			{
				update_ou_background_stimuli_gpu<T>(gen_states_->mutable_gpu_data(),
													sub_bcounts_->cpu_data(),
													ou_current_params_,
													delta_t_ / t_steps_,
													mean,
													stdv,
													i_ou_background_stimuli_->mutable_gpu_data(),
													i_synaptics_->mutable_gpu_data(),
													NULL,
													stream);
			}
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::set_I_ttype_ca_param(const vector<unsigned int>& brain_indice,
												const vector<float>& h_init_vals,
												const vector<float>& g_ts,
												const vector<float>& tao_h_minuss,
												const vector<float>& tao_h_pluss,
												const vector<float>& v_hs,
												const vector<float>& v_ts,
												hipStream_t stream)
{
	for(auto index : brain_indice)
	{
		unsigned int offset = 0;
		if(1 == sub_bids_->count())
		{
			if(index != sub_bids_->cpu_data()[0])
				continue;
		}
		else
		{
			offset = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), index) - sub_bids_->cpu_data();
			if(offset == sub_bids_->count())
			{
				continue;
			}
		}

		std::stringstream ss;
		if(ca_current_params_.find(offset) == ca_current_params_.end())
		{
			auto param = ca_current_params_.emplace(std::make_pair(offset, T_Type_Ca_Current_Param()));
			assert(param.second);
			param.first->second.g_t_ = g_ts[offset];
			param.first->second.tao_h_minus_ = tao_h_minuss[offset];
			param.first->second.tao_h_plus_ = tao_h_pluss[offset];
			param.first->second.v_h_ = v_hs[offset];
			param.first->second.v_t_ = v_ts[offset];

			if(nullptr == h_ttype_ca_stimuli_)
			{
				h_ttype_ca_stimuli_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
			}

			reset_partial_gpu<T>(static_cast<T>(h_init_vals[offset]),
							offset,
							sub_bcounts_->cpu_data(),
							h_ttype_ca_stimuli_->mutable_gpu_data(),
							stream);

			ss << "add map: " << "[" << offset << "]" << std::endl;
			ss << "h_init_val: " << h_init_vals[offset] << std::endl;
		}
		else
		{
			assert(nullptr != h_ttype_ca_stimuli_);
			auto& param = ca_current_params_[offset];
			ss << "modify map:" << "[" << offset << "]" << std::endl;

			param.g_t_ = g_ts[offset];
			param.tao_h_minus_ = tao_h_minuss[offset];
			param.tao_h_plus_ = tao_h_pluss[offset];
			param.v_h_ = v_hs[offset];
			param.v_t_ = v_ts[offset];
		}

		ss << "g_t: " << g_ts[offset] << std::endl;
		ss << "tao_h_minus: " <<  tao_h_minuss[offset] << std::endl;
		ss << "tao_h_plus: " << tao_h_pluss[offset] << std::endl;
		ss << "v_h: " << v_hs[offset] << std::endl;
		ss << "v_t: " << v_ts[offset] << std::endl;
		
		LOG_INFO << ss.str();
	}
	
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_ttype_ca_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(nullptr != h_ttype_ca_stimuli_)
		{	
			update_ttype_ca_stimuli_gpu<T>(v_membranes_->gpu_data(),
										sub_bcounts_->cpu_data(),
										ca_current_params_,
										h_ttype_ca_stimuli_->gpu_data(),
										i_synaptics_->mutable_gpu_data(),
										stream);
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_H_ttype_ca_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(nullptr != h_ttype_ca_stimuli_)
		{	
			update_h_ttype_ca_gpu<T>(v_membranes_->gpu_data(),
										sub_bcounts_->cpu_data(),
										ca_current_params_,
										delta_t_ / t_steps_,
										h_ttype_ca_stimuli_->mutable_gpu_data(),
										stream);
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::set_I_dopamine_param(const vector<unsigned int>& brain_indice,
												const vector<float>& v_das,
												const vector<float>& g_das)
{
	for(auto index : brain_indice)
	{
		unsigned int offset = 0;
		if(1 == sub_bids_->count())
		{
			if(index != sub_bids_->cpu_data()[0])
				continue;
		}
		else
		{
			offset = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), index) - sub_bids_->cpu_data();
			if(offset == sub_bids_->count())
			{
				continue;
			}
		}

		std::stringstream ss;
		if(da_current_params_.find(offset) == da_current_params_.end())
		{
			auto param = da_current_params_.emplace(std::make_pair(offset, Dopamine_Current_Param()));
			assert(param.second);
			param.first->second.v_dopamine_ = v_das[offset];
			param.first->second.g_dopamine_ = g_das[offset];
			ss << "add map: " << "[" << offset << "]" << std::endl;
		}
		else
		{
			auto& param = da_current_params_[offset];
			param.v_dopamine_ = v_das[offset];
			param.g_dopamine_ = g_das[offset];
			ss << "modify map: " << "[" << offset << "]" << std::endl;
		}

		ss << "v_dopamine: " << v_das[offset] << std::endl;
		ss << "g_dopamine: " << g_das[offset] << std::endl;

		LOG_INFO << ss.str();
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_dopamine_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_dopamine_stimuli_gpu<T>(v_membranes_->gpu_data(),
									sub_bcounts_->cpu_data(),
									da_current_params_,
									i_synaptics_->mutable_gpu_data(),
									stream);
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::set_I_adaptation_param(const vector<unsigned int>& brain_indice,
												const vector<float>& ca_init_vals,
												const vector<float>& ca_decays,
												const vector<float>& alpha_constants,
												const vector<float>& v_ks,
												const vector<float>& g_ahps,
												hipStream_t stream)
{
	for(auto index : brain_indice)
	{
		unsigned int offset = 0;
		if(1 == sub_bids_->count())
		{
			if(index != sub_bids_->cpu_data()[0])
				continue;
		}
		else
		{
			offset = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), index) - sub_bids_->cpu_data();
			if(offset == sub_bids_->count())
			{
				continue;
			}
		}

		std::stringstream ss;
		if(adapt_current_params_.find(offset) == adapt_current_params_.end())
		{
			auto param = adapt_current_params_.emplace(std::make_pair(offset, Adaptation_Current_Param()));
			assert(param.second);
			param.first->second.ca_decay_ = std::exp(((T)-1) * (delta_t_ / t_steps_ ) / ca_decays[offset]);
			param.first->second.alpha_constant_ = alpha_constants[offset];
			param.first->second.v_k_ = v_ks[offset];
			param.first->second.g_ahp_ = g_ahps[offset];
			if(nullptr == ahp_ca_concentration_)
			{
				ahp_ca_concentration_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
			}

			reset_partial_gpu<T>(static_cast<T>(ca_init_vals[offset]),
								offset,
								sub_bcounts_->cpu_data(),
								ahp_ca_concentration_->mutable_gpu_data(),
								stream);
			
			ss << "add map: " << "[" << offset << "]" << std::endl;
			ss << "ca_init_val: " << ca_init_vals[offset] << std::endl;
		}
		else
		{
			assert(nullptr != ahp_ca_concentration_);
			auto& param = adapt_current_params_[offset];
			ss << "modify map:" << "[" << offset << "]" << std::endl;
			
			param.ca_decay_ = std::exp(-1.f * static_cast<float>(delta_t_ / t_steps_)  / ca_decays[offset]);
			param.alpha_constant_ = alpha_constants[offset];
			param.v_k_ = v_ks[offset];
			param.g_ahp_ = g_ahps[offset];
			ss << "modify map: " << "[" << offset << "]" << std::endl;
		}

		ss << "ca_decay: " << ca_decays[offset] << std::endl;
		ss << "alpha_constant: " << alpha_constants[offset] << std::endl;
		ss << "v_k: " << v_ks[offset] << std::endl;
		ss << "g_ahp: " << g_ahps[offset] << std::endl;

		LOG_INFO << ss.str();
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_adaptation_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(nullptr != ahp_ca_concentration_)
		{
			update_adaptation_stimuli_gpu<T>(v_membranes_->gpu_data(),
											sub_bcounts_->cpu_data(),
											adapt_current_params_,
											ahp_ca_concentration_->mutable_gpu_data(),
											i_synaptics_->mutable_gpu_data(),
											stream);
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_ca_concentration_gpu(bool use_recorded, hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(nullptr != ahp_ca_concentration_)
		{
			if(use_recorded)
			{
				update_ahp_ca_concentration_gpu<T>(f_recorded_inner_actives_->gpu_data(),
												sub_bcounts_->cpu_data(),
												adapt_current_params_,
												ahp_ca_concentration_->mutable_gpu_data(),
												stream);
			}
			else
			{
				update_ahp_ca_concentration_gpu<T>(f_inner_actives_->gpu_data(),
												sub_bcounts_->cpu_data(),
												adapt_current_params_,
												ahp_ca_concentration_->mutable_gpu_data(),
												stream);
			}
		}
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_I_synaptic_gpu(bool have_receptor_imeans, hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(have_receptor_imeans)
		{
			update_synaptic_current_gpu<T, T2>(j_ex_presynaptics_->gpu_data(),
									j_in_presynaptics_->gpu_data(),
									g_ex_conducts_->gpu_data(),
									g_in_conducts_->gpu_data(),
									v_ex_membranes_->gpu_data(),
									v_in_membranes_->gpu_data(),
									v_membranes_->gpu_data(),
									total_neurons_,
									j_ex_presynaptic_deltas_->mutable_gpu_data(),
									j_in_presynaptic_deltas_->mutable_gpu_data(),
									i_synaptics_->mutable_gpu_data(),
									stream);
		}
		else
		{
			update_synaptic_current_gpu<T, T2>(j_ex_presynaptics_->gpu_data(),
									j_in_presynaptics_->gpu_data(),
									g_ex_conducts_->gpu_data(),
									g_in_conducts_->gpu_data(),
									v_ex_membranes_->gpu_data(),
									v_in_membranes_->gpu_data(),
									v_membranes_->gpu_data(),
									total_neurons_,
									NULL,
									NULL,
									i_synaptics_->mutable_gpu_data(),
									stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_V_membrane_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_membrane_voltage_gpu<T>(i_synaptics_->gpu_data(),
								i_ext_stimuli_->gpu_data(),
								v_resets_->gpu_data(),
								v_thresholds_->gpu_data(),
								c_membrane_reciprocals_->gpu_data(),
								v_leakages_->gpu_data(),
								g_leakages_->gpu_data(),
								t_refs_->gpu_data(),
								total_neurons_,
								delta_t_ / t_steps_,
								static_cast<int>(t_steps_),
								f_inner_actives_->mutable_gpu_data(),
								t_actives_->mutable_gpu_data(),
								v_membranes_->mutable_gpu_data(),
								stream);
		
	}
	else
	{
		update_membrane_voltage_for_input_gpu<T>(v_resets_->gpu_data(),
												v_thresholds_->gpu_data(),
												f_inner_actives_->gpu_data(),
												total_neurons_,
												v_membranes_->mutable_gpu_data(),
												stream);
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_F_active_gpu(const T a,
											   const T b,
											   bool saving_sample,
											   hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(saving_sample)
		{
			if(nullptr == uniform_samples_)
			{
				uniform_samples_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
			}
			
			update_spike_gpu<T>(gen_states_->mutable_gpu_data(),
								total_neurons_,
								noise_rates_->gpu_data(),
								f_inner_actives_->gpu_data(),
								f_recorded_inner_actives_->mutable_gpu_data(),
								a,
								b,
								uniform_samples_->mutable_gpu_data(),
								stream);
		}
		else
		{
			update_spike_gpu<T>(gen_states_->mutable_gpu_data(),
								total_neurons_,
								noise_rates_->gpu_data(),
								f_inner_actives_->gpu_data(),
								f_recorded_inner_actives_->mutable_gpu_data(),
								a,
								b,
								NULL,
								stream);
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_F_accumulated_spike_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_accumulated_spike_gpu(f_inner_actives_->gpu_data(),
								total_neurons_,
								f_recorded_inner_actives_->mutable_gpu_data());
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::record_F_sending_actives(const map<unsigned short, vector<unsigned int>>& rank_map)
{
	if(rank_map.empty())
		return;

	unsigned int n = 0;
	vector<unsigned int> sending_rowptrs;
	vector<unsigned int> sending_colinds;
	sending_rowptrs.push_back(n);
	
	for (auto it = rank_map.begin(); it != rank_map.end(); ++it)
	{
		f_sending_bids_.push_back(it->first);
		n += it->second.size();
		sending_rowptrs.push_back(n);
	}

	LOG_INFO << "total number of sent to other node: " << n  << std::endl; 
	f_sending_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), sending_rowptrs.size() * sizeof(unsigned int));
	memcpy(f_sending_rowptrs_->mutable_cpu_data(), sending_rowptrs.data(), f_sending_rowptrs_->size());
	HIP_CHECK(hipMemcpy(f_sending_rowptrs_->mutable_gpu_data(), f_sending_rowptrs_->cpu_data(), f_sending_rowptrs_->size(), hipMemcpyHostToDevice));
	f_sending_colinds_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), n * sizeof(unsigned int));
	
	unsigned int blocks = divide_up<unsigned int>(n, HIP_THREADS_PER_BLOCK * HIP_ITEMS_PER_THREAD);
	f_sending_block_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), 2 * (blocks + 1) * sizeof(unsigned int));
	f_sending_block_rowptrs_->gpu_data();
	f_sending_active_rowptrs_ = make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), 2 * sending_rowptrs.size() * sizeof(unsigned int));
	f_sending_active_rowptrs_->gpu_data();
	f_sending_active_rowptrs_->cpu_data();

	size_t storage_size_bytes = 0;
	count_F_sending_actives_temporary_storage_size(n,
												sending_rowptrs.size() - 1,
												f_sending_block_rowptrs_->mutable_gpu_data(),
												f_sending_active_rowptrs_->mutable_gpu_data(),
												storage_size_bytes);
	
	f_sending_active_colinds_= make_shared<DataAllocator<unsigned int>>(static_cast<int>(bid_ + 1), storage_size_bytes, false);
	f_sending_active_colinds_->cpu_data();
	f_sending_actives_= make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), n * sizeof(unsigned char));
	f_sending_actives_->gpu_data();
	
	n = 0;
	for (auto it = rank_map.begin(); it != rank_map.end(); ++it, n++)
	{
		assert(it->second.size() == (sending_rowptrs[n + 1] - sending_rowptrs[n]));
		HIP_CHECK(hipMemcpy(f_sending_colinds_->mutable_gpu_data() + sending_rowptrs[n], it->second.data(), it->second.size() * sizeof(unsigned int), hipMemcpyHostToDevice));
	}
	
	HIP_CHECK(hipMemcpy(f_sending_colinds_->mutable_cpu_data(), f_sending_colinds_->gpu_data(), f_sending_colinds_->size(), hipMemcpyDeviceToHost));
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::count_F_sending_actives_temporary_storage_size(const unsigned int sending_count,
																			const unsigned int segments,
																			unsigned int* block_rowptrs,
																			unsigned int* active_rowptrs,
																			size_t& storage_size_bytes,
																			hipStream_t stream)
{
	
	count_sending_spikes_temporary_storage_size(sending_count,
											segments,
											block_rowptrs,
											active_rowptrs,
											storage_size_bytes,
											stream);
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_sending_actives_gpu(bool use_recorded, hipStream_t stream)
{
	if(!f_sending_bids_.empty())
	{
		if(use_recorded)
		{
			update_sending_spikes_gpu(f_recorded_inner_actives_->gpu_data(),
							f_sending_rowptrs_->gpu_data(),
							f_sending_colinds_->gpu_data(),
							f_sending_rowptrs_->count() - 1,
							f_sending_colinds_->count(),
							f_sending_actives_->mutable_gpu_data(),
							f_sending_block_rowptrs_->mutable_gpu_data(),
							f_sending_active_rowptrs_->mutable_gpu_data(),
							f_shared_active_colinds_->mutable_gpu_data(),
							stream);
		}
		else
		{
			update_sending_spikes_gpu(f_inner_actives_->gpu_data(),
								f_sending_rowptrs_->gpu_data(),
								f_sending_colinds_->gpu_data(),
								f_sending_rowptrs_->count() - 1,
								f_sending_colinds_->count(),
								f_sending_actives_->mutable_gpu_data(),
								f_sending_block_rowptrs_->mutable_gpu_data(),
								f_sending_active_rowptrs_->mutable_gpu_data(),
								f_shared_active_colinds_->mutable_gpu_data(),
								stream);
		}
	}
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_F_recving_actives_gpu(hipStream_t stream)
{
	update_recving_spikes_gpu(f_receiving_rowptrs_->gpu_data(), 
							f_receiving_active_rowptrs_->gpu_data(),
							f_receiving_rowptrs_->count() - 1,
							f_shared_active_colinds_->mutable_gpu_data(),
							stream);
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_presynaptic_voltage_gpu<T, T2>(tao_ex_constants_->gpu_data(),
											tao_in_constants_->gpu_data(),
											total_neurons_,
											j_ex_presynaptics_->mutable_gpu_data(),
											j_ex_presynaptic_deltas_->mutable_gpu_data(),
											j_in_presynaptics_->mutable_gpu_data(),
											j_in_presynaptic_deltas_->mutable_gpu_data(),
											stream);
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_per_step_gpu(hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		update_presynaptic_voltage_gpu<T, T2>(tao_ex_constants_->gpu_data(),
											tao_in_constants_->gpu_data(),
											total_neurons_,
											j_ex_presynaptics_->mutable_gpu_data(),
											NULL,
											j_in_presynaptics_->mutable_gpu_data(),
											NULL,
											stream);
	}
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_inner_gpu(bool use_recorded, hipStream_t stream)
{
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		HIP_CHECK(hipMemsetAsync(j_ex_presynaptic_deltas_->mutable_gpu_data(), 0x00, j_ex_presynaptic_deltas_->size(), stream));
		HIP_CHECK(hipMemsetAsync(j_in_presynaptic_deltas_->mutable_gpu_data(), 0x00, j_in_presynaptic_deltas_->size(), stream));
		if(nullptr != inner_conninds_)
		{
			if(use_recorded)
			{
				update_presynaptic_voltage_inner_gpu<T, T2>(inner_rowptrs_->gpu_data(),
														inner_colinds_->gpu_data(),
														inner_w_synaptics_.type_,
														inner_w_synaptics_.data_->gpu_data(),
														inner_connkinds_->gpu_data(),
														inner_conninds_->count(),
														inner_conninds_->gpu_data(),
														f_recorded_inner_actives_->gpu_data(),
														j_ex_presynaptic_deltas_->mutable_gpu_data(),
														j_in_presynaptic_deltas_->mutable_gpu_data(),
														stream);
			}
			else
			{
				update_presynaptic_voltage_inner_gpu<T, T2>(inner_rowptrs_->gpu_data(),
														inner_colinds_->gpu_data(),
														inner_w_synaptics_.type_,
														inner_w_synaptics_.data_->gpu_data(),
														inner_connkinds_->gpu_data(),
														inner_conninds_->count(),
														inner_conninds_->gpu_data(),
														f_inner_actives_->gpu_data(),
														j_ex_presynaptic_deltas_->mutable_gpu_data(),
														j_in_presynaptic_deltas_->mutable_gpu_data(),
														stream);
			}
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::update_J_presynaptic_outer_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(BLOCK_TYPE_NORMAL == block_type_)
	{
		if(!f_receiving_bids_.empty())
		{
			assert(nullptr != f_receiving_active_rowptrs_ && f_receiving_active_rowptrs_->count() > 0);
			unsigned int n = f_receiving_active_rowptrs_->cpu_data()[f_receiving_active_rowptrs_->count() - 1];
			if(n > 0)
			{
				
				update_presynaptic_voltage_outer_gpu<T, T2>(outer_rowptrs_->gpu_data(),
													outer_colinds_->gpu_data(),
													outer_w_synaptics_.type_,
													outer_w_synaptics_.data_->gpu_data(),
													outer_connkinds_->gpu_data(),
													f_shared_active_colinds_->gpu_data(),
													n,
													j_ex_presynaptic_deltas_->mutable_gpu_data(),
													j_in_presynaptic_deltas_->mutable_gpu_data(),
													stream);
			}
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Vmeans_and_Imeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != imeans_ && nullptr != vmeans_)
	{
		assert(imeans_->count() == (sub_bcounts_->count() - 1) &&
			vmeans_->count() == (sub_bcounts_->count() - 1));

		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_bcounts_->gpu_data(),
									NULL,
									sub_bcounts_->count() - 1,
									NULL,
									v_membranes_->gpu_data(),
									i_synaptics_->gpu_data(),
									vmeans_->mutable_gpu_data(),
									imeans_->mutable_gpu_data(),
									stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_bcounts_->gpu_data(),
									f_exclusive_counts_->gpu_data(),
									sub_bcounts_->count() - 1,
									f_exclusive_flags_->gpu_data(),
									v_membranes_->gpu_data(),
									i_synaptics_->gpu_data(),
									vmeans_->mutable_gpu_data(),
									imeans_->mutable_gpu_data(),
									stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Freqs_gpu(bool is_char, hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr == freqs_)
		return;

#if 0
	if(is_char)
	{
		assert(freqs_->count() == (sub_bcounts_->count() - 1));
	}
	else
	{
		assert(freqs_->count() == ((sub_bcounts_->count() - 1) * sizeof(unsigned int)));
	}
#endif
	if(nullptr == f_exclusive_flags_)
	{
		assert(nullptr == f_exclusive_counts_);

		if(is_char)
		{
			if(nullptr != i_ou_background_stimuli_ && t_steps_ == 1)
			{
				stat_freqs_gpu<unsigned char>(sub_bcounts_->gpu_data(),
											NULL,
											sub_bcounts_->count() - 1,
											NULL,
											f_inner_actives_->gpu_data(),
											freqs_->mutable_gpu_data(),
											stream);
			}
			else
			{
				stat_freqs_gpu<unsigned char>(sub_bcounts_->gpu_data(),
											NULL,
											sub_bcounts_->count() - 1,
											NULL,
											f_recorded_inner_actives_->gpu_data(),
											freqs_->mutable_gpu_data(),
											stream);
			}
		}
		else
		{
			if(nullptr != i_ou_background_stimuli_ && t_steps_ == 1)
			{
				stat_freqs_gpu<unsigned int>(sub_bcounts_->gpu_data(),
											NULL,
											sub_bcounts_->count() - 1,
											NULL,
											f_inner_actives_->gpu_data(),
											reinterpret_cast<unsigned int*>(freqs_->mutable_gpu_data()),
											stream);
			}
			else
			{
				
				stat_freqs_gpu<unsigned int>(sub_bcounts_->gpu_data(),
											NULL,
											sub_bcounts_->count() - 1,
											NULL,
											f_recorded_inner_actives_->gpu_data(),
											reinterpret_cast<unsigned int*>(freqs_->mutable_gpu_data()),
											stream);
			}
		}
	}
	else
	{
		assert(nullptr != f_exclusive_counts_);
		if(is_char)
		{
			if(nullptr != i_ou_background_stimuli_ && t_steps_ == 1)
			{
				stat_freqs_gpu<unsigned char>(sub_bcounts_->gpu_data(),
											f_exclusive_counts_->gpu_data(),
											sub_bcounts_->count() - 1,
											f_exclusive_flags_->gpu_data(),
											f_inner_actives_->gpu_data(),
											freqs_->mutable_gpu_data(),
											stream);
			}
			else
			{
				stat_freqs_gpu<unsigned char>(sub_bcounts_->gpu_data(),
										f_exclusive_counts_->gpu_data(),
										sub_bcounts_->count() - 1,
										f_exclusive_flags_->gpu_data(),
										f_recorded_inner_actives_->gpu_data(),
										freqs_->mutable_gpu_data(),
										stream);
			}
		}
		else
		{
			if(nullptr != i_ou_background_stimuli_ && t_steps_ == 1)
			{
				stat_freqs_gpu<unsigned int>(sub_bcounts_->gpu_data(),
											f_exclusive_counts_->gpu_data(),
											sub_bcounts_->count() - 1,
											f_exclusive_flags_->gpu_data(),
											f_inner_actives_->gpu_data(),
											reinterpret_cast<unsigned int*>(freqs_->mutable_gpu_data()),
											stream);
			}
			else
			{
				stat_freqs_gpu<unsigned int>(sub_bcounts_->gpu_data(),
											f_exclusive_counts_->gpu_data(),
											sub_bcounts_->count() - 1,
											f_exclusive_flags_->gpu_data(),
											f_recorded_inner_actives_->gpu_data(),
											reinterpret_cast<unsigned int*>(freqs_->mutable_gpu_data()),
											stream);
			}
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Vmeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != vmeans_)
	{
		assert(vmeans_->count() == (sub_bcounts_->count() - 1));
		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_bcounts_->gpu_data(),
										NULL,
										sub_bcounts_->count() - 1,
										NULL,
										v_membranes_->gpu_data(),
										NULL,
										vmeans_->mutable_gpu_data(),
										NULL,
										stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_bcounts_->gpu_data(),
										f_exclusive_counts_->gpu_data(),
										sub_bcounts_->count() - 1,
										f_exclusive_flags_->gpu_data(),
										v_membranes_->gpu_data(),
										NULL,
										vmeans_->mutable_gpu_data(),
										NULL,
										stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Imeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != imeans_)
	{
		assert(imeans_->count() == (sub_bcounts_->count() - 1));
		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_bcounts_->gpu_data(),
										NULL,
										sub_bcounts_->count() - 1,
										NULL,
										NULL,
										i_synaptics_->gpu_data(),
										NULL,
										imeans_->mutable_gpu_data(),
										stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_vmeans_and_imeans_gpu<T>(sub_bcounts_->gpu_data(),
										f_exclusive_counts_->gpu_data(),
										sub_bcounts_->count() - 1,
										f_exclusive_flags_->gpu_data(),
										NULL,
										i_synaptics_->gpu_data(),
										NULL,
										imeans_->mutable_gpu_data(),
										stream);
		}
	}
}

template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_receptor_Imeans_gpu(hipStream_t stream)
{
	//HIP_CHECK(cudaSetDevice(gpu_id_));
	if(nullptr != ampa_imeans_)
	{
		assert(imeans_->count() == (sub_bcounts_->count() - 1));
		if(nullptr == f_exclusive_flags_)
		{
			assert(nullptr == f_exclusive_counts_);
			stat_receptor_imeans_gpu<T, T2>(sub_bcounts_->gpu_data(),
										NULL,
										sub_bids_->count(),
										NULL,
										j_ex_presynaptic_deltas_->gpu_data(),
										j_in_presynaptic_deltas_->gpu_data(),
										ampa_imeans_->mutable_gpu_data(),
										nmda_imeans_->mutable_gpu_data(),
										gabaa_imeans_->mutable_gpu_data(),
										gabab_imeans_->mutable_gpu_data(),
										stream);
		}
		else
		{
			assert(nullptr != f_exclusive_counts_);
			stat_receptor_imeans_gpu<T, T2>(sub_bcounts_->gpu_data(),
										f_exclusive_counts_->gpu_data(),
										sub_bids_->count(),
										f_exclusive_flags_->gpu_data(),
										j_ex_presynaptic_deltas_->gpu_data(),
										j_in_presynaptic_deltas_->gpu_data(),
										ampa_imeans_->mutable_gpu_data(),
										nmda_imeans_->mutable_gpu_data(),
										gabaa_imeans_->mutable_gpu_data(),
										gabab_imeans_->mutable_gpu_data(),
										stream);
		}
	}
}


template<typename T, typename T2> 
void BrainBlock<T, T2>::stat_Samples_gpu(const unsigned int* samples,
											const unsigned int n,
											char* spikes,
											float* vmembs,
											float* isynaptics,
											const bool use_recorded,
											float* ious,
											hipStream_t stream)
{
	if(!use_recorded)
	{
		stat_samples_gpu<T>(samples,
						  n,
						  f_inner_actives_->gpu_data(),
						  v_membranes_->gpu_data(),
						  i_synaptics_->gpu_data(),
						  i_ou_background_stimuli_->gpu_data(),
						  spikes,
						  vmembs,
						  isynaptics,
						  ious,
						  stream);
	}
	else
	{
		stat_samples_gpu<T>(samples,
						  n,
						  f_recorded_inner_actives_->gpu_data(),
						  v_membranes_->gpu_data(),
						  i_synaptics_->gpu_data(),
						  NULL,
						  spikes,
						  vmembs,
						  isynaptics,
						  ious,
						  stream);
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_Tao_constant_gpu()
{
	const T delta_t = delta_t_ / t_steps_;
	update_tao_constant_gpu<T, T2>(delta_t, tao_ex_constants_->cpu_data(), tao_ex_constants_->count(), tao_ex_constants_->mutable_gpu_data());
	update_tao_constant_gpu<T, T2>(delta_t, tao_in_constants_->cpu_data(), tao_in_constants_->count(), tao_in_constants_->mutable_gpu_data());
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_Props_gpu(const unsigned int* d_neruon_indice,
											const unsigned int* h_neruon_indice,
											const unsigned int* d_prop_indice,
											const unsigned int* h_prop_indice,
											const float* d_prop_vals,
											const float* h_prop_vals,
											const unsigned int n,
											hipStream_t stream)
{
	Properties<T, T2> prop;
	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();

	update_props_gpu<T, T2>(d_neruon_indice, d_prop_indice, d_prop_vals, n, prop, stream);
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int pid = h_prop_indice[i];
		unsigned int nid = h_neruon_indice[i];
		if(pid == PropType::TAO_AMPA)
		{
			tao_ex_constants_->mutable_cpu_data()[nid].x = h_prop_vals[i];
		}
		else if(pid == PropType::TAO_NMDA)
		{
			tao_ex_constants_->mutable_cpu_data()[nid].y = h_prop_vals[i];
		}
		else if(pid == PropType::TAO_GABAa)
		{
			tao_in_constants_->mutable_cpu_data()[nid].x = h_prop_vals[i];
		}
		else if(pid == PropType::TAO_GABAb)
		{
			tao_in_constants_->mutable_cpu_data()[nid].y = h_prop_vals[i];
		}
	}
	
}

template<typename T, typename T2>
void BrainBlock<T, T2>::update_Prop_Cols_gpu(const unsigned int* d_prop_indice,
											const unsigned int* h_prop_indice,
											const unsigned int* d_brain_indice,
											const unsigned int* h_brain_indice,
											const float* d_hp_vals,
											const unsigned int n,
											hipStream_t stream)
{
	Properties<T, T2> prop;
	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();
	
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int pid = h_prop_indice[i];
		unsigned int bid = h_brain_indice[i];
		if(pid == PropType::TAO_AMPA || pid == PropType::TAO_NMDA)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_ex_constants_->mutable_gpu_data() + beg, tao_ex_constants_->cpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyHostToDevice, stream));
		}
		else if(pid == PropType::TAO_GABAa || pid == PropType::TAO_GABAb)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_in_constants_->mutable_gpu_data() + beg, tao_in_constants_->cpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyHostToDevice, stream));
		}
	}
	
	update_prop_cols_gpu<T, T2>(sub_bids_->gpu_data(),
							sub_bcounts_->gpu_data(),
							sub_bids_->count(),
							d_prop_indice,
							d_brain_indice,
							d_hp_vals,
							n,
							prop,
							stream);

	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int pid = h_prop_indice[i];
		unsigned int bid = h_brain_indice[i];
		if(pid == PropType::TAO_AMPA || pid == PropType::TAO_NMDA)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_ex_constants_->mutable_cpu_data() + beg, tao_ex_constants_->gpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyDeviceToHost, stream));
		}
		else if(pid == PropType::TAO_GABAa || pid == PropType::TAO_GABAb)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_in_constants_->mutable_cpu_data() + beg, tao_in_constants_->gpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyDeviceToHost, stream));
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::assign_Prop_Cols_gpu(const unsigned int* d_prop_indice,
											const unsigned int* h_prop_indice,
											const unsigned int* d_brain_indice,
											const unsigned int* h_brain_indice,
											const float* d_hp_vals,
											const unsigned int n,
											hipStream_t stream)
{
	Properties<T, T2> prop;
	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();
	assign_prop_cols_gpu<T, T2>(sub_bids_->gpu_data(),
							sub_bcounts_->gpu_data(),
							sub_bids_->count(),
							d_prop_indice,
							d_brain_indice,
							d_hp_vals,
							n,
							prop,
							stream);

	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int pid = h_prop_indice[i];
		unsigned int bid = h_brain_indice[i];
		if(pid == PropType::TAO_AMPA || pid == PropType::TAO_NMDA)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_ex_constants_->mutable_cpu_data() + beg, tao_ex_constants_->gpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyDeviceToHost, stream));
		}
		else if(pid == PropType::TAO_GABAa || pid == PropType::TAO_GABAb)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_in_constants_->mutable_cpu_data() + beg, tao_in_constants_->gpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyDeviceToHost, stream));
		}
	}
}


template<typename T, typename T2>
void BrainBlock<T, T2>::update_Gamma_Prop_Cols_gpu(const unsigned int* d_prop_indice,
															const unsigned int* h_prop_indice,
															const unsigned int* d_brain_indice,
															const unsigned int* h_brain_indice,
															const float* d_alphas,
															const float* d_betas,
															const unsigned int n,
															hipStream_t stream)
{
	assert(n > 0);
	Properties<T, T2> prop;

	prop.n = total_neurons_;
	prop.i_ext_stimuli = i_ext_stimuli_->mutable_gpu_data();
	prop.c_membrane_reciprocals = c_membrane_reciprocals_->mutable_gpu_data();
	prop.t_refs = t_refs_->mutable_gpu_data();
	prop.g_leakages = g_leakages_->mutable_gpu_data();
	prop.v_leakages = v_leakages_->mutable_gpu_data();
	prop.v_thresholds = v_thresholds_->mutable_gpu_data();
	prop.v_resets = v_resets_->mutable_gpu_data();
	prop.g_ex_conducts = g_ex_conducts_->mutable_gpu_data();
	prop.g_in_conducts = g_in_conducts_->mutable_gpu_data();
	prop.v_ex_membranes = v_ex_membranes_->mutable_gpu_data();
	prop.v_in_membranes = v_in_membranes_->mutable_gpu_data();
	prop.tao_ex_constants = tao_ex_constants_->mutable_gpu_data();
	prop.tao_in_constants = tao_in_constants_->mutable_gpu_data();
	prop.noise_rates = noise_rates_->mutable_gpu_data();

	gamma_gpu<T, T2>(gen_states_->mutable_gpu_data(),
				sub_bids_->gpu_data(),
				sub_bcounts_->gpu_data(),
				sub_bids_->count(),
				d_prop_indice,
				d_brain_indice,
				d_alphas,
				d_betas,
				n,
				prop,
			 	stream);

	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int pid = h_prop_indice[i];
		unsigned int bid = h_brain_indice[i];
		if(pid == PropType::TAO_AMPA || pid == PropType::TAO_NMDA)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_ex_constants_->mutable_cpu_data() + beg, tao_ex_constants_->gpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyDeviceToHost, stream));
		}
		else if(pid == PropType::TAO_GABAa || pid == PropType::TAO_GABAb)
		{
			const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), h_brain_indice[i]);
			const unsigned int bid = iter - sub_bids_->cpu_data();
			const unsigned int beg = sub_bcounts_->cpu_data()[bid];
			const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
			HIP_CHECK(hipMemcpyAsync(tao_in_constants_->mutable_cpu_data() + beg, tao_in_constants_->gpu_data() + beg, (end - beg) * sizeof(T2), hipMemcpyDeviceToHost, stream));
		}
	}
	
}

template<typename T, typename T2>
void BrainBlock<T, T2>::reset_F_active_gpu(hipStream_t stream)
{
	reset_spike_gpu<T>(total_neurons_,
					v_membranes_->gpu_data(),
					v_thresholds_->gpu_data(),
					f_inner_actives_->mutable_gpu_data(),
					stream);
}


template<typename T, typename T2>
void BrainBlock<T, T2>::fetch_sample_neuron_offsets(const std::vector<unsigned short>& sample_bids,
														const std::vector<std::vector<unsigned int>>& sample_nids,
														std::vector<unsigned int>& begin_offsets,
														std::vector<unsigned int>& end_offsets,
														std::vector<unsigned short>& matched_sample_bids,
														std::vector<unsigned int>& matched_sample_nids,
														std::tuple<unsigned int, unsigned int>& inner_offset)
{
	
	bool inner_once = false;
	for(unsigned int i = 0; i < sample_bids.size(); i++)
	{
		if(sample_bids[i] != bid_)
		{
			if(!f_receiving_bids_.empty())
			{
				unsigned short* bid_it = thrust::find(f_receiving_bids_.data(), f_receiving_bids_.data() + f_receiving_bids_.size(), sample_bids[i]);
				if(bid_it != (f_receiving_bids_.data() + f_receiving_bids_.size()))
				{
					unsigned int bidx = bid_it - f_receiving_bids_.data();
					thrust::host_vector<unsigned int> h_rowptrs(f_receiving_rowptrs_->cpu_data()[bidx + 1] - f_receiving_rowptrs_->cpu_data()[bidx] + 1);
					thrust::copy(thrust::device_pointer_cast(outer_rowptrs_->gpu_data()) + f_receiving_rowptrs_->cpu_data()[bidx],
							thrust::device_pointer_cast(outer_rowptrs_->gpu_data()) + f_receiving_rowptrs_->cpu_data()[bidx + 1] + 1,
							h_rowptrs.begin());
					for(unsigned int j = 0; j < sample_nids[i].size(); j++)
					{
						const unsigned int* nid_it = thrust::find(f_receiving_colinds_->cpu_data() + f_receiving_rowptrs_->cpu_data()[bidx], f_receiving_colinds_->cpu_data() + f_receiving_rowptrs_->cpu_data()[bidx + 1], (sample_nids[i])[j]); 
						if(nid_it != f_receiving_colinds_->cpu_data() + f_receiving_rowptrs_->cpu_data()[bidx + 1])
						{
							unsigned int nidx = nid_it - (f_receiving_colinds_->cpu_data() + f_receiving_rowptrs_->cpu_data()[bidx]);
							begin_offsets.push_back(h_rowptrs[nidx]);
							end_offsets.push_back(h_rowptrs[nidx + 1]);
							matched_sample_bids.push_back(sample_bids[i]);
							matched_sample_nids.push_back((sample_nids[i])[j]);
						}
					}
				}
			}
		}
		else
		{
			if(nullptr != inner_conninds_)
			{
				thrust::host_vector<unsigned int> nids(inner_conninds_->count());
				thrust::copy(thrust::device_pointer_cast(inner_conninds_->gpu_data()),
							thrust::device_pointer_cast(inner_conninds_->gpu_data()) + inner_conninds_->count(),
							nids.begin());
				thrust::host_vector<unsigned int> h_rowptrs(inner_rowptrs_->count());
				thrust::copy(thrust::device_pointer_cast(inner_rowptrs_->gpu_data()),
							thrust::device_pointer_cast(inner_rowptrs_->gpu_data()) + inner_rowptrs_->count(),
							h_rowptrs.begin());

				for(unsigned int j = 0; j < sample_nids[i].size(); j++)
				{
					auto nid_it = thrust::find(nids.begin(), nids.end(), (sample_nids[i])[j]); 
					if(nid_it != nids.end())
					{
						if(!inner_once)
						{
							std::get<0>(inner_offset) = matched_sample_bids.size();
							inner_once = true;
						}
						unsigned int nidx = nid_it - nids.begin();
						begin_offsets.push_back(h_rowptrs[nidx]);
						end_offsets.push_back(h_rowptrs[nidx + 1]);
						matched_sample_bids.push_back(sample_bids[i]);
						matched_sample_nids.push_back((sample_nids[i])[j]);
					}
				}

				if(inner_once)
				{
					std::get<1>(inner_offset) = matched_sample_bids.size();
				}
			}

			if(!inner_once)
			{
				std::get<0>(inner_offset) = 0;
				std::get<1>(inner_offset) = 0;
			}
		}
	}
}

template<typename T, typename T2>
void BrainBlock<T, T2>::fetch_sample_neuron_weights(const std::vector<unsigned short>& sample_bids,
														const std::vector<unsigned int>& sample_nids,
														const std::vector<unsigned int>& begin_offsets,
														const std::vector<unsigned int>& end_offsets,
														const std::tuple<unsigned int, unsigned int>& inner_offset,
														const unsigned int sample_id,
														std::vector<unsigned short>& matched_sample_bids,
														std::vector<unsigned int>& matched_sample_nids,
														Weights& matched_sample_weights,
														std::vector<unsigned char>& matched_sample_channels)
{
	thrust::host_vector<char> h_weights;
	thrust::host_vector<unsigned char> h_channels;
	thrust::host_vector<unsigned int> h_indice;
	DataType dtype;

	auto fetch_weights_func
	    = [&begin_offsets, &end_offsets, sample_id, &h_indice, &h_channels, &dtype, &h_weights](const unsigned int beg, const unsigned int end, const unsigned int* d_colinds, const unsigned char* d_channels, const Weights& weights)
	{
		unsigned int count = end - beg;
		if(0 == count || nullptr == d_colinds)
			return;
		
		thrust::device_vector<unsigned int> d_begin_offsets(count);
		thrust::device_vector<unsigned int> d_end_offsets(count);
		thrust::device_vector<unsigned int> d_results(count);
		thrust::device_vector<unsigned int> d_maps(count);
		thrust::copy(begin_offsets.data() + beg, begin_offsets.data() + end, d_begin_offsets.begin());
		thrust::copy(end_offsets.data() + beg, end_offsets.data() + end, d_end_offsets.begin());
		dtype = weights.type_;

		for(unsigned int i = 0; i < count; i++)
		{
			if(end_offsets[beg + i] <= begin_offsets[beg + i])
				continue;
			unsigned int elems = end_offsets[beg + i] - begin_offsets[beg + i];
			thrust::host_vector<unsigned int> h_colinds(elems);
			thrust::copy(thrust::device_pointer_cast(d_colinds) + begin_offsets[beg + i], 
						thrust::device_pointer_cast(d_colinds) + end_offsets[beg + i],
						h_colinds.begin());
		}

		fetch_sample_offset_gpu(d_colinds,
						count, 
						thrust::raw_pointer_cast(d_begin_offsets.data()),
						thrust::raw_pointer_cast(d_end_offsets.data()),
						sample_id,
						thrust::raw_pointer_cast(d_results.data()));
		HIP_CHECK(hipDeviceSynchronize());
		 
		unsigned int invalid_count = thrust::count(d_results.begin(), d_results.end(), static_cast<unsigned int>(-1));
		//LOG_INFO << "fetch_weights_func invalid count: " << invalid_count << std::endl;
		count = count - invalid_count;
		if(0 == count)
			return;

		thrust::sequence(d_maps.begin(), d_maps.end(), beg);
		thrust::sort_by_key(d_results.begin(), d_results.end(), d_maps.begin());
		unsigned int orig_size = h_indice.size();
		h_indice.resize(orig_size + count);
		thrust::copy(d_maps.begin(), d_maps.begin() + count, h_indice.begin() + orig_size);

		if(weights.type_ == DOUBLE)
		{
			thrust::device_vector<double2> d_weights(count);
			thrust::gather(d_results.begin(), d_results.begin() + count, thrust::device_pointer_cast(reinterpret_cast<const double2*>(weights.data_->gpu_data())), d_weights.begin());
			h_weights.resize((orig_size + count)* sizeof(double2));
			thrust::copy(d_weights.begin(), d_weights.end(), reinterpret_cast<double2*>(h_weights.data()) + orig_size);
		}
		else if(weights.type_ == FLOAT)
		{
			thrust::device_vector<float2> d_weights(count);
			thrust::gather(d_results.begin(), d_results.begin() + count, thrust::device_pointer_cast(reinterpret_cast<const float2*>(weights.data_->gpu_data())), d_weights.begin());
			h_weights.resize((orig_size + count)* sizeof(float2));
			thrust::copy(d_weights.begin(), d_weights.end(), reinterpret_cast<float2*>(h_weights.data()) + orig_size);
		}
		else if(weights.type_ == FLOAT16)
		{
			thrust::device_vector<half2> d_weights(count);
			thrust::gather(d_results.begin(), d_results.begin() + count, thrust::device_pointer_cast(reinterpret_cast<const half2*>(weights.data_->gpu_data())), d_weights.begin());
			h_weights.resize((orig_size + count)* sizeof(half2));
			thrust::copy(d_weights.begin(), d_weights.end(), reinterpret_cast<half2*>(h_weights.data()) + orig_size);
		}
		else
		{
			assert(weights.type_ == INT8);
			thrust::device_vector<uchar2> d_weights(count);
			thrust::gather(d_results.begin(), d_results.begin() + count, thrust::device_pointer_cast(reinterpret_cast<const uchar2*>(weights.data_->gpu_data())), d_weights.begin());
			h_weights.resize((orig_size + count)* sizeof(uchar2));
			thrust::copy(d_weights.begin(), d_weights.end(), reinterpret_cast<uchar2*>(h_weights.data()) + orig_size);
		}

		thrust::device_vector<unsigned char> d_matched_channels(count);
		thrust::gather(d_results.begin(), d_results.begin() + count, thrust::device_pointer_cast(d_channels), d_matched_channels.begin());
		h_channels.resize(orig_size + count);
		thrust::copy(d_matched_channels.begin(), d_matched_channels.end(), h_channels.begin() + orig_size);
	};

	const unsigned int inner_begin = std::get<0>(inner_offset);
	const unsigned int inner_end = std::get<1>(inner_offset);
	
	if(inner_begin > 0)
	{
		fetch_weights_func(0, inner_begin, outer_colinds_->gpu_data(), outer_connkinds_->gpu_data(), outer_w_synaptics_);
	}
	
	if(inner_end > inner_begin)
	{
		fetch_weights_func(inner_begin, inner_end, inner_colinds_->gpu_data(), inner_connkinds_->gpu_data(), inner_w_synaptics_);
	}

	if(inner_end < sample_bids.size())
	{
		fetch_weights_func(inner_end, sample_bids.size(), outer_colinds_->gpu_data(), outer_connkinds_->gpu_data(), outer_w_synaptics_);
	}

	
	if(!h_indice.empty())
	{
		matched_sample_bids.resize(h_indice.size());
		matched_sample_nids.resize(h_indice.size());
		thrust::gather(h_indice.begin(), h_indice.end(), sample_bids.data(), matched_sample_bids.data());
		thrust::gather(h_indice.begin(), h_indice.end(), sample_nids.data(), matched_sample_nids.data());
	}

	if(!h_weights.empty())
	{
		matched_sample_weights.type_ = dtype;
		matched_sample_weights.data_ = make_shared<DataAllocator<char>>(static_cast<int>(bid_ + 1), h_weights.size(), false);
		thrust::copy(h_weights.begin(), h_weights.end(), matched_sample_weights.data_->mutable_cpu_data());
	}

	if(!h_channels.empty())
	{
		matched_sample_channels.resize(h_indice.size());
		thrust::copy(h_channels.begin(), h_channels.end(), matched_sample_channels.data());
	}
	
}

template<typename T, typename T2>
void BrainBlock<T, T2>::fetch_props(const unsigned int* neuron_indice,
									const unsigned int* prop_indice,
									const unsigned int n,
									vector<T>& result)
{
	#define FETCH_T_TYPE(CASE, DATA, PROP) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		result[i] = DATA[nid]; \
	break;

	#define FETCH_T2_TYPE(CASE, DATA, PROP, FILED) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		result[i] = DATA[nid].FILED; \
	break;

	result.resize(n);
	vector<T> i_ext_stimuli;
	vector<T> c_membrane_reciprocals;
	vector<T> t_refs;
	vector<T> g_leakages;
	vector<T> v_leakages;
	vector<T> v_thresholds;
	vector<T> v_resets;
	vector<T2> g_ex_conducts;
	vector<T2> g_in_conducts;
	vector<T2> v_ex_membranes;
	vector<T2> v_in_membranes;
	vector<T2> tao_ex_constants;
	vector<T2> tao_in_constants;
	vector<T> noise_rates;
	
	for(unsigned int i = 0; i < n; i++)
	{
		unsigned int nid = neuron_indice[i];
		unsigned int pid = prop_indice[i];
		switch(pid)
		{
			FETCH_T_TYPE(EXT_STIMULI_I, i_ext_stimuli, i_ext_stimuli_)
			FETCH_T_TYPE(MEMBRANE_C, c_membrane_reciprocals, c_membrane_reciprocals_)
			FETCH_T_TYPE(REF_T, t_refs, t_refs_)
			FETCH_T_TYPE(LEAKAGE_G, g_leakages, g_leakages_)
			FETCH_T_TYPE(LEAKAGE_V, v_leakages, v_leakages_)
			FETCH_T_TYPE(THRESHOLD_V, v_thresholds, v_thresholds_)
			FETCH_T_TYPE(RESET_V, v_resets, v_resets_)
			FETCH_T2_TYPE(CONDUCT_G_AMPA, g_ex_conducts, g_ex_conducts_, x)
			FETCH_T2_TYPE(CONDUCT_G_NMDA, g_ex_conducts, g_ex_conducts_, y)
			FETCH_T2_TYPE(CONDUCT_G_GABAa, g_in_conducts, g_in_conducts_, x)
			FETCH_T2_TYPE(CONDUCT_G_GABAb, g_in_conducts, g_in_conducts_, y)
			
			FETCH_T2_TYPE(MEMBRANE_V_AMPA, v_ex_membranes, v_ex_membranes_, x)
			FETCH_T2_TYPE(MEMBRANE_V_NMDA, v_ex_membranes, v_ex_membranes_, y)
			FETCH_T2_TYPE(MEMBRANE_V_GABAa, v_in_membranes, v_in_membranes_, x)
			FETCH_T2_TYPE(MEMBRANE_V_GABAb, v_in_membranes, v_in_membranes_, y)

			FETCH_T2_TYPE(TAO_AMPA, tao_ex_constants, tao_ex_constants_, x)
			FETCH_T2_TYPE(TAO_NMDA, tao_ex_constants, tao_ex_constants_, y)
			FETCH_T2_TYPE(TAO_GABAa, tao_in_constants, tao_in_constants_, x)
			FETCH_T2_TYPE(TAO_GABAb, tao_in_constants, tao_in_constants_, y)
			FETCH_T_TYPE(NOISE_RATE, noise_rates, noise_rates_)
			default:
				assert(0);
			break;
		}
	}

	#undef FETCH_T_TYPE
	#undef FETCH_T2_TYPE
}

template<typename T, typename T2>
void BrainBlock<T, T2>::fetch_prop_cols(const unsigned int* prop_indice,
										const unsigned int* brain_indice,
										const unsigned int n,
										vector<vector<T>>& results)
{
	#define FETCH_T_TYPE(CASE, DATA, PROP) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		results[i][total] = DATA[j]; \
	break;

	#define FETCH_T2_TYPE(CASE, DATA, PROP, FILED) \
	case CASE: \
		if(DATA.empty()) \
		{ \
			DATA.resize(total_neurons_); \
			HIP_CHECK(hipMemcpy(DATA.data(), PROP->gpu_data(), PROP->size(), hipMemcpyDeviceToHost)); \
		} \
		results[i][total] = DATA[j].FILED; \
	break;
	
	vector<T> i_ext_stimuli;
	vector<T> c_membrane_reciprocals;
	vector<T> t_refs;
	vector<T> g_leakages;
	vector<T> v_leakages;
	vector<T> v_thresholds;
	vector<T> v_resets;
	vector<T2> g_ex_conducts;
	vector<T2> g_in_conducts;
	vector<T2> v_ex_membranes;
	vector<T2> v_in_membranes;
	vector<T2> tao_ex_constants;
	vector<T2> tao_in_constants;
	vector<T> noise_rates;
	
	results.resize(n);
	for(unsigned int i = 0; i < n; i++)
	{
		const unsigned int pid = prop_indice[i];
		const unsigned int* iter = thrust::find(sub_bids_->cpu_data(), sub_bids_->cpu_data() + sub_bids_->count(), brain_indice[i]);
		unsigned int bid = iter - sub_bids_->cpu_data();
		if(bid == sub_bids_->count())
			continue;
		
		const unsigned int beg = sub_bcounts_->cpu_data()[bid];
		const unsigned int end = sub_bcounts_->cpu_data()[bid + 1];
		assert(end > beg);
		results[i].resize((end - beg));
		
		unsigned int total = 0;
		for(unsigned int j = beg; j < end; j++)
		{
			switch(pid)
			{
				FETCH_T_TYPE(EXT_STIMULI_I, i_ext_stimuli, i_ext_stimuli_)
				FETCH_T_TYPE(MEMBRANE_C, c_membrane_reciprocals, c_membrane_reciprocals_)
				FETCH_T_TYPE(REF_T, t_refs, t_refs_)
				FETCH_T_TYPE(LEAKAGE_G, g_leakages, g_leakages_)
				FETCH_T_TYPE(LEAKAGE_V, v_leakages, v_leakages_)
				FETCH_T_TYPE(THRESHOLD_V, v_thresholds, v_thresholds_)
				FETCH_T_TYPE(RESET_V, v_resets, v_resets_)
				FETCH_T2_TYPE(CONDUCT_G_AMPA, g_ex_conducts, g_ex_conducts_, x)
				FETCH_T2_TYPE(CONDUCT_G_NMDA, g_ex_conducts, g_ex_conducts_, y)
				FETCH_T2_TYPE(CONDUCT_G_GABAa, g_in_conducts, g_in_conducts_, x)
				FETCH_T2_TYPE(CONDUCT_G_GABAb, g_in_conducts, g_in_conducts_, y)
				
				FETCH_T2_TYPE(MEMBRANE_V_AMPA, v_ex_membranes, v_ex_membranes_, x)
				FETCH_T2_TYPE(MEMBRANE_V_NMDA, v_ex_membranes, v_ex_membranes_, y)
				FETCH_T2_TYPE(MEMBRANE_V_GABAa, v_in_membranes, v_in_membranes_, x)
				FETCH_T2_TYPE(MEMBRANE_V_GABAb, v_in_membranes, v_in_membranes_, y)

				FETCH_T2_TYPE(TAO_AMPA, tao_ex_constants, tao_ex_constants_, x)
				FETCH_T2_TYPE(TAO_NMDA, tao_ex_constants, tao_ex_constants_, y)
				FETCH_T2_TYPE(TAO_GABAa, tao_in_constants, tao_in_constants_, x)
				FETCH_T2_TYPE(TAO_GABAb, tao_in_constants, tao_in_constants_, y)
				FETCH_T_TYPE(NOISE_RATE, noise_rates, noise_rates_)
				
				default:
					assert(0);
				break;
			}  
			total++;
		}
	}

	#undef FETCH_T_TYPE
	#undef FETCH_T2_TYPE
}

template class BrainBlock<float, float2>;
template class BrainBlock<double, double2>;
}//namespace dtb


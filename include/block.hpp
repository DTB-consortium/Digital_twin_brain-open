#pragma once

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <tuple>
#include <cassert>
#include <hiprand_kernel.h>
#include "common.hpp"
#include "data_allocator.hpp"
#include "weights.hpp"
#include "stage.hpp"

using namespace std;

//digital twin brain
namespace dtb {

template<typename T, typename T2>
class BrainBlock
{
public:

	enum BLOCK_TYPE
	{
		BLOCK_TYPE_INPUT = 0,
		BLOCK_TYPE_NORMAL
	};

	BrainBlock(const unsigned short block_id,
			const int gpu_id,
			const T delta_t = 0.1f,
			const unsigned long long seed = gen_seed());
	
	~BrainBlock(){};

	void init_config_params_gpu(const std::string& filename);
	void init_connection_table_gpu(const std::string& filename);
	void init_all_stages_gpu();
	void init_random_state();

	unsigned int update_F_input_spike_gpu(const unsigned int timestamp, 
										const unsigned int offset = 0,
										hipStream_t stream = NULL);

	void set_I_ou_current_param(const vector<unsigned int>& brain_indice,
									const vector<float>& means,
									const vector<float>& deviations,
									const vector<float>& correlation_times,
									hipStream_t stream = NULL);
												
	void update_I_ou_background_gpu(const T mean = 0.f,
											const T stdv = 1.f,
											bool saving_sample = false,
											hipStream_t stream = NULL);

	void set_I_ttype_ca_param(const vector<unsigned int>& brain_indice,
								const vector<float>& h_init_vals,
								const vector<float>& g_ts,
								const vector<float>& tao_h_minuss,
								const vector<float>& tao_h_pluss,
								const vector<float>& v_hs,
								const vector<float>& v_ts,
								hipStream_t stream = NULL);
	void update_I_ttype_ca_gpu(hipStream_t stream = NULL);
	void update_H_ttype_ca_gpu(hipStream_t stream = NULL);

	void set_I_dopamine_param(const vector<unsigned int>& brain_indeice,
								const vector<float>& v_das,
								const vector<float>& g_das);

	void update_I_dopamine_gpu(hipStream_t stream = NULL);

	void set_I_adaptation_param(const vector<unsigned int>& brain_indeice,
								const vector<float>& ca_init_vals,
								const vector<float>& ca_decays,
								const vector<float>& alpha_constants,
								const vector<float>& v_ks,
								const vector<float>& g_ahps,
								hipStream_t stream = NULL);

	void update_I_adaptation_gpu(hipStream_t stream = NULL);

	void update_ca_concentration_gpu(bool use_recorded = true, hipStream_t stream = NULL);
	
	void update_I_synaptic_gpu(bool have_receptor_imeans = false, hipStream_t stream = NULL);
	void update_V_membrane_gpu(hipStream_t stream = NULL);
	void reset_F_active_gpu(hipStream_t stream = NULL);
	void update_F_active_gpu(const T a = 0.f,
								const T b = 1.f,
								bool saving_sample = false,
								hipStream_t stream = NULL);
	void update_F_accumulated_spike_gpu(hipStream_t stream = NULL);
	void update_J_presynaptic_per_step_gpu(hipStream_t stream = NULL);
	void update_J_presynaptic_inner_gpu(bool use_recorded = true, hipStream_t stream = NULL);
	void update_J_presynaptic_outer_gpu(hipStream_t stream = NULL);
	void update_J_presynaptic_gpu(hipStream_t stream = NULL);
	void reset_V_membrane_gpu(hipStream_t stream = NULL);
	void stat_Vmeans_and_Imeans_gpu(hipStream_t stream = NULL);

	void stat_Freqs_gpu(bool is_char = false, hipStream_t stream = NULL);

	void stat_Vmeans_gpu(hipStream_t stream = NULL);

	void stat_Imeans_gpu(hipStream_t stream = NULL);

	void stat_receptor_Imeans_gpu(hipStream_t stream);

	void update_Tao_constant_gpu();

	void update_Props_gpu(const unsigned int* d_neruon_indice,
							const unsigned int* h_neruon_indice,
							const unsigned int* d_prop_indice,
							const unsigned int* h_prop_indice,
							const float* d_prop_vals,
							const float* h_prop_vals,
							const unsigned int n,
							hipStream_t stream = NULL);

	void update_Prop_Cols_gpu(const unsigned int* d_prop_indice,
							const unsigned int* h_prop_indice,
							const unsigned int* d_brain_indice,
							const unsigned int* h_brain_indice,
							const float* d_hp_vals,
							const unsigned int n,
							hipStream_t stream = NULL);

	void assign_Prop_Cols_gpu(const unsigned int* d_prop_indice,
									const unsigned int* h_prop_indice,
									const unsigned int* d_brain_indice,
									const unsigned int* h_brain_indice,
									const float* d_hp_vals,
									const unsigned int n,
									hipStream_t stream = NULL);

	void update_Gamma_Prop_Cols_gpu(const unsigned int* d_prop_indice,
											const unsigned int* h_prop_indice,
											const unsigned int* d_brain_indice,
											const unsigned int* h_brain_indice,
											const float* d_alphas,
											const float* d_betas,
											const unsigned int n,
											hipStream_t stream = NULL);

	void stat_Samples_gpu(const unsigned int* samples,
							const unsigned int n,
							char* spikes,
							float* vmembs,
							float* isynaptics,
							const bool use_recorded = true,
							float* ious = NULL,
							hipStream_t stream = NULL);

	void record_F_sending_actives(const map<unsigned short, vector<unsigned int>>& rank_map);

	void count_F_sending_actives_temporary_storage_size(const unsigned int sending_count,
																const unsigned int segments,
																unsigned int* block_rowptrs,
																unsigned int* active_rowptrs,
																size_t& storage_size_bytes,
																hipStream_t stream = NULL);
	
	void update_F_sending_actives_gpu(bool use_recorded = true, hipStream_t stream = NULL);


	void update_F_recving_actives_gpu(hipStream_t stream = NULL);


	bool is_input_block() const
	{
		return BLOCK_TYPE_INPUT == block_type_;
	}
	
	void set_t_step(const unsigned int step)
	{
		t_steps_ = step;
	}
	
	const unsigned long long get_seed() const
	{
		return seed_;
	}

	const unsigned int get_input_timestamp_size() const
	{
		if(nullptr != input_timestamps_)
		{
			assert(BLOCK_TYPE_INPUT == block_type_);
			return input_timestamps_->count();
		}
		else
		{
			return 0;
		}
	}
	
	const unsigned int* get_input_timestamp_data() const
	{
		if(nullptr != input_timestamps_)
        {
            assert(BLOCK_TYPE_INPUT == block_type_);	
			return input_timestamps_->cpu_data();
		}
		else
		{
			return NULL;
		}
	}

	unsigned short get_block_id() const
	{
		return bid_;
	}
	
	unsigned int get_total_neurons() const
	{
		return total_neurons_;
	}

	const unsigned char* get_F_actives_gpu() const
	{
		return f_inner_actives_->gpu_data();
	}

	const unsigned char* get_F_actives_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(f_inner_actives_->mutable_cpu_data(), f_inner_actives_->gpu_data(), f_inner_actives_->size(), hipMemcpyDeviceToHost));
		return f_inner_actives_->cpu_data();
	}

	void reset_F_recorded_actives_gpu(hipStream_t stream = NULL)
	{
		HIP_CHECK(hipMemsetAsync(f_recorded_inner_actives_->mutable_gpu_data(), 0x00, f_recorded_inner_actives_->size(), stream));
	}

	void reset_F_recorded_actives_cpu()
	{
		memset(f_recorded_inner_actives_->mutable_cpu_data(), 0x00, f_recorded_inner_actives_->size());
	}
	
	const unsigned char* get_F_recorded_actives_gpu() const
	{
		return f_recorded_inner_actives_->gpu_data();
	}

	const unsigned char* get_F_recorded_actives_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(f_recorded_inner_actives_->mutable_cpu_data(), f_recorded_inner_actives_->gpu_data(), f_recorded_inner_actives_->size(), hipMemcpyDeviceToHost));
		return f_recorded_inner_actives_->cpu_data();
	}

	const T* get_I_ou_background_stimuli_gpu() const
	{
		if(nullptr == i_ou_background_stimuli_)
			return NULL;
		
		return i_ou_background_stimuli_->gpu_data();
	}

	T* get_I_ou_background_stimuli_mutable_gpu(bool is_created = false)
	{
		if(is_created && nullptr == i_ou_background_stimuli_)
		{
			i_ou_background_stimuli_ = make_shared<DataAllocator<T>>(static_cast<int>(bid_ + 1), total_neurons_ * sizeof(T));
		}

		if(nullptr == i_ou_background_stimuli_)
			return NULL;

		
		return i_ou_background_stimuli_->mutable_gpu_data();
	}

	const T* get_h_ttype_ca_stimuli_gpu() const
	{
		if(nullptr == h_ttype_ca_stimuli_)
			return NULL;
		
		return h_ttype_ca_stimuli_->gpu_data();
	}

	const T* get_ahp_ca_concentration_gpu() const
	{
		if(nullptr == ahp_ca_concentration_)
			return NULL;
		
		return ahp_ca_concentration_->gpu_data();
	}

	const T* get_V_membranes_gpu() const
	{
		return v_membranes_->gpu_data();
	}

	T* get_V_membranes_mutable_gpu()
	{
		return v_membranes_->mutable_gpu_data();
	}

	const int* get_T_actives_gpu() const
	{
		return t_actives_->gpu_data();
	}

	int* get_T_actives_mutable_gpu()
	{
		return t_actives_->mutable_gpu_data();
	}

	const T2* get_J_ex_presynaptics_gpu() const
	{
		return j_ex_presynaptics_->gpu_data();
	}

	T2* get_J_ex_presynaptics_mutable_gpu()
	{
		return j_ex_presynaptics_->mutable_gpu_data();
	}
	
	const T2* get_J_in_presynaptics_gpu() const
	{
		return j_in_presynaptics_->gpu_data();
	}

	T2* get_J_in_presynaptics_mutable_gpu() const
	{
		return j_in_presynaptics_->mutable_gpu_data();
	}
	
	const T* get_I_synaptics_gpu() const
	{
		return i_synaptics_->gpu_data();
	}

	T* get_I_synaptics_mutable_gpu()
	{
		return i_synaptics_->mutable_gpu_data();
	}

	const T* get_I_ext_stimuli_gpu() const
	{
		return i_ext_stimuli_->gpu_data();
	}

	T* get_I_ext_stimuli_mutable_gpu()
	{
		return i_ext_stimuli_->mutable_gpu_data();
	}

	const T* get_noise_rate_gpu() const
	{
		return noise_rates_->gpu_data();
	}

	unsigned int get_total_subblocks() const
	{
		assert(nullptr != sub_bids_);
		return sub_bids_->count();
	}

	const unsigned int* get_sub_bids_cpu() const
	{
		return sub_bids_->cpu_data();
	}

	const unsigned int* get_sub_bcounts_cpu() const
	{
		return sub_bcounts_->cpu_data();
	}

	const unsigned int* get_sub_exclusive_counts_cpu() const
	{
		if(nullptr == f_exclusive_counts_)
		{
			return NULL;
		}
		
		return f_exclusive_counts_->cpu_data();
	}

	void set_freqs(bool has_freq, bool is_char)
	{
		if(has_freq)
		{
			unsigned int size = is_char ? (get_total_subblocks() * sizeof(unsigned char)) : (get_total_subblocks() * sizeof(unsigned int));

			if(nullptr == freqs_)
			{
				freqs_ = make_shared<DataAllocator<unsigned char>>(static_cast<int>(bid_ + 1), size);
				assert(nullptr != freqs_);
				freqs_->gpu_data();
			}
			else if(freqs_->size() != size)
			{
				freqs_.reset(new DataAllocator<unsigned char>(static_cast<int>(bid_ + 1), size));
				assert(nullptr != freqs_);
				freqs_->gpu_data();
			}
		}
		else if(nullptr != freqs_)
		{
			freqs_.reset();
			freqs_ = nullptr;
		}
	}
	
	const unsigned char* get_freqs_gpu() const
	{
		if(nullptr == freqs_)
			return NULL;
		
		return freqs_->gpu_data();
	}

	const unsigned char* get_freqs_cpu(bool synced = true)
	{
		if(nullptr == freqs_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(freqs_->mutable_cpu_data(), freqs_->gpu_data(), freqs_->size(), hipMemcpyDeviceToHost));
		return freqs_->cpu_data();
	}

	void set_vmeans(bool has_vmean)
	{
		if(has_vmean)
		{
			if(nullptr == vmeans_)
			{
				vmeans_ = make_shared<DataAllocator<float>>(static_cast<int>(bid_ + 1), sub_bids_->count() * sizeof(float));
				vmeans_->gpu_data();
			}
		}
		else if(nullptr != vmeans_)
		{
			vmeans_.reset();
			vmeans_ = nullptr;
		}
	}

	const float* get_vmeans_gpu() const
	{
		if(nullptr == vmeans_)
			return NULL;
		
		return vmeans_->gpu_data();
	}


	const float* get_vmeans_cpu(bool synced = true)
	{
		if(nullptr == vmeans_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(vmeans_->mutable_cpu_data(), vmeans_->gpu_data(), vmeans_->size(), hipMemcpyDeviceToHost));
		return vmeans_->cpu_data();
	}

	void set_imeans(bool has_imean)
	{
		if(has_imean)
		{
			if(nullptr == imeans_)
			{
				imeans_ = make_shared<DataAllocator<float>>(static_cast<int>(bid_ + 1), sub_bids_->count() * sizeof(float));
				imeans_->gpu_data();
			}
		}
		else if(nullptr != imeans_)
		{
			imeans_.reset();
			imeans_ = nullptr;
		}
	}

	const float* get_imeans_gpu() const
	{
		if(nullptr == imeans_)
			return NULL;
		
		return imeans_->gpu_data();
	}

	const float* get_imeans_cpu(bool synced = true)
	{
		if(nullptr == imeans_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(imeans_->mutable_cpu_data(), imeans_->gpu_data(), imeans_->size(), hipMemcpyDeviceToHost));
		return imeans_->cpu_data();
	}

	void set_receptor_imeans(bool have_receptor_imeans)
	{
		if(have_receptor_imeans)
		{
			if(nullptr == ampa_imeans_)
			{
				ampa_imeans_ = make_shared<DataAllocator<float>>(static_cast<int>(bid_ + 1), sub_bids_->count() * sizeof(float));
				ampa_imeans_->gpu_data();

				nmda_imeans_ = make_shared<DataAllocator<float>>(static_cast<int>(bid_ + 1), sub_bids_->count() * sizeof(float));
				nmda_imeans_->gpu_data();

				gabaa_imeans_ = make_shared<DataAllocator<float>>(static_cast<int>(bid_ + 1), sub_bids_->count() * sizeof(float));
				gabaa_imeans_->gpu_data();

				gabab_imeans_ = make_shared<DataAllocator<float>>(static_cast<int>(bid_ + 1), sub_bids_->count() * sizeof(float));
				gabab_imeans_->gpu_data();
			}
		}
		else if(nullptr != ampa_imeans_)
		{
			ampa_imeans_.reset();
			ampa_imeans_ = nullptr;

			nmda_imeans_.reset();
			nmda_imeans_ = nullptr;

			gabaa_imeans_.reset();
			gabaa_imeans_ = nullptr;

			gabab_imeans_.reset();
			gabab_imeans_ = nullptr;
		}
	}

	const float* get_ampa_imeans_gpu() const
	{
		if(nullptr == ampa_imeans_)
			return NULL;
		
		return ampa_imeans_->gpu_data();
	}

	const float* get_ampa_imeans_cpu(bool synced = true)
	{
		if(nullptr == ampa_imeans_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(ampa_imeans_->mutable_cpu_data(), ampa_imeans_->gpu_data(), ampa_imeans_->size(), hipMemcpyDeviceToHost));
		return ampa_imeans_->cpu_data();
	}

	const float* get_nmda_imeans_gpu() const
	{
		if(nullptr == nmda_imeans_)
			return NULL;
		
		return nmda_imeans_->gpu_data();
	}

	const float* get_nmda_imeans_cpu(bool synced = true)
	{
		if(nullptr == nmda_imeans_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(nmda_imeans_->mutable_cpu_data(), nmda_imeans_->gpu_data(), nmda_imeans_->size(), hipMemcpyDeviceToHost));
		return nmda_imeans_->cpu_data();
	}

	const float* get_gabaa_imeans_gpu() const
	{
		if(nullptr == gabaa_imeans_)
			return NULL;
		
		return gabaa_imeans_->gpu_data();
	}

	const float* get_gabaa_imeans_cpu(bool synced = true)
	{
		if(nullptr == gabaa_imeans_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(gabaa_imeans_->mutable_cpu_data(), gabaa_imeans_->gpu_data(), gabaa_imeans_->size(), hipMemcpyDeviceToHost));
		return gabaa_imeans_->cpu_data();
	}

	const float* get_gabab_imeans_gpu() const
	{
		if(nullptr == gabab_imeans_)
			return NULL;
		
		return gabab_imeans_->gpu_data();
	}

	const float* get_gabab_imeans_cpu(bool synced = true)
	{
		if(nullptr == gabab_imeans_)
			return NULL;
		
		if(synced)
			HIP_CHECK(hipMemcpy(gabab_imeans_->mutable_cpu_data(), gabab_imeans_->gpu_data(), gabab_imeans_->size(), hipMemcpyDeviceToHost));
		return gabab_imeans_->cpu_data();
	}

	const size_t get_inner_conninds_size() const
	{
		if(nullptr == inner_conninds_)
			return 0;
		return inner_conninds_->count();
	}

	const unsigned int* get_inner_conninds_gpu() const
	{
		return inner_conninds_->gpu_data();
	}

	const unsigned int* get_inner_conninds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_conninds_->mutable_cpu_data(), inner_conninds_->gpu_data(), inner_conninds_->size(), hipMemcpyDeviceToHost));
		return inner_conninds_->cpu_data();
	}

	DataType get_inner_w_synaptics_type() const
	{
		return inner_w_synaptics_.type_;
	}
	
	const char* get_inner_w_synaptics_gpu() const
	{
		return inner_w_synaptics_.data_->gpu_data();
	}
	const size_t get_inner_w_synaptics_size()
	{
		return inner_w_synaptics_.count();
	}

	const char* get_inner_w_synaptics_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_w_synaptics_.data_->mutable_cpu_data(), inner_w_synaptics_.data_->gpu_data(), inner_w_synaptics_.count() * inner_w_synaptics_.elem_size(), hipMemcpyDeviceToHost));
		return inner_w_synaptics_.data_->cpu_data();
	}
	
	const unsigned int* get_inner_colinds_gpu() const
	{
		return inner_colinds_->gpu_data();
	}

	const unsigned int* get_inner_colinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_colinds_->mutable_cpu_data(), inner_colinds_->gpu_data(), inner_colinds_->size(), hipMemcpyDeviceToHost));
		return inner_colinds_->cpu_data();
	}
	
	const unsigned int* get_inner_rowptrs_gpu() const
	{
		return inner_rowptrs_->gpu_data();
	}

	const unsigned int* get_inner_rowptrs_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_rowptrs_->mutable_cpu_data(), inner_rowptrs_->gpu_data(), inner_rowptrs_->size(), hipMemcpyDeviceToHost));
		return inner_rowptrs_->cpu_data();
	}

	const unsigned char* get_inner_connkinds_gpu() const
	{
		return inner_connkinds_->gpu_data();
	}

	const unsigned char* get_inner_connkinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(inner_connkinds_->mutable_cpu_data(), inner_connkinds_->gpu_data(), inner_connkinds_->size(), hipMemcpyDeviceToHost));
		return inner_connkinds_->cpu_data();
	}

	DataType get_outer_w_synaptics_type() const
	{
		return inner_w_synaptics_.type_;
	}

	const char* get_outer_w_synaptics_gpu() const
	{
		return outer_w_synaptics_.data_->gpu_data();
	}

	const size_t get_outer_w_synaptics_size()
	{
		return outer_w_synaptics_.count();
	}

	const char* get_outer_w_synaptics_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_w_synaptics_.data_->mutable_cpu_data(), outer_w_synaptics_.data_->gpu_data(), outer_w_synaptics_.count() * outer_w_synaptics_.elem_size(), hipMemcpyDeviceToHost));
		return outer_w_synaptics_.data_->cpu_data();
	}

	const unsigned int* get_outer_colinds_gpu() const
	{
		return outer_colinds_->gpu_data();
	}

	const unsigned int* get_outer_colinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_colinds_->mutable_cpu_data(), outer_colinds_->gpu_data(), outer_colinds_->size(), hipMemcpyDeviceToHost));
		return outer_colinds_->cpu_data();
	}
	
	const unsigned int* get_outer_rowptrs_gpu() const
	{
		return outer_rowptrs_->gpu_data();
	}

	const unsigned int* get_outer_rowptrs_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_rowptrs_->mutable_cpu_data(), outer_rowptrs_->gpu_data(), outer_rowptrs_->size(), hipMemcpyDeviceToHost));
		return outer_rowptrs_->cpu_data();
	}

	const unsigned char* get_outer_connkinds_gpu() const
	{
		return outer_connkinds_->gpu_data();
	}

	const unsigned char* get_outer_connkinds_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(outer_connkinds_->mutable_cpu_data(), outer_connkinds_->gpu_data(), outer_connkinds_->size(), hipMemcpyDeviceToHost));
		return outer_connkinds_->cpu_data();
	}

	const T* get_uniform_samples_gpu() const
	{
		return uniform_samples_->gpu_data();
	}

	const T* get_uniform_samples_cpu(bool synced = true)
	{
		if(synced)
			HIP_CHECK(hipMemcpy(uniform_samples_->mutable_cpu_data(), uniform_samples_->gpu_data(), uniform_samples_->size(), hipMemcpyDeviceToHost));
		return uniform_samples_->cpu_data();
	}

	void fetch_sample_neuron_offsets(const std::vector<unsigned short>& sample_bids,
											const std::vector<std::vector<unsigned int>>& sample_nids,
											std::vector<unsigned int>& begin_offsets,
											std::vector<unsigned int>& end_offsets,
											std::vector<unsigned short>& matched_sample_bids,
											std::vector<unsigned int>& matched_sample_nids,
											std::tuple<unsigned int, unsigned int>& inner_offset);

	void fetch_sample_neuron_weights(const std::vector<unsigned short>& sample_bids,
											const std::vector<unsigned int>& sample_nids,
											const std::vector<unsigned int>& begin_offsets,
											const std::vector<unsigned int>& end_offsets,
											const std::tuple<unsigned int, unsigned int>& inner_offset,
											const unsigned int sample_id,
											std::vector<unsigned short>& matched_sample_bids,
											std::vector<unsigned int>& matched_sample_nids,
											Weights& matched_sample_weights,
											std::vector<unsigned char>& matched_sample_channels);

	void fetch_props(const unsigned int* neuron_indice,
					const unsigned int* prop_indice,
					const unsigned int n,
					vector<T>& result);

	void fetch_prop_cols(const unsigned int* prop_indice,
						const unsigned int* brain_indice,
						const unsigned int n,
						vector<vector<T>>& result);

protected:
	int gpu_id_;
	unordered_map<unsigned int, OU_Background_Current_Param> ou_current_params_;
	unordered_map<unsigned int, T_Type_Ca_Current_Param> ca_current_params_;
	unordered_map<unsigned int, Dopamine_Current_Param> da_current_params_;
	unordered_map<unsigned int, Adaptation_Current_Param> adapt_current_params_;
	shared_ptr<DataAllocator<T>> noise_rates_;
	shared_ptr<DataAllocator<T2>> g_ex_conducts_;
	shared_ptr<DataAllocator<T2>> g_in_conducts_;
	shared_ptr<DataAllocator<T2>> v_ex_membranes_;
	shared_ptr<DataAllocator<T2>> v_in_membranes_;
	shared_ptr<DataAllocator<T2>> tao_ex_constants_;
	shared_ptr<DataAllocator<T2>> tao_in_constants_;
	shared_ptr<DataAllocator<T>> v_resets_;
	shared_ptr<DataAllocator<T>> v_thresholds_;
	shared_ptr<DataAllocator<T>> c_membrane_reciprocals_;
	shared_ptr<DataAllocator<T>> v_leakages_;
	shared_ptr<DataAllocator<T>> g_leakages_;
	shared_ptr<DataAllocator<T>> t_refs_;

	shared_ptr<DataAllocator<unsigned int>> sub_bids_;
	shared_ptr<DataAllocator<unsigned int>> sub_bcounts_;
	shared_ptr<DataAllocator<unsigned char>> f_exclusive_flags_;
	shared_ptr<DataAllocator<unsigned int>> f_exclusive_counts_;

	shared_ptr<DataAllocator<T2>> j_ex_presynaptics_;
	shared_ptr<DataAllocator<T2>> j_ex_presynaptic_deltas_;
	shared_ptr<DataAllocator<T2>> j_in_presynaptics_;
	shared_ptr<DataAllocator<T2>> j_in_presynaptic_deltas_;
	shared_ptr<DataAllocator<T>> v_membranes_;
	shared_ptr<DataAllocator<int>> t_actives_;
	shared_ptr<DataAllocator<T>> i_synaptics_;
	shared_ptr<DataAllocator<T>> i_ext_stimuli_;
	
	unsigned long long seed_;
	shared_ptr<DataAllocator<hiprandStatePhilox4_32_10_t>> gen_states_;

	shared_ptr<DataAllocator<T>> i_ou_background_stimuli_;
	shared_ptr<DataAllocator<T>> h_ttype_ca_stimuli_;
	shared_ptr<DataAllocator<T>> ahp_ca_concentration_;
	shared_ptr<DataAllocator<T>> uniform_samples_;
	
	//input block spike
	shared_ptr<DataAllocator<unsigned int>> input_timestamps_;
	shared_ptr<DataAllocator<unsigned int>> input_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> input_colinds_;
	
	//intra block connecting spike
	shared_ptr<DataAllocator<unsigned char>> f_inner_actives_;
	shared_ptr<DataAllocator<unsigned char>> f_recorded_inner_actives_;

	shared_ptr<DataAllocator<unsigned int>> inner_conninds_;
	//intra block connecting weight table in csr format
	Weights inner_w_synaptics_;
	//Points to the integer array that contains the row indices 
	//of the corresponding nonzero elements in array csr_inner_w_synaptics_
	shared_ptr<DataAllocator<unsigned int>> inner_colinds_;
	//Points to the integer array of length n+1 (n rows in the sparse matrix)
	shared_ptr<DataAllocator<unsigned int>> inner_rowptrs_;
	shared_ptr<DataAllocator<unsigned char>> inner_connkinds_;

	//inter block connecting spike
	//shared_ptr<DataAllocator<unsigned char>> f_outer_actives_;

	//inter block connecting weight table in csr format
	Weights outer_w_synaptics_;
	//Points to the integer array that contains the row indices 
	//of the corresponding nonzero elements in array csr_outer_w_synaptics_
	shared_ptr<DataAllocator<unsigned int>> outer_colinds_;
	//Points to the integer array of length n+1 (n rows in the sparse matrix)
	shared_ptr<DataAllocator<unsigned int>> outer_rowptrs_;
	shared_ptr<DataAllocator<unsigned char>> outer_connkinds_;

	unsigned int total_neurons_;  // numbers of neurons in this brain block
	unsigned short bid_;  // block id
	T delta_t_;
	unsigned int t_steps_;

	shared_ptr<DataAllocator<unsigned char>> freqs_;
	shared_ptr<DataAllocator<float>> vmeans_;
	shared_ptr<DataAllocator<float>> ampa_imeans_;
	shared_ptr<DataAllocator<float>> nmda_imeans_;
	shared_ptr<DataAllocator<float>> gabaa_imeans_;
	shared_ptr<DataAllocator<float>> gabab_imeans_;
	shared_ptr<DataAllocator<float>> imeans_;

public:
	BLOCK_TYPE block_type_;

	vector<unsigned short> f_sending_bids_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_colinds_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_block_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_sending_active_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>>	f_sending_active_colinds_;
	shared_ptr<DataAllocator<unsigned char>> f_sending_actives_;
	
	vector<unsigned short> f_receiving_bids_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_colinds_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_active_rowptrs_;
	shared_ptr<DataAllocator<unsigned int>> f_receiving_active_colinds_;

	shared_ptr<DataAllocator<unsigned int>>	f_shared_active_colinds_;
	
private:
	DISABLE_COPY_AND_ASSIGN(BrainBlock);
};

}//namespace dtb

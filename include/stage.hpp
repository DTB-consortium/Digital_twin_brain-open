#pragma once
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#include <unordered_map>
#include "weights.hpp"

using namespace std;

namespace dtb {

#define MTGP32_MAX_NUM_BLOCKS 200

struct OU_Background_Current_Param
{
	float mean_;
	float deviation_;
	float correlation_time_;
};

struct Dopamine_Current_Param
{
	float v_dopamine_;
	float g_dopamine_;
};


struct T_Type_Ca_Current_Param
{
	float g_t_;
	float tao_h_minus_;
	float tao_h_plus_;
	float v_h_;
	float v_t_;
};

struct Adaptation_Current_Param
{
	float ca_decay_;
	float alpha_constant_;
	float v_k_;
	float g_ahp_;
};

void update_input_spike_gpu(const unsigned int* neuron_inds,
								const unsigned int n,
								unsigned char* f_actives,
								hipStream_t stream = NULL);


template<typename T>
void update_ou_background_stimuli_gpu(hiprandStatePhilox4_32_10_t* states,
											const unsigned int* rowptrs,
											const std::unordered_map<unsigned int, OU_Background_Current_Param>& param_map,
											const T delta_t,
											const T mean,
											const T stdv,
											T* i_backgrounds,
											T* i_synaptics,
											T* samples,
											hipStream_t stream = NULL);

template<typename T>
void update_ttype_ca_stimuli_gpu(const T* v_membranes,
								const unsigned int* rowptrs,
								const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
								const T* h_ttype_ca,
								T* i_ttype_ca,
								hipStream_t stream = NULL);

template<typename T>
void update_h_ttype_ca_gpu(const T* v_membranes,
								const unsigned int* rowptrs,
								const std::unordered_map<unsigned int, T_Type_Ca_Current_Param>& param_map,
								const T delta_t,
								T* h_ttype_ca,
								hipStream_t stream = NULL);

template<typename T>
void reset_partial_gpu(const T init,
						const unsigned int offset,
						const unsigned int* rowptrs,
						T* out,
						hipStream_t stream = NULL);

template<typename T>
void update_dopamine_stimuli_gpu(const T* v_membranes,
									const unsigned int* rowptrs,
									const std::unordered_map<unsigned int, Dopamine_Current_Param>& param_map,
									T* i_synaptics,
									hipStream_t stream = NULL);

template<typename T>
void update_adaptation_stimuli_gpu(const T* v_membranes,
										const unsigned int* rowptrs,
										const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
										const T* ahp_ca_concentrations,
										T* i_synaptics,
										hipStream_t stream = NULL);

template<typename T>
void update_ahp_ca_concentration_gpu(const unsigned char* f_spikes,
											const unsigned int* rowptrs,
											const std::unordered_map<unsigned int, Adaptation_Current_Param>& param_map,
											T* ahp_ca_concentrations,
											hipStream_t stream = NULL);


template<typename T>
void init_membrane_voltage_gpu(const T* v_ths,
									const T* v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream = NULL);

template<typename T>
void reset_membrane_voltage_gpu(const T*  v_rsts,
									const unsigned int n,
									T* v_membranes,
									hipStream_t stream = NULL);

template<typename T>
void update_membrane_voltage_for_input_gpu(const T* v_rsts,
													const T* v_ths,
													const unsigned char* f_actives,
													const unsigned int n,
													T* v_membranes,
													hipStream_t stream = NULL);

template<typename T>
void update_membrane_voltage_gpu(const T* i_synaptics,
										const T* i_ext_stimuli,
										const T* v_rsts,
										const T* v_ths,
										const T* c_membranes,
										const T* v_leakages,
										const T* g_leakages,
										const T* t_refs,
										const unsigned int n,
										const T delta_t,
										const int t_steps,
										unsigned char* f_actives,
										int* t_actives,
										T* v_membranes,
										hipStream_t stream = NULL);

template<typename T>
void reset_spike_gpu(const unsigned int n,
						const T* v_membranes,
						const T* v_thresholds,
						unsigned char* f_actives,
						hipStream_t stream = NULL);

template<typename T>
void init_spike_time_gpu(const unsigned int n,
							const T* vals,
							const T scale,
							int* t_actives,
							hipStream_t stream = NULL);

template<typename T>
void randint(hiprandStatePhilox4_32_10_t* states,
			const unsigned int n,
			const T a,
			const T b,
			T* samples,
			hipStream_t stream = NULL);

template<typename T>
void update_spike_gpu(hiprandStatePhilox4_32_10_t* states,
						const unsigned int n,
						const T* noise_rates,
						const unsigned char* f_actives,
						unsigned char* f_recorded_actives,
						const T a = 0.f,
						const T b = 1.f,
						T* samples = NULL,
						hipStream_t stream = NULL);

void update_accumulated_spike_gpu(const unsigned char* f_actives,
										const unsigned int n,
										unsigned char* f_recorded_actives,
										hipStream_t stream = NULL);

void update_recving_spikes_gpu(const unsigned int* inputs,
									const unsigned int* rowptrs,
									const unsigned int  segments,
									unsigned int* outputs,	
									hipStream_t stream = NULL);

void count_sending_spikes_temporary_storage_size(const unsigned int sending_count,
														const unsigned int segments,
														unsigned int* block_rowptrs,
														unsigned int* active_rowptrs,
														size_t& storage_size_bytes,
														hipStream_t stream = NULL);

void update_sending_spikes_gpu(const unsigned char* f_actives,
									const unsigned int* sending_rowptrs,
									const unsigned int* sending_colinds,
									const unsigned int segments_count,
									const unsigned int sending_count,
									unsigned char* f_sending_actives,
									unsigned int* block_rowptrs,
									unsigned int* active_rowptrs,
									unsigned int* active_colinds,
									hipStream_t stream = NULL);

template<typename T2>
void init_presynaptic_voltage_gpu(const unsigned int n,
									const T2 val,
									T2* j_u_presynaptics,
									hipStream_t stream = NULL);


template<typename T, typename T2> 
void update_presynaptic_voltage_gpu(const T2* tao_ex_constants,
												const T2* tao_in_constants,
												const unsigned int n,
												T2* j_ex_presynaptics,
												T2* j_ex_presynaptic_deltas,
												T2* j_in_presynaptics,
												T2* j_in_presynaptic_deltas,
												hipStream_t stream = NULL);

template<typename T, typename T2>
void update_presynaptic_voltage_inner_gpu(const unsigned int* rowptrs,
										const unsigned int* colinds,
										const DataType weight_type,
										const char* w_synaptics,
										const unsigned char* connkinds,
										const unsigned int n,
										const unsigned int* f_indices,
										const unsigned char* f_actives,
										T2* j_ex_presynaptics,
										T2* j_in_presynaptics,
										hipStream_t stream = NULL);

template<typename T, typename T2>
void update_presynaptic_voltage_outer_gpu(const unsigned int* rowptrs,
										const unsigned int* colinds,
										const DataType weight_type,
										const char* w_synaptics,
										const unsigned char* connkinds,
										const unsigned int* active_colinds,
										const unsigned int n,
										T2* j_ex_presynaptics,
										T2* j_in_presynaptics,
										hipStream_t stream = NULL);

template<typename T, typename T2>
void update_synaptic_current_gpu(const T2* j_ex_presynaptics,
									const T2* j_in_presynaptics,
									const T2* g_ex_conducts,
									const T2* g_in_conducts,
									const T2* v_ex_membranes,
									const T2* v_in_membranes,
									const T* v_membranes,
									const unsigned int n,
									T2* i_ex_synaptics,
									T2* i_in_synaptics,
									T* i_synaptics,
									hipStream_t stream = NULL);
template<typename T>
void stat_freqs_gpu(const unsigned int* sub_bcounts,
					const unsigned int* exclusive_counts,
					const unsigned int n,
					const unsigned char* exclusive_flags,
					const unsigned char* f_actives,
					T* freqs,
					hipStream_t stream = NULL);

template<typename T>
void stat_vmeans_and_imeans_gpu(const unsigned int* sub_bcounts,
								const unsigned int* exclusive_counts,
								const unsigned int n,
								const unsigned char* exclusive_flags,
								const T* v_membranes,
								const T* i_synapses,
								float* vmeans,
								float* imeans,
								hipStream_t stream = NULL);

template<typename T, typename T2>
void stat_receptor_imeans_gpu(const unsigned int* sub_bcounts,
								const unsigned int* exclusive_counts,
								const unsigned int n,
								const unsigned char* exclusive_flags,
								const T2* i_ex_synapses,
								const T2* i_in_synapses,
								float* ampa_imeans,
								float* nmda_imeans,
								float* gabaa_imeans,
								float* gabab_imeans,
								hipStream_t stream = NULL);


template<typename T>
void stat_samples_gpu(const unsigned int* samples,
								const unsigned int n,
								const unsigned char* f_actives,
								const T* v_membranes,
								const T* i_synaptics,
								const T* i_ou_background_stimuli,
								char* spikes,
								float* vmembs,
								float* isynaptics,
								float* ious,
								hipStream_t stream = NULL);

void fetch_sample_offset_gpu(const unsigned int* colinds,
								const unsigned int segments,
								const unsigned int* begin_offsets,
								const unsigned int* end_offsets,
								const unsigned int nid,
								unsigned int* results,
								hipStream_t stream = NULL,
								bool debug_synchronous = false);

enum PropType{
	EXT_STIMULI_I = 0,
	MEMBRANE_C = 1,
	REF_T = 2,
	LEAKAGE_G = 3,
	LEAKAGE_V = 4,
	THRESHOLD_V = 5,
	RESET_V = 6,
	CONDUCT_G_AMPA = 7,
	CONDUCT_G_NMDA = 8,
	CONDUCT_G_GABAa = 9,
	CONDUCT_G_GABAb = 10,
	MEMBRANE_V_AMPA = 11,
	MEMBRANE_V_NMDA = 12,
	MEMBRANE_V_GABAa = 13,
	MEMBRANE_V_GABAb = 14,
	TAO_AMPA = 15,
	TAO_NMDA = 16,
	TAO_GABAa = 17,
	TAO_GABAb = 18,
	NOISE_RATE = 19
};

template<typename T, typename T2>
struct Properties
{
	T* noise_rates;
	T* i_ext_stimuli;
	T* c_membrane_reciprocals;
	T* t_refs;
	T* g_leakages;
	T* v_leakages;
	T* v_thresholds;
	T* v_resets;
	T2* g_ex_conducts;
	T2* g_in_conducts;
	T2* v_ex_membranes;
	T2* v_in_membranes;
	T2* tao_ex_constants;
	T2* tao_in_constants;
	unsigned int n;
};

template<typename T, typename T2>
void update_tao_constant_gpu(const T delta_t,
							const T2* h_tao_constants,
							const unsigned int n,
							T2* d_tao_constants);

template<typename T>
void update_refractory_period_gpu(const T delta_t,
								const unsigned int n,
								T* d_t_refs);


template<typename T, typename T2>
void update_props_gpu(const unsigned int* neuron_indice,
						const unsigned int* prop_indice,
						const float* prop_vals,
						const unsigned int n,
						Properties<T, T2>& prop,
						hipStream_t stream = NULL);

template<typename T, typename T2>
void assign_prop_cols_gpu(const unsigned int* sub_bids,
							const unsigned int* sub_bcounts,
							const unsigned int m,
							const unsigned int* prop_indice,
							const unsigned int* brain_indice,
							const float* hp_vals,
							const unsigned int n,
							Properties<T, T2>& prop,
							hipStream_t stream = NULL);

template<typename T, typename T2>
void update_prop_cols_gpu(const unsigned int* sub_bids,
							const unsigned int* sub_bcounts,
							const unsigned int m,
							const unsigned int* prop_indice,
							const unsigned int* brain_indice,
							const float* hp_vals,
							const unsigned int n,
							Properties<T, T2>& prop,
							hipStream_t stream = NULL);

template<typename T, typename T2>
void gamma_gpu(hiprandStatePhilox4_32_10_t* states,
					const unsigned int* sub_bids,
					const unsigned int* sub_bcounts,
					const unsigned int m,
					const unsigned int* prop_indice,
					const unsigned int* brain_indice,
					const float* alphas,
					const float* betas,
					const unsigned int n,
					Properties<T, T2>& prop,
					hipStream_t stream = NULL);

void init_random_gpu(const unsigned long long seed,
					const unsigned long long offset,
					const unsigned int n,
					hiprandStatePhilox4_32_10_t *states,
					hipStream_t stream = NULL);


}//namespace istbi 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "block.hpp"

namespace py = pybind11;
using namespace wdm;

class BlockWrapper
{
public:
	BlockWrapper(py::array_t<float> props, py::array_t<unsigned int> nids,
					py::array_t<unsigned short> conn_bids, py::array_t<unsigned int> conn_nids,
					py::array_t<unsigned char> conn_kinds, py::array_t<float> weights,
					const unsigned int block_id, const unsigned int gpu_id, const float delta_t)
	:gpu_id_(gpu_id)
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		// Clear error status
  		HIP_CHECK(hipGetLastError());
		auto arr_prop = props.mutable_unchecked<2>();
		auto arr_nid = nids.mutable_unchecked<1>();
		auto arr_conn_bid = conn_bids.mutable_unchecked<1>();
		auto arr_conn_nid = conn_nids.mutable_unchecked<1>();
		auto arr_conn_kind = conn_kinds.mutable_unchecked<1>();
		auto arr_weight = weights.mutable_unchecked<2>();
		assert(arr_weight.shape(0) == arr_nid.shape(0) &&
				arr_weight.shape(1) == 2);
		shared_blk_ = make_shared<BrainBlock<float, float2>>(arr_prop.shape(0), static_cast<unsigned short>(block_id), gpu_id, delta_t);
		BrainBlock<float, float2>* block = shared_blk_.get();
		block->init_connection_table_gpu(arr_nid.mutable_data(0),
										arr_conn_bid.mutable_data(0),
										arr_conn_nid.mutable_data(0),
										arr_conn_kind.mutable_data(0),
										reinterpret_cast<float2*>(arr_weight.mutable_data(0, 0)),
										arr_nid.shape(0));
		block->init_config_params_gpu(arr_prop.data(0, 0), arr_prop.shape(0), arr_prop.shape(1));
		block->init_all_stages_gpu();
		block->reset_V_membrane_gpu();
		HIP_CHECK(hipDeviceSynchronize());

		j_presynaptics_ = make_shared<DataAllocator<float>>(sizeof(float) * 4 * block->get_total_neurons());
		j_presynaptics_->cpu_data();
		j_presynaptics_->gpu_data();
	}

	void run(const float noise_rate)
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		BrainBlock<float, float2>* block = shared_blk_.get();
		//auto f = freqs.mutable_unchecked<1>();
		//auto v = v_means.mutable_unchecked<1>();
		block->update_time();
		block->update_V_membrane_gpu();
		block->stat_Freqs_and_Vmeans_gpu();
		block->update_F_active_gpu(noise_rate);
		block->update_J_presynaptic_inner_gpu();
		block->update_J_presynaptic_outer_gpu();
		block->update_J_presynaptic_gpu();
		block->update_I_synaptic_gpu();
		HIP_CHECK(hipDeviceSynchronize());
	}

	py::array_t<unsigned int> get_freqs()
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		BrainBlock<float, float2>* block = shared_blk_.get();
		if(block->get_total_subblocks() > 0)
		{
			return py::array_t<unsigned int>(block->get_total_subblocks(), block->get_freqs_cpu());
		}
		
		throw std::runtime_error("There are no subblocks");
	}
	
	py::array_t<float> get_vmeans()
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		BrainBlock<float, float2>* block = shared_blk_.get();
		if(block->get_total_subblocks() > 0)
		{
			return py::array_t<float>(block->get_total_subblocks(), block->get_vmeans_cpu());
		}

		throw std::runtime_error("There are no subblocks");
	}

	py::array_t<float> get_v_membranes()
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		BrainBlock<float, float2>* block = shared_blk_.get();
		return py::array_t<float>(block->get_total_neurons(), block->get_V_membranes_cpu());
	}

	py::array_t<float> get_j_presynaptics()
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		shared_blk_.get()->get_J_presynaptic_cpu(j_presynaptics_);
		return py::array_t<float>(j_presynaptics_->count(), j_presynaptics_->cpu_data());
	}

	py::array_t<unsigned char> get_f_actives()
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		BrainBlock<float, float2>* block = shared_blk_.get();
		return py::array_t<unsigned char>(block->get_total_neurons(), block->get_F_actives_cpu());
	}

	py::array_t<float> get_t_actives()
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		BrainBlock<float, float2>* block = shared_blk_.get();
		return py::array_t<float>(block->get_total_neurons(), block->get_T_actives_cpu());
	}

	void update_properity(py::array_t<float> props)
	{
		HIP_CHECK(hipSetDevice(gpu_id_));
		auto p = props.unchecked<2>();
		BrainBlock<float, float2>* block = shared_blk_.get();
		block->init_config_params_gpu(p.data(0, 0), p.shape(0), p.shape(1));
	}

protected:
	unsigned int gpu_id_;
	shared_ptr<BrainBlock<float, float2>> shared_blk_;
	shared_ptr<DataAllocator<float>> j_presynaptics_;
};

PYBIND11_MODULE(BrainBlock, m) {
    py::class_<BlockWrapper>(m, "BlockWrapper")
        .def(py::init<py::array_t<float>, py::array_t<unsigned int>, py::array_t<unsigned short>,
        			py::array_t<unsigned int>, py::array_t<unsigned char>,
				    py::array_t<float>, const unsigned int, const unsigned int, const float>())
        .def("run", &BlockWrapper::run)
        .def("update_properity", &BlockWrapper::update_properity)
        .def("get_v_membranes", &BlockWrapper::get_v_membranes)
        .def("get_t_actives", &BlockWrapper::get_t_actives)
        .def("get_j_presynaptics", &BlockWrapper::get_j_presynaptics)
        .def("get_f_actives", &BlockWrapper::get_f_actives)
        .def("get_freqs", &BlockWrapper::get_freqs)
        .def("get_vmeans", &BlockWrapper::get_vmeans);
}


#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <future>
#include <chrono>
#include <functional>
#include <thread>
#include <string>
#include <cassert>
#include <cstring>
#include <set>
#include <unordered_set>
#include <stdint.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "common.hpp"
#include "block.hpp"
#include "blocking_queue.hpp"
#include "device_function.hpp"
#include "util/transpose.hpp"
#include "util/cmd_arg.hpp"
#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>
#include "snn.grpc.pb.h"
#include "logging.hpp"
#include "notification.hpp"
#include "unique.hpp"
#include "util/cnpy.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "route.hpp"
#include "version.hpp"

#define ENV_LOCAL_RANK 		"OMPI_COMM_WORLD_LOCAL_RANK"
#define MPI_MASTER_RANK		0

#define ID2RANK(id) (static_cast<int>(id))
#define RANK2ID(rank) static_cast<unsigned short>((rank))

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   					\
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static constexpr auto INVALID_ACTIVE_OFFSET = -1;
using namespace std;
using namespace dtb;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::ServerReaderWriter;
using grpc::Status;
using snn::Empty;
using snn::Version;
using snn::SnnStatus;
using snn::SubblockInfo;
using snn::InitRequest;
using snn::InitResponse;
using snn::RunRequest;
using snn::RunResponse;
using snn::PropType;
using snn::UpdatePropRequest;
using snn::UpdatePropResponse;
using snn::UpdateGammaRequest;
using snn::UpdateGammaResponse;
using snn::UpdateHyperParaRequest;
using snn::UpdateHyperParaResponse;
using snn::UpdateSampleRequest;
using snn::UpdateSampleResponse;
using snn::UpdateOUBackgroundParamRequest;
using snn::UpdateOUBackgroundParamResponse;
using snn::UpdateTTypeCaCurrentParamRequest;
using snn::UpdateTTypeCaCurrentParamResponse;
using snn::UpdateDopamineCurrentParamRequest;
using snn::UpdateDopamineCurrentParamResponse;
using snn::UpdateAdaptationCurrentParamRequest;
using snn::UpdateAdaptationCurrentParamResponse;
using snn::ShutdownRequest;
using snn::ShutdownResponse;
using snn::Snn;
using chrono::high_resolution_clock;
using chrono::time_point;
using chrono::duration;
using chrono::duration_cast;

enum Command
{
	SNN_INIT,
	SNN_RUN,
	SNN_UPDATE_PROP,
	SNN_UPDATE_GAMMA,
	SNN_UPDATE_HYPERPARA,
	SNN_SAMPLE,
	SNN_UPDATE_OU_BACKGROUND_PARAM,
	SNN_UPDATE_TType_CA_CURRENT_PARAM,
	SNN_UPDATE_DOPAMINE_CURRENT_PARAM,
	SNN_UPDATE_ADAPTATION_CURRENT_PARAM,
	SNN_SHUTDOWN
};

enum Tag
{
	TAG_UPDATE_PROP = 1000,
	TAG_UPDATE_GAMMA = 2000,
	TAG_UPDATE_HYPERPARA = 3000,
	TAG_UPDATE_SAMPLE = 4000,
	TAG_UPDATE_TTYPE_CA_CURRENT_PARAM = 5000,
	TAG_UPDATE_DOPAMINE_CURRENT_PARAM = 6000,
	TAG_UPDATE_OU_BACKGROUND_PARAM = 7000,
	TAG_UPDATE_ADAPTATION_CURRENT_PARAM = 8000
};

struct RunReportInfo
{
	unique_ptr<DataAllocator<unsigned char>> freqs_;
	unique_ptr<DataAllocator<float>> vmeans_;
	unique_ptr<DataAllocator<char>> spikes_;
	unique_ptr<DataAllocator<float>> vmembs_;
	unique_ptr<DataAllocator<float>> ious_;
	unique_ptr<DataAllocator<float>> isynaptics_;
	unique_ptr<DataAllocator<float>> ampa_imeans_;
	unique_ptr<DataAllocator<float>> nmda_imeans_;
	unique_ptr<DataAllocator<float>> gabaa_imeans_;
	unique_ptr<DataAllocator<float>> gabab_imeans_;
	unique_ptr<DataAllocator<float>> imeans_;
};

struct MPIInfo
{
	MPIInfo(const int rank, const int size, MPI_Comm comm, const int prime_rank, const int prime_size, MPI_Comm prime_comm)
	:rank_(rank),
	size_(size),
	comm_(comm),
	prime_rank_(prime_rank),
	prime_size_(prime_size),
	prime_comm_(prime_comm){}

	MPIInfo() = delete;
	MPIInfo(MPIInfo&&) = default;
	
	int rank_;
	int size_;
	MPI_Comm comm_;

	int prime_rank_;
	int prime_size_;
	MPI_Comm prime_comm_;
};

struct TransConfig
{
	void clear()
	{
		sending_ranks_.clear();
		routing_ranks_.clear();
		recving_ranks_.clear();
	}

	//Storing ranks to be sent by local rank.
	set<int> sending_ranks_;
	//Storing level of routing hierarchy in first element of tuple structure, and ranks to be routed in 2nd.
	unordered_map<int, set<int>> routing_ranks_;
	//Storing ranks to be received by local rank.
	map<int, int> recving_ranks_;
};

class TransTable
{
public:
	//Storing pointer of  buffer in first element, the number to be sent in 2nd. 
	using BufferRecord = tuple<unsigned int*, int>;
	
	TransTable() = delete;
	explicit TransTable(int mask)
	{
		route_mask_ = align_up<1, int>(mask);
	}
	
	enum Mode
	{
		MODE_POINT_TO_POINT,
		MODE_ROUTE
	};

	struct Record
	{
		int rank_;
		int max_len_;
		set<int> merged_set_;
		BufferRecord buffer_;
	};

	void clear()
	{
		send_records_.clear();
		send_buffs_.clear();
		recv_records_.clear();
		recv_buffs_.clear();
		route_buffs_.clear();
	}

	bool is_routing_mark(int mark)
	{
		return mark & route_mask_;
	}

	int get_native_mark(int mark)
	{
		return mark & ~route_mask_;
	}

	int make_routing_mark(int mark)
	{
		return mark | route_mask_;
	}

	void add_sending_buff(const int rank, 
							vector<unsigned int>& buff, 
							map<unsigned short, vector<unsigned int>>& records,
							map<int, set<int>>& infos)
	{
		if(!buff.empty())
		{
			auto send_buff = send_buffs_.emplace(rank, vector<unsigned int>(buff.size()));
			assert(send_buff.second);
			auto info = infos.emplace(rank, set<int>());
			assert(info.second);
			parse_merged_payload(buff, records, info.first->second);
		}
	}
	
	BufferRecord add_routing_record(const int rank, const int local_rank,
										const int tag, const map<int, BufferRecord>& records)
	{
		if(records.empty())
		{
			BufferRecord record = {NULL, 0};
			return record;
		}
		
		auto route_record = route_records_.emplace(tag, Record());
		assert(route_record.second);
		auto& route_meta = route_record.first->second;
		route_meta.rank_ = rank;

		int total_len = 0;
		for(auto it = records.begin(); it != records.end(); it++)
		{
			assert(std::get<1>(it->second) > 0);
			total_len += (std::get<1>(it->second) + 2);
		}
		route_meta.max_len_ = total_len;
		
		route_buffs_.push_back(vector<unsigned int>(total_len));
		unsigned int* buff = route_buffs_.back().data();
		int offset = 0;
		vector<int> buff_lens;
		buff_lens.reserve(records.size());
		for(auto it = records.begin(); it != records.end(); it++)
		{
			int elems = std::get<1>(it->second);
			assert(elems > 0);
			buff[offset++] = static_cast<unsigned int>(elems + 1);
			buff[offset++] = it->first;
			memcpy(buff + offset, std::get<0>(it->second), elems * sizeof(unsigned int));
			assert(route_meta.merged_set_.insert(it->first).second);
			offset += elems;
			buff_lens.push_back(elems);
		}

		std::get<0>(route_meta.buffer_) = buff;
		std::get<1>(route_meta.buffer_) = total_len;
		
		auto queue_record = route_records_for_queue_.emplace(tag, vector<Record>());
		assert(queue_record.second);
		auto& queue_meta = queue_record.first->second;

		offset = 0;
		for(auto merged_rank : route_meta.merged_set_)
		{
			if(merged_rank == local_rank)
			{
				int native_tag = get_native_mark(tag);
				auto recv_record = recv_records_.emplace(native_tag, Record());
				assert(recv_record.second);
				auto& recv_meta = recv_record.first->second;
				
				recv_meta.max_len_ = buff_lens[offset++];
				continue;
			}
			queue_meta.push_back(Record());
			auto& record = queue_meta.back();
			record.rank_ = merged_rank;
			record.max_len_ = buff_lens[offset++];
		}

		return route_meta.buffer_;
	}

	void make_sending_record(int local_rank, const set<int>& send_ranks, const map<int, set<int>>& send_infos)
	{
		if(send_ranks.empty())
			return;

		send_indice_.reserve(send_ranks.size());
		for(const auto rank : send_ranks)
		{
			assert(rank != local_rank);
			send_indice_.push_back(rank);

			Record record;
			auto send_it = send_buffs_.find(rank);
			if(send_it != send_buffs_.end())
			{
				record.rank_ = make_routing_mark(rank);
				record.max_len_ = static_cast<int>(send_it->second.size());
				std::get<0>(record.buffer_) = send_it->second.data();
				auto info_it = send_infos.find(rank);
				assert(info_it != send_infos.end() && !info_it->second.empty());
				record.merged_set_ = info_it->second;
			}
			else
			{
				record.rank_ = rank;
			}
			send_records_.push_back(record);
		}

		thrust::device_vector<int> d_ranks = send_indice_;
		assert(thrust::is_sorted(d_ranks.begin(), d_ranks.end()));
		auto it = thrust::upper_bound(d_ranks.begin(), d_ranks.end(), local_rank);
		unsigned int start = it - d_ranks.begin();
		for(auto& index : send_indice_)
		{
			index = static_cast<int>((start++) % send_indice_.size());
		}
	}

	void add_recving_record(const int rank, const int tag, const int size)
	{
		if(size > 0)
		{
			recv_buffs_.push_back(vector<unsigned int>(size));
			auto recv_record = recv_records_.emplace(tag, Record());
			assert(recv_record.second);
			auto& record = recv_record.first->second;
			record.rank_ = rank;
			record.max_len_ = size;
			std::get<0>(record.buffer_) = recv_buffs_.back().data();
		}
	}

	void parse_merged_payload(const int local_rank, const int tag, Record& record)
	{
		auto queue_it = route_records_for_queue_.find(tag);
		assert(queue_it != route_records_for_queue_.end());
		
		unsigned int queue_idx = 0;
		unsigned int* content = std::get<0>(record.buffer_);
		int size = std::get<1>(record.buffer_);
		int content_read;

		for(auto rank : record.merged_set_)
		{
			if(rank == local_rank)
			{
				int native_tag = get_native_mark(tag);
				auto recv_it = recv_records_.find(native_tag);
				assert(recv_it != recv_records_.end());
				if(size > 0)
				{
					content_read = static_cast<int>(content[0]);
					assert(content_read > 1);
					int merge_rank = static_cast<int>(content[1]);
					if(merge_rank == rank)
					{
						content++;
						size--;
						
						std::get<0>(recv_it->second.buffer_) = content + 1;
						std::get<1>(recv_it->second.buffer_) = content_read - 1;
						content +=  content_read;
						size -= content_read;
					}
					else
					{
						assert(rank < merge_rank);
						std::get<0>(recv_it->second.buffer_) = NULL;
						std::get<1>(recv_it->second.buffer_) = 0;
					}
				}
				else
				{
					std::get<0>(recv_it->second.buffer_) = NULL;
					std::get<1>(recv_it->second.buffer_) = 0;
				}
			}
			else
			{
				auto& queue_record = queue_it->second[queue_idx++];
				assert(queue_record.rank_ == rank);
				if(size > 0)
				{
					content_read = static_cast<int>(content[0]);
					assert(content_read > 1);
					int merge_rank = static_cast<int>(content[1]);
					if(rank == merge_rank)
					{
						assert(queue_record.max_len_ >= (content_read - 1));
						// modify position to unread content
						content++;
						size--;
						
						std::get<0>(queue_record.buffer_) = content + 1;
						std::get<1>(queue_record.buffer_) = content_read - 1;
						content +=  content_read;
						size -= content_read;
					}
					else
					{
						assert(rank < merge_rank);
						std::get<0>(queue_record.buffer_) = NULL;
						std::get<1>(queue_record.buffer_) = 0;
					}
				}
				else
				{
					std::get<0>(queue_record.buffer_) = NULL;
					std::get<1>(queue_record.buffer_) = 0;
				}
			}
		}
		assert(0 == size && queue_idx == queue_it->second.size());
	}

	
	Mode mode_;
	int recv_count_;
	//key indicates rank of receiver
	vector<Record> send_records_;
	map<int,Record> recv_records_;
	map<int,Record> route_records_;
	map<int,vector<Record>> route_records_for_queue_;

	vector<int> send_indice_;
	vector<MPI_Request> send_requests_;

protected:
	void parse_merged_payload(vector<unsigned int>& buff,
								map<unsigned short, vector<unsigned int>>& records,
								set<int>& merge_ranks)
	{
		int content_read;
		unsigned int* content = buff.data();
		int content_size = static_cast<int>(buff.size());
		while(content_size > 0) 
		{
			content_read = static_cast<int>(content[0]);
			// modify position to unread content
			content++;
			content_size--;

			assert(content_read > 1);
			assert(merge_ranks.insert(static_cast<int>(content[0])).second);
			auto record = records.emplace(RANK2ID(content[0]), vector<unsigned int>(content_read - 1));
			assert(record.second);
			memcpy(record.first->second.data(), (content + 1), (content_read - 1) * sizeof(unsigned int));
			content +=  content_read;
			content_size -= content_read;
			assert(thrust::is_sorted(record.first->second.begin(),record.first->second.end()));
		}
		assert(0 == content_size);
	}

	int route_mask_;
	map<int, vector<unsigned int>> send_buffs_;
	//key indicates target rank of sender and value indicates meta data of operation to each sender
	//in which key indicates  rank of sender.
	vector<vector<unsigned int>> recv_buffs_;
	vector<vector<unsigned int>> route_buffs_;
};

template<typename T, typename T2>
class NodeInfo
{
public:
	
	NodeInfo(const int rank, const int size, const MPI_Comm comm, 
			const int prime_rank, const int prime_size, MPI_Comm prime_comm, 
			const int gpu_id, const string& name, const set<int>& rank_in_same_node, 
			int max_rank_in_same_node)
	:info_(new MPIInfo(rank, size, comm, prime_rank, prime_size, prime_comm)),
	gid_(gpu_id),
	rank_in_same_node_(rank_in_same_node),
	max_rank_in_same_node_(max_rank_in_same_node),
	use_ou_background_stimuli_(false)
	{
		name_ = name + string("-") + std::to_string(rank);
		stream1_ = HipStream::create();
		stream2_ = HipStream::create();
		if(MPI_COMM_NULL != prime_comm)
		{
			trans_table_ = make_unique<TransTable>(prime_size);
		}
		spikes_.reset(nullptr);
		vmembs_.reset(nullptr);
		ious_.reset(nullptr);
		isynaptics_.reset(nullptr);
	}

	void clear()
	{
		block_.reset(nullptr);
		samples_.reset(nullptr);
		spikes_.reset(nullptr);
		vmembs_.reset(nullptr);
		trans_table_->clear();
		reporting_queue_.clear();	
	}

	int gid_;
	string name_;
	unique_ptr<MPIInfo> info_;
	set<int> rank_in_same_node_;
	int max_rank_in_same_node_;
	unique_ptr<BrainBlock<T, T2>> block_;
	shared_ptr<HipStream> stream1_;
	shared_ptr<HipStream> stream2_;

	unique_ptr<TransTable> trans_table_;
	
	//per node
	std::vector<unsigned short> sample_bids_;
	std::vector<unsigned int> sample_nids_;
	std::vector<unsigned int> sample_begin_offsets_;
	std::vector<unsigned int> sample_end_offsets_;
	std::tuple<unsigned int, unsigned int> sample_inner_offset_;
	
	unique_ptr<DataAllocator<unsigned int>> samples_;
	unique_ptr<DataAllocator<char>> spikes_;
	unique_ptr<DataAllocator<float>> vmembs_;
	unique_ptr<DataAllocator<float>> ious_;
	unique_ptr<DataAllocator<float>> isynaptics_;

	BlockingQueue<int> routing_queue_;

	Command cmd_;
	string path_;
	//T noise_rate_;
	T delta_t_;

	int iter_;
	int iter_offset_;
	bool has_freq_;
	bool freq_is_char_;
	bool has_vmean_;
	bool has_sample_spike_;
	bool has_sample_vmemb_;
	bool has_sample_iou_;
	bool has_sample_isynaptic_;
	bool has_imean_;
	bool have_receptor_imeans_;

	bool use_ou_background_stimuli_;
	//gamma
	vector<unsigned int> prop_indice_;
	vector<unsigned int> brain_indice_;
	
	Notification reporting_notification_;
	BlockingQueue<shared_ptr<RunReportInfo>> reporting_queue_;
	BlockingQueue<shared_ptr<RunReportInfo>> reporting_free_queue_;

	double used_cpu_mem_;
	double total_gpu_mem_;
	double used_gpu_mem_;
};

static void init_mpi_env(int* argc, char*** argv, int& rank, int& gpu_id, int& size, string& name)
{
	// Setting the device here will have an effect only for the CUDA-aware MPI
	char* local_rank_str = NULL;
	
	// We extract the local rank initialization using an environment variable
	if((local_rank_str = getenv(ENV_LOCAL_RANK)) != NULL)
	{
		rank = atoi(local_rank_str);
	}

	int provided;
	MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);

	if(provided != MPI_THREAD_MULTIPLE)  
	{  
	    cerr << "MPI do not Support Multiple thread" << endl;  
	    exit(0);  
	} 
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(MPI_MASTER_RANK == rank)
	{
		cout << "Multi thread provide " << provided << " support." << endl;
		gpu_id = -1;
	}
	else
	{
		int dev_count = 0;
		HIP_CHECK(hipGetDeviceCount(&dev_count));
		gpu_id = (rank % dev_count);
		HIP_CHECK(hipSetDevice(gpu_id));
		hipDeviceProp_t deviceProp;
        hipGetDeviceProperties(&deviceProp, gpu_id);
		name = string(deviceProp.name);
	}
}

static int wait_handle(Command& cmd, const MPIInfo& info)
{
	return MPI_Bcast(&cmd, 1, MPI_INT, MPI_MASTER_RANK, info.comm_);
}

static int snn_group_sync(const MPIInfo& info)
{
	if(MPI_COMM_NULL != info.prime_comm_)
		return MPI_Barrier(info.prime_comm_);
	return MPI_SUCCESS;
}

static int snn_sync(const MPIInfo& info)
{
	return MPI_Barrier(info.comm_);
}

static int snn_gather(const MPIInfo& info,
						const void *sendbuf,
						int sendcount,
						MPI_Datatype sendtype,
						void *recvbuf,
						int recvcount,
						MPI_Datatype recvtype,
						int root = MPI_MASTER_RANK)
{
	return MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, info.comm_);
}

static int snn_gatherv(const MPIInfo& info,
						const void *sendbuf,
						int sendcount,
						MPI_Datatype sendtype,
						void *recvbuf,
						const int* recvcounts,
						const int* displs,
						MPI_Datatype recvtype,
						int root = MPI_MASTER_RANK)
{
	return MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
			displs, recvtype, root, info.comm_);
}


static void snn_init_report(const MPIInfo& info,
							int* recvcounts,
							int* displs,
							int* neurons_per_block,
							vector<int>& subids,
							vector<int>& subcounts)
{

	int neurons = 0;
	int subblocks = 0;

	MPICHECK(snn_gather(info, &neurons, 1, MPI_INT, neurons_per_block, 1, MPI_INT));
	MPICHECK(snn_gather(info, &subblocks, 1, MPI_INT, recvcounts, 1, MPI_INT));
	
	int total = 0;
	for(int i = 0; i < info.size_; i++)
	{
		total += recvcounts[i];
	}
	subids.resize(total);
	subcounts.resize(total);
	
	thrust::exclusive_scan(recvcounts, recvcounts + info.size_, displs);

	MPICHECK(snn_gatherv(info, NULL, subblocks, MPI_INT, subids.data(), recvcounts, displs, MPI_INT));
	MPICHECK(snn_gatherv(info, NULL, subblocks, MPI_INT, subcounts.data(), recvcounts, displs, MPI_INT));

}

template<typename T, typename T2>
static void snn_init_report(NodeInfo<T, T2>& node)
{
	int neurons = static_cast<int>(node.block_->get_total_neurons());
	int subblocks = static_cast<int>(node.block_->get_total_subblocks());
	vector<int> sub_bcounts(subblocks);
	vector<int> sub_bids(subblocks);
	memcpy(sub_bids.data(), node.block_->get_sub_bids_cpu(), subblocks * sizeof(int));
	
	MPICHECK(snn_gather(*node.info_, &neurons, 1, MPI_INT, NULL, 1, MPI_INT));
	MPICHECK(snn_gather(*node.info_, &subblocks, 1, MPI_INT, NULL, 1, MPI_INT));

	for(int i = 0; i < subblocks; i++)
	{
		unsigned int sub_count = node.block_->get_sub_bcounts_cpu()[i + 1] - node.block_->get_sub_bcounts_cpu()[i];
		if(NULL != node.block_->get_sub_exclusive_counts_cpu())
		{
			sub_bcounts[i] = static_cast<int>(sub_count - node.block_->get_sub_exclusive_counts_cpu()[i]);
		}
		else
		{
			sub_bcounts[i] = static_cast<int>(sub_count);
		}
		
	}

	MPICHECK(snn_gatherv(*node.info_, sub_bids.data(),
					subblocks, MPI_INT, NULL, NULL, NULL, MPI_INT));
	MPICHECK(snn_gatherv(*node.info_, sub_bcounts.data(),
					subblocks, MPI_INT, NULL, NULL, NULL, MPI_INT));
}

static void snn_init(const MPIInfo& info,
					string& path,
					float& delta_t,
					int& comm_mode)
{
	size_t len = 0;
	vector<char> vpath;
	
	if(MPI_MASTER_RANK == info.rank_)
	{
		len = path.length();
	}
	
	MPICHECK(MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, MPI_MASTER_RANK, info.comm_));
	if(len > 0)
	{
		vpath.resize(len + 1);
		vpath[len] = 0;

		if(MPI_MASTER_RANK == info.rank_)
		{
			path.copy(vpath.data(), len);
		}
	
		MPICHECK(MPI_Bcast(vpath.data(), len, MPI_CHAR, MPI_MASTER_RANK, info.comm_));

		if(MPI_MASTER_RANK != info.rank_)
		{
			path.assign(vpath.data());
		}
	}
	
	MPICHECK(MPI_Bcast(&delta_t, 1, MPI_FLOAT, MPI_MASTER_RANK, info.comm_));

	MPICHECK(MPI_Bcast(&comm_mode, 1, MPI_INT, MPI_MASTER_RANK, info.comm_));
}


template<typename T, typename T2>
static void config_conn_table(const std::string& filename,
								NodeInfo<T, T2>& node)

{
	node.block_ = make_unique<BrainBlock<T, T2>>((node.info_->rank_ - 1), node.gid_, node.delta_t_);
	node.block_->init_connection_table_gpu(filename);
	node.block_->init_config_params_gpu(filename);
	node.block_->init_all_stages_gpu();

	//node.block_->reset_V_membrane_gpu();
	HIP_CHECK(hipDeviceSynchronize());

	LOG_INFO << "the total neurons: " << node.block_->get_total_neurons() << endl;

}

template<typename T, typename T2>
static bool config_route_table(NodeInfo<T, T2>& node,
								TransConfig& conf)
{
	bool ret = true;
	MPIInfo& info = *node.info_;
	std::vector<int32_t> dims(2);
	dims[1] = node.max_rank_in_same_node_;
	dims[0] = (info.prime_size_ + dims[1] - 1) / dims[1];
	std::vector<int32_t> route_table;
	
	generate_route(info.prime_size_, dims, route_table);

	const int n = info.prime_size_ - 1;
	std::stringstream ss;
	for(int i = 0; i < route_table.size(); i++)
	{
		if(i % n == 0)
			ss << std::endl;
		ss << route_table[i] << "\t";
	}
	ss << std::endl;
	LOG_INFO << ss.str();

	thrust::host_vector<int> h_sranks(n);
	thrust::host_vector<int> h_dranks(n);
	thrust::device_vector<int> d_sranks(n);
	thrust::device_vector<int> d_dranks(n);
	for(int dst_rank = 0; dst_rank < info.prime_size_; dst_rank++)
	{	
		for(int j = 0; j < n; ++j)
		{
			h_sranks[j] = route_table[dst_rank * n + j];
			if(j >= dst_rank)
			{
				h_dranks[j] = j + 1;
			}
			else
			{
				h_dranks[j] = j;
			}
		}

		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_sranks.data()), h_sranks.data(), h_sranks.size() * sizeof(int), hipMemcpyHostToDevice));
		HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_dranks.data()), h_dranks.data(), h_dranks.size() * sizeof(int), hipMemcpyHostToDevice));
				
		auto it = thrust::unique(d_dranks.begin(), d_dranks.end());
		if(it != d_dranks.end())
		{
			std::cerr << "Rank" << info.rank_ << " config route table failed. fall back peer to peer" << std::endl;
			conf.clear();
			ret = false;
			break;
		}

		if(dst_rank != info.prime_rank_)
		{
			it = thrust::find(d_dranks.begin(), d_dranks.end(), info.prime_rank_);
			int next_rank = -1;
			if(it != d_dranks.end())
			{
				assert(1 == thrust::count(d_dranks.begin(), d_dranks.end(), info.prime_rank_));
				next_rank = h_sranks[it - d_dranks.begin()];
				assert(conf.recving_ranks_.emplace(std::make_pair(dst_rank, next_rank)).second);
			}
			
			if(!thrust::is_sorted(d_sranks.begin(), d_sranks.end()))
			{
				thrust::sort_by_key(d_sranks.begin(), d_sranks.end(), d_dranks.begin());
			}

			int count = thrust::count(d_sranks.begin(), d_sranks.end(), info.prime_rank_);
			if(count > 0)
			{
				if(next_rank < 0)
				{
					std::cerr << "Rank" << info.rank_ << " config route table failed. fall back peer to peer" << std::endl;
					conf.clear();
					ret = false;
					break;
				}
				assert(next_rank == dst_rank);
				it = thrust::lower_bound(d_sranks.begin(), d_sranks.end(), info.prime_rank_);
				assert(it != d_sranks.end());
				int offset = it - d_sranks.begin();
				HIP_CHECK(hipMemcpy(h_dranks.data(), thrust::raw_pointer_cast(d_dranks.data() + offset), count * sizeof(int), hipMemcpyDeviceToHost));
				for(int j = 0; j < count; j++)
				{
					assert(h_dranks[j] != info.prime_rank_);
				}
				auto route = conf.routing_ranks_.emplace(std::make_pair(dst_rank, set<int>()));
				assert(route.second);
				
				for(int j = 0; j < count; j++)
				{
					assert((route.first->second).insert(h_dranks[j]).second);
				}
			}
		}
		else
		{
			if(!thrust::is_sorted(d_sranks.begin(), d_sranks.end()))
			{
				thrust::sort_by_key(d_sranks.begin(), d_sranks.end(), d_dranks.begin());
			}

			int count = thrust::count(d_sranks.begin(), d_sranks.end(), info.prime_rank_);
			if(count > 0)
			{
				it = thrust::lower_bound(d_sranks.begin(), d_sranks.end(), info.prime_rank_);
				if(it != d_sranks.end())
				{
					int offset = it - d_sranks.begin();
					HIP_CHECK(hipMemcpy(h_dranks.data(), thrust::raw_pointer_cast(d_dranks.data() + offset), count * sizeof(int), hipMemcpyDeviceToHost));
					for(unsigned int j = 0; j < count; j++)
					{
						assert(conf.sending_ranks_.insert(h_dranks[j]).second);
					}
				}
			}
			else
			{
				std::cerr << "Rank" << info.rank_ << " config route failed. fall back peer to peer" << std::endl;
				conf.clear();
				ret = false;
				break;
			}
		}
	}

	if(!ret)
	{
		for(int rank = 0; rank < info.prime_size_; rank++)
		{
			if(rank == info.prime_rank_)
				continue;

			assert(conf.recving_ranks_.emplace(std::make_pair(rank, rank)).second);
			assert(conf.sending_ranks_.insert(rank).second);
		}
	}

	return ret;
}

template<typename T, typename T2>
static void send_init(const NodeInfo<T, T2>& node,
					TransConfig& conf,
					unordered_map<int, map<int, TransTable::BufferRecord>>& route_records,
					vector<MPI_Request>& requests)
{
	const unsigned short* receiver_bids = NULL;
	const unsigned int* receiver_rowptrs = NULL;
	const unsigned int* receiver_colinds = NULL;

	if(!node.block_->f_receiving_bids_.empty())
	{
		receiver_bids = node.block_->f_receiving_bids_.data();
		receiver_rowptrs = node.block_->f_receiving_rowptrs_->cpu_data();
		receiver_colinds = node.block_->f_receiving_colinds_->cpu_data();
	}
	
	unsigned int last_index = 0;
	for (auto recv_it = conf.recving_ranks_.begin(); recv_it != conf.recving_ranks_.end();)
	{
		int count = 0;
		if(last_index < node.block_->f_receiving_bids_.size() && 
			recv_it->first == ID2RANK(receiver_bids[last_index]))
		{
			count = static_cast<int>(receiver_rowptrs[last_index + 1] - receiver_rowptrs[last_index]);
			assert(count > 0);
		}

		auto route_it = conf.routing_ranks_.find(recv_it->first);
		//Point-to-point traffic or no routing is required
		if(conf.routing_ranks_.empty() || route_it == conf.routing_ranks_.end())
		{
			MPI_Request request;
			const void* send_buff = NULL;
			if(count > 0)
			{
				send_buff = receiver_colinds + receiver_rowptrs[last_index];
				last_index++; 
			}
			
			MPICHECK(MPI_Isend(send_buff,
							count,
							MPI_UNSIGNED,
							recv_it->second,
							recv_it->first,
							node.info_->prime_comm_,
							&request));
			requests.push_back(request);

			if(0 == count)
			{
				recv_it = conf.recving_ranks_.erase(recv_it);
			}
			else
			{
				node.trans_table_->add_recving_record(recv_it->second, recv_it->first, count);
				recv_it++;
			}
		}
		else
		{
			if(count > 0)
			{
				auto route = route_records.emplace(recv_it->first, map<int, TransTable::BufferRecord>());
				assert(route.second);
				auto info = route.first->second.emplace(node.info_->prime_rank_, TransTable::BufferRecord());
				assert(info.second);
				std::get<0>(info.first->second) = const_cast<unsigned int*>(receiver_colinds) + receiver_rowptrs[last_index];
				std::get<1>(info.first->second) = count;
				last_index++;
			}
			recv_it++;
		}
	}	
	assert(last_index == node.block_->f_receiving_bids_.size());
}

template<typename T, typename T2>
static void check_init(NodeInfo<T, T2>& node)
{
	vector<MPI_Request> requests(node.block_->f_sending_bids_.size());
	MPI_Status status;
	for(unsigned int i = 0; i < node.block_->f_sending_bids_.size(); i++)
	{
		int count = node.block_->f_sending_rowptrs_->cpu_data()[i + 1] - node.block_->f_sending_rowptrs_->cpu_data()[i];
		assert(count > 0);
		MPICHECK(MPI_Isend(node.block_->f_sending_colinds_->cpu_data() + node.block_->f_sending_rowptrs_->cpu_data()[i],
							count,
							MPI_UNSIGNED,
							ID2RANK(node.block_->f_sending_bids_[i]),
							node.info_->prime_rank_,
							node.info_->prime_comm_,
							&requests[i]));
	}

	assert(thrust::is_sorted(node.block_->f_receiving_bids_.begin(), node.block_->f_receiving_bids_.end()));
	for(unsigned int i = 0; i < node.block_->f_receiving_bids_.size(); i++)
	{
		MPICHECK(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, node.info_->prime_comm_, &status));
		int rank = status.MPI_SOURCE;
		int tag = status.MPI_TAG;
		int elems;
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		auto it = thrust::find(node.block_->f_receiving_bids_.begin(), node.block_->f_receiving_bids_.end(), tag);
		//auto it = thrust::lower_bound(node.block_->f_receiving_bids_.begin(), node.block_->f_receiving_bids_.end(), tag);
		assert(it != node.block_->f_receiving_bids_.end());
		unsigned int index = it - node.block_->f_receiving_bids_.begin();
		assert(elems == static_cast<int>(node.block_->f_receiving_rowptrs_->cpu_data()[index + 1] - node.block_->f_receiving_rowptrs_->cpu_data()[index]));
		
		vector<unsigned int> buff(elems);

		MPICHECK(MPI_Recv(buff.data(),
						elems,
						MPI_UNSIGNED,
						rank,
						tag,
						node.info_->prime_comm_,
						MPI_STATUS_IGNORE));

		assert(thrust::equal(buff.begin(), buff.end(), node.block_->f_receiving_colinds_->cpu_data() + node.block_->f_receiving_rowptrs_->cpu_data()[index]));
	}
	
	if(!requests.empty())
	{
		MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}
}

template<typename T, typename T2>
static void route_init(NodeInfo<T, T2>& node,
						TransConfig& conf,
						const int rank,
						const int tag,
						const int elems,
						vector<vector<unsigned int>>& route_buffs,
						unordered_map<int, map<int, TransTable::BufferRecord>>& route_records,
						vector<MPI_Request>& requests)
{
	MPI_Request request;
	auto route_it = conf.routing_ranks_.find(tag);
	assert(route_it != conf.routing_ranks_.end());
	if((route_it->second).find(rank) == (route_it->second).end())
	{
		LOG_ERROR<< "[route_init]: invalid routing table configuration!" << std::endl;
		assert(0);
	}

	unsigned int* buff = NULL;
	auto record_it = route_records.find(tag);
	if(0 < elems)
	{
		route_buffs.push_back(vector<unsigned int>(elems));
		buff = route_buffs.back().data();
		if(record_it == route_records.end())
		{
			auto record = route_records.emplace(tag, map<int, TransTable::BufferRecord>());
			assert(record.second);
			record_it = record.first;
		}

		auto info = record_it->second.emplace(rank, TransTable::BufferRecord());
		assert(info.second);
		std::get<0>(info.first->second) = buff;
		std::get<1>(info.first->second) = elems;
	}
	
	MPICHECK(MPI_Recv(buff,
					elems,
					MPI_UNSIGNED,
					rank,
					tag,
					node.info_->prime_comm_,
					MPI_STATUS_IGNORE));

	assert(1 == (route_it->second).erase(rank));
	
	if(route_it->second.empty())
	{
		auto recv_it = conf.recving_ranks_.find(tag);
		assert(recv_it != conf.recving_ranks_.end());
		if(record_it == route_records.end())
		{
			assert(recv_it->first == tag);
			MPICHECK(MPI_Isend(NULL,
							0,
							MPI_UNSIGNED,
							recv_it->second,
							tag,
							node.info_->prime_comm_,
							&request));
			conf.recving_ranks_.erase(recv_it);
		}
		else if(1 == record_it->second.size() && record_it->second.begin()->first == node.info_->prime_rank_)
		{
			assert(recv_it->first == tag);
			int count = std::get<1>(record_it->second.begin()->second);
			MPICHECK(MPI_Isend(std::get<0>(record_it->second.begin()->second),
							count,
							MPI_UNSIGNED,
							recv_it->second,
							tag,
							node.info_->prime_comm_,
							&request));

			node.trans_table_->add_recving_record(recv_it->second, tag, elems);
		}
		else
		{
			assert(record_it->second.size() >= 1);
			int route_tag = node.trans_table_->make_routing_mark(tag);
			TransTable::BufferRecord record = node.trans_table_->add_routing_record(recv_it->second, node.info_->prime_rank_, route_tag, record_it->second);
			
			MPICHECK(MPI_Isend(std::get<0>(record),
							std::get<1>(record),
							MPI_UNSIGNED,
							recv_it->second,
							route_tag,
							node.info_->prime_comm_,
							&request));
		}
		requests.push_back(request);
	}
}

template<typename T, typename T2>
static void recv_init(NodeInfo<T, T2>& node,
					TransConfig& conf,
					unordered_map<int, map<int, TransTable::BufferRecord>>& route_records,
					vector<MPI_Request>& requests)
{
	map<unsigned short, vector<unsigned int>> sending_records;
	map<int, set<int>> sending_infos;
	vector<vector<unsigned int>> routing_buffs;
	MPI_Status status;
	
	unsigned int recving_count = conf.sending_ranks_.size();
	for(const auto& pair : conf.routing_ranks_)
	{
		assert(pair.first != node.info_->prime_rank_);
		recving_count += pair.second.size();
	}

	if(0 == recving_count)
		return;

	for(unsigned int idx = 0; idx < recving_count; idx++)
	{	
		int elems, rank, tag;
		
		MPICHECK(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, node.info_->prime_comm_, &status));
		assert(status.MPI_SOURCE != node.info_->prime_rank_);
		rank = status.MPI_SOURCE;
		tag = status.MPI_TAG;
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		
		if(node.trans_table_->is_routing_mark(tag))
		{
			auto send_it = conf.sending_ranks_.find(rank);
			assert(send_it != conf.sending_ranks_.end());
			assert(node.trans_table_->get_native_mark(tag) == node.info_->prime_rank_);
			assert(elems > 0);
			vector<unsigned int> buff(elems);
			MPICHECK(MPI_Recv(buff.data(),
							elems,
							MPI_UNSIGNED,
							rank,
							tag,
							node.info_->prime_comm_,
							MPI_STATUS_IGNORE));

			node.trans_table_->add_sending_buff(rank, buff, sending_records, sending_infos);
		}
		else if(node.info_->prime_rank_ == tag)
		{
			auto send_it = conf.sending_ranks_.find(rank);
			assert(send_it != conf.sending_ranks_.end());
			unsigned int* buff = NULL;
			if(elems > 0)
			{
				auto record = sending_records.emplace(RANK2ID(rank), vector<unsigned int>());
				assert(record.second);
				record.first->second.resize(elems);
				buff = record.first->second.data();
			}
			else
			{
				assert(0 == elems);
				conf.sending_ranks_.erase(send_it);
			}

			MPICHECK(MPI_Recv(buff,
						elems,
						MPI_UNSIGNED,
						rank,
						tag,
						node.info_->prime_comm_,
						MPI_STATUS_IGNORE));

			if(elems > 0)
			{
				assert(thrust::is_sorted(buff, buff + elems));
			}
		}
		else if(!conf.routing_ranks_.empty())
		{
			route_init(node, conf, rank, tag, elems, routing_buffs, route_records, requests);
		}
		else
		{
			LOG_INFO << "[ERROR]: unexpected point-to-point traffic!" << std::endl;
			assert(0);
		}
	}
	
	assert(node.trans_table_->recv_records_.size() == node.block_->f_receiving_bids_.size());
	node.block_->record_F_sending_actives(sending_records);
	sending_records.clear();

	node.trans_table_->make_sending_record(node.info_->prime_rank_, conf.sending_ranks_, sending_infos);
	assert(thrust::is_sorted(node.block_->f_sending_bids_.begin(), node.block_->f_sending_bids_.end()));
	for(auto& record : node.trans_table_->send_records_)
	{
		if(record.merged_set_.empty())
		{
			for(int i = 0; i < static_cast<int>(node.block_->f_sending_bids_.size()); i++)
			{
				if(record.rank_ == ID2RANK(node.block_->f_sending_bids_[i]))
				{
					assert(record.merged_set_.insert(i).second);
					break;
				}
			}
			continue;
		}
		
		int offset = 0;
		set<int> indice;
		for(auto rank : record.merged_set_)
		{
			for(int i = offset; i < static_cast<int>(node.block_->f_sending_bids_.size()); i++)
			{
				if(rank == ID2RANK(node.block_->f_sending_bids_[i]))
				{
					assert(indice.insert(i).second);
					offset = i + 1;
					break;
				}
			}
		}
		assert(indice.size() == record.merged_set_.size());
		assert(thrust::is_sorted(indice.begin(), indice.end()));
		record.merged_set_ = indice;
	}
	
	if(!requests.empty())
	{
		MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}

	assert(thrust::is_sorted(node.block_->f_receiving_bids_.begin(), node.block_->f_receiving_bids_.end()));

	unsigned int n = 0;
	n += node.trans_table_->send_records_.size();
	for(auto& pair : node.trans_table_->route_records_for_queue_)
	{
		n += pair.second.size();
	}
	node.trans_table_->send_requests_.resize(n);

	n = 0;
	n += node.trans_table_->recv_records_.size();
	for(auto& pair : node.trans_table_->route_records_)
	{
		if(pair.second.merged_set_.find(node.info_->prime_rank_) == pair.second.merged_set_.end())
		{
			n++;
		}
	}
	node.trans_table_->recv_count_ = n;
	
	n = 0;
	if(nullptr != node.block_->f_receiving_active_colinds_)
	{
		n = node.block_->f_receiving_active_colinds_->size();
	}
	
	if(nullptr != node.block_->f_sending_active_colinds_)
	{
		n = MAX(n, node.block_->f_sending_active_colinds_->size());
	}

	if(n > 0)
	{
		node.block_->f_shared_active_colinds_= make_shared<DataAllocator<unsigned int>>(node.info_->rank_, n);
		node.block_->f_shared_active_colinds_->gpu_data();
	}
}

template<typename T, typename T2>
static void snn_init(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	assert(MPI_MASTER_RANK != info.rank_);
	node.clear();
	
	LOG_INFO << "==========Before snn init=============" << endl;
	report_dev_info();
	report_mem_info();

	float delta_t;
	int mode;
	snn_init(info, node.path_, delta_t, mode);
	node.delta_t_ = static_cast<T>(delta_t);
	node.trans_table_->mode_ = static_cast<TransTable::Mode>(mode);

	config_conn_table<T, T2>(node.path_ + string("/block_") + to_string(info.rank_ - 1) + string(".npz"),
							node);

	MPICHECK(snn_group_sync(info));
	
	LOG_INFO << "==========After parsing connection table=============" << endl;
	report_dev_info();
	report_mem_info();

	if(info.prime_comm_ != MPI_COMM_NULL)
	{
		vector<MPI_Request> requests;
		unordered_map<int, map<int, TransTable::BufferRecord>> route_records;
		TransConfig conf;
		if(node.trans_table_->mode_ == TransTable::Mode::MODE_ROUTE)
		{
			if(!config_route_table(node, conf))
			{
				node.trans_table_->mode_ = TransTable::Mode::MODE_POINT_TO_POINT;
				LOG_INFO << "Rank" << info.prime_rank_ << " config route failed. fall back peer to peer" << std::endl;
			}
		}
		else
		{
			for(int rank = 0; rank < info.prime_size_; rank++)
			{
				if(rank == info.prime_rank_)
					continue;

				assert(conf.recving_ranks_.emplace(std::make_pair(rank, rank)).second);
				assert(conf.sending_ranks_.insert(rank).second);
			}
		}
		
		send_init<T, T2>(node, conf, route_records, requests);
		recv_init<T, T2>(node, conf, route_records, requests);
	}
	MPICHECK(snn_group_sync(info));
	check_init(node);
	MPICHECK(snn_group_sync(info));

	LOG_INFO << "==========After snn init=============" << endl;
	report_dev_info();
	report_mem_info();
}

static void snn_run_report(const MPIInfo& info,
							const bool has_freq,
							const bool freq_is_char,
							unsigned char* freqs,
							const int* stat_freq_recvcounts,
							const int* stat_freq_displs,
							const bool has_vmean,
							float* vmeans,
							const bool have_receptor_imeans,
							float* ampa_imeans,
							float* nmda_imeans,
							float* gabaa_imeans,
							float* gabab_imeans,
							const bool has_imean,
							float* imeans,
							const int* stat_recvcounts,
							const int* stat_displs,
							const bool has_sample_spike,
							const bool has_sample_vmemb,
							const bool has_sample_iou,
							const bool has_sample_isynaptic,
							char* spikes,
							float* vmembs,
							float* isynaptics,
							float* ious,
							const int* sample_recvcounts,
							const int* sample_displs)
{
	if(has_freq)
	{
		assert(NULL != freqs);
		MPICHECK(snn_gatherv(info, NULL, 0,
				MPI_UNSIGNED_CHAR, freqs, stat_freq_recvcounts, stat_freq_displs, MPI_UNSIGNED_CHAR));
	}
	
	if(has_vmean)
	{
		assert(NULL != vmeans);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, vmeans, stat_recvcounts, stat_displs, MPI_FLOAT));
			
	}

	if(has_imean)
	{
		assert(NULL != imeans);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, imeans, stat_recvcounts, stat_displs, MPI_FLOAT));
	}

	if(have_receptor_imeans)
	{
		assert(NULL != ampa_imeans);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, ampa_imeans, stat_recvcounts, stat_displs, MPI_FLOAT));
			
		assert(NULL != nmda_imeans);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, nmda_imeans, stat_recvcounts, stat_displs, MPI_FLOAT));
			

		assert(NULL != gabaa_imeans);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, gabaa_imeans, stat_recvcounts, stat_displs, MPI_FLOAT));
		
		assert(NULL != gabab_imeans);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, gabab_imeans, stat_recvcounts, stat_displs, MPI_FLOAT));
	}

	if(has_sample_spike)
	{
		assert(NULL != spikes);
		MPICHECK(snn_gatherv(info, NULL, 0,
				MPI_CHAR, spikes, sample_recvcounts, sample_displs, MPI_CHAR));
	}
	
	if(has_sample_vmemb)
	{
		assert(NULL != vmembs);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, vmembs, sample_recvcounts, sample_displs, MPI_FLOAT));
	}

	if(has_sample_isynaptic)
	{
		assert(NULL != isynaptics);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, isynaptics, sample_recvcounts, sample_displs, MPI_FLOAT));	
	}

	if(has_sample_iou)
	{
		assert(NULL != ious);
		MPICHECK(snn_gatherv(info, NULL, 0,
						MPI_FLOAT, ious, sample_recvcounts, sample_displs, MPI_FLOAT));
	}
}


template<typename T, typename T2>
static void snn_run_report(NodeInfo<T, T2>& node)
{
	high_resolution_clock::time_point time_start;
	duration<double> diff;
	
	for(int i = 0; i < node.iter_; i++)
	{
		shared_ptr<RunReportInfo> report = node.reporting_queue_.pop();
		assert(nullptr != report);

		time_start = high_resolution_clock::now();

		if(node.has_freq_)
		{
			assert(nullptr != report->freqs_);
			MPICHECK(snn_gatherv(*node.info_, report->freqs_->cpu_data(), static_cast<int>(report->freqs_->count()),
					MPI_UNSIGNED_CHAR, NULL, NULL, NULL, MPI_UNSIGNED_CHAR));
		}
		
		if(node.has_vmean_)
		{
			assert(nullptr != report->vmeans_);
			MPICHECK(snn_gatherv(*node.info_, report->vmeans_->cpu_data(), static_cast<int>(report->vmeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
		}

		if(node.has_imean_)
		{
			assert(nullptr != report->imeans_);
			MPICHECK(snn_gatherv(*node.info_, report->imeans_->cpu_data(), static_cast<int>(report->imeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
		}

		if(node.have_receptor_imeans_)
		{
			assert(nullptr != report->ampa_imeans_);
			MPICHECK(snn_gatherv(*node.info_, report->ampa_imeans_->cpu_data(), static_cast<int>(report->ampa_imeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));

			assert(nullptr != report->nmda_imeans_);
			MPICHECK(snn_gatherv(*node.info_, report->nmda_imeans_->cpu_data(), static_cast<int>(report->nmda_imeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));

			assert(nullptr != report->gabaa_imeans_);
			MPICHECK(snn_gatherv(*node.info_, report->gabaa_imeans_->cpu_data(), static_cast<int>(report->gabaa_imeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));

			assert(nullptr != report->gabab_imeans_);
			MPICHECK(snn_gatherv(*node.info_, report->gabab_imeans_->cpu_data(), static_cast<int>(report->gabab_imeans_->count()),
							MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
		}

		if(nullptr == node.samples_)
		{
			if(node.has_sample_spike_)
			{
				assert(nullptr == report->spikes_);
				MPICHECK(snn_gatherv(*node.info_, NULL, 0,
						MPI_CHAR, NULL, NULL, NULL, MPI_CHAR));
			}

			if(node.has_sample_vmemb_)
			{
				MPICHECK(snn_gatherv(*node.info_, NULL,  0,
								MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));		
			}

			if(node.has_sample_isynaptic_)
			{
				assert(nullptr == report->isynaptics_);
				MPICHECK(snn_gatherv(*node.info_, NULL,  0,
								MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
			}

			if(node.has_sample_iou_)
			{
				assert(nullptr == report->ious_);
				MPICHECK(snn_gatherv(*node.info_, NULL, 0,
								MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
			}
		}
		else
		{
			if(node.has_sample_spike_)
			{
				assert(nullptr != report->spikes_);
				MPICHECK(snn_gatherv(*node.info_, report->spikes_->cpu_data(), static_cast<int>(report->spikes_->count()),
						MPI_CHAR, NULL, NULL, NULL, MPI_CHAR));
			}

			if(node.has_sample_vmemb_)
			{
				assert(nullptr != report->vmembs_);
				MPICHECK(snn_gatherv(*node.info_, report->vmembs_->cpu_data(),  static_cast<int>(report->vmembs_->count()),
								MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
			}

			if(node.has_sample_isynaptic_)
			{
				assert(nullptr != report->isynaptics_);
				MPICHECK(snn_gatherv(*node.info_, report->isynaptics_->cpu_data(),  static_cast<int>(report->isynaptics_->count()),
								MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
			}

			if(node.has_sample_iou_)
			{
				assert(nullptr != report->ious_);
				MPICHECK(snn_gatherv(*node.info_, report->ious_->cpu_data(), static_cast<int>(report->ious_->count()),
								MPI_FLOAT, NULL, NULL, NULL, MPI_FLOAT));
			}
		}
		
		node.reporting_free_queue_.push(report);

		diff = duration_cast<duration<double>>(high_resolution_clock::now() - time_start);
		LOG_INFO << "the " << i << "th report cost time: " << diff.count() << std::endl;
		
	}
	
	node.reporting_free_queue_.clear();
	MPICHECK(snn_sync(*node.info_));
	
}

static void snn_run(const MPIInfo& info,
					int& iter,
					int& iter_offset,
					bool& has_freq,
					bool& freq_is_char,
					bool& has_vmean,
					bool& has_sample_spike,
					bool& has_sample_vmemb,
					bool& has_sample_iou,
					bool& has_sample_isynaptic,
					bool& has_isynapse,
					bool& use_ou_background_stimuli,
					int& t_steps,
					bool& has_receptor_isynapse)
{
	vector<int> record(13);
		
	if(MPI_MASTER_RANK == info.rank_)
	{
		record[0] = iter;
		record[1] = iter_offset;
		record[2] = has_freq;
		record[3] = freq_is_char;
		record[4] = has_vmean;
		record[5] = has_sample_spike;
		record[6] = has_sample_vmemb;
		record[7] = has_sample_iou;
		record[8] = has_sample_isynaptic;
		record[9] = has_isynapse;
		record[10] = use_ou_background_stimuli;
		record[11] = t_steps;
		record[12] = has_receptor_isynapse;
	}

	MPICHECK(MPI_Bcast(record.data(), record.size(), MPI_INT, MPI_MASTER_RANK, info.comm_));

	if(MPI_MASTER_RANK != info.rank_)
	{
		iter = record[0];
		iter_offset = record[1];
		has_freq = record[2];
		freq_is_char = record[3];
		has_vmean = record[4];
		has_sample_spike = record[5];
		has_sample_vmemb = record[6];
		has_sample_iou = record[7];
		has_sample_isynaptic = record[8];
		has_isynapse = record[9];
		use_ou_background_stimuli = record[10];
		t_steps = record[11];
		has_receptor_isynapse = record[12];
	}
}

template<typename T, typename T2>
static void snn_send(NodeInfo<T, T2>& node)
{	
	int request_idx = 0;
	for(auto index : node.trans_table_->send_indice_)
	{
		int rank, tag;
		auto& record = node.trans_table_->send_records_[index];
		if(node.trans_table_->is_routing_mark(record.rank_))
		{
			rank = node.trans_table_->get_native_mark(record.rank_);
			int offset = 0;
			unsigned int* buff = std::get<0>(record.buffer_);
			for(auto sidx : record.merged_set_)
			{
				int len = static_cast<int>(node.block_->f_sending_active_rowptrs_->cpu_data()[sidx + 1] - node.block_->f_sending_active_rowptrs_->cpu_data()[sidx]);
				if(len > 0)
				{
					buff[offset++] = len + 1;
					buff[offset++] = ID2RANK(node.block_->f_sending_bids_[sidx]);
					memcpy(buff + offset, node.block_->f_sending_active_colinds_->cpu_data() + node.block_->f_sending_active_rowptrs_->cpu_data()[sidx], len * sizeof(unsigned int));
					offset += len;
				}
			}
			std::get<1>(record.buffer_) = offset;
			assert(offset <= record.max_len_);
			tag = node.trans_table_->make_routing_mark(node.info_->prime_rank_);
		}
		else
		{
			rank = record.rank_;
			tag = node.info_->prime_rank_;
			assert(1 == record.merged_set_.size());
			int sidx = *record.merged_set_.begin();
			std::get<0>(record.buffer_) = node.block_->f_sending_active_colinds_->mutable_cpu_data() + node.block_->f_sending_active_rowptrs_->cpu_data()[sidx];
			std::get<1>(record.buffer_) = static_cast<int>(node.block_->f_sending_active_rowptrs_->cpu_data()[sidx + 1] - node.block_->f_sending_active_rowptrs_->cpu_data()[sidx]);
		}
		
		MPICHECK(MPI_Isend(std::get<0>(record.buffer_),
						std::get<1>(record.buffer_),
						MPI_UNSIGNED,
						rank,
						tag,
						node.info_->prime_comm_,
						&node.trans_table_->send_requests_[request_idx++]));
	}

	for(unsigned int i = 0; i < node.trans_table_->route_records_for_queue_.size(); i++)
	{
		int tag = node.routing_queue_.pop();
		auto route_it = node.trans_table_->route_records_for_queue_.find(tag);
		assert(route_it != node.trans_table_->route_records_for_queue_.end());
		int native_tag = node.trans_table_->get_native_mark(tag);
		for(auto& record : route_it->second)
		{
			MPICHECK(MPI_Isend(std::get<0>(record.buffer_),
						std::get<1>(record.buffer_),
						MPI_UNSIGNED,
						record.rank_,
						native_tag,
						node.info_->prime_comm_,
						&node.trans_table_->send_requests_[request_idx++]));
		}
	}

	assert(request_idx == static_cast<int>(node.trans_table_->send_requests_.size()));
	MPICHECK(MPI_Waitall(node.trans_table_->send_requests_.size(), node.trans_table_->send_requests_.data(), MPI_STATUSES_IGNORE));

}

template<typename T, typename T2>
static void snn_recv(NodeInfo<T, T2>& node)
{
	int recv_count = 0;
	MPI_Status status;
	int elems;
	
	while(recv_count < node.trans_table_->recv_count_)
	{
		MPICHECK(MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, node.info_->prime_comm_, &status));
		int rank = status.MPI_SOURCE;
		int tag = status.MPI_TAG;
		void* buff;
		int buff_len;

		if(node.trans_table_->is_routing_mark(tag))
		{
			auto it = node.trans_table_->route_records_.find(tag);
			assert(it != node.trans_table_->route_records_.end() && rank == it->second.rank_);
			buff = std::get<0>(it->second.buffer_);
			buff_len =  it->second.max_len_;
			
			MPICHECK(MPI_Recv(buff,
						buff_len,
						MPI_UNSIGNED,
						rank,
						tag,
						node.info_->prime_comm_,
						&status));
			MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
			std::get<1>(it->second.buffer_) = elems;

			node.trans_table_->parse_merged_payload(node.info_->prime_rank_, tag, it->second);
			node.routing_queue_.push(tag);
		}
		else
		{
			auto it = node.trans_table_->recv_records_.find(tag);
			assert(it != node.trans_table_->recv_records_.end() && rank == it->second.rank_);
			buff = std::get<0>(it->second.buffer_);
			buff_len =  it->second.max_len_;
			
			MPICHECK(MPI_Recv(buff,
						buff_len,
						MPI_UNSIGNED,
						rank,
						tag,
						node.info_->prime_comm_,
						&status));
			MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
			std::get<1>(it->second.buffer_) = elems;
		}

		recv_count++;
	}
}

template<typename T, typename T2>
static void snn_run(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;	
	assert(MPI_MASTER_RANK != info.rank_);
	assert(node.reporting_queue_.empty());
	int t_steps;
	snn_run(info, node.iter_, node.iter_offset_, node.has_freq_, node.freq_is_char_, node.has_vmean_, 
		    node.has_sample_spike_, node.has_sample_vmemb_, node.has_sample_iou_, node.has_sample_isynaptic_,
		    node.has_imean_, node.use_ou_background_stimuli_, t_steps, node.have_receptor_imeans_);
	{
		node.block_->init_random_state();
		node.block_->set_t_step(static_cast<unsigned int>(t_steps));
		node.block_->update_Tao_constant_gpu();
		node.block_->set_freqs(node.has_freq_, node.freq_is_char_);
		node.block_->set_vmeans(node.has_vmean_);
		node.block_->set_imeans(node.has_imean_);
		node.block_->set_receptor_imeans(node.have_receptor_imeans_);

		if(nullptr != node.samples_)
		{
			if(node.has_sample_spike_)
			{
				assert(node.samples_->count() > 0);
				if(nullptr == node.spikes_)
				{
					node.spikes_ = make_unique<DataAllocator<char>>(node.info_->rank_, sizeof(char) * node.samples_->count());
					assert(nullptr != node.spikes_);
					node.spikes_->gpu_data();
				}
				else if(node.samples_->count() != static_cast<int>(node.spikes_->count()))
				{
					node.spikes_.reset(new DataAllocator<char>(node.info_->rank_, sizeof(char) * node.samples_->count()));
					assert(nullptr != node.spikes_);
					node.spikes_->gpu_data();
				}
			}
			else if(nullptr != node.spikes_)
			{
				node.spikes_.reset(nullptr);
			}
			
			if(node.has_sample_vmemb_)
			{
				if(nullptr == node.vmembs_)
				{
					node.vmembs_ = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * node.samples_->count());
					assert(nullptr != node.vmembs_);
					node.vmembs_->gpu_data();
				}
				else if(node.samples_->count() != static_cast<int>(node.vmembs_->count()))
				{
					node.vmembs_.reset(new DataAllocator<float>(node.info_->rank_, sizeof(float) * node.samples_->count()));
					assert(nullptr != node.vmembs_);
					node.vmembs_->gpu_data();
				}
			}
			else if(nullptr != node.vmembs_)
			{
				node.vmembs_.reset(nullptr);
			}
		
			if(node.has_sample_isynaptic_)
			{
				assert(node.samples_->count() > 0);
				if(nullptr == node.isynaptics_)
	            {
	            	node.isynaptics_ = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * node.samples_->count());
					assert(nullptr != node.isynaptics_);
					node.isynaptics_->gpu_data();
				}
				else if(node.samples_->count() != node.isynaptics_->count())
				{
	                node.isynaptics_.reset(new DataAllocator<float>(node.info_->rank_, sizeof(float) * node.samples_->count()));
					assert(nullptr != node.isynaptics_);
					node.isynaptics_->gpu_data();
	            }
			}
			else if(nullptr != node.isynaptics_)
			{
				node.isynaptics_.reset(nullptr);
			}
				
			if(node.has_sample_iou_)
			{
				assert(node.samples_->count() > 0);
				if(nullptr == node.ious_)
				{
					 node.ious_ = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * node.samples_->count());
					 assert(nullptr != node.ious_);
					 node.ious_->gpu_data();
				}
				else if(node.samples_->count() != node.ious_->count())
	            {
	                node.ious_.reset(new DataAllocator<float>(node.info_->rank_, sizeof(float) * node.samples_->count()));
					assert(nullptr != node.ious_);
					node.ious_->gpu_data();
				}
			}
			else if(nullptr != node.ious_)
			{
				node.ious_.reset(nullptr);
			}
		}
	}

	shared_ptr<RunReportInfo> report;
	auto fetch_report = [&node, t_steps]() -> shared_ptr<RunReportInfo>{
		if(!node.reporting_free_queue_.empty())
		{
			return node.reporting_free_queue_.pop();
		}
		
		shared_ptr<RunReportInfo> report = make_shared<RunReportInfo>();
		if(node.has_freq_)
		{
			if(node.freq_is_char_)
			{
				report->freqs_ = make_unique<DataAllocator<unsigned char>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(unsigned char), false);
			}
			else
			{
				report->freqs_ = make_unique<DataAllocator<unsigned char>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(unsigned int), false);
			}
			assert(nullptr != report->freqs_);

			report->freqs_->cpu_data();
		}
					
		if(node.has_vmean_)
		{
			report->vmeans_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(float), false);
			assert(nullptr != report->vmeans_);

			report->vmeans_->cpu_data();
		}

		if(node.has_imean_)
		{
			report->imeans_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(float), false);
			assert(nullptr != report->imeans_);

			report->imeans_->cpu_data();
		}

		if(node.have_receptor_imeans_)
		{
			report->ampa_imeans_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(float), false);
			assert(nullptr != report->ampa_imeans_);

			report->ampa_imeans_->cpu_data();

			report->nmda_imeans_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(float), false);
			assert(nullptr != report->nmda_imeans_);

			report->nmda_imeans_->cpu_data();

			report->gabaa_imeans_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(float), false);
			assert(nullptr != report->gabaa_imeans_);

			report->gabaa_imeans_->cpu_data();

			report->gabab_imeans_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.block_->get_total_subblocks() * sizeof(float), false);
			assert(nullptr != report->gabab_imeans_);

			report->gabab_imeans_->cpu_data();
		}

		if(nullptr != node.samples_)
		{
			
			if(node.has_sample_spike_)
			{
				report->spikes_= make_unique<DataAllocator<char>>(node.info_->rank_, node.spikes_->size(), false);
				assert(nullptr != report->spikes_);
				report->spikes_->cpu_data();
			}

			if(node.has_sample_vmemb_)
			{
				report->vmembs_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.vmembs_->size(), false);
				assert(nullptr != report->vmembs_);
				report->vmembs_->cpu_data();
			}

			if(node.has_sample_isynaptic_)
			{
				report->isynaptics_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.isynaptics_->size(), false);
				assert(nullptr != report->isynaptics_);
				report->isynaptics_->cpu_data();
			}

			if(node.has_sample_iou_)
			{
				report->ious_ = make_unique<DataAllocator<float>>(node.info_->rank_, node.ious_->size(), false);
				assert(nullptr != report->ious_);
				report->ious_->cpu_data();
			}
		}
		
		return report;
	};

	std::thread sending_thread;
    Notification sending_notification;
    Notification sending_done_notification;
    if(!node.trans_table_->send_requests_.empty())
    {
        sending_thread = std::thread(
                [&node, &sending_notification, &sending_done_notification]()
		{
			for(int i = 0; i < node.iter_; i++)
            {
               sending_notification.Wait();
			   snn_send(node);
			   sending_done_notification.Notify();
			}
    	});
    }

	std::thread recving_thread;
    Notification recving_notification;
	Notification recving_done_notification;
    if(node.trans_table_->recv_count_ > 0)
	{   
        recving_thread = std::thread(
                [&node, &recving_notification, &recving_done_notification]()
        {
            for(int i = 0; i < node.iter_; i++)
            {
               recving_notification.Wait();
			   snn_recv(node);
			   recving_done_notification.Notify();
			}
        });
    }
	
	unsigned int timestamp_offset = 0;
	if(node.iter_offset_ > 0)
	{
		if(node.block_->get_input_timestamp_size() > 0)
		{
			thrust::device_vector<unsigned int> d_timestamps(node.block_->get_input_timestamp_size());
			HIP_CHECK(hipMemcpy(thrust::raw_pointer_cast(d_timestamps.data()), node.block_->get_input_timestamp_data(), node.block_->get_input_timestamp_size() * sizeof(unsigned int), hipMemcpyHostToDevice));
			thrust::device_vector<unsigned int>::iterator it = thrust::lower_bound(d_timestamps.begin(), d_timestamps.end(), static_cast<unsigned int>(node.iter_offset_));
			timestamp_offset = it - d_timestamps.begin();
		}
	}

	node.reporting_notification_.Notify();
	MPICHECK(snn_sync(info));

	for(int i = 0; i < node.iter_; i++)
	{
		timestamp_offset = node.block_->update_F_input_spike_gpu(static_cast<unsigned int>(i + node.iter_offset_),timestamp_offset);

		if(!node.use_ou_background_stimuli_ || t_steps > 1)
		{
			node.block_->reset_F_recorded_actives_gpu();
		}
		
		if(node.trans_table_->recv_count_ > 0)
		{
			recving_notification.Notify();
		}

		int loop = 0;
		do{
			if(node.use_ou_background_stimuli_)
			{
				node.block_->update_I_ou_background_gpu();
			}

			node.block_->update_I_ttype_ca_gpu();
			node.block_->update_I_dopamine_gpu();
			node.block_->update_I_adaptation_gpu();
			node.block_->update_V_membrane_gpu();

			if(t_steps > 1)
			{
				assert(node.use_ou_background_stimuli_);
				node.block_->update_F_accumulated_spike_gpu();
				if(loop < (t_steps - 1))
				{
					node.block_->update_J_presynaptic_per_step_gpu();
					node.block_->update_I_synaptic_gpu();
					node.block_->update_H_ttype_ca_gpu();
					node.block_->update_ca_concentration_gpu();
				}
			}
			HIP_CHECK(hipDeviceSynchronize());
			loop++;
		} while(loop < t_steps);
			
		if(!node.use_ou_background_stimuli_)
		{
			node.block_->update_F_active_gpu();
			node.block_->update_F_sending_actives_gpu();
			node.block_->update_ca_concentration_gpu();
		}
		else
		{
			if(t_steps > 1)
			{
				node.block_->update_F_sending_actives_gpu();
			}
			else
			{
				node.block_->update_F_sending_actives_gpu(false);
			}

			node.block_->update_ca_concentration_gpu(false);
		}
		node.block_->update_H_ttype_ca_gpu();
		HIP_CHECK(hipDeviceSynchronize());
		
		node.block_->update_J_presynaptic_inner_gpu(true, node.stream1_->get());

		if(!node.trans_table_->send_requests_.empty())
		{
			HIP_CHECK(hipMemcpyAsync(node.block_->f_sending_active_rowptrs_->mutable_cpu_data(), node.block_->f_sending_active_rowptrs_->gpu_data(), node.block_->f_sending_active_rowptrs_->size(), hipMemcpyDeviceToHost, node.stream2_->get()));
			HIP_CHECK(hipStreamSynchronize(node.stream2_->get()));
			unsigned int count = node.block_->f_sending_active_rowptrs_->cpu_data()[node.block_->f_sending_rowptrs_->count() - 1];
			if(count > 0)
			{
				HIP_CHECK(hipMemcpyAsync(node.block_->f_sending_active_colinds_->mutable_cpu_data(),
										node.block_->f_shared_active_colinds_->gpu_data(),
										count * sizeof(unsigned int),
										hipMemcpyDeviceToHost,
										node.stream2_->get()));
				HIP_CHECK(hipStreamSynchronize(node.stream2_->get()));
			}
			
			sending_notification.Notify();
		}

		if(node.trans_table_->recv_count_ > 0)
		{
			recving_done_notification.Wait();
		
			node.block_->f_receiving_active_rowptrs_->mutable_cpu_data()[0] = 0;
			unsigned int total = 0;
			unsigned int j = 0;
			for(auto& record : node.trans_table_->recv_records_)
			{
				assert(record.first == ID2RANK(node.block_->f_receiving_bids_[j]));
				int count = std::get<1>(record.second.buffer_);
				assert(count <= (node.block_->f_receiving_rowptrs_->cpu_data()[j + 1] - node.block_->f_receiving_rowptrs_->cpu_data()[j]));
				node.block_->f_receiving_active_rowptrs_->mutable_cpu_data()[j + 1] = node.block_->f_receiving_active_rowptrs_->cpu_data()[j] + static_cast<unsigned int>(count);
				
				if(count > 0)
				{
					unsigned int* h_colinds = node.block_->f_receiving_active_colinds_->mutable_cpu_data() + node.block_->f_receiving_active_rowptrs_->cpu_data()[j];
					memcpy(h_colinds, std::get<0>(record.second.buffer_), static_cast<unsigned int>(count) * sizeof(unsigned int));
					assert(thrust::is_sorted(h_colinds, h_colinds + count));
					assert(h_colinds[count - 1] <  node.block_->f_receiving_colinds_->count());
					total += static_cast<unsigned int>(count);
				}
				j++;
			}
			
			assert(total <= node.block_->f_receiving_active_colinds_->count());			
			if(total > 0)
			{
				HIP_CHECK(hipMemcpyAsync(node.block_->f_receiving_active_rowptrs_->mutable_gpu_data(),
										node.block_->f_receiving_active_rowptrs_->cpu_data(),
										node.block_->f_receiving_active_rowptrs_->size(),
										hipMemcpyHostToDevice,
										node.stream2_->get()));
				HIP_CHECK(hipMemcpyAsync(node.block_->f_shared_active_colinds_->mutable_gpu_data(),
										node.block_->f_receiving_active_colinds_->cpu_data(),
										total * sizeof(unsigned int),
										hipMemcpyHostToDevice,
										node.stream2_->get()));
				
				node.block_->update_F_recving_actives_gpu(node.stream2_->get());
				HIP_CHECK(hipStreamSynchronize(node.stream2_->get()));
			}

			LOG_INFO << "receive count: " << total << std::endl;
		}
		
		node.block_->update_J_presynaptic_outer_gpu(node.stream1_->get());
		node.block_->update_J_presynaptic_gpu(node.stream1_->get());
        node.block_->update_I_synaptic_gpu(node.have_receptor_imeans_, node.stream1_->get());
           
		report = fetch_report();
		if(node.has_freq_)
		{
			node.block_->stat_Freqs_gpu(node.freq_is_char_, node.stream1_->get());
			HIP_CHECK(hipMemcpyAsync(report->freqs_->mutable_cpu_data(), node.block_->get_freqs_gpu(), report->freqs_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
		}
		
		if(node.has_vmean_)
		{
			node.block_->stat_Vmeans_gpu(node.stream1_->get());
			HIP_CHECK(hipMemcpyAsync(report->vmeans_->mutable_cpu_data(), node.block_->get_vmeans_gpu(), report->vmeans_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
		}
		
		if(node.has_imean_)
		{
			node.block_->stat_Imeans_gpu(node.stream1_->get());
			HIP_CHECK(hipMemcpyAsync(report->imeans_->mutable_cpu_data(), node.block_->get_imeans_gpu(), report->imeans_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
		}

		if(node.have_receptor_imeans_)
		{
			node.block_->stat_receptor_Imeans_gpu(node.stream1_->get());
			HIP_CHECK(hipMemcpyAsync(report->ampa_imeans_->mutable_cpu_data(), node.block_->get_ampa_imeans_gpu(), report->ampa_imeans_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			HIP_CHECK(hipMemcpyAsync(report->nmda_imeans_->mutable_cpu_data(), node.block_->get_nmda_imeans_gpu(), report->nmda_imeans_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			HIP_CHECK(hipMemcpyAsync(report->gabaa_imeans_->mutable_cpu_data(), node.block_->get_gabaa_imeans_gpu(), report->gabaa_imeans_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			HIP_CHECK(hipMemcpyAsync(report->gabab_imeans_->mutable_cpu_data(), node.block_->get_gabab_imeans_gpu(), report->gabab_imeans_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
		}

		if(nullptr != node.samples_)
		{
			char* d_spikes = NULL;
			float* d_vmembs = NULL;
			float* d_isynaptics = NULL;
			float* d_ious = NULL;
			bool use_record = false;

			if(node.has_sample_spike_)
			{
				assert(nullptr != node.spikes_);
				d_spikes = node.spikes_->mutable_gpu_data();
			}

			if(node.has_sample_vmemb_)
			{
				assert(nullptr != node.vmembs_);
				d_vmembs = node.vmembs_->mutable_gpu_data();
			}

			if(node.has_sample_isynaptic_)
			{
				assert(nullptr != node.isynaptics_);
				d_isynaptics = node.isynaptics_->mutable_gpu_data();
			}
			
			if(node.has_sample_iou_)
			{
				assert(nullptr != node.ious_);
				d_ious = node.ious_->mutable_gpu_data();
			}

			if(!node.use_ou_background_stimuli_ || t_steps > 1)
			{
				use_record = true;
			}

			if(NULL != d_spikes ||
				NULL != d_vmembs ||
				NULL != d_isynaptics ||
				NULL != d_ious)			
			{
				node.block_->stat_Samples_gpu(node.samples_->gpu_data(),
										node.samples_->count(),
										d_spikes,
										d_vmembs,
										d_isynaptics,
										use_record,
										d_ious,
										node.stream1_->get());
			}
			
			if(NULL != d_spikes)
			{
				HIP_CHECK(hipMemcpyAsync(report->spikes_->mutable_cpu_data(), d_spikes, node.spikes_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			}

			if(NULL != d_vmembs)
			{
				HIP_CHECK(hipMemcpyAsync(report->vmembs_->mutable_cpu_data(), d_vmembs, node.vmembs_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			}

			if(NULL != d_isynaptics)
			{
				HIP_CHECK(hipMemcpyAsync(report->isynaptics_->mutable_cpu_data(), d_isynaptics, node.isynaptics_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			}
			
			if(NULL != d_ious)
			{
				HIP_CHECK(hipMemcpyAsync(report->ious_->mutable_cpu_data(), d_ious, node.ious_->size(), hipMemcpyDeviceToHost, node.stream1_->get()));
			}
		}
		
		HIP_CHECK(hipDeviceSynchronize());
		node.reporting_queue_.push(report);	

		if(!node.trans_table_->send_requests_.empty())
		{
			sending_done_notification.Wait();
		}
		MPICHECK(snn_group_sync(info));		
	}

	if(!node.trans_table_->send_requests_.empty())
	{
		if (sending_thread.joinable())
		{
			sending_thread.join();
		}
	}

	if(node.trans_table_->recv_count_ > 0)
	{
		if (recving_thread.joinable())
		{
			recving_thread.join();
		}
	}
}

static void snn_update_prop(const MPIInfo& info,
								const int rank,
								const unsigned int* neuron_indice,
								const unsigned int* prop_indice,
								const float* prop_vals,
								const int n)
{
	int tag = info.size_ + Tag::TAG_UPDATE_PROP;
	vector<MPI_Request> requests(3);
	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	
	MPICHECK(MPI_Isend(neuron_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
	
	tag++;
	MPICHECK(MPI_Isend(prop_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[1]));
	
	tag++;
	MPICHECK(MPI_Isend(prop_vals, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
	
	MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
}

template<typename T, typename T2>
static void snn_update_prop(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_PROP;
	int elems;
	MPI_Status status;
	
	unique_ptr<DataAllocator<unsigned int>> neuron_indice = nullptr;
	unique_ptr<DataAllocator<unsigned int>> prop_indice = nullptr;
	unique_ptr<DataAllocator<float>> prop_vals = nullptr;

	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		neuron_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		MPICHECK(MPI_Recv(neuron_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == neuron_indice->count());
		HIP_CHECK(hipMemcpy(neuron_indice->mutable_gpu_data(), neuron_indice->cpu_data(), neuron_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		prop_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		MPICHECK(MPI_Recv(prop_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == prop_indice->count());
		HIP_CHECK(hipMemcpy(prop_indice->mutable_gpu_data(), prop_indice->cpu_data(), prop_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
	
	if(elems > 0)
	{
		prop_vals = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * elems);
		MPICHECK(MPI_Recv(prop_vals->mutable_cpu_data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == prop_vals->count());
		HIP_CHECK(hipMemcpy(prop_vals->mutable_gpu_data(), prop_vals->cpu_data(), prop_vals->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == 0);	
	}

	if(elems > 0)
	{
		assert(neuron_indice->count() == elems &&
			prop_indice->count() == elems &&
			prop_vals->count() == elems);
		node.block_->update_Props_gpu(neuron_indice->gpu_data(),
									neuron_indice->cpu_data(),
									prop_indice->gpu_data(),
									prop_indice->cpu_data(),
									prop_vals->gpu_data(),
									prop_vals->cpu_data(),
									static_cast<unsigned int>(elems));
		HIP_CHECK(hipDeviceSynchronize());
	}
}

static void snn_update_gamma(const MPIInfo& info,
									const int rank,
									const unsigned int* prop_indice,
									const unsigned int* brain_indice,
									const float* alphas,
									const float* betas,
									const unsigned int n)
{
	int tag = info.size_ + Tag::TAG_UPDATE_GAMMA;
	vector<MPI_Request> requests(4);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);

	MPICHECK(MPI_Isend(prop_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
	
	tag++;
	MPICHECK(MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[1]));
	
	tag++;
	MPICHECK(MPI_Isend(alphas, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
	
	tag++;
	MPICHECK(MPI_Isend(betas, n, MPI_FLOAT, rank, tag, info.comm_, &requests[3]));
	
	MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
}

template<typename T, typename T2>
static void snn_update_gamma(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_GAMMA;
	int elems, n;
	MPI_Status status;
	
	unique_ptr<DataAllocator<unsigned int>> prop_indice = nullptr;
	unique_ptr<DataAllocator<unsigned int>> brain_indice = nullptr;
	unique_ptr<DataAllocator<float>> alphas = nullptr;
	unique_ptr<DataAllocator<float>> betas = nullptr;

	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	n = elems; 
	
	if(elems > 0)
	{
		node.prop_indice_.resize(elems);
		prop_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		MPICHECK(MPI_Recv(node.prop_indice_.data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == prop_indice->count());
		HIP_CHECK(hipMemcpy(prop_indice->mutable_gpu_data(), node.prop_indice_.data(), prop_indice->size(), hipMemcpyHostToDevice));
		node.prop_indice_.clear();
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	assert(n == elems);
	
	if(elems > 0)
	{
		node.brain_indice_.resize(elems);
		brain_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		MPICHECK(MPI_Recv(node.brain_indice_.data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == brain_indice->count());
		HIP_CHECK(hipMemcpy(brain_indice->mutable_gpu_data(), node.brain_indice_.data(), brain_indice->size(), hipMemcpyHostToDevice));
		node.brain_indice_.clear();
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		
	assert(n == elems);

	if(elems > 0)
	{
		alphas = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * elems);
		vector<float> buff(elems);
		MPICHECK(MPI_Recv(buff.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == alphas->count());
		HIP_CHECK(hipMemcpy(alphas->mutable_gpu_data(), buff.data(), alphas->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == 0);
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
	assert(n == elems);
	
	if(elems > 0)
	{
		betas = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * elems);
		vector<float> buff(elems);
		MPICHECK(MPI_Recv(buff.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == alphas->count());
		HIP_CHECK(hipMemcpy(betas->mutable_gpu_data(), buff.data(), betas->size(), hipMemcpyHostToDevice));
	}
	else
	{		
		MPICHECK(MPI_Recv(NULL, elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == 0);		
	}

	if(n > 0)
	{
		node.block_->update_Gamma_Prop_Cols_gpu(prop_indice->gpu_data(),
												prop_indice->cpu_data(),
												brain_indice->gpu_data(),
												brain_indice->cpu_data(),
												alphas->gpu_data(),
												betas->gpu_data(),
												n);
		
		HIP_CHECK(hipDeviceSynchronize());
	}
	
}

static void snn_update_hyperpara(const MPIInfo& info,
									const int rank,
									const unsigned int* prop_indice,
									const unsigned int* brain_indice,
									const float* hyperpara_vals,
									const int n,
									bool assigned)
{
	int tag = info.size_ + Tag::TAG_UPDATE_HYPERPARA;
	vector<MPI_Request> requests(4);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	MPICHECK(MPI_Isend(prop_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
	
	tag++;
	MPICHECK(MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[1]));
	
	tag++;
	MPICHECK(MPI_Isend(hyperpara_vals, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
	
	tag++;
	MPICHECK(MPI_Isend(&assigned, 1, MPI_INT, rank, tag, info.comm_, &requests[3]));
		
	MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
}

template<typename T, typename T2>
static void snn_update_hyperpara(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_HYPERPARA;
	int elems;
	bool assigned;
	MPI_Status status;
	
	unique_ptr<DataAllocator<unsigned int>> prop_indice = nullptr;
	unique_ptr<DataAllocator<unsigned int>> brain_indice = nullptr;
	unique_ptr<DataAllocator<float>> hyperpara_vals = nullptr;
	
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		prop_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		MPICHECK(MPI_Recv(prop_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		HIP_CHECK(hipMemcpy(prop_indice->mutable_gpu_data(), prop_indice->cpu_data(), prop_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, 0, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		assert(prop_indice!= nullptr && elems == static_cast<int>(prop_indice->count()));
		brain_indice = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		MPICHECK(MPI_Recv(brain_indice->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		HIP_CHECK(hipMemcpy(brain_indice->mutable_gpu_data(), brain_indice->cpu_data(), brain_indice->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, 0, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
	}

	tag++;
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
	
	if(elems > 0)
	{
		assert(brain_indice!= nullptr && elems == static_cast<int>(brain_indice->count()));
		hyperpara_vals = make_unique<DataAllocator<float>>(node.info_->rank_, sizeof(float) * elems);
		MPICHECK(MPI_Recv(hyperpara_vals->mutable_cpu_data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		HIP_CHECK(hipMemcpy(hyperpara_vals->mutable_gpu_data(), hyperpara_vals->cpu_data(), hyperpara_vals->size(), hipMemcpyHostToDevice));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, 0, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
	}

	tag++;
	MPICHECK(MPI_Recv(&assigned, 1, MPI_INT, MPI_MASTER_RANK, tag, info.comm_, &status));
	
	if(prop_indice != nullptr)
	{
		if(assigned)
		{
			node.block_->assign_Prop_Cols_gpu(prop_indice->gpu_data(),
									prop_indice->cpu_data(),
									brain_indice->gpu_data(),
									brain_indice->cpu_data(),
									hyperpara_vals->gpu_data(),
									static_cast<unsigned int>(elems));
		}
		else
		{
			node.block_->update_Prop_Cols_gpu(prop_indice->gpu_data(),
									prop_indice->cpu_data(),
									brain_indice->gpu_data(),
									brain_indice->cpu_data(),
									hyperpara_vals->gpu_data(),
									static_cast<unsigned int>(elems));
		}
		HIP_CHECK(hipDeviceSynchronize());
	}
}

static void snn_update_sample(MPIInfo& info,
								const int rank,
								const unsigned int* sample_indice,
								const int n)
{
	int tag = info.size_ + Tag::TAG_UPDATE_SAMPLE;

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);		
	MPICHECK(MPI_Send(sample_indice, n, MPI_UNSIGNED, rank, tag, info.comm_));
}


template<typename T, typename T2>
static void snn_update_sample(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_SAMPLE;
	int elems;
	MPI_Status status;
	
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		if(nullptr == node.samples_)
		{
			node.samples_ = make_unique<DataAllocator<unsigned int>>(node.info_->rank_, sizeof(unsigned int) * elems);
		}
		else if(elems != static_cast<int>(node.samples_->count()))
		{
			node.samples_.reset(new DataAllocator<unsigned int>(node.info_->rank_, sizeof(unsigned int) * elems));
		}

		MPICHECK(MPI_Recv(node.samples_->mutable_cpu_data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == node.samples_->count());

		HIP_CHECK(hipMemcpy(node.samples_->mutable_gpu_data(), node.samples_->cpu_data(), node.samples_->size(), hipMemcpyDeviceToHost));
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, 0, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		if(nullptr != node.samples_)
		{
			node.samples_.reset(nullptr);
		}
	}
}


static void snn_update_ou_background_param(const MPIInfo& info,
													const int rank,
													const unsigned int* brain_indice,
													const unsigned int n,
                                                    const float* mean,
                                                    const float* deviation,
                                                    const float* correlation_time)
{
	int tag = info.size_ + Tag::TAG_UPDATE_OU_BACKGROUND_PARAM;
	vector<MPI_Request> requests(4);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	
	if(n > 0)
	{
		MPICHECK(MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		tag++;
		MPICHECK(MPI_Isend(mean, n, MPI_FLOAT, rank, tag, info.comm_, &requests[1]));
		tag++;
		MPICHECK(MPI_Isend(deviation, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
		tag++;
		MPICHECK(MPI_Isend(correlation_time, n, MPI_FLOAT, rank, tag, info.comm_, &requests[3]));
		MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}
	else
	{
		MPICHECK(MPI_Isend(NULL, 0, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		MPICHECK(MPI_Wait(&requests[0], MPI_STATUS_IGNORE));
	}
}

template<typename T, typename T2>
static void snn_update_ou_background_param(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_OU_BACKGROUND_PARAM;
	int elems;
	MPI_Status status;
	std::vector<unsigned int> brain_indice;
	std::vector<float> means;
	std::vector<float> deviations;
	std::vector<float> correlation_times;
	
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		brain_indice.resize(elems);
		MPICHECK(MPI_Recv(brain_indice.data(), elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		means.resize(elems);
		MPICHECK(MPI_Recv(means.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		deviations.resize(elems);
		MPICHECK(MPI_Recv(deviations.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		correlation_times.resize(elems);
		MPICHECK(MPI_Recv(correlation_times.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		node.block_->set_I_ou_current_param(brain_indice, means, deviations, correlation_times);
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}
}

static void snn_update_ttype_ca_current_param(const MPIInfo& info,
									const int rank,
									const unsigned int* brain_indice,
									const unsigned int n,
									const float* h_init_vals,
									const float* g_ts,
									const float* tao_h_minuss,
									const float* tao_h_pluss,
									const float* v_hs,
									const float* v_ts)
{
	int tag = info.size_ + Tag::TAG_UPDATE_TTYPE_CA_CURRENT_PARAM;
	vector<MPI_Request> requests(7);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	if(n > 0)
	{
		MPICHECK(MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		tag++;
		MPICHECK(MPI_Isend(h_init_vals, n, MPI_FLOAT, rank, tag, info.comm_, &requests[1]));
		tag++;
		MPICHECK(MPI_Isend(g_ts, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
		tag++;
		MPICHECK(MPI_Isend(tao_h_minuss, n, MPI_FLOAT, rank, tag, info.comm_, &requests[3]));
		tag++;
		MPICHECK(MPI_Isend(tao_h_pluss, n, MPI_FLOAT, rank, tag, info.comm_, &requests[4]));
		tag++;
		MPICHECK(MPI_Isend(v_hs, n, MPI_FLOAT, rank, tag, info.comm_, &requests[5]));
		tag++;
		MPICHECK(MPI_Isend(v_ts, n, MPI_FLOAT, rank, tag, info.comm_, &requests[6]));
		MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}
	else
	{
		MPICHECK(MPI_Isend(NULL, 0, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		MPICHECK(MPI_Wait(&requests[0], MPI_STATUS_IGNORE));
	}
	
}

template<typename T, typename T2>
static void snn_update_ttype_ca_current_param(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_TTYPE_CA_CURRENT_PARAM;
	int elems;
	MPI_Status status;
	std::vector<unsigned int> brain_indice;
	std::vector<float> h_init_vals;
	std::vector<float> g_ts;
	std::vector<float> tao_h_minuss;
	std::vector<float> tao_h_pluss;
	std::vector<float> v_hs;
	std::vector<float> v_ts;
	
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		brain_indice.resize(elems);
		MPICHECK(MPI_Recv(brain_indice.data(), brain_indice.size(), MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		h_init_vals.resize(elems);
		MPICHECK(MPI_Recv(h_init_vals.data(), h_init_vals.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		g_ts.resize(elems);
		MPICHECK(MPI_Recv(g_ts.data(), g_ts.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		tao_h_minuss.resize(elems);
		MPICHECK(MPI_Recv(tao_h_minuss.data(), tao_h_minuss.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		tao_h_pluss.resize(elems);
		MPICHECK(MPI_Recv(tao_h_pluss.data(), tao_h_pluss.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		v_hs.resize(elems);
		MPICHECK(MPI_Recv(v_hs.data(), v_hs.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		v_ts.resize(elems);
		MPICHECK(MPI_Recv(v_ts.data(), v_ts.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		node.block_->set_I_ttype_ca_param(brain_indice, h_init_vals, g_ts, tao_h_minuss, tao_h_pluss, v_hs, v_ts);
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}
}


static void snn_update_dopamine_current_param(const MPIInfo& info,
									const int rank,
									const unsigned int* brain_indice,
									const unsigned int n,
									const float* v_das,
									const float* g_das)
{
	int tag = info.size_ + Tag::TAG_UPDATE_DOPAMINE_CURRENT_PARAM;
	vector<MPI_Request> requests(3);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);

	if(n > 0)
	{
		MPICHECK(MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));

		tag++;
		MPICHECK(MPI_Isend(v_das, n, MPI_FLOAT, rank, tag, info.comm_, &requests[1]));

		tag++;
		MPICHECK(MPI_Isend(g_das, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
		
		MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}
	else
	{
		MPICHECK(MPI_Isend(NULL, 0, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		MPICHECK(MPI_Wait(&requests[0], MPI_STATUS_IGNORE));
	}
}

template<typename T, typename T2>
static void snn_update_dopamine_current_param(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_DOPAMINE_CURRENT_PARAM;
	int elems;
	MPI_Status status;
	std::vector<unsigned int> brain_indice;
	std::vector<float> v_das;
	std::vector<float> g_das;
	
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		brain_indice.resize(elems);
		MPICHECK(MPI_Recv(brain_indice.data(), brain_indice.size(), MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		v_das.resize(elems);
		MPICHECK(MPI_Recv(v_das.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		g_das.resize(elems);
		MPICHECK(MPI_Recv(g_das.data(), elems, MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		node.block_->set_I_dopamine_param(brain_indice, v_das, g_das);
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, 0, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}
}

static void snn_update_adaptation_current_param(const MPIInfo& info,
														const int rank,
														const unsigned int* brain_indice,
														const unsigned int n,
														const float* ca_init_vals,
														const float* ca_decays,
														const float* alpha_constants,
														const float* v_ks,
														const float* g_ahps)
{
	int tag = info.size_ + Tag::TAG_UPDATE_TTYPE_CA_CURRENT_PARAM;
	vector<MPI_Request> requests(6);

	assert(MPI_MASTER_RANK == info.rank_ && MPI_MASTER_RANK != rank);
	if(n > 0)
	{
		MPICHECK(MPI_Isend(brain_indice, n, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		tag++;
		MPICHECK(MPI_Isend(ca_init_vals, n, MPI_FLOAT, rank, tag, info.comm_, &requests[1]));
		tag++;
		MPICHECK(MPI_Isend(ca_decays, n, MPI_FLOAT, rank, tag, info.comm_, &requests[2]));
		tag++;
		MPICHECK(MPI_Isend(alpha_constants, n, MPI_FLOAT, rank, tag, info.comm_, &requests[3]));
		tag++;
		MPICHECK(MPI_Isend(v_ks, n, MPI_FLOAT, rank, tag, info.comm_, &requests[4]));
		tag++;
		MPICHECK(MPI_Isend(g_ahps, n, MPI_FLOAT, rank, tag, info.comm_, &requests[5]));
		MPICHECK(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
	}
	else
	{
		MPICHECK(MPI_Isend(NULL, 0, MPI_UNSIGNED, rank, tag, info.comm_, &requests[0]));
		MPICHECK(MPI_Wait(&requests[0], MPI_STATUS_IGNORE));
	}
	
}

template<typename T, typename T2>
static void snn_update_adaptation_current_param(NodeInfo<T, T2>& node)
{
	MPIInfo& info = *node.info_;
	int tag = info.size_ + Tag::TAG_UPDATE_ADAPTATION_CURRENT_PARAM;
	int elems;
	MPI_Status status;
	std::vector<unsigned int> brain_indice;
	std::vector<float> ca_init_vals;
	std::vector<float> ca_decays;
	std::vector<float> alpha_constants;
	std::vector<float> v_ks;
	std::vector<float> g_ahps;
	
	MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
	MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
	if(elems > 0)
	{
		brain_indice.resize(elems);
		MPICHECK(MPI_Recv(brain_indice.data(), brain_indice.size(), MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		ca_init_vals.resize(elems);
		MPICHECK(MPI_Recv(ca_init_vals.data(), ca_init_vals.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		ca_decays.resize(elems);
		MPICHECK(MPI_Recv(ca_decays.data(), ca_decays.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		alpha_constants.resize(elems);
		MPICHECK(MPI_Recv(alpha_constants.data(), alpha_constants.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		v_ks.resize(elems);
		MPICHECK(MPI_Recv(v_ks.data(), v_ks.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));

		tag++;
		MPICHECK(MPI_Probe(MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_FLOAT, &elems));
		assert(elems == brain_indice.size());
		g_ahps.resize(elems);
		MPICHECK(MPI_Recv(g_ahps.data(), g_ahps.size(), MPI_FLOAT, MPI_MASTER_RANK, tag, info.comm_, &status));
		
		node.block_->set_I_adaptation_param(brain_indice, ca_init_vals, ca_decays, alpha_constants, v_ks, g_ahps);
	}
	else
	{
		MPICHECK(MPI_Recv(NULL, elems, MPI_UNSIGNED, MPI_MASTER_RANK, tag, info.comm_, &status));
		MPICHECK(MPI_Get_count(&status, MPI_UNSIGNED, &elems));
		assert(elems == 0);
	}
}

static void snn_done(MPIInfo& info, int* total = NULL)
{
	int done = 1;
	MPICHECK(MPI_Reduce(&done, total, 1, MPI_INT, MPI_SUM, MPI_MASTER_RANK, info.comm_));
}

class SnnImpl final : public Snn::Service
{
 public:
 	SnnImpl(const int rank, const int size, const MPI_Comm comm, 
			const int prime_rank, const int prime_size, const MPI_Comm prime_comm)
	:info_(new MPIInfo(rank, size, comm, prime_rank, prime_size, prime_comm)),
	total_subblocks_(0),
	total_samples_(0),
	stat_recvcounts_(size),
	stat_displs_(size),
	stat_recvcounts_in_bytes_(size),
	stat_displs_in_bytes_(size),
	sample_recvcounts_(size),
	sample_displs_(size)
 	{
 	}

	Status Init(ServerContext* context, const InitRequest* request, ServerWriter<InitResponse>* writer) override
	{
		string path = request->file_path();
		float delta_t = request->delta_t();
		int comm_mode = request->mode();
		
		cout << "Path: " << path << endl;
		cout << "Delta time: " << delta_t << endl;
		cout << "Comm mode: " << ((comm_mode == InitRequest::COMM_P2P)
								? "POINT TO POINT" : "ROUTE") << endl;
		
		if(path.empty())
		{
			InitResponse response;
			response.set_status(SnnStatus::SNN_INVALID_PARAMETER);
			writer->Write(response);
			return Status::OK;
		}

		Command cmd = SNN_INIT;
		MPICHECK(wait_handle(cmd, *info_));
		snn_init(*info_, path, delta_t, comm_mode);
		
		vector<int> neurons_per_block(info_->size_);
		vector<int> subids;
		vector<int> subcounts;

		snn_init_report(*info_,
						stat_recvcounts_.data(),
						stat_displs_.data(),
						neurons_per_block.data(),
						subids,
						subcounts);
		{
			thrust::constant_iterator<int> constant(4);
			thrust::multiplies<int> op;
			thrust::transform(stat_recvcounts_.data(), stat_recvcounts_.data() + stat_recvcounts_.size(), constant, stat_recvcounts_in_bytes_.data(), op);
			thrust::transform(stat_displs_.data(), stat_displs_.data() + stat_displs_.size(), constant, stat_displs_in_bytes_.data(), op);
		}
		
		total_subblocks_ = subids.size();
		cout << "Total subblocks: " << total_subblocks_ << endl;
		if(0 == total_subblocks_)
		{
			InitResponse response;
			response.set_status(SnnStatus::SNN_INVALID_NETWORK);
			writer->Write(response);
			return Status::OK;
		}

		int idx = 0;
		for(int i = 1; i < info_->size_; i++)
		{
			InitResponse response;
			response.set_status(SnnStatus::SNN_OK);
			response.set_block_id(i - 1);
			response.set_neurons_per_block(neurons_per_block[i]);
			SubblockInfo* info;
			for(int j = 0; j < stat_recvcounts_[i]; j++)
			{
				info = response.add_subblk_info();
				info->set_subblk_id(subids[idx]);
				info->set_subblk_num(subcounts[idx]);
				idx++;
			}
			writer->Write(response);
		}
		
		return Status::OK;
	}

	Status Run(ServerContext* context, const RunRequest* request, ServerWriter<RunResponse>* writer) override
	{
		duration<double> diff;
		auto time_start = high_resolution_clock::now();

		iter_ = request->iter();
		int iter_offset = request->iter_offset();
		bool has_freq = request->output_freq();
		bool freq_is_char = request->freq_char();
		bool has_vmean = request->output_vmean();
		bool has_sample_spike = request->output_sample_spike();
		bool has_sample_vmemb = request->output_sample_vmemb();
		bool has_sample_iou = request->output_sample_iou();
		bool has_sample_isynaptic = request->output_sample_isynaptic();
		bool has_imean = request->output_imean();
		bool use_ou_background = request->use_ou_background();
		bool have_receptor_imeans = request->output_receptor_imeas();
		int t_steps = request->t_steps();

		if(iter_ <= 0 ||
			(has_sample_iou && !use_ou_background) ||
			((has_sample_spike || has_sample_vmemb || has_sample_iou || has_sample_isynaptic) && (0 == total_samples_)) ||
			(!use_ou_background && t_steps > 1))
		{
			RunResponse response;
			response.set_status(SnnStatus::SNN_INVALID_PARAMETER);
			writer->Write(response);
			return Status::OK;
		}

		cout << "[Run]: " << endl;
		cout << "Iteration: " << iter_ << endl;
		cout << "Iteration offset: " << iter_offset << endl;
		cout << "Output frequencies: " << has_freq << endl;
		cout << "freq is char: " << freq_is_char << endl;
		cout << "Output vmeans: " << has_vmean << endl;
		cout << "Output sample spikes: " << has_sample_spike << endl;
		cout << "Output sample vmembs: " << has_sample_vmemb << endl;
		cout << "Output sample ious: " << has_sample_iou << endl;
		cout << "Output sample isynaptics: " << has_sample_isynaptic << endl;
		cout << "Output imeans: " << has_imean << endl;
		cout << "Use OU background: " << use_ou_background << endl;
		cout << "T steps: " << t_steps << endl;
		
		vector<unsigned char> freqs;
		vector<float> vmeans;
		vector<char> spikes;
		vector<float> vmembs;
		vector<float> isynaptics;
		vector<float> ious;
		vector<float> ampa_imeans;
		vector<float> nmda_imeans;
		vector<float> gabaa_imeans;
		vector<float> gabab_imeans;
		vector<float> imeans;
		RunResponse response;
		response.set_status(SnnStatus::SNN_OK);

		if(has_freq)
		{
			assert(total_subblocks_ > 0);
			if(freq_is_char)
			{
				freqs.resize(total_subblocks_);
			}
			else
			{
				freqs.resize(total_subblocks_ * sizeof(unsigned int));
			}
			response.add_freq(freqs.data(), freqs.size());
		}

		if(has_vmean)
		{
			assert(total_subblocks_ > 0);
			vmeans.resize(total_subblocks_);
			response.add_vmean(vmeans.data(), vmeans.size() * sizeof(float));
			
		}
		
		if(has_sample_spike)
		{
			spikes.resize(total_samples_);
			response.add_sample_spike(spikes.data(), spikes.size());
		}
		
		if(has_sample_vmemb)
		{
			vmembs.resize(total_samples_);
			response.add_sample_vmemb(vmembs.data(), vmembs.size() * sizeof(float));
		}

		if(has_sample_isynaptic)
		{
			isynaptics.resize(total_samples_);
			response.add_sample_isynaptic(isynaptics.data(), isynaptics.size() * sizeof(float));
		}
		
		if(has_sample_iou)
		{
			ious.resize(total_samples_);
			response.add_sample_iou(ious.data(), ious.size() * sizeof(float));
		}

		if(has_imean)
		{
			assert(total_subblocks_ > 0);
			imeans.resize(total_subblocks_);
			response.add_imean(imeans.data(), imeans.size() * sizeof(float));
		}

		if(have_receptor_imeans)
		{
			assert(total_subblocks_ > 0);
			ampa_imeans.resize(total_subblocks_);
			response.add_ampa_imean(ampa_imeans.data(), ampa_imeans.size() * sizeof(float));

			nmda_imeans.resize(total_subblocks_);
			response.add_nmda_imean(nmda_imeans.data(), nmda_imeans.size() * sizeof(float));

			gabaa_imeans.resize(total_subblocks_);
			response.add_gabaa_imean(gabaa_imeans.data(), gabaa_imeans.size() * sizeof(float));

			gabab_imeans.resize(total_subblocks_);
			response.add_gabab_imean(gabab_imeans.data(), gabab_imeans.size() * sizeof(float));
		}

		Command cmd = SNN_RUN;
		MPICHECK(wait_handle(cmd, *info_));
		snn_run(*info_, iter_, iter_offset, has_freq, freq_is_char, has_vmean, has_sample_spike, has_sample_vmemb, has_sample_iou, has_sample_isynaptic, has_imean, use_ou_background, t_steps, have_receptor_imeans);
		MPICHECK(snn_sync(*info_));

		auto time_end = high_resolution_clock::now();
		diff = duration_cast<duration<double>>(time_end - time_start);
		std::cout << "===initializing time of run stage: " << diff.count() << std::endl;
		time_start = time_end;
		
		for(int i = 0; i < iter_; i++)
		{
			if(freq_is_char)
			{
				snn_run_report(*info_,
								has_freq,
								freq_is_char,
								freqs.data(),
								stat_recvcounts_.data(),
								stat_displs_.data(),
								has_vmean,
								vmeans.data(),
								have_receptor_imeans,
								ampa_imeans.data(),
								nmda_imeans.data(),
								gabaa_imeans.data(),
								gabab_imeans.data(),
								has_imean,
								imeans.data(),
								stat_recvcounts_.data(),
								stat_displs_.data(),
								has_sample_spike,
								has_sample_vmemb,
								has_sample_iou,
								has_sample_isynaptic,
								spikes.data(),
								vmembs.data(),
								isynaptics.data(),
								ious.data(),
								sample_recvcounts_.data(),
								sample_displs_.data());
			}
			else
			{
				snn_run_report(*info_,
								has_freq,
								freq_is_char,
								freqs.data(),
								stat_recvcounts_in_bytes_.data(),
								stat_displs_in_bytes_.data(),
								has_vmean,
								vmeans.data(),
								have_receptor_imeans,
								ampa_imeans.data(),
								nmda_imeans.data(),
								gabaa_imeans.data(),
								gabab_imeans.data(),
								has_imean,
								imeans.data(),
								stat_recvcounts_.data(),
								stat_displs_.data(),
								has_sample_spike,
								has_sample_vmemb,
								has_sample_iou,
								has_sample_isynaptic,
								spikes.data(),
								vmembs.data(),
								isynaptics.data(),
								ious.data(),
								sample_recvcounts_.data(),
								sample_displs_.data());
			}
			
			if(has_freq)
			{
				response.set_freq(0, freqs.data(), freqs.size());
			}

			if(has_vmean)
			{
				response.set_vmean(0, vmeans.data(), vmeans.size() * sizeof(float));
			}
			
			if(has_sample_spike)
			{
				response.set_sample_spike(0, spikes.data(), spikes.size());
			}
			
			if(has_sample_vmemb)
			{
				response.set_sample_vmemb(0, vmembs.data(), vmembs.size() * sizeof(float));
			}

			if(has_sample_isynaptic)
			{
				response.set_sample_isynaptic(0, isynaptics.data(), isynaptics.size() * sizeof(float));
			}
			
			if(has_sample_iou)
			{
				response.set_sample_iou(0, ious.data(), ious.size() * sizeof(float));
			}

			if(has_imean)
			{
				response.set_imean(0, imeans.data(), imeans.size() * sizeof(float));
			}
			
			if(have_receptor_imeans)
			{
				response.set_ampa_imean(0, ampa_imeans.data(), ampa_imeans.size() * sizeof(float));
				response.set_nmda_imean(0, nmda_imeans.data(), nmda_imeans.size() * sizeof(float));
				response.set_gabaa_imean(0, gabaa_imeans.data(), gabaa_imeans.size() * sizeof(float));
				response.set_gabab_imean(0, gabab_imeans.data(), gabab_imeans.size() * sizeof(float));
			}
			if(i < (iter_ - 1))
			{
				writer->Write(response);
			}
		}

		MPICHECK(snn_sync(*info_));
		writer->Write(response);

		diff = duration_cast<duration<double>>(high_resolution_clock::now() - time_start);
                std::cout << "===runing time of run stage: " << diff.count() << std::endl;

		return Status::OK;
	}

	Status Updateprop(ServerContext* context, ServerReader<UpdatePropRequest>* reader,
                     UpdatePropResponse* response) override
    {
		UpdatePropRequest prop;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));
		
		Command cmd = SNN_UPDATE_PROP;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (reader->Read(&prop))
		{
			assert(prop.neuron_id_size() == prop.prop_id_size() &&
				prop.neuron_id_size() == prop.prop_val_size());
			
			int bid = prop.block_id();
			int n = prop.neuron_id_size();
			
			if(1 == has_bids[bid])
				continue;

			snn_update_prop(*info_,
							bid + 1,
							reinterpret_cast<const unsigned int*>(prop.neuron_id().data()),
							reinterpret_cast<const unsigned int*>(prop.prop_id().data()),
							prop.prop_val().data(),
							n);
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_prop(*info_,
								i + 1,
								NULL,
								NULL,
								NULL,
								0);
			}
		}
		
		{
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}

		response->set_success(true);
		return Status::OK;
	}

	Status Updategamma(ServerContext* context, ServerReader<UpdateGammaRequest>* reader,
                     UpdateGammaResponse* response) override
    {
		UpdateGammaRequest request;
		vector<int> has_bids(info_->size_ - 1, 0);

		Command cmd = SNN_UPDATE_GAMMA;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (reader->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));
			int n = request.prop_id_size();
			assert(n == request.brain_id_size() &&
				n == request.gamma_concentration_size() &&
				n == request.gamma_rate_size());

			if(1 == has_bids[bid])
				continue;
			
			snn_update_gamma(*info_,
							bid + 1,
							reinterpret_cast<const unsigned int*>(request.prop_id().data()),
							reinterpret_cast<const unsigned int*>(request.brain_id().data()),
							request.gamma_concentration().data(),
							request.gamma_rate().data(),
							n);
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_gamma(*info_,
								i + 1,
								NULL,
								NULL,
								NULL,
								NULL,
								0);
			}
		}
		
	    {
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status Updatehyperpara(ServerContext* context, ServerReader<UpdateHyperParaRequest>* reader,
                     UpdateHyperParaResponse* response) override
	{
		UpdateHyperParaRequest hp;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));

		Command cmd = SNN_UPDATE_HYPERPARA;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);
		
		while (reader->Read(&hp))
		{
			assert(hp.prop_id_size() == hp.brain_id_size() &&
				hp.prop_id_size() == hp.hpara_val_size());

			int bid = hp.block_id();
			int n = hp.prop_id_size();
			bool assigned = hp.assigned();
			
			if(1 == has_bids[bid])
				continue;
			
			snn_update_hyperpara(*info_,
								bid + 1,
								reinterpret_cast<const unsigned int*>(hp.prop_id().data()),
								reinterpret_cast<const unsigned int*>(hp.brain_id().data()),
								hp.hpara_val().data(),
								n,
								assigned);
			
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			bool assigned = false;
			if(0 == has_bids[i])
			{
				snn_update_hyperpara(*info_,
								i + 1,
								NULL,
								NULL,
								NULL,
								0,
								assigned);
			}
		}

		{
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status Updatesample(ServerContext* context, ServerReader<UpdateSampleRequest>* reader,
                     						UpdateSampleResponse* response) override
    {
    	UpdateSampleRequest sample;
		int total = 0;
		vector<int> has_bids(info_->size_ - 1);
		memset(has_bids.data(), 0, has_bids.size() * sizeof(int));
		memset(sample_recvcounts_.data(), 0, sample_recvcounts_.size() * sizeof(int));

		Command cmd = SNN_SAMPLE;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);
			
		while(reader->Read(&sample))
		{
			int bid = sample.block_id();
			assert(bid < (info_->size_ - 1));
			int n = sample.sample_idx_size();
			if(1 == has_bids[bid])
				continue;

			snn_update_sample(*info_,
							bid + 1,
							reinterpret_cast<const unsigned int*>(sample.sample_idx().data()),
							n);
			
			has_bids[bid] = 1;
			sample_recvcounts_[bid + 1] = n;
			total += n;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_sample(*info_, i + 1, NULL, 0);
			}
		}

		thrust::exclusive_scan(sample_recvcounts_.begin(), sample_recvcounts_.end(), sample_displs_.begin());
		total_samples_ = total;
		cout << "number of sample: " << total_samples_ << endl;
		{	
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}

		response->set_success(true);
		return Status::OK;
	}

	Status Updateoubackgroundparam(ServerContext* context, ServerReader<UpdateOUBackgroundParamRequest>* reader, UpdateOUBackgroundParamResponse* response) override
	{
		UpdateOUBackgroundParamRequest request;
		vector<int> has_bids(info_->size_ - 1, 0);

		Command cmd = SNN_UPDATE_OU_BACKGROUND_PARAM;
		MPICHECK(wait_handle(cmd, *info_));
		
		while (reader->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));
			assert(request.brain_id_size() == request.mean_size() &&
				request.mean_size() == request.deviation_size() &&
				request.deviation_size() == request.correlation_time_size());
			
			if(1 == has_bids[bid])
				continue;

			snn_update_ou_background_param(*info_, 
											bid + 1,
											reinterpret_cast<const unsigned int*>(request.brain_id().data()),
											request.brain_id_size(),
											request.mean().data(),
											request.deviation().data(),
											request.correlation_time().data());
			has_bids[bid] = 1;
		}
		
		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_ou_background_param(*info_,
								i + 1,
								NULL,
								0,
								NULL,
								NULL,
								NULL);
			}
		}
		
		{
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}

		response->set_success(true);
		return Status::OK;
	}

	Status Updatettypecacurrentparam(ServerContext* context,  ServerReader<UpdateTTypeCaCurrentParamRequest>* reader, UpdateTTypeCaCurrentParamResponse* response) override
	{
		UpdateTTypeCaCurrentParamRequest request;
		vector<int> has_bids(info_->size_ - 1, 0);
		Command cmd = SNN_UPDATE_TType_CA_CURRENT_PARAM;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (reader->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));

			if(1 == has_bids[bid])
				continue;

			assert(request.brain_id_size() == request.h_init_size() &&
				request.h_init_size() == request.g_t_size() &&
				request.g_t_size() == request.tao_h_minus_size() &&
				request.tao_h_minus_size() == request.tao_h_plus_size() &&
				request.tao_h_plus_size() == request.v_h_size() &&
				request.v_h_size() == request.v_t_size());
			
			snn_update_ttype_ca_current_param(*info_,
								bid + 1,
								reinterpret_cast<const unsigned int*>(request.brain_id().data()),
								request.brain_id_size(),
								request.h_init().data(),
								request.g_t().data(),
								request.tao_h_minus().data(),
								request.tao_h_plus().data(),
								request.v_h().data(),
								request.v_t().data());
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_ttype_ca_current_param(*info_,
								i + 1,
								NULL,
								0,
								NULL,
								NULL,
								NULL,
								NULL,
								NULL,
								NULL);
			}
		}
		
	    {
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status Updatedopaminecurrentparam(ServerContext* context,  ServerReader<UpdateDopamineCurrentParamRequest>* reader, UpdateDopamineCurrentParamResponse* response) override
	{
		UpdateDopamineCurrentParamRequest request;
		vector<int> has_bids(info_->size_ - 1, 0);
		Command cmd = SNN_UPDATE_DOPAMINE_CURRENT_PARAM;
		MPICHECK(wait_handle(cmd, *info_));

		while (reader->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));

			assert(request.brain_id_size() == request.v_da_size() &&
				request.v_da_size() == request.g_da_size());

			if(1 == has_bids[bid])
				continue;
			
			snn_update_dopamine_current_param(*info_,
								bid + 1,
								reinterpret_cast<const unsigned int*>(request.brain_id().data()),
								request.brain_id_size(),
								request.v_da().data(),
								request.g_da().data());
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_dopamine_current_param(*info_,
								i + 1,
								NULL,
								0,
								NULL,
								NULL);
			}
		}
		
	    {
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status Updateadaptationcurrentparam(ServerContext* context,  ServerReader<UpdateAdaptationCurrentParamRequest>* reader, UpdateAdaptationCurrentParamResponse* response) override
	{
		UpdateAdaptationCurrentParamRequest request;
		vector<int> has_bids(info_->size_ - 1, 0);
		Command cmd = SNN_UPDATE_ADAPTATION_CURRENT_PARAM;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);

		while (reader->Read(&request))
		{
			int bid = request.block_id();
			assert(bid < (info_->size_ - 1));

			if(1 == has_bids[bid])
				continue;

			assert(request.brain_id_size() == request.ca_init_size() &&
				request.ca_init_size() == request.ca_decay_size() &&
				request.ca_decay_size() == request.alpha_constant_size() &&
				request.alpha_constant_size() == request.v_k_size() &&
				request.v_k_size() == request.g_ahp_size());
			
			snn_update_adaptation_current_param(*info_,
								bid + 1,
								reinterpret_cast<const unsigned int*>(request.brain_id().data()),
								request.brain_id_size(),
								request.ca_init().data(),
								request.ca_decay().data(),
								request.alpha_constant().data(),
								request.v_k().data(),
								request.g_ahp().data());
			
			has_bids[bid] = 1;
		}

		for(int i = 0; i < has_bids.size(); i++)
		{
			if(0 == has_bids[i])
			{
				snn_update_adaptation_current_param(*info_,
								i + 1,
								NULL,
								0,
								NULL,
								NULL,
								NULL,
								NULL,
								NULL);
			}
		}
		
	    {
			int dones = 0;
			snn_done(*info_, &dones);
			assert(dones == info_->size_);
		}
		
		response->set_success(true);
		return Status::OK;
	}

	Status GetVersion(ServerContext* context, const Empty* request, Version* response) override
	{
		static constexpr int32_t kDTB_VERSION_IMPL
                      = (DTB_MAJOR * 1000) + (DTB_MINOR * 100) + DTB_PATCH; // major, minor, patch
		response->set_version(kDTB_VERSION_IMPL);
		return Status::OK;
	}
	
	Status Shutdown(ServerContext* context, const ShutdownRequest* request, ShutdownResponse* response) override
	{
		Command cmd = SNN_SHUTDOWN;
		int err = wait_handle(cmd, *info_);
		assert(err == MPI_SUCCESS);
		
		cout << "Ready for shutdown server...\n";

		int dones = 0;
		snn_done(*info_, &dones);
		assert(dones == info_->size_);
		
		response->set_shutdown(true);
		promise_.set_value();
		
		return Status::OK;
	}

	void Setserver(shared_ptr<Server> server)
	{
		server_ = server;
	}

	shared_ptr<Server> Getserver()
	{
		return server_;
	}

	std::future<void> Getfuture()
	{
		return std::move(promise_.get_future());
	}
	
  private:
  	std::promise<void> promise_;
	
  	unique_ptr<MPIInfo> info_;
	shared_ptr<Server> server_;
	
	size_t total_subblocks_;
	size_t total_samples_;
	
	vector<int> stat_recvcounts_;
	vector<int> stat_displs_;

	vector<int> stat_recvcounts_in_bytes_;
	vector<int> stat_displs_in_bytes_;
	
	vector<int> sample_recvcounts_;
	vector<int> sample_displs_;

	int iter_;
};

static void server_shutdown(void* arg)
{
	SnnImpl* snn = reinterpret_cast<SnnImpl*>(arg);
	
	 auto shutdown_future = snn->Getfuture();
     if (shutdown_future.valid())
	 {
     	shutdown_future.get();
     }

	std::this_thread::sleep_for(std::chrono::seconds(1));
	snn->Getserver()->Shutdown();
	std::cout << "shutdown grpc server.\n";
}

template<typename T, typename T2>
static void node_handle(void* arg)
{
	NodeInfo<T, T2>* node = reinterpret_cast<NodeInfo<T, T2>*>(arg);
	bool quit = false;
	HIP_CHECK(hipSetDevice(node->gid_));
	do{
		int err = wait_handle(node->cmd_, *node->info_);
		assert(err == MPI_SUCCESS);
		switch(node->cmd_)
		{
			case SNN_INIT:
				snn_init<T, T2>(*node);
			break;
			case SNN_RUN:
				snn_run<T, T2>(*node);
			break;
			case SNN_UPDATE_PROP:
				snn_update_prop<T, T2>(*node);
			break;
			case SNN_UPDATE_GAMMA:
				snn_update_gamma<T, T2>(*node);
			break;
			case SNN_UPDATE_HYPERPARA:
				snn_update_hyperpara<T, T2>(*node);
			break;
			case SNN_SAMPLE:
				snn_update_sample<T, T2>(*node);
			break;
			case SNN_UPDATE_OU_BACKGROUND_PARAM:
				snn_update_ou_background_param<T, T2>(*node);
			break;
			case SNN_UPDATE_TType_CA_CURRENT_PARAM:
				snn_update_ttype_ca_current_param<T, T2>(*node);
			break;
			case SNN_UPDATE_DOPAMINE_CURRENT_PARAM:
				snn_update_dopamine_current_param<T, T2>(*node);
			break;
			case SNN_UPDATE_ADAPTATION_CURRENT_PARAM:
				snn_update_adaptation_current_param<T, T2>(*node);
			break;
			case SNN_SHUTDOWN:
				quit = true;
			break;
			default:
				assert(0);
			break;
		}

		if(SNN_RUN != node->cmd_)
		{
			node->reporting_notification_.Notify();
		}
	}while(!quit);
}

template<typename T, typename T2>
static void report_handle(void* arg)
{
	NodeInfo<T, T2>* node = reinterpret_cast<NodeInfo<T, T2>*>(arg);
	bool quit = false;
	do{
		
		node->reporting_notification_.Wait();
		
		switch(node->cmd_)
		{
			case SNN_INIT:
				snn_init_report<T, T2>(*node);
			break;
			
			case SNN_RUN:
				snn_run_report<T, T2>(*node);
				assert(node->reporting_queue_.empty());
			break;
			
			case SNN_UPDATE_PROP:
			case SNN_SAMPLE:
			case SNN_UPDATE_HYPERPARA:
			case SNN_UPDATE_OU_BACKGROUND_PARAM:
			case SNN_UPDATE_TType_CA_CURRENT_PARAM:
			case SNN_UPDATE_DOPAMINE_CURRENT_PARAM:
			case SNN_UPDATE_ADAPTATION_CURRENT_PARAM:
			case SNN_SHUTDOWN:
				snn_done(*node->info_);
				if(SNN_SHUTDOWN == node->cmd_)
					quit = true;
			break;
			default:
				assert(0);
			break;
		}
	}while(!quit);
}

static int get_ipaddr_by_hostname(const char *hostname, char *ip_addr, size_t size)
{
	struct addrinfo *result = NULL, hints;
	int ret = -1;
 
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_DGRAM;
	ret = getaddrinfo(hostname, NULL, &hints, &result);
 
	if (ret == 0)
	{
		struct in_addr addr = ((struct sockaddr_in *)result->ai_addr)->sin_addr;
		const char *re_ntop = inet_ntop(AF_INET, &addr, ip_addr, size);
		if (re_ntop == NULL)
			ret = -1;	
	}
 
	freeaddrinfo(result);
	return ret;
}

static int get_ibaddr_by_name(const char *ibname, char **ip_addr)
{
	int ret = -1;
 
	struct sockaddr_in  sin;
    struct ifreq        ifr;

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
	assert(fd != -1);

	strncpy(ifr.ifr_name, ibname, IFNAMSIZ - 1);      //Interface name

    if(ioctl(fd, SIOCGIFADDR, &ifr) == 0) 
	{
        memcpy(&sin, &ifr.ifr_addr, sizeof(ifr.ifr_addr));
        *ip_addr = inet_ntoa(sin.sin_addr);
		ret = 0;
    } 
	
	return ret;
}

static void get_host_name(char* hostname, int maxlen) 
{
	gethostname(hostname, maxlen);
	for (int i = 0; i < maxlen; i++) 
	{
		if (hostname[i] == '.') 
		{
		    hostname[i] = '\0';
		    return;
		}
	}
}

static uint64_t get_host_hash(const char* string) 
{
	// Based on DJB2, result = result * 33 + char
	uint64_t result = 5381;
	for (int c = 0; string[c] != '\0'; c++)
	{
		result = ((result << 5) + result) + string[c];
	}
	
	return result;
}

int main(int argc, char **argv)
{
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank, size;
	int prime_rank = -1, prime_size = -1;
	MPI_Comm prime_comm = MPI_COMM_NULL;
	
	int gpu_id;
	string device;
	int use_double_percision = get_cmdline_argint(argc, (const char**)argv, "use_double_percision");
	int check_memory = get_cmdline_argint(argc, (const char**)argv, "cm");
	char* log_path = NULL;
	get_cmdline_argstring(argc, (const char**)argv, "log", &log_path);
	set<int> rank_in_same_node;
	int max_rank_in_same_node;

	//double time_start;
	init_mpi_env(&argc, &argv, rank, gpu_id, size, device);
	if(rank == MPI_MASTER_RANK)
	{
		if(use_double_percision)
		{
			cout << "using double percision..." << endl;
		}
		else
		{
			cout << "using float percision..." << endl;
		}
		cout << "Init mpi env done ..." << endl;
	}
	
	{
		char hostname[1024];
		get_host_name(hostname, sizeof(hostname));

		vector<uint64_t> host_hashs(size);
  
		host_hashs[rank] = get_host_hash(hostname);
		MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashs.data(), sizeof(uint64_t), MPI_BYTE, comm));

		if(rank == MPI_MASTER_RANK)
		{
			cout << "MPI_Allgather done..." << endl;
		}

		MPICHECK(MPI_Barrier(comm));
		if(MPI_MASTER_RANK != rank)
		{
			for (int p = 1; p < size; p++) 
			{
				 if (p == rank)
				 {
				 	continue;
				 }

				 if (host_hashs[p] == host_hashs[rank])
				 {
				 	assert(rank_in_same_node.insert(p).second);
				 }
			}
		}
		int rank_size_in_node = static_cast<int>(rank_in_same_node.size()) + 1;
		MPICHECK(MPI_Allreduce(&rank_size_in_node, &max_rank_in_same_node, 1, MPI_INT, MPI_MAX, comm));

		if(rank != MPI_MASTER_RANK && check_memory)
		{
			HIP_CHECK(hipGetLastError());
			bool ret = report_gpu_mem(rank, hostname, 15.0);
			MPICHECK(MPI_Barrier(comm));
			if(!ret)
			{
				MPI_Abort(comm, MPI_ERR_NO_MEM);
				return 1;
			}
		}

		if(check_memory)
		{
		    bool ret = report_mem(rank, hostname, 88.0);
			MPICHECK(MPI_Barrier(comm));
			if(!ret)
			{
				MPI_Abort(comm, MPI_ERR_NO_MEM);
				return 1;
			}
		}

		cout << "The rank (" << rank << ") within the node " << hostname << "." << endl;
	}

	if(size > 2)
	{
		vector<int> ranks(size - 1);
		for(int i = 1; i < size; i++)
		{
			ranks[i - 1] = i;
		}
		
		MPI_Group word_group;
		MPICHECK(MPI_Comm_group(comm, &word_group));

		MPI_Group prime_group;
		MPICHECK(MPI_Group_incl(word_group, ranks.size(), ranks.data(), &prime_group));

		// `MPI_Comm_create` can be flaky in certain cases.
		constexpr int max_retries = 3;
		bool comm_updated = false;
		MPICHECK(MPI_Barrier(comm));
		for (int i = 0; i < max_retries; i++) 
		{
			if(MPI_SUCCESS == MPI_Comm_create(comm, prime_group, &prime_comm))
			{
				comm_updated = true;
				break;
			}
		}

		MPICHECK(MPI_Group_free(&word_group));
		MPICHECK(MPI_Group_free(&prime_group));
		assert(comm_updated);

		MPICHECK(MPI_Comm_rank(prime_comm, &prime_rank));
      	MPICHECK(MPI_Comm_size(prime_comm, &prime_size));

		if(rank == MPI_MASTER_RANK)
		{
			assert(prime_rank < 0 && 0 == prime_size);
		}
		else
		{
			assert(prime_rank == (rank - 1) && prime_size == (size - 1));
			assert(prime_comm != MPI_COMM_NULL);
		}
	}
	
	if(rank == MPI_MASTER_RANK)
	{
		assert(-1 == gpu_id && device.empty());
		string server_address;
		{
			char* ipaddr;
			int ret = get_ibaddr_by_name("ib0", &ipaddr);
			assert(!ret);
			server_address = string(ipaddr) + string(":50051");
		}
  		SnnImpl service(rank, size, comm, prime_rank, prime_size, prime_comm);

		ServerBuilder builder;
		builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
		builder.RegisterService(&service);
		builder.SetMaxMessageSize(INT_MAX);
		shared_ptr<Server> server(builder.BuildAndStart());
		cout << "Server listening on " << server_address << endl;
		service.Setserver(server);
		
		std::thread thrd(server_shutdown, &service);

		server->Wait();
		
		thrd.join();
	}
	else
	{
		{
			string logfile;
			if(NULL == log_path)
			{
				char link[1024];
	    		char exe_path[1024];
				sprintf(link, "/proc/%d/exe", getpid());
	    		int n = readlink(link, exe_path, sizeof(exe_path));
	    		exe_path[n] = '\0';
				string str(exe_path);
				n = str.rfind("/");
				logfile = str.substr(0, n) + string("/output_") + to_string(rank - 1) + string(".log");
			}
			else
			{
				logfile = string(log_path) + string("/output_") + to_string(rank - 1) + string(".log");
			}

			Logger::instance(logfile.c_str());
		}
		//LOG_SET_VERBOSE(1);
		if(use_double_percision)
		{
			shared_ptr<NodeInfo<double, double2>> shrd_node = make_shared<NodeInfo<double, double2>>(rank, size, comm, prime_rank, prime_size, prime_comm, gpu_id, device, rank_in_same_node, max_rank_in_same_node);
			std::thread thrd(report_handle<double, double2>, shrd_node.get());
			node_handle<double, double2>(shrd_node.get());
			thrd.join();
		}
		else
		{
			shared_ptr<NodeInfo<float, float2>> shrd_node = make_shared<NodeInfo<float, float2>>(rank, size, comm, prime_rank, prime_size, prime_comm, gpu_id, device, rank_in_same_node, max_rank_in_same_node);
			std::thread thrd(report_handle<float, float2>, shrd_node.get());
			node_handle<float, float2>(shrd_node.get());
			thrd.join();
		}
	}
	
	MPI_Finalize();
	DEVICE_RESET
	return 0;
}


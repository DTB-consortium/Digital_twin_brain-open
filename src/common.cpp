#include <sys/types.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include "common.hpp"
#include "logging.hpp"

namespace dtb {

unsigned int warp_size()
{
	hipDeviceProp_t props;
	int dev_id;
	HIP_CHECK(hipGetDevice(&dev_id));
   	HIP_CHECK(hipGetDeviceProperties(&props, dev_id));
	return props.warpSize;
}

unsigned long long gen_seed()
{
  unsigned long long s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  //System entropy source not available, using fallback algorithm to generate seed instead.
  if (f)
    fclose(f);

  pid = static_cast<unsigned long long>(getpid());
  s = static_cast<unsigned long long>(time(NULL));
  seed = static_cast<unsigned long long>(((s * 181) * ((pid - 83) * 359)) % 104729);
  
  return seed;
}

static double bytesToGB(size_t s)
{
	return (double)s / (1024.0 * 1024.0 * 1024.0); 
}
 
static size_t GBTobytes(double s)
{
	return (size_t)(s * 1024.0 * 1024.0 * 1024.0); 
}

bool report_gpu_mem(const int rank, const char* hostname, const double limit)
{
	const int w1 = 34;
	void* gpu_ptr;
	if(hipMalloc((void**)&gpu_ptr, GBTobytes(limit)))
    {
	    std::cout << std::setw(w1) << "rank " << rank << " ("  << hostname << ") GPU memeory less thean " << std::fixed << std::setprecision(2)
         << limit << " GB" << std::endl;
    	return false;
    }
	
	hipFree(gpu_ptr);
	return true;
}


bool report_mem(const int rank, const char* hostname, const double limit)
{
	const int w1 = 34;
	bool ret = true;
	size_t size = 0UL;
	FILE* file = fopen("/proc/meminfo", "r");
	char line[128];
	while (fgets(line, sizeof(line), file) != nullptr) 
	{
		if(strncmp(line, "SwapFree:", 9) == 0 ||
			strncmp(line, "MemAvailable:", 13) == 0)
		{
			size_t len = strlen(line);
			const char* p = line;
			while (*p <'0' || *p > '9') p++;
			line[len - 3] = '\0';
			size += (size_t)atol(p);
		}
	}

	fclose(file);

	if(bytesToGB(size * 1024) < limit)
	{
		std::cout << std::setw(w1) << "rank " << rank << " ("  << hostname << ") available memory: " << std::fixed << std::setprecision(2)
         << bytesToGB(size * 1024) << " GB" << std::endl;
		ret = false;
	}

	return ret;
}


void report_mem_info(double* used_mem) 
{
	size_t i = 0UL;
	int count = 0;
	FILE* file = fopen("/proc/self/status", "r");
	char line[128];
	while (fgets(line, sizeof(line), file) != nullptr) 
	{
		if(strncmp(line, "VmRSS:", 6) == 0 || strncmp(line, "VmPin:", 6) == 0)
		{
			size_t j = strlen(line);
			const char* p = line;
			while (*p <'0' || *p > '9') p++;
			line[j - 3] = '\0';
			i += (size_t)atol(p);
			count++;
			
			if(2 == count)
			{
				break;
			}
		}
	}
	fclose(file);

	if(used_mem)
	{
		*used_mem = bytesToGB(i);
		LOG_INFO << "Used CPU Memory: " << std::fixed << std::setprecision(2)
	         << *used_mem << " GB" << std::endl;
	}
	else
	{
		LOG_INFO << "Used CPU Memory: " << std::fixed << std::setprecision(2)
	         << bytesToGB(i) << " GB" << std::endl;
	}
}

void report_dev_info(double* used_mem, double* total_mem)
{
	const int w1 = 34;
	size_t free;
	size_t total;
	int device;
	HIP_CHECK(hipGetDevice(&device));

	std::stringstream ss;
	ss << std::left;
	ss << std::setw(w1)
		 << "--------------------------------------------------------------------------------"
		 << std::endl;
	ss << std::setw(w1) << "device#" << device << std::endl;

	hipDeviceProp_t props;
	HIP_CHECK(hipGetDeviceProperties(&props, device));
	HIP_CHECK(hipMemGetInfo(&free, &total));

	assert(total == props.totalGlobalMem);
	if(total_mem)
	{
		*total_mem = bytesToGB(total);

		if(used_mem)
		{
			*used_mem = *total_mem - bytesToGB(free);
		}

		ss << std::setw(w1) << "Total GPU Memory: " << std::fixed << std::setprecision(2)
	         << *total_mem << " GB" << std::endl;
	}
	else
	{
		ss << std::setw(w1) << "Total GPU Memory: " << std::fixed << std::setprecision(2)
	         << bytesToGB(total) << " GB" << std::endl;
	}

	if(used_mem)
	{
		ss << std::setw(w1) << "Used GPU Memory: " << std::fixed << std::setprecision(2)
	         << *used_mem << " GB" << std::endl;
	}
	else
	{
		ss << std::setw(w1) << "Used GPU Memory: " << std::fixed << std::setprecision(2)
	         << (bytesToGB(total) - bytesToGB(free)) << " GB" << std::endl;
	}

	LOG_INFO << std::endl << ss.str();
}


void check_malloc(hipError_t result, int rank, char const *const func, const char *const file, int const line)
{
	if(result)
    {
        fprintf(stderr, "Rank %d HIP error at %s:%d code=%d(%s) \"%s\" \n",
                rank, file, line, static_cast<unsigned int>(result), hipGetErrorString(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}


void check_hip(hipError_t result, char const *const func, const char *const file, int const line)
{
	if(result)
    {
        fprintf(stderr, "HIP error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), hipGetErrorString(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

void check_hiprand(hiprandStatus_t result, char const *const func, const char *const file, int const line)
{
    if(result)
    {
        fprintf(stderr, "HIP RAND error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), hiprand_cpp::error::to_string(result).c_str(), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

void check_kernel(const char* err_msg, const char* file, const int line)
{
	hipError_t err = hipGetLastError();

    if (hipSuccess != err)
    {
        fprintf(stderr, "%s(%i) : HIP error : %s : (%d) %s.\n",
                file, line, err_msg, (int)err, hipGetErrorString(err));
        DEVICE_RESET
        exit(EXIT_FAILURE);
    }
}

HipStream::HipStream(bool high_priority)
{
  if (high_priority) {
    int leastPriority, greatestPriority;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    HIP_CHECK(hipStreamCreateWithPriority(&stream_, hipStreamDefault, greatestPriority));
  } else {
    HIP_CHECK(hipStreamCreate(&stream_));
  }
}

HipStream::~HipStream()
{
  int current_device;  // Just to check CUDA status:
  hipError_t status = hipGetDevice(&current_device);
  if (stream_ != nullptr) {
    	HIP_CHECK(hipStreamDestroy(stream_));
  }
}

HipRandGenerator::HipRandGenerator(unsigned long long seed, hipStream_t stream)
{
	HIPRAND_CHECK(hiprandCreateGenerator(&handle_, HIPRAND_RNG_PSEUDO_DEFAULT));
    HIPRAND_CHECK(hiprandSetPseudoRandomGeneratorSeed(handle_, seed));
	HIPRAND_CHECK(hiprandSetGeneratorOffset(handle_, 0));
	if(stream != nullptr)
		HIPRAND_CHECK(hiprandSetStream(handle_, stream));
}

HipRandGenerator::~HipRandGenerator()
{
  int current_device;  // Just to check CUDA status:
  hipError_t status = hipGetDevice(&current_device);
  if (handle_ != nullptr) {
    HIPRAND_CHECK(hiprandDestroyGenerator(handle_));
  }
}

}//namespace istbi

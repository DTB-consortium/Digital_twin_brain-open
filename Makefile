# Location of the hip
HIP_PATH ?=  $(wildcard /opt/rocm/hip)

HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC =$(HIP_PATH)/bin/hipcc

LDFLAGS  :=

CXX ?= `which g++`

MPI_PATH := /opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1
#MPI_PATH := /opt/hpc/software/mpi/hpcx/v2.4.1/gcc-7.3.1
#MPICXX = $(MPI_PATH)/bin/mpic++
#MPICFLAGS = -DMPI_DEBUG -I$(MPI_PATH)/include 
MPICFLAGS = -I$(MPI_PATH)/include

PROTOS_PATH = ./protos
PROTOC = protoc
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

vpath %.proto $(PROTOS_PATH)


CXXFLAGS     := -std=c++14 -fPIC
HIPFLAGS := -m64 --offload-arch=gfx906

INCLUDES  := -I./include -I$(HIP_PATH)/../include -I$(HIP_PATH)/../rocrand/include -I$(HIP_PATH)/../hiprand/include -I$(HIP_PATH)/include -I$(HIP_PATH)/../include -I$(MPI_PATH)/include
LIBRARIES += -L$(HIP_PATH)/../hiprand/lib -lhiprand -L$(MPI_PATH)/lib -lmpi
################################################################################
SRCOBJS := src/common.o src/util/transpose.o src/util/cnpy.o src/util/cmd_arg.o src/data_allocator.o src/configuration.o src/route.o\
                   src/stage/membrane_voltage.o src/stage/presynaptic_voltage.o src/stage/random.o \
                   src/stage/spike.o src/stage/synaptic_current.o src/stage/stat_result.o src/stage/extern_stimuli.o \
                   src/stage/sample.o src/stage/property.o src/block.o src/logging.o src/weights.o
KERNELOBJS :=  src/stage/membrane_voltage.o src/stage/presynaptic_voltage.o src/stage/random.o \
                   src/stage/spike.o src/stage/synaptic_current.o src/stage/stat_result.o src/stage/extern_stimuli.o \
                   src/stage/sample.o src/stage/property.o

# Target rules
#all: test/test_block python/BrainBlock test/test_multiblock test/test_result tools/dist_simulator
#test: python/BrainBlock test/test_block test/test_multiblock test/test_result tools/dist_simulator
all: tools/dist_simulator
benchmark: tools/benchmark
test: test/test_cnpy

#src/stage/extern_stimuli.cpp : src/stage/extern_stimuli.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/membrane_voltage.cpp : src/stage/membrane_voltage.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/presynaptic_voltage.cpp : src/stage/presynaptic_voltage.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/random.cpp : src/stage/random.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/spike.cpp : src/stage/spike.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/synaptic_current.cpp : src/stage/synaptic_current.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/stat_result.cpp : src/stage/stat_result.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/sample.cpp : src/stage/sample.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/stage/property.cpp : src/stage/property.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@

#src/configuration.cpp : src/configuration.cu
#	$(HIP_PATH)/bin/hipify-perl $< > $@
	
#src/util/transpose.cpp: src/util/transpose.cu
	$(HIP_PATH)/bin/hipify-perl $< > $@

#test/save_result.cpp : test/save_result.cu
	$(HIP_PATH)/bin/hipify-perl $< > $@

src/util/cnpy.o : src/util/cnpy.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
	
src/data_allocator.o : src/data_allocator.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS)  $(INCLUDES) -o $@ -c $<

src/stage/extern_stimuli.o : src/stage/extern_stimuli.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/membrane_voltage.o : src/stage/membrane_voltage.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/presynaptic_voltage.o : src/stage/presynaptic_voltage.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/random.o : src/stage/random.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/spike.o : src/stage/spike.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/synaptic_current.o : src/stage/synaptic_current.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/stat_result.o : src/stage/stat_result.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/sample.o : src/stage/sample.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/stage/property.o : src/stage/property.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/configuration.o : src/configuration.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<
	
src/util/transpose.o: src/util/transpose.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/common.o : src/common.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

#src/blocking_queue.o : src/blocking_queue.cpp
#	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/util/cmd_arg.o : src/util/cmd_arg.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

src/logging.o : src/logging.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

src/weights.o : src/weights.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

src/route.o : src/route.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

tools/dist_simulator.o : tools/dist_simulator.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) `pkg-config --cflags protobuf grpc` $(MPICFLAG)  -o $@ -c $<

tools/benchmark.o : tools/benchmark.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) `pkg-config --cflags protobuf grpc` $(MPICFLAG)  -o $@ -c $<

src/block.o : src/block.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

test/save_result.o : test/save_result.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

test/check.o : test/check.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ -c $<

test/test_block: test/test_block.cpp $(SRCOBJS) src/util/cmd_arg.o test/save_result.o test/check.o
	$(HIPCC) $(CXXFLAGS) -std=c++11 $(INCLUDES) -o $@ $+ -L$(CUDA_HOME)/lib64 $(LIBRARIES)

tools/simulator: tools/simulator.cpp $(SRCOBJS) src/util/cmd_arg.o test/save_result.o test/check.o
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ $+ -L$(CUDA_HOME)/lib64 $(LIBRARIES)

test/test_multiblock: test/test_multiblock.cpp $(SRCOBJS) src/util/cmd_arg.o test/save_result.o test/check.o
	$(MPICXX) $(CXXFLAGS) $(MPICFLAGS) $(INCLUDES) -o $@ $+

test/test_result: test/test_result.cpp $(SRCOBJS) src/util/cmd_arg.o test/save_result.o test/check.o
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ $+

test/test_kernel: test/test_kernel.cpp $(SRCOBJS) src/util/cmd_arg.o
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ $+

test/test_spike: test/test_spike.cpp  $(KERNELOBJS)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(INCLUDES) -o $@ $+

#python/BrainBlock: python/blockwraper.cpp $(SRCOBJS)
#	$(HIPCC) $(CXXFLAGS) -shared $(HIPFLAGS) $(INCLUDES) -I$(HOME)/software/pybind11/include `pkg-config --cflags python3` -o $@`python3-config --extension-suffix` $+ -L$(CUDA_HOME)/lib64 $(LIBRARIES)

tools/snn.pb.o: tools/snn.pb.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) `pkg-config --cflags protobuf grpc` -o $@ -c $<

tools/snn.grpc.pb.o: tools/snn.grpc.pb.cc 
	$(CXX) $(CXXFLAGS) $(INCLUDES) `pkg-config --cflags protobuf grpc` -o $@ -c $<

tools/dist_simulator: tools/dist_simulator.o $(SRCOBJS) tools/snn.pb.o tools/snn.grpc.pb.o 
	$(HIPCC) -o $@ $+ $(LIBRARIES) `pkg-config --libs protobuf grpc++` -pthread -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed -ldl

tools/benchmark: tools/benchmark.o $(SRCOBJS) tools/snn.pb.o tools/snn.grpc.pb.o 
	$(HIPCC) -o $@ $+ $(LIBRARIES) `pkg-config --libs protobuf grpc++` -pthread -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed -ldl

test/test_cnpy.o : test/test_cnpy.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<

test/test_cnpy: test/test_cnpy.o src/util/cnpy.o
	$(CXX) $(CXXFLAGS) $(INCLUDES)  -o $@ $+ -lz

%.grpc.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=./tools --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<
%.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --cpp_out=./tools $<

clean:
	rm -f  $(SRCOBJS) test/test_block test/test_multiblock test/test_result tools/simulator python/*.so test/check.o test/save_result.o tools/*.o tools/dist_simulator


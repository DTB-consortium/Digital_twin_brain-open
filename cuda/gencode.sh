#!/bin/sh
make snn.grpc.pb.cc snn.pb.cc
python3 -m grpc_tools.protoc -I./protos --python_out=./python --grpc_python_out=./python ./protos/snn.proto

#!/bin/bash

# (Creates python callable GRPC logic from the .proto file)
# You can run this script from anywhere,
# and it will generate the new files in the protobuf folder in this script's directory

THIS_SCRIPTS_DIR="`dirname \"$0\"`"
cd "$THIS_SCRIPTS_DIR" || (echo "Couldn't cd into $THIS_SCRIPTS_DIR" && exit)

python3 -m grpc_tools.protoc \
  -I protobuf \
  --python_out protobuf \
  --grpc_python_out protobuf \
  protobuf/lock_server.proto


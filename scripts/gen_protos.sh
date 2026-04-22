#!/bin/bash

REPO_ROOT=$(dirname $(dirname $(realpath $0)))

is_clean=false
if [ "$1" == "--clean" ]; then
    is_clean=true
fi

if [ "$is_clean" == true ]; then
    rm -rf ${REPO_ROOT}/control_stubs/control_stubs/*.py
    rm -rf ${REPO_ROOT}/control_stubs/control_stubs/*.pyi
    rm -rf ${REPO_ROOT}/control_stubs/control_stubs/*.pb.cc
    rm -rf ${REPO_ROOT}/control_stubs/control_stubs/*.pb.h
    rm -rf ${REPO_ROOT}/control_stubs/*.py
    rm -rf ${REPO_ROOT}/control_stubs/*.pyi
    touch ${REPO_ROOT}/control_stubs/__init__.py
    rm -rf ${REPO_ROOT}/control_stubs/*.pb.cc
    rm -rf ${REPO_ROOT}/control_stubs/*.pb.h
    exit 0
fi

# generate for python
python3 -m grpc_tools.protoc \
    -I${REPO_ROOT}/control_stubs \
    --python_out=${REPO_ROOT}/control_stubs \
    --grpc_python_out=${REPO_ROOT}/control_stubs \
    --pyi_out=${REPO_ROOT}/control_stubs \
    ${REPO_ROOT}/control_stubs/control_stubs/*.proto
cp ${REPO_ROOT}/control_stubs/control_stubs/*.py ${REPO_ROOT}/control_stubs
cp ${REPO_ROOT}/control_stubs/control_stubs/*.pyi ${REPO_ROOT}/control_stubs

# generate for C++
GRPC_CPP_PLUGIN=$(which grpc_cpp_plugin)
protoc \
    --proto_path=${REPO_ROOT}/control_stubs \
    --cpp_out=${REPO_ROOT}/control_stubs \
    --grpc_out=${REPO_ROOT}/control_stubs \
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} \
    ${REPO_ROOT}/control_stubs/control_stubs/*.proto

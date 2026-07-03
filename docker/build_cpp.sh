#!/usr/bin/env bash
# Build the ported C++ TensorRT inference app (deploy/cpp).
#
#   docker compose -f docker/docker-compose.yml run --rm fastbev-cpp bash docker/build_cpp.sh
#
# Produces deploy/cpp/build/fastbev. The app reads configs/{trt,param,calib}.yaml
# from the parent of the working dir (it uses current_path().parent_path()), so
# run it from deploy/cpp/build.
set -euo pipefail

CPP_DIR="${CPP_DIR:-/workspace/deploy/cpp}"
BUILD_DIR="${CPP_DIR}/build"

echo ">> gcc: $(gcc-10 --version | head -1)"
echo ">> nvcc: $(nvcc --version | grep release)"

cmake -S "${CPP_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-10 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10

cmake --build "${BUILD_DIR}" -j"$(nproc)"
echo ">> built: ${BUILD_DIR}/fastbev"

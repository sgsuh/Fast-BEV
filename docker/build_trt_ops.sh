#!/usr/bin/env bash
# Build ONLY the TensorRT custom-op plugin library from third_party/mmdeploy.
#
#   docker compose -f docker/docker-compose.yml run --rm fastbev-trt \
#       bash docker/build_trt_ops.sh
#
# Produces third_party/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so which
# carries the project_2d_to_3d / emc / bev_pool_v2 plugins. The SDK is not built
# (MMDEPLOY_BUILD_SDK defaults OFF); we only need the plugin .so, loaded later
# via ctypes -- no mmdeploy python package required.
set -euo pipefail

MMDEPLOY_DIR="${MMDEPLOY_DIR:-/workspace/third_party/mmdeploy}"
BUILD_DIR="${MMDEPLOY_DIR}/build"
TENSORRT_DIR="${TENSORRT_DIR:-/usr}"
CUDNN_DIR="${CUDNN_DIR:-/usr}"

echo ">> TensorRT headers: $(ls ${TENSORRT_DIR}/include/x86_64-linux-gnu/NvInfer.h /usr/include/NvInfer.h 2>/dev/null | head -1)"
echo ">> nvcc: $(nvcc --version | grep release || true)"

cmake -S "${MMDEPLOY_DIR}" -B "${BUILD_DIR}" \
    -DMMDEPLOY_TARGET_BACKENDS=trt \
    -DTENSORRT_DIR="${TENSORRT_DIR}" \
    -DCUDNN_DIR="${CUDNN_DIR}" \
    -DCMAKE_CUDA_ARCHITECTURES="86-real;86-virtual" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --target mmdeploy_tensorrt_ops -j"$(nproc)"

# Mirror the install() rule so ctypes get_ops_path() finds it under mmdeploy/lib.
mkdir -p "${MMDEPLOY_DIR}/mmdeploy/lib"
find "${BUILD_DIR}" -name 'libmmdeploy_tensorrt_ops.so' -exec cp -v {} "${MMDEPLOY_DIR}/mmdeploy/lib/" \;

SO="${MMDEPLOY_DIR}/mmdeploy/lib/libmmdeploy_tensorrt_ops.so"
echo ">> built: ${SO}"
echo ">> registered plugins:"
python - "$SO" <<'PY'
import ctypes, sys
import tensorrt as trt
ctypes.CDLL(sys.argv[1])
reg = trt.get_plugin_registry()
names = sorted({c.name for c in reg.plugin_creator_list})
print("   " + ", ".join(names))
assert {"project_2d_to_3d", "emc"} <= set(names), "FastBEV plugins missing!"
print(">> OK: project_2d_to_3d + emc registered")
PY

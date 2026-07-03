# Fast-BEV C++ TensorRT inference (nuScenes)

Standalone C++ TensorRT inference for the Fast-BEV deploy model, ported from
ThorDrive's `revolt.fast-bev-infer` (built for Jetson Orin / sm_87) to this
repo's deploy environment: **x86_64, CUDA 11.3, TensorRT 8.6.1, RTX 4070
(Ada / sm_89)**.

The custom TRT plugins (`project_2d_to_3d`, `emc`) are compiled straight into
the executable (`src/trt_*`), so the engine is built from the ONNX and run here
without the mmdeploy plugin `.so`. Inputs match the validated Python deploy path:
`input_0` = 6 preprocessed views, `input_1` = lidar2img extrinsic (6×4×4),
`input_2` = identity `tf_vec`.

## What was changed from the upstream source
- Dropped the DDS/Lucid app (`apps/fastdds.cpp`) and `external/thor/*`; only
  `apps/nuscenes.cpp` is built.
- New `CMakeLists.txt`: nuScenes app only, no `Thor::` links, C++17, CUDA arch
  `86-real;86-virtual` (compute_86 PTX → sm_89 JIT), Qhull imported-target fallback.
- `FindTensorRT.cmake`: search the Debian multiarch `*/x86_64-linux-gnu` dirs.
- `trt_fastbev.{h,cpp}`: added headless `saveBev()` → writes `bev_<frame>.jpg` to
  `save_dir` and prints per-frame detection counts (upstream only had
  `#define VIS` + `cv::imshow`, unusable in the container).
- `configs/*` are regenerated for nuScenes-mini by `tools/prep_cpp_nuscenes.py`.

## Build & run (from repo root)
```bash
docker compose -f docker/docker-compose.yml build fastbev-cpp
docker compose -f docker/docker-compose.yml run --rm fastbev-cpp bash docker/build_cpp.sh

# generate configs + lay out one scene's frames (needs mmdet3d -> fastbev-trt)
docker compose -f docker/docker-compose.yml run --rm fastbev-trt \
    bash -lc 'python tools/prep_cpp_nuscenes.py --start 0 --num 20'

# first run parses work_dirs/fastbev_m0_f4.onnx -> FP16 engine, then streams frames
docker compose -f docker/docker-compose.yml run --rm fastbev-cpp \
    bash -lc 'cd /workspace/deploy/cpp/build && ./fastbev'
```
BEV renders land in `work_dirs/cpp_nuscenes/bev/`. On the 4070 this runs at
~28 ms/frame; detection counts and spatial layout match the Python deploy path.

## Known limitations
- `calib` is static per scene, so AB3DMOT tracking does not see per-frame ego
  motion (single-frame detection is correct; multi-frame tracking would need
  per-frame ego pose).
- Same "single-frame baked volume" ONNX quirk as the Python engine (no temporal
  accumulation).
- Preprocessing uses `initUndistortRectifyMap` with `dist_coef = 0` (nuScenes is
  treated as pinhole, matching FastBEV); this is a near-exact, not bit-exact,
  match of the Python resize/crop.

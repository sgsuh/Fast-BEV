# Fast-BEV

[Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline](https://arxiv.org/abs/2301.12511)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/fast-bev++.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark_setting.png)
![image](https://github.com/Sense-GVT/Fast-BEV/blob/main/benchmark.png)

## Better Inference Implementation
Thanks to the repository [CUDA-FastBEV](https://github.com/Mandylove1993/CUDA-FastBEV) inference using CUDA & TensorRT. And provide PTQ and QAT int8 quantization code.
You can refer to it to get faster speed.

## Usage

[usage](https://github.com/Sense-GVT/Fast-BEV/blob/dev/tools/fastbev_run.sh)

### Installation

* CUDA>=9.2
* GCC>=5.4
* Python>=3.6
* Pytorch>=1.8.1
* Torchvision>=0.9.1
* MMCV-full==1.4.0
* MMDetection==2.14.0
* MMSegmentation==0.14.1

### Dataset preparation

```
  .
  ├── data
  │   └── nuscenes
  │       ├── maps
  │       ├── maps_bev_seg_gt_2class
  │       ├── nuscenes_infos_test_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_train_4d_interval3_max60.pkl
  │       ├── nuscenes_infos_val_4d_interval3_max60.pkl
  │       ├── v1.0-test
  │       └── v1.0-trainval
```

[download](https://drive.google.com/drive/folders/10KyLm0xW3QiLhAefxBbXR-Hw_7nel_tm?usp=sharing)

### Pretraining

```
  .
  ├── pretrained_models
  │   ├── cascade_mask_rcnn_r18_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5110_segm_mAP_0.4070.pth
  │   ├── cascade_mask_rcnn_r34_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5190_segm_mAP_0.4140.pth
  │   └── cascade_mask_rcnn_r50_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5400_segm_mAP_0.4300.pth
```

[download](https://drive.google.com/drive/folders/19BD4totDHtwnHtOqTdn0xYJh7stwYd9l?usp=sharing)

### Training

```
  .
  ├── work_dirs
    └── fastbev
      └── exp
          └── paper
              └── fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.02062323.txt
              │   └── log.test.fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.02062309.txt
              ├── fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4.02080000.txt
              │   └── log.test.fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4.02072346.txt
              ├── fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.02080021.txt
              │   └── log.test.fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4.02080005.txt
              ├── fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4
              │   ├── epoch_20.pth
              │   ├── latest.pth -> epoch_20.pth
              │   ├── log.eval.fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.02080021.txt
              │   └── log.test.fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4.02080005.txt
              └── fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4
                  ├── epoch_20.pth
                  ├── latest.pth -> epoch_20.pth
                  ├── log.eval.fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.02080021.txt
                  └── log.test.fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4.02080001.txt
```

[download](https://drive.google.com/drive/folders/1Ja9mqOE0iGPysVxmLSrZyUoCEBYu5fMH?usp=sharing)

### Deployment (this fork, `deploy` branch)

This fork adds a self-contained deploy/inference stack for running a trained
Fast-BEV model on a consumer GPU, validated end-to-end on **NVIDIA RTX 4070
Laptop (Ada / sm_89, 8 GB, WSL2)** against **nuScenes-mini**. Everything runs in
Docker; no training or full-dataset setup is required.

The repo pins the legacy stack (MMCV-full 1.4.0 / MMDet 2.14.0 / MMSeg 0.14.1),
which only builds against PyTorch ~1.10 (CUDA 11.3). Ada normally needs CUDA
11.8+, but the 4070 runs this old stack via **PTX forward-compatibility**: the
CUDA ops are compiled with an explicit `compute_86` PTX entry and the driver
JIT-compiles them to sm_89. See `docker/README.md` for the full guide.

**Environment (Docker images, built in order):**

| Image | Adds | Purpose |
|-------|------|---------|
| `fastbev-deploy:cu113` | torch 1.10.1+cu113, MMCV 1.4.0 (source, +PTX) | PyTorch inference |
| `fastbev-deploy:trt`   | TensorRT 8.6.1 + cuDNN 8.9 | ONNX → TRT engine |
| `fastbev-deploy:cpp`   | gcc-10, OpenCV/Eigen/yaml-cpp/Qhull | C++ TRT inference |

```bash
docker compose -f docker/docker-compose.yml build fastbev        # base
docker compose -f docker/docker-compose.yml build fastbev-trt    # + TensorRT
docker compose -f docker/docker-compose.yml build fastbev-cpp    # + C++ toolchain
```

**1. nuScenes-mini info + PyTorch inference.** Generate mini info pkls and run the
standard evaluator (`tools/test.py`) or the streaming deploy-mode detector
(`FastBEVCus`). The deploy path takes 3 fixed inputs — `img (1,6,3,256,704)`,
`lidar2img extrinsic`, identity `tf_vec` — and accumulates temporal context in a
rolling volume fused by the EMC op.

```bash
# accuracy check (m0, epoch_20, mini_val 81 samples): mAP 0.2742 / NDS 0.3495
docker compose ... run --rm fastbev bash -lc \
  'python tools/test.py configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
   work_dirs/.../epoch_20.pth --eval bbox'

# deploy-mode streaming accuracy vs the 4-frame reference
docker compose ... run --rm fastbev bash -lc 'python tools/test_deploy_seq.py --start 0 --num 20'
```

**2. TensorRT (Python).** The custom TRT plugins (`project_2d_to_3d`, `emc`) live
in the `third_party/mmdeploy` submodule (a fork of mmdeploy 1.3.1); build the
plugin `.so`, export ONNX, build an FP16 engine, and verify.

```bash
git submodule update --init third_party/mmdeploy
docker compose ... run --rm fastbev-trt bash docker/build_trt_ops.sh              # libmmdeploy_tensorrt_ops.so
docker compose ... run --rm fastbev-trt bash -lc 'python tools/export_onnx.py'    # pth → ONNX
docker compose ... run --rm fastbev-trt bash -lc 'python tools/export_trt.py --fp16'
docker compose ... run --rm fastbev-trt bash -lc 'python tools/test_trt_seq.py --start 0 --num 20'
```
Verified (scene0, 16 frames): TRT recall 0.644 / prec 0.245, on par with the
PyTorch deploy path (0.613 / 0.234). Note the ONNX bakes the rolling-volume
history as a zeros constant, so the engine is effectively single-frame.

**3. C++ TensorRT inference** (`deploy/cpp/`) — a standalone app (ported from
ThorDrive's fast-bev-infer) that builds the engine from the ONNX and runs the
full native pipeline (GPU preprocess → engine → decode → scale-NMS → tracking).
See `deploy/cpp/README.md`.

```bash
docker compose ... run --rm fastbev-cpp bash docker/build_cpp.sh
docker compose ... run --rm fastbev-trt bash -lc 'python tools/prep_cpp_nuscenes.py --start 0 --num 20'
docker compose ... run --rm fastbev-cpp bash -lc 'cd /workspace/deploy/cpp/build && ./fastbev'
```
Runs at ~28 ms/frame on the 4070; frame-0 detections (142) and spatial layout
match the Python reference.

## View Transformation Latency on device
[2D-to-3D on CUDA & CPU](https://github.com/Sense-GVT/Fast-BEV/tree/dev/script/view_tranform_cuda)

## Citation
```
@article{li2023fast,
  title={Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline},
  author={Li, Yangguang and Huang, Bin and Chen, Zeren and Cui, Yufeng and Liang, Feng and Shen, Mingzhu and Liu, Fenggang and Xie, Enze and Sheng, Lu and Ouyang, Wanli and others},
  journal={arXiv preprint arXiv:2301.12511},
  year={2023}
}
```

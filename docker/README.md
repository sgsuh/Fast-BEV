# Fast-BEV Docker environment (RTX 4070 / deployment & inference)

A Docker environment for **deploying / running inference** with a trained Fast-BEV
model. It is built to run on an RTX 4070 (Ada, `sm_89`) under WSL2. All installation
happens **inside the Docker image only** (nothing is installed on the host).

## Why this setup

This repo requires the legacy stack `MMCV-full 1.4.0 / MMDet 2.14.0 / MMSeg 0.14.1`,
which only builds against PyTorch ~1.10 (CUDA 11.3). An Ada GPU normally needs
CUDA 11.8+, but the RTX 4070 runs this legacy stack thanks to **PTX
forward-compatibility**: the driver JIT-compiles `compute_86` PTX to `sm_89` at
runtime. This was verified empirically — cuDNN conv, batchnorm, indexing, matmul
and MMCV's CUDA NMS all execute correctly on the 4070.

To keep that guarantee for the CUDA extensions we compile ourselves (MMCV and this
repo's `mmdet3d`), we build them with `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"`
so the resulting `.so` files carry JIT-able PTX for `sm_89`.

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04` |
| Python | 3.8 |
| PyTorch | 1.10.1+cu113 / torchvision 0.11.2+cu113 |
| MMCV-full | 1.4.0 (built from source, with PTX) |
| MMDet / MMSeg | 2.14.0 / 0.14.1 |

## Build

```bash
docker compose -f docker/docker-compose.yml build
```

> The first build takes ~30-45 min (downloading PyTorch + compiling MMCV from source).

## Run

The whole repo is bind-mounted at `/workspace` inside the container, so code,
configs and checkpoints edited on the host are immediately visible in the container.
This repo's `mmdet3d` custom CUDA ops are compiled **once on first start**
(see `entrypoint.sh`, ~3-5 min); subsequent starts come up in a few seconds.

```bash
# Open a shell in the container
docker compose -f docker/docker-compose.yml run --rm fastbev

# Check the GPU is visible
docker compose -f docker/docker-compose.yml run --rm fastbev \
    python -c "import torch; print(torch.cuda.get_device_name(0))"

# Full environment check (torch / mmcv NMS / mmdet3d ops / model build)
docker compose -f docker/docker-compose.yml run --rm fastbev \
    python docker/verify_env.py
```

## Model inference (deploy)

A trained checkpoint already ships with the repo:
`work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth`

Single-GPU inference / evaluation (inside the container):

```bash
docker compose -f docker/docker-compose.yml run --rm fastbev bash
# --- then, inside the container shell ---
python tools/test.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth \
    --eval bbox
```

### Generating the nuScenes-mini info files

The configs reference the temporal annotation file
`data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl`. For the full dataset
these are downloadable, but for the **nuScenes-mini** dataset shipped here they
must be generated. Run the two steps below **inside the container** (`data/nuscenes`
must already contain the `v1.0-mini` split, `samples/`, `sweeps/` and `maps/`):

```bash
docker compose -f docker/docker-compose.yml run --rm fastbev bash
# --- inside the container shell ---
export PYTHONPATH=/workspace

# Step 1 - base mini infos (nuscenes_infos_{train,val}.pkl)
python -c "from tools.data_converter import nuscenes_converter as nc; \
nc.create_nuscenes_infos('./data/nuscenes', 'nuscenes', version='v1.0-mini', max_sweeps=10)"

# Step 2 - temporal '4d' infos (nuscenes_infos_{train,val}_4d_interval3_max60.pkl)
python tools/data_converter/nuscenes_seq_converter_mini.py \
    --root-path ./data/nuscenes/ --version v1.0-mini --sets train val
```

`nuscenes_seq_converter_mini.py` is a mini/general variant of the original
`nuscenes_seq_converter.py` (which is hard-coded to the v1.0-test split). It is
parametrized by `--version`, `--sets`, `--interval` and `--max-adj`.

### Running inference / evaluation on mini

```bash
# inside the container shell
python tools/test.py \
    configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py \
    work_dirs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth \
    --eval bbox
```

This runs the trained m0 model over the 81 mini-val samples on the GPU and reports
nuScenes detection metrics (mAP / NDS). Classes that barely appear in mini-val
(trailer, construction_vehicle, barrier) score ~0 by construction — this is a
property of the tiny mini split, not a bug.

Two repo changes make this work on mini (both committed):
- `mmdet3d/datasets/nuscenes_monocular_dataset_map_2.py` now auto-detects the
  nuScenes DB version (mini/trainval/test) instead of hard-coding `v1.0-trainval`,
  and tolerates a missing map-expansion (BEV-seg GT is only needed for seg models).
- The m0 config uses the local `disk` file-client backend instead of the original
  `petrel` (s3/ceph) backend.

The GPU environment and the model graph can also be verified without any data via
`python docker/verify_env.py`.

## Notes

- On WSL2, `nvidia-smi` must work inside the container. If it doesn't, check
  Docker Desktop's WSL integration and the latest NVIDIA Windows driver.
- The RTX 4070 Laptop has 8 GB VRAM, so keep `samples_per_gpu` (batch size) small.

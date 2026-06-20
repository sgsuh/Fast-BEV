#!/usr/bin/env bash
# Installs this repo (editable) and builds its mmdet3d CUDA ops, then runs CMD.
# Heavy/stable deps (torch, mmcv, mmdet, mmseg) are baked into the image; only the
# in-tree custom ops are compiled here, against the bind-mounted source.
#
# `docker compose run` creates a fresh container each time, so the egg-link must be
# re-created on every start. We therefore ALWAYS run the editable install: the very
# first run compiles the CUDA ops (~3-5 min) into the bind-mounted tree
# (mmdet3d/ops/*.so, persisted on the host); later runs find the .so up-to-date and
# only re-link the package in seconds.
set -e

cd /workspace

ARCH="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6+PTX}"

echo "[entrypoint] Ensuring Fast-BEV (mmdet3d) is installed in this container (TORCH_CUDA_ARCH_LIST=${ARCH}) ..."
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${ARCH}" \
    pip install -e . --no-deps --no-build-isolation -q
echo "[entrypoint] Ready."

exec "$@"

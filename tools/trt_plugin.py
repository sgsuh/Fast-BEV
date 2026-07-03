"""
Load the FastBEV custom TensorRT plugins (project_2d_to_3d / emc) without pulling
in the mmdeploy *python* package.

The plugin library is built from the third_party/mmdeploy submodule (see
docker/build_trt_ops.sh) and only needs a ctypes CDLL to self-register its
creators with TensorRT's global plugin registry -- which is exactly what
mmdeploy.backend.tensorrt.init_plugins.load_tensorrt_plugin does. We avoid that
import because mmdeploy 1.x depends on mmcv 2.x / mmengine, incompatible with the
mmcv 1.4.0 pinned by this repo.

Create: 2026.07.03
"""

import ctypes
import glob
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [
    'third_party/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so',
    'third_party/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so',
]

_loaded = False


def load_fastbev_trt_plugins(verbose=True):
    """ctypes-load libmmdeploy_tensorrt_ops.so so project_2d_to_3d / emc register."""
    global _loaded
    if _loaded:
        return True

    lib_path = None
    for rel in _CANDIDATES:
        p = os.path.join(_REPO_ROOT, rel)
        if os.path.exists(p):
            lib_path = p
            break
    if lib_path is None:
        hits = glob.glob(os.path.join(_REPO_ROOT, 'third_party/mmdeploy', '**',
                                      'libmmdeploy_tensorrt_ops.so'), recursive=True)
        lib_path = hits[0] if hits else None

    if lib_path is None:
        raise FileNotFoundError(
            'libmmdeploy_tensorrt_ops.so not found. Build it first:\n'
            '  docker compose -f docker/docker-compose.yml run --rm fastbev-trt '
            'bash docker/build_trt_ops.sh')

    ctypes.CDLL(lib_path)
    _loaded = True
    if verbose:
        print('[trt_plugin] loaded {}'.format(lib_path))
    return True


if __name__ == '__main__':
    load_fastbev_trt_plugins()
    import tensorrt as trt
    reg = trt.get_plugin_registry()
    names = sorted({c.name for c in reg.plugin_creator_list})
    print('registered plugins:', ', '.join(names))

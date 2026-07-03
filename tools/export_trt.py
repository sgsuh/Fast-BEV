"""
Build a TensorRT engine from the FastBEVCus ONNX, binding the custom plugins
(project_2d_to_3d / emc) from third_party/mmdeploy.

Ported from thor-fast-bev/tools/export_trt.py (SG.SUH, 2023): the mmdeploy python
plugin loader is replaced by tools/trt_plugin.load_fastbev_trt_plugins (ctypes),
so no mmcv-2.x mmdeploy install is needed.

Create: 2026.07.03
"""

import sys

sys.path.append('.')
sys.path.append('tools')

import argparse
import onnx
import tensorrt as trt

from trt_plugin import load_fastbev_trt_plugins


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='work_dirs/fastbev_m0_f4.onnx')
    parser.add_argument('--trt_path', type=str, default='work_dirs/fastbev_m0_f4_fp16.engine')
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--fp32', dest='fp16', action='store_false')
    parser.add_argument('--workspace', type=int, default=32, help='workspace size, 1<<N bytes')
    return parser.parse_args()


def main():
    args = parse_args()
    load_fastbev_trt_plugins()

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)
    onnx_model = onnx.load(args.onnx_path)
    if not parser.parse(onnx_model.SerializeToString()):
        for i in range(parser.num_errors):
            print('  onnx-parse error:', parser.get_error(i))
        raise RuntimeError('Failed to parse ONNX.')

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << args.workspace
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.TF32)

    print('building TensorRT engine ({}) ...'.format('FP16' if args.fp16 else 'TF32'))
    engine = builder.build_engine(network, config)
    assert engine is not None, 'Failed to create TensorRT engine'

    with open(args.trt_path, 'wb') as f:
        f.write(bytearray(engine.serialize()))
    print('serialized engine -> {}'.format(args.trt_path))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build a TensorRT engine from the single ONNX of make_single_onnx.py using the
TensorRT Python API (for machines without `trtexec`, e.g. a pip-installed
tensorrt on Windows).  Equivalent to:
    trtexec --onnx=<onnx> --saveEngine=<engine> --fp16

Usage
  python scripts/spatial/build_trt_engine.py --onnx work/onnx_1088x1920/fast_foundationstereo.onnx \
      --engine work/onnx_1088x1920/fast_foundationstereo.engine [--fp32] [--workspace_gb 8]
"""
import argparse, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--engine', default=None)
    ap.add_argument('--fp32', action='store_true', help='disable fp16')
    ap.add_argument('--workspace_gb', type=float, default=8.0)
    args = ap.parse_args()
    import tensorrt as trt
    engine_path = args.engine or os.path.splitext(args.onnx)[0] + '.engine'
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) \
        if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH') else builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    t0 = time.perf_counter()
    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit('ONNX parse failed')
    print(f'parsed {args.onnx} in {time.perf_counter()-t0:.1f} s; inputs:',
          [(network.get_input(i).name, tuple(network.get_input(i).shape)) for i in range(network.num_inputs)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    fp16 = False
    if not args.fp32:
        if hasattr(trt.BuilderFlag, 'FP16'):          # TensorRT <= 10
            config.set_flag(trt.BuilderFlag.FP16); fp16 = True
        else:                                          # TensorRT >= 11: strongly typed, precision comes from the ONNX dtypes
            print('this TensorRT has no FP16 builder flag (strongly-typed networks); the engine keeps the ONNX precision (fp32 here). '
                  'Export a half-precision ONNX to get an fp16 engine.')
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit('engine build failed')
    with open(engine_path, 'wb') as f:
        f.write(blob)
    print(f'engine {engine_path}: {os.path.getsize(engine_path)/1e6:.1f} MB, built in {time.perf_counter()-t0:.0f} s, TensorRT {trt.__version__}, fp16 flag={fp16}')


if __name__ == '__main__':
    main()

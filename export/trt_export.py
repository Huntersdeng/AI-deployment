import tensorrt as trt
from PIL import Image
import os
import argparse
import numpy as np
import pycuda.driver as cuda
import torchvision.transforms as transforms
from PIL import Image

# TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
# TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def get_engine(
    onnx_file_path,
    engine_file_path="",
    precision="fp32",
):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(f"ONNX file {onnx_file_path} not found")
                exit(0)
            print(f"Loading ONNX file from path {onnx_file_path}...")
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
                print("Completed parsing of ONNX file")

            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            outputs = [network.get_output(i)
                       for i in range(network.num_outputs)]

            print("Network Description")
            for input in inputs:
                batch_size = input.shape[0]
                print(
                    f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}"
                )
            for output in outputs:
                print(
                    f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}"
                )
            assert batch_size > 0

            # config.max_workspace_size = 1 << 31  # 29 : 512MiB, 30 : 1024MiB
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

            if precision == "fp16":
                if not builder.platform_has_fast_fp16:
                    print("FP16 is not supported natively on this platform/device")
                else:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("Using FP16 mode.")
            elif precision == "int8":
                if not builder.platform_has_fast_int8:
                    print("INT8 is not supported natively on this platform/device")
                else:
                    config.set_flag(trt.BuilderFlag.INT8)
                    print("Using INT8 mode.")

            elif precision == "fp32":
                print("Using FP32 mode.")
            else:
                raise NotImplementedError(
                    f"Currently hasn't been implemented: {precision}."
                )

            print(
                f"Building an engine from file {onnx_file_path}; this may take a while..."
            )
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")

            with open(engine_file_path, "wb") as f:
                f.write(plan)

            return engine

    print(engine_file_path)

    return build_engine()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="ONNX weights")
    parser.add_argument("--precision", type=str, default="fp32", help="Precision ")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    onnx_model_path = args.onnx
    engine_file_path = onnx_model_path.replace(".onnx", ".trt")

    get_engine(
        onnx_model_path, engine_file_path, args.precision)


if __name__ == "__main__":
    main()

from rknn.api import RKNN
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, required=True, help="ONNX weights")
    parser.add_argument("--target", type=str, default="rk3588", help="target device")
    parser.add_argument("--device-id", type=str, default="", help="device id")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="txt file containing path of quantization dataset",
    )
    parser.add_argument(
        "--accuracy-analysis",
        action="store_true",
        help="compute accuracy of each layer's output",
    )
    parser.add_argument(
        "--eval-perf", action="store_true", help="evaluate running speed"
    )
    args = parser.parse_args()
    return args


def main(args):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print("--> Config model")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
    )
    print("done")

    # Load ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(model=args.weights)
    # ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['/model.24/m.0/Conv_output_0', '/model.24/m.1/Conv_output_0', '/model.24/m.2/Conv_output_0'])
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    # Build model
    print("--> Building model")
    ret = rknn.build(rknn_batch_size=1, do_quantization=True, dataset=args.dataset)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    # Export RKNN model
    save_path = args.weights.replace("onnx", "rknn")
    print("--> Export rknn model")
    ret = rknn.export_rknn(save_path)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    # Accuracy analysis
    if args.accuracy_analysis:
        with open(args.dataset, "r") as f:
            analysis_dataset = f.read().split()
        print("--> Accuracy analysis")
        ret = rknn.accuracy_analysis(
            inputs=analysis_dataset, target=args.target, device_id=args.device_id
        )
        if ret != 0:
            print("Accuracy analysis failed!")
        print("done")

    # 对模型性能进行评估
    if args.eval_perf:
        # Init runtime environment
        print("--> Init runtime environment")
        # ret = rknn.init_runtime(target="RK3588", device_id="90ce0632eb5338b5", perf_debug=True)
        ret = rknn.init_runtime(
            target=args.target, device_id=args.device_id, perf_debug=True
        )
        if ret != 0:
            print("Init runtime environment failed!")
        print("done")
        perf_detail = rknn.eval_perf(is_print=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

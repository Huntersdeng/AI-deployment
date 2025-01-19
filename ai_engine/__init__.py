def setup_model(args):
    model_path = args.model_path
    if model_path.endswith(".pt") or model_path.endswith(".torchscript"):
        platform = "pytorch"
        from .ai_engine_torch import AiEngineTorch
        model = AiEngineTorch(args.model_path)
    elif model_path.endswith(".rknn"):
        platform = "rknn"
        from .ai_engine_rknn import AiEngineRknn
        model = AiEngineRknn(args.model_path)
    elif model_path.endswith("onnx"):
        platform = "onnx"
        from .ai_engine_onnx import AiEngineOnnx
        model = AiEngineOnnx(args.model_path)
    elif model_path.endswith("trt"):
        platform = "TensorRT"
        from .ai_engine_trt import AiEngineTrt
        model = AiEngineTrt(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print("Model-{} is {} model, starting inference".format(model_path, platform))
    return model, platform

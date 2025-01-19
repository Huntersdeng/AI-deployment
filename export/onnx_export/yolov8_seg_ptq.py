import argparse
from io import BytesIO
import os
from PIL import Image

import onnx
import torch
import torchvision.transforms as transforms
from torch import nn
from ultralytics import YOLO
from pytorch_quantization import quant_modules
from tqdm import tqdm
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
import py_quant_utils as quant_utils

try:
    import onnxsim
except ImportError:
    onnxsim = None


class PostSeg(nn.Module):
    export = True
    shape = None
    dynamic = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size
        mc = [self.cv4[i](x[i]) for i in range(self.nl)]
        x = self.forward_det(x)
        bo = len(x) // 3
        relocated = []
        for i in range(len(mc)):
            relocated.extend(x[i * bo: (i + 1) * bo])
            relocated.extend([mc[i]])
        relocated.extend([p])
        return relocated

    def forward_det(self, x):
        shape = x[0].shape
        y = []
        for i in range(self.nl):
            y.append(self.cv2[i](x[i]))
            y.append(self.cv3[i](x[i]))
        return y


def optim(module: nn.Module):
    s = str(type(module))[6:-2].split(".")[-1]
    if s == "Segment":
        setattr(module, "__class__", PostSeg)

class LetterBox:
    def __init__(self, ouptut_size, padding):
        self.output_size = ouptut_size
        self.padding = padding
        
    def __call__(self, image):
        """
        将图像调整为目标大小，同时保持宽高比，使用填充。

        Args:
            image (PIL.Image): 输入图像。
            target_size (tuple): 目标大小 (宽, 高)。
            color (tuple): 填充颜色，默认为黑色 (0, 0, 0)。

        Returns:
            PIL.Image: 调整后的图像。
        """
        # 获取原始图像的宽和高
        original_width, original_height = image.size
        target_width, target_height = self.output_size

        # 计算宽高比
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # 调整图像大小
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)

        # 创建一个新的图像，填充为目标大小
        new_image = Image.new("RGB", self.output_size, self.padding)

        # 计算填充的位置
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # 将调整后的图像粘贴到填充图像上
        new_image.paste(resized_image, (x_offset, y_offset))

        return new_image

def prepare_model(weight):
    quant_utils.initialize_calib_method(per_channel_quantization=True, calib_method="histogram")
    yolo = YOLO(weight)
    model = yolo.model
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()
    for m in model.modules():
        optim(m)

    model.cuda()

    # quant_utils.replace_to_quantization_module(model, ignore_policy=[])

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="PyTorch yolov8 weights"
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--sim", action="store_true", help="simplify onnx model")
    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 640, 640],
        help="Model input shape only for api builder",
    )
    parser.add_argument("--quant-method", type=str, default="mse",
                        help="Quantization method, [percentile, mse, entropy]")
    parser.add_argument("--calib-data", type=str, default="",
                        help="Calibration data path")
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    model = prepare_model(args.weights)
    # model.info(True)
    transform = transforms.Compose([LetterBox((640, 640), (114,114,114)), transforms.ToTensor()])
    dataset = quant_utils.CalibDataset(args.calib_data, transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=2)
    with torch.no_grad():
        quant_utils.collect_stats(model, dataloader, num_batches=4)
        quant_utils.compute_amax(model, method=args.quant_method)

    fake_input = torch.randn(args.input_shape).cuda()
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace(".pt", "-ptq.onnx")
    output_names = [
        "output0",
        "output1",
        "output2",
        "output3",
        "output4",
        "output5",
        "output6",
        "output7",
        "output8",
        "proto",
    ]
    quant_utils.quant_nn.TensorQuantizer.use_fb_fake_quant = True
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=["images"],
            output_names=output_names,
        )
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")
    onnx.save(onnx_model, save_path)
    print(f"ONNX export success, saved as {save_path}")


if __name__ == "__main__":
    main(parse_args())

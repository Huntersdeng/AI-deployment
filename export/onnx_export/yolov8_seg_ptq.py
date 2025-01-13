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


class CalibDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file,  image_size):
        with open(txt_file, 'r') as f:
            self.image_list = f.read().split()
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int):
        img_path = self.image_list[index]
        if not os.path.exists(img_path):
            print(f"{img_path} is not found")
            return None

        img = Image.open(img_path)
        img = self.letterbox(img, self.image_size, (114, 114, 114))
        return self.transform(img)

    def letterbox(self, image, target_size, color=(0, 0, 0)):
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
        target_width, target_height = target_size

        # 计算宽高比
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # 调整图像大小
        resized_image = image.resize((new_width, new_height), Image.BILINEAR)

        # 创建一个新的图像，填充为目标大小
        new_image = Image.new("RGB", target_size, color)

        # 计算填充的位置
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # 将调整后的图像粘贴到填充图像上
        new_image.paste(resized_image, (x_offset, y_offset))

        return new_image


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""
    print("Feed data to the network and collect statistic")
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, image in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(**kwargs, strict=False)
            print(F"{name:40}: {module}")
    model.cuda()


def prepare_model(weight):
    quant_utils.initialize_calib_method(per_channel_quantization=True, calib_method="histogram")
    yolo = YOLO(weight)
    model = yolo.model
    model.float()
    model.eval()
    for m in model.modules():
        optim(m)

    with torch.no_grad():
        model.fuse()
    model.cuda()

    quant_utils.replace_to_quantization_module(model, ignore_policy=[])

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
    dataset = CalibDataset(args.calib_data, [args.input_shape[3], args.input_shape[2]])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=2)
    with torch.no_grad():
        collect_stats(model, dataloader, num_batches=4)
        compute_amax(model, method=args.quant_method)

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

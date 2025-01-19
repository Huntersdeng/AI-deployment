import onnx
import torch
import argparse
from io import BytesIO
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import models.crnn as crnn
try:
    import onnxsim
except ImportError:
    onnxsim = None

import py_quant_utils as quant_utils


class CRNN(nn.Module):
    def __init__(self, weights):
        super(CRNN, self).__init__()
        self.crnn = crnn.CRNN(32, 1, 37, 256)
        print('loading pretrained model from %s' % weights)
        self.crnn.load_state_dict(torch.load(weights))

    def forward(self, x):
        output = self.crnn(x)
        scores, labels = output.transpose(0, 1).max(dim=-1, keepdim=True)
        return labels.to(torch.float32)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.convert('L')
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch crnn weights')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[1, 1, 32, 100],
                        help='Model input shape only for api builder')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Export ONNX device')
    parser.add_argument("--quant-method", type=str, default="mse",
                        help="Quantization method, [percentile, mse, entropy]")
    parser.add_argument("--calib-data", type=str, default="",
                        help="Calibration data path")
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def prepare_model(weight):
    quant_utils.quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_utils.initialize_calib_method(per_channel_quantization=True, calib_method="histogram")
    model = CRNN(weight)

    model.cuda().eval()

    return model


def main(args):
    model = prepare_model(args.weights)
    # model = CRNN(args.weights)
    print(model)

    transform = resizeNormalize((100, 32))
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

    save_path = args.weights.replace('.pth', '-ptq.onnx')
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=['images'],
            output_names=['output'])
        f.seek(0)
        onnx_model = onnx.load(f)

    onnx.checker.check_model(onnx_model)

    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    main(parse_args())

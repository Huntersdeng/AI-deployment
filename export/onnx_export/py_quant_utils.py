from typing import List
import json
import os

# PyTorch
import torch

# Pytorch Quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from absl import logging as quant_logging
from tqdm import tqdm
from PIL import Image
        
class CalibDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, transform):
        with open(txt_file, 'r') as f:
            self.image_list = f.read().split()
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index: int):
        img_path = self.image_list[index]
        if not os.path.exists(img_path):
            print(f"{img_path} is not found")
            return None

        img = Image.open(img_path)
        return self.transform(img)
        
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


# Initialize PyTorch Quantization
def initialize_calib_method(per_channel_quantization=True, calib_method="histogram"):
    # Initialize quantization, model and data loaders
    if per_channel_quantization:
        quant_desc_input = QuantDescriptor(calib_method=calib_method)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLSTM.set_default_quant_desc_input(quant_desc_input)
        quant_logging.set_verbosity(quant_logging.ERROR)

    else:
        # Force per tensor quantization for onnx runtime
        quant_desc_input = QuantDescriptor(calib_method=calib_method, axis=None)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        quant_desc_weight = QuantDescriptor(calib_method=calib_method, axis=None)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantLSTM.set_default_quant_desc_weight(quant_desc_weight)
        quant_logging.set_verbosity(quant_logging.ERROR)
    quant_modules.initialize()


def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):

        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(
            self.__class__)

        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def replace_to_quantization_module(model: torch.nn.Module, ignore_policy: List[str] = None):
    module_dict = {}

    for entry in quant_modules._DEFAULT_QUANT_MAP:

        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix + "." + name

            recursive_and_replace_module(submodule, path)
            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                if ignore_policy is not None and path in ignore_policy:
                    continue

                module._modules[name] = transfer_torch_to_quantization(
                    submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)

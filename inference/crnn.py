import os
import cv2
import time
import argparse
import numpy as np
import torch
import collections
import sys
sys.path.append('.')
from ai_engine import setup_model

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return np.array(text), np.array(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        """
        if len(length) == 1:
            length = length[0]
            assert len(t) == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], np.array([l]), raw=raw))
                index += l
            return texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with YOLOv8")
    # basic params
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="model path, could be [onnx, rknn, pt] file",
    )

    # data params
    parser.add_argument(
        "--img-folder", type=str, default="../model", help="img folder path"
    )

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)
    
    img_list = sorted(os.listdir(args.img_folder))
        
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    converter = strLabelConverter(alphabet)
    
    # warm up
    for i in range(100):
        img_name = img_list[0]
        img_path = os.path.join(args.img_folder, img_name)

        img_src = cv2.imread(img_path)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 32))
        
        # preprocee if not rknn model
        if platform in ["pytorch", "onnx", "TensorRT"]:
            input_data = img.reshape(1, 1, *img.shape).astype(np.float32)
            input_data = input_data / 255.0
            input_data = (input_data - 0.5) / 0.5
        else:
            input_data = img
        outputs = model.run([input_data])
        
    # run test
    inference_time = 0.0
    for i in range(len(img_list)):
        print("infer {}/{}".format(i + 1, len(img_list)), end="\r")

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 32))
        
        # preprocee if not rknn model
        if platform in ["pytorch", "onnx", "TensorRT"]:
            input_data = img.reshape(1, 1, *img.shape).astype(np.float32)
            input_data = input_data / 255.0
            input_data = (input_data - 0.5) / 0.5
        else:
            input_data = img
            
        start = time.time()
        outputs = model.run([input_data])
        inference_time += time.time() - start
        
        preds = np.array(outputs['output'][0, :, 0], dtype=np.int32)

        preds_size = np.array([len(preds)])
        raw_pred = converter.decode(preds, preds_size, raw=True)
        sim_pred = converter.decode(preds, preds_size, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
    print("average inference time: ", inference_time / len(img_list))
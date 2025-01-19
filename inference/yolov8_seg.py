import os
import cv2
import sys
import argparse
import torch
import json
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from copy import copy
import time

sys.path.append('.')
from ai_engine import setup_model

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
MAX_DETECT = 300

# The follew two param is for mAP test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)


class Letter_Box_Info:
    def __init__(self, shape, new_shape, w_ratio, h_ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw
        self.dh = dh
        self.pad_color = pad_color


class COCO_test_helper:
    def __init__(self, enable_letter_box=False) -> None:
        self.record_list = []
        self.enable_ltter_box = enable_letter_box
        if self.enable_ltter_box is True:
            self.letter_box_info_list = []
        else:
            self.letter_box_info_list = None

    def letter_box(self, im, new_shape, pad_color=(0, 0, 0), info_need=False):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
        )  # add border

        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(
                Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color)
            )
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def direct_resize(self, im, new_shape, info_need=False):
        shape = im.shape[:2]
        h_ratio = new_shape[0] / shape[0]
        w_ratio = new_shape[1] / shape[1]
        if self.enable_ltter_box is True:
            self.letter_box_info_list.append(
                Letter_Box_Info(shape, new_shape, w_ratio, h_ratio, 0, 0, (0, 0, 0))
            )
        im = cv2.resize(im, (new_shape[1], new_shape[0]))
        return im

    def get_real_box(self, box, in_format="xyxy"):
        bbox = copy(box)
        if self.enable_ltter_box == True:
            # unletter_box result
            if in_format == "xyxy":
                bbox[:, 0] -= self.letter_box_info_list[-1].dw
                bbox[:, 0] /= self.letter_box_info_list[-1].w_ratio
                bbox[:, 0] = np.clip(
                    bbox[:, 0], 0, self.letter_box_info_list[-1].origin_shape[1]
                )

                bbox[:, 1] -= self.letter_box_info_list[-1].dh
                bbox[:, 1] /= self.letter_box_info_list[-1].h_ratio
                bbox[:, 1] = np.clip(
                    bbox[:, 1], 0, self.letter_box_info_list[-1].origin_shape[0]
                )

                bbox[:, 2] -= self.letter_box_info_list[-1].dw
                bbox[:, 2] /= self.letter_box_info_list[-1].w_ratio
                bbox[:, 2] = np.clip(
                    bbox[:, 2], 0, self.letter_box_info_list[-1].origin_shape[1]
                )

                bbox[:, 3] -= self.letter_box_info_list[-1].dh
                bbox[:, 3] /= self.letter_box_info_list[-1].h_ratio
                bbox[:, 3] = np.clip(
                    bbox[:, 3], 0, self.letter_box_info_list[-1].origin_shape[0]
                )
        return bbox

    def get_real_seg(self, seg):
        #! fix side effect
        dh = int(self.letter_box_info_list[-1].dh)
        dw = int(self.letter_box_info_list[-1].dw)
        origin_shape = self.letter_box_info_list[-1].origin_shape
        new_shape = self.letter_box_info_list[-1].new_shape
        if (dh == 0) and (dw == 0) and origin_shape == new_shape:
            return seg
        elif dh == 0 and dw != 0:
            seg = seg[:, :, dw:-dw]  # a[0:-0] = []
        elif dw == 0 and dh != 0:
            seg = seg[:, dh:-dh, :]
        seg = np.where(seg, 1, 0).astype(np.uint8).transpose(1, 2, 0)
        seg = cv2.resize(
            seg, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        if len(seg.shape) < 3:
            return seg[None, :, :]
        else:
            return seg.transpose(2, 0, 1)

    def add_single_record(
        self, image_id, category_id, bbox, score, in_format="xyxy", pred_masks=None
    ):
        if self.enable_ltter_box == True:
            # unletter_box result
            if in_format == "xyxy":
                bbox[0] -= self.letter_box_info_list[-1].dw
                bbox[0] /= self.letter_box_info_list[-1].w_ratio

                bbox[1] -= self.letter_box_info_list[-1].dh
                bbox[1] /= self.letter_box_info_list[-1].h_ratio

                bbox[2] -= self.letter_box_info_list[-1].dw
                bbox[2] /= self.letter_box_info_list[-1].w_ratio

                bbox[3] -= self.letter_box_info_list[-1].dh
                bbox[3] /= self.letter_box_info_list[-1].h_ratio
                # bbox = [value/self.letter_box_info_list[-1].ratio for value in bbox]

        if in_format == "xyxy":
            # change xyxy to xywh
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
        else:
            assert (
                False
            ), "now only support xyxy format, please add code to support others format"

        def single_encode(x):
            from pycocotools.mask import encode

            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        if pred_masks is None:
            self.record_list.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x, 3) for x in bbox],
                    "score": round(score, 5),
                }
            )
        else:
            rles = single_encode(pred_masks)
            self.record_list.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x, 3) for x in bbox],
                    "score": round(score, 5),
                    "segmentation": rles,
                }
            )

    def export_to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.record_list, f)


def readClasses(filename):
    with open(filename, "rb") as f:
        classes = f.read().split()
    classes = [str(i) for i in classes]
    id_list = [i + 1 for i in range(len(classes))]
    return classes, id_list


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score >= OBJ_THRESH)
    scores = (class_max_score)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part


def dfl(position):
    # Distribution Focal Loss (DFL)
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(
        1, 2, 1, 1
    )

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    # input_data[0], input_data[4], and input_data[8] are detection box information
    # input_data[1], input_data[5], and input_data[9] are category score information
    # input_data[2], input_data[6], and input_data[10] are confidence score information
    # input_data[3], input_data[7], and input_data[11] are segmentation information
    # input_data[12] is the proto information
    proto = input_data['proto']
    boxes, scores, classes_conf, seg_part = [], [], [], []
    defualt_branch = 3
    pair_per_branch = 3
    input_data_list = [input_data['output0'], input_data['output1'], input_data['output2'],
                       input_data['output3'], input_data['output4'], input_data['output5'],
                       input_data['output6'], input_data['output7'], input_data['output8'],]

    for i in range(defualt_branch):
        boxes.append(box_process(input_data_list[pair_per_branch * i]))
        conf = sigmoid(input_data_list[pair_per_branch * i + 1])
        classes_conf.append(conf)
        conf_sum = np.clip(np.sum(conf, axis=1, keepdims=True), 0, 1)
        scores.append(np.ones_like(conf_sum, dtype=np.float32))
        seg_part.append(input_data_list[pair_per_branch * i + 2])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_part = [sp_flatten(_v) for _v in seg_part]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_part = np.concatenate(seg_part)

    # filter according to threshold
    boxes, classes, scores, seg_part = filter_boxes(
        boxes, scores, classes_conf, seg_part
    )

    zipped = zip(boxes, classes, scores, seg_part)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)

    max_nms = 30000
    n = boxes.shape[0]  # number of boxes
    if not n:
        return None, None, None, None
    elif n > max_nms:  # excess boxes
        boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_part = [np.array(x) for x in result]

    # nms
    nboxes, nclasses, nscores, nseg_part = [], [], [], []
    agnostic = 0
    max_wh = 7680
    c = classes * (0 if agnostic else max_wh)
    ids = torchvision.ops.nms(
        torch.tensor(boxes, dtype=torch.float32)
        + torch.tensor(c, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(scores, dtype=torch.float32),
        NMS_THRESH,
    )
    real_keeps = ids.tolist()[:MAX_DETECT]
    nboxes.append(boxes[real_keeps])
    nclasses.append(classes[real_keeps])
    nscores.append(scores[real_keeps])
    nseg_part.append(seg_part[real_keeps])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_part = np.concatenate(nseg_part)

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_part.shape[-1], -1)
    seg_img = np.matmul(seg_part, proto)
    seg_img = sigmoid(seg_img)
    seg_img = seg_img.reshape(-1, ph, pw)

    seg_threadhold = 0.5

    # crop seg outside box
    seg_img = F.interpolate(
        torch.tensor(seg_img)[None],
        torch.Size([640, 640]),
        mode="bilinear",
        align_corners=False,
    )[0]
    seg_img_t = _crop_mask(seg_img, torch.tensor(boxes))

    seg_img = seg_img_t.numpy()
    seg_img = seg_img > seg_threadhold
    return boxes, classes, scores, seg_img


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print(
            "%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score)
        )
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            image,
            "{0} {1:.2f}".format(CLASSES[cl], score),
            (top, left - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def _crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[
        None, None, :
    ]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[
        None, :, None
    ]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def merge_seg(image, seg_img, classes):
    color = Colors()
    for i in range(len(seg_img)):
        seg = seg_img[i]
        seg = seg.astype(np.uint8)
        seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
        seg = seg * color(classes[i])
        seg = seg.astype(np.uint8)
        image = cv2.add(image, seg)
    return image


def img_check(path):
    img_type = [".jpg", ".jpeg", ".png", ".bmp"]
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


CLASSES = ("PAPER", "MASK", "THREAD", "OBSTACLE", "BOTTLE", "BLOOD", "PERSON", "REFLECTION", "BASE", "CHAIR", "WAITING_BENCH",
           "ESCALATOR_ENTRANCE", "ROD", "COFFEE", "MILK", "CARPET", "FLOOR_LAMP", "WHEELCHAIR", "HOSPITAL_BED", "DOOR_STOPPER", "DOOR ")

coco_id_list = [i for i in range(1, 22)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with YOLOv8")
    # basic params
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="model path, could be [onnx, rknn, pt] file",
    )

    parser.add_argument(
        "--img-save", action="store_true", default=False, help="save the result"
    )

    # data params
    parser.add_argument(
        "--img-folder", type=str, default="../model", help="img folder path"
    )
    parser.add_argument("--save-path", type=str, default="./", help="img folder path")

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)

    file_list = sorted(os.listdir(args.img_folder))
    co_helper = COCO_test_helper(enable_letter_box=True)
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.img_save:
        image_path = os.path.join(args.save_path, "image")
        if not os.path.exists(image_path):
            os.mkdir(image_path)

    result_path = os.path.join(args.save_path, "result")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

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
        h, w, _ = img_src.shape
        if img_src is None:
            continue

        img = co_helper.letter_box(
            im=img_src.copy(),
            new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
            pad_color=(114, 114, 114),
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocee if not rknn model
        if platform in ["pytorch", "onnx", "TensorRT"]:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.0
        else:
            input_data = img

        start = time.time()
        outputs = model.run([input_data])
        inference_time += time.time() - start
        boxes, classes, scores, seg_img = post_process(outputs)

        txt_file_path = os.path.join(result_path, img_name[:-4] + ".txt")
        a = open(txt_file_path, "w")

        if boxes is not None:
            real_boxs = co_helper.get_real_box(boxes)
            real_segs = co_helper.get_real_seg(seg_img)
            img_p = merge_seg(img_src, real_segs, classes)

            for idx, mask in enumerate(real_segs):
                # print(f'mask.shape : {mask.shape}')
                line = ""
                str_seg_li = []
                mask = np.uint8(mask)
                edges = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]

                for edge in edges:
                    edge = edge.reshape((-1, 2)).astype(np.float32)
                    edge[:, 0] /= w
                    edge[:, 1] /= h
                    edge = edge.reshape((-1))
                    str_seg = "%g " * len(edge) % tuple(edge)
                    str_seg_li.append(str_seg)

                line += str(classes[idx]) + "; "
                # boxes
                cur_left_norm = real_boxs[idx, 0] / w
                cur_right_norm = real_boxs[idx, 2] / w
                cur_top_norm = real_boxs[idx, 1] / h
                cur_bottom_norm = real_boxs[idx, 3] / h
                line += (
                    str(cur_left_norm)
                    + " "
                    + str(cur_top_norm)
                    + " "
                    + str(cur_right_norm)
                    + " "
                    + str(cur_bottom_norm)
                    + " "
                    + "; "
                )
                for str_seg in str_seg_li:
                    line += str_seg + ";"
                line += " " + str(scores[idx])
                a.write(line + "\n")

            if args.img_save:
                print("\n\nIMG: {}".format(img_name))

                draw(img_p, real_boxs, scores, classes)
                img_path = os.path.join(image_path, img_name)
                cv2.imwrite(img_path, img_p)
                print("The segmentation results have been saved to {}".format(img_path))
        else:
            if args.img_save:
                print("\n\nIMG: {}".format(img_name))

                img_path = os.path.join(image_path, img_name)
                cv2.imwrite(img_path, img_src)
                print("The segmentation results have been saved to {}".format(img_path))

        a.close()
    print("average inference time: ", inference_time / len(img_list))

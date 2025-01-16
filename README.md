# AI Deployment

This repository contains the code for deploying an AI model using python api, including
1. Model Export
- export pytorch model to ONNX format
- export ONNX model to TensorRT engine
- export ONNX model to RKNN model
2. Quantization
- int8 quantization with pytorch-quantization for TensorRT
- int8 quantization with rknn-toolkit for RKNN
3. Inference
- inference with pytorch model
- inference with ONNX model
- inference with RKNN model
- inference with TensorRT engine
4. Now support models (Updating):
- yolov8

# Environment
## Build docker image
`docker build . -t ai-deployment:latest`
## Run docker container
`docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --gpus all --shm-size 8G ai-deployment:latest /bin/bash`
## Install pytorch-quantization
`cd /app/pytorch-quantization && python3 setup install`

# Usage
1. export pytorch model to TensorRT engine (without quantization)
- step1: export pytorch model to ONNX Format \
`python3 export/onnx_export/yolov8_seg.py -w weights/best.pt --opset 13 --sim`

- step2: export ONNX model to TensorRT engine \
`python3 export/trt_export.py --onnx weights/best.onnx --precision fp32`
- step3: inference with TensorRT engine \
Create a folder and put your images in it. \
`python3 inference/yolov8_seg.py --model-path weights/best.trt --img-folder ./images --save-path ./output/trt --img-save`

2. export pytorch model to TensorRT engine (with int8 quantization)
- step1: Use calibration data to quantize model and export pytorch model to ONNX Format \
`python3 export/onnx_export/yolov8_seg_ptq.py -w weights/best.pt --calib-data ./calib_data/calib_data.txt --opset 13 --sim`. \
You should create a txt file that contains the path of calibration images, like this: 
```
/app/images/cam2.1702989848.748686000.jpg
/app/images/cam2.1703734518.928999936.jpg
/app/images/cam2.1702989842.674320000.jpg
/app/images/cam2.1703064274.487586000.jpg
/app/images/cam2.1702990031.959408000.jpg
/app/images/cam2.1703063118.673907000.jpg
/app/images/cam2.1702990039.614835000.jpg
/app/images/cam2.1679942868.912214000.jpg
/app/images/cam1.1703734428.895000064.jpg
/app/images/cam2.1703064120.593955000.jpg
/app/images/cam2.1703734399.968999936.jpg
/app/images/cam2.1703064237.528560000.jpg
```

- step2: export ONNX model to TensorRT engine \
`python3 export/trt_export.py --onnx weights/best-ptq.onnx --precision int8`
- step3: inference with TensorRT engine \
Create a folder and put your images in it. \
`python3 inference/yolov8_seg.py --model-path weights/best-ptq.trt --img-folder ./images --save-path ./output/trt-int8 --img-save`
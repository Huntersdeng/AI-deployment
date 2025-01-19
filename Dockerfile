FROM nvcr.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

RUN sed -i 's|http://ports.ubuntu.com/|http://mirrors.ustc.edu.cn/|g' /etc/apt/sources.list
RUN sed -i 's|http://archive.ubuntu.com/|http://mirrors.ustc.edu.cn/|g' /etc/apt/sources.list
RUN sed -i 's|http://security.ubuntu.com/|http://mirrors.ustc.edu.cn/|g' /etc/apt/sources.list
RUN apt-get update

RUN apt-get install -y --no-install-recommends tensorrt-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer8=8.6.0.12-1+cuda12.0 \
    libnvinfer-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-plugin-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-plugin8=8.6.0.12-1+cuda12.0 \
    libnvinfer-headers-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-headers-plugin-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-lean-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-lean8=8.6.0.12-1+cuda12.0 \
    libnvinfer-dispatch-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-dispatch8=8.6.0.12-1+cuda12.0 \
    libnvinfer-vc-plugin-dev=8.6.0.12-1+cuda12.0 \
    libnvinfer-vc-plugin8=8.6.0.12-1+cuda12.0 \
    libnvonnxparsers8=8.6.0.12-1+cuda12.0 \
    libnvonnxparsers-dev=8.6.0.12-1+cuda12.0 \
    libnvparsers8=8.6.0.12-1+cuda12.0 \
    libnvparsers-dev=8.6.0.12-1+cuda12.0 \
    cuda-toolkit-config-common=12.3.101-1 \
    cuda-toolkit-12-config-common=12.3.101-1 \
    libcublas-12-4- cuda-toolkit-12-4-config-common- \
    libcublas-12-5- cuda-toolkit-12-5-config-common- \
    libcublas-12-6- cuda-toolkit-12-6-config-common-

RUN apt update
RUN apt install -y --no-install-recommends cmake
RUN apt install -y --no-install-recommends python3 python3-dev python3-pip 
RUN apt install -y --no-install-recommends python3-libnvinfer-dev=8.6.0.12-1+cuda12.0 \ 
    python3-libnvinfer-lean=8.6.0.12-1+cuda12.0 \
    python3-libnvinfer-lean=8.6.0.12-1+cuda12.0 \
    python3-libnvinfer=8.6.0.12-1+cuda12.0 \
    python3-libnvinfer-dispatch=8.6.0.12-1+cuda12.0

RUN apt install -y --no-install-recommends libgl1 libglib2.0-0

WORKDIR /app
COPY requirements.txt .

RUN pip install -r requirements.txt  -i https://pypi.mirrors.ustc.edu.cn/simple
WORKDIR /app/pytorch-quantization
COPY pytorch-quantization .
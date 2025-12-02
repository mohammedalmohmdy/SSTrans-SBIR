FROM nvidia/cuda:11.7-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip
COPY . /workspace
WORKDIR /workspace
RUN pip3 install -r requirements.txt
ENTRYPOINT ["bash"]

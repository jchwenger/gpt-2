# FROM nvidia/cuda:10.2-base-ubuntu18.04
# FROM python:3.7.3-slim-stretch
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# RUN apt-get -y update && apt-get -y install gcc

# RUN pip3 --no-cache-dir install \
# 	numpy==1.16.4 \
# 	tensorflow-gpu==1.14.0

ENV LANG=C.UTF-8
RUN mkdir /gpt-2
WORKDIR /gpt-2
ADD . /gpt-2

RUN pip3 install -r requirements.txt

# # Clean up APT when done.
# RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PYTHONPATH "${PYTHONPATH}:src"

# RUN python3 download_model.py 117M
# RUN python3 download_model.py 345M

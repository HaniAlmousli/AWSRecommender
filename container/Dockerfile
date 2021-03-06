# Build an image that can do training and inference in SageMaker
# This is a Python 3.5 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Hani <hani.mousli@gmail.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3.5 \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Here we get all python packages.

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.5 get-pip.py
RUN pip install numpy scipy scikit-learn pandas flask gevent gunicorn 
RUN (cd /usr/local/lib/python3.5/dist-packages/scipy/.libs; rm *; ln ../../numpy/.libs/* .) && \
        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY collaborative_filtering /opt/program
WORKDIR /opt/program


FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
MAINTAINER ethylomat

USER root
RUN mkdir -p /app/log  && \
	mkdir -p /app/data && \
	mkdir -p /input && \
	mkdir -p /output && \
	apt-get update && apt-get install --no-install-recommends -y python3-opencv && apt-get clean && rm -rf /var/lib/apt/lists/* && \
	pip install --no-cache scikit-image opencv-python tqdm nibabel argparse

ADD models /app/models
ADD src /app/src
ADD models /app/models
RUN mkdir -p /app/log
RUN mkdir -p /app/data

ENTRYPOINT ["/bin/sh"] 
CMD [] 
USER 0
WORKDIR /app
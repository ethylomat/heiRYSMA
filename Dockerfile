FROM bitnami/pytorch
MAINTAINER ethylomat

USER root
RUN mkdir -p /app/log  && \
	mkdir -p /app/data && \
	mkdir -p /input && \
	mkdir -p /output && \
	apt-get update && apt-get install --no-install-recommends -y python3-opencv && apt-get clean && rm -rf /var/lib/apt/lists/* && \
	pip install --no-cache scikit-image opencv-python tqdm nibabel

ADD models /app/models
ADD src /app/src

ENTRYPOINT ["/bin/sh", "-c"]
CMD []
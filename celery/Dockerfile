FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
MAINTAINER ethylomat

USER root
RUN mkdir -p /app/log  && \
	mkdir -p /app/data && \
	mkdir -p /input && \
	mkdir -p /output && \
	apt-get update && apt-get install --no-install-recommends -y python3-opencv && apt-get clean && rm -rf /var/lib/apt/lists/* && \
	pip install --no-cache scikit-image opencv-python tqdm nibabel

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0
ENV C_FORCE_ROOT true

COPY . /queue
RUN mkdir -p /queue/src
RUN mkdir -p /queue/models
WORKDIR /queue

RUN pip install -r requirements.txt

# production
#ENTRYPOINT celery -A tasks worker --loglevel=info

# hot code reloading
ENTRYPOINT watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery -A tasks worker --concurrency=1 --loglevel=INFO

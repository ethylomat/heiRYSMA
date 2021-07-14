FROM bitnami/pytorch

USER root
RUN apt-get update &&\
	apt-get install -y python3-opencv
RUN pip install --no-cache scikit-image opencv-python tqdm nibabel

ADD src /app/src
ADD models /app/models
RUN mkdir -p /app/log
RUN mkdir -p /app/data
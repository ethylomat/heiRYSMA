import os
import sys
import time
from celery import Celery

from src.main_challenge import main

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379'),
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379')

celery = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@celery.task(name='tasks.add')
def add(x: int, y: int) -> int:
    time.sleep(5)
    return x + y


@celery.task(name='tasks.process')
def process(sample_id: str) -> str:
    model = "model__BCE__256_256_0__o01__b10__lr0001"
    main(data=os.path.join("/temp", sample_id), model="/queue/models/" + model, resolution=[256,256,0], overlap=1, loss=None, webapp=True)
    return sample_id
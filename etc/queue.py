#!/usr/bin/python
from submission_helper import submit

"""
batch_size=8, 
overlap=30, 
resolution=[64, 64, 64], 
learning_rate="0.001", 
wall_time="01:59:00", 
loss="DIC", ["DIC", "BCE", "FOC"]
partition="gpu_8", ["dev_gpu_4", "gpu_4", gpu_8]
resizing=False
"""

if __name__ == '__main__':
    for b in [4]:
        submit(prefix="test_", learning_rate= "0.001", batch_size=b, resolution=[256, 256, 0], overlap=1, loss="BCE", wall_time="00:30:00", partition="dev_gpu_4", resizing=True)
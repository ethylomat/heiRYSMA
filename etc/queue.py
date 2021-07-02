#!/usr/bin/python
import os, time

def main():
    for i in range(4):
        submit(learning_rate= "0.0001", overlap=30)
        submit(learning_rate=  "0.001", overlap=30)
        submit(learning_rate=   "0.01", overlap=30)

def submit(batch_size=8, overlap=30, resolution=64, learning_rate="0.001", wall_time="01:59:00", gpu=8):
    job_name = f"o{overlap}lr{str(learning_rate).replace('0.', '')}"
    try:
        os.mkdir(os.path.join("log", job_name))
    except:
        pass
    output = f"log/{job_name}/%J_info.txt"
    error = f"log/{job_name}/%J_error.txt"
    command = f'sbatch -d singleton --export=BATCH_SIZE={batch_size},OVERLAP={overlap},RESOLUTION="{resolution} {resolution} {resolution}",LEARNING_RATE={learning_rate},ID="{job_name}" --job-name="{job_name}" --partition=gpu_{gpu} --gres=gpu:{gpu} -t {wall_time} --output={output} --error={error} submit.sh'
    print("Running command: ", command)
    os.system(command)
    time.sleep(1)

if __name__ == '__main__':
    main()
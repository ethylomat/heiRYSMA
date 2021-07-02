#!/usr/bin/python
import os, time

prefix = ""

def main():
    for i in range(1):
        submit(learning_rate= "0.0001", overlap=30)

def submit(batch_size=8, overlap=30, resolution=64, learning_rate="0.001", wall_time="01:59:00", loss="DIC", partition="gpu_8"):
    job_name = f"{prefix}o{overlap}lr{str(learning_rate).replace('0.', '')}_{loss}"
    try:
        os.mkdir(os.path.join("log", job_name))
    except:
        pass
    output = f"log/{job_name}/%J_output.txt"
    error = f"log/{job_name}/%J_error.txt"

    gpu_reservation = ""
    if "gpu_8" in partition:
        gpu_reservation = " --gres=gpu:8"
    if "gpu_4" in partition:
        gpu_reservation = " --gres=gpu:4"

    command = f'sbatch -d singleton --export=BATCH_SIZE={batch_size},OVERLAP={overlap},RESOLUTION="{resolution} {resolution} {resolution}",LEARNING_RATE={learning_rate},ID="{job_name}",LOSS={loss} --job-name="{job_name}" --partition={partition}{gpu_reservation} -t {wall_time} --output={output} --error={error} submit.sh'
    print("Running command: ", command)
    os.system(command)
    time.sleep(1)

if __name__ == '__main__':
    main()
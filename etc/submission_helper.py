import os, time

def submit(batch_size=8, overlap=30, resolution=[64, 64, 64], learning_rate="0.001", wall_time="01:59:00", loss="DIC", partition="gpu_8", prefix="", resizing=False):
    job_name = f"{prefix}b_{batch_size}o{overlap}lr{str(learning_rate).replace('0.', '')}_{loss}"
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

    resizing_arg = ""
    if resizing:
        resizing_arg = "RESIZING=1,"

    command = f'sbatch -d singleton --export={resizing_arg}BATCH_SIZE={batch_size},OVERLAP={overlap},RESOLUTION="{resolution[0]} {resolution[1]} {resolution[2]}",LEARNING_RATE={learning_rate},ID="{job_name}",LOSS={loss} --job-name="{job_name}" --partition={partition}{gpu_reservation} -t {wall_time} --output={output} --error={error} submit.sh'
    print("Running command: ", command)
    os.system(command)
    time.sleep(1)
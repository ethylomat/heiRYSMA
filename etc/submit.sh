#!/bin/bash
# Submission script for the bw-uni-cluster

####### General Settings (set name + executable)
#SBATCH --job-name="heiRYSMA"
#SBATCH --export=ALL

####### Notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

####### Output
#SBATCH --output=%x_%J_info.txt
#SBATCH --error=%x_%J_error.txt

####### Partition type (dev_single, single, dev_multiple, multiple, dev_multiple_e, multiple_e, fat, dev_gpu_4, gpu_4, gpu_8)
# See: https://wiki.bwhpc.de/e/BwUniCluster_2.0_Batch_Queues
#SBATCH --partition=dev_gpu_4

####### Resources
#SBATCH --time=0-01:59:30
#SBATCH --nodes=1

module load 'numlib/python_scipy/1.5.2_numpy_1.19.1_python_3.8.6_intel_19.1'
runcommand="./run.sh"
echo $runcommand
exec $runcommand
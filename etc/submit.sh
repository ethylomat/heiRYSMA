#!/bin/bash
# Submission script for the bw-uni-cluster

####### General Settings (set name + executable)
#SBATCH --job-name="heiRYSMA"
#SBATCH --export=ALL,EXECUTABLE=./run.sh

####### Notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

####### Output
#SBATCH --output=info.txt
#SBATCH --error=error.txt

####### Partition type (dev_single, single, dev_multiple, multiple, dev_multiple_e, multiple_e, fat, dev_gpu_4, gpu_4, gpu_8)
# See: https://wiki.bwhpc.de/e/BwUniCluster_2.0_Batch_Queues
#SBATCH --partition=dev_gpu_4

####### Resources
#SBATCH --time=0-00:03:00
#SBATCH --nodes=1


######## OpenMPI
#module load ${MPI_MODULE}
#export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))
#export MPIRUN_OPTIONS="--bind-to core --map-by socket:PE=${OMP_NUM_THREADS} -report-bindings"
#export NUM_CORES=${SLURM_NTASKS}*${OMP_NUM_THREADS}
#echo "${EXECUTABLE} running on ${NUM_CORES} cores with ${SLURM_NTASKS} MPI-tasks and ${OMP_NUM_THREADS} threads"

#runcommand="time mpirun -n ${SLURM_NTASKS} ${MPIRUN_OPTIONS} ${EXECUTABLE} > output"

module load 'numlib/python_scipy/1.5.2_numpy_1.19.1_python_3.8.6_intel_19.1'
runcommand="${EXECUTABLE}"
echo $runcommand
exec $runcommand

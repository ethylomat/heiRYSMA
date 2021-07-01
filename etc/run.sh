#!/bin/bash
source ~/heiRYSMA/venv/bin/activate
cd ~/heiRYSMA

# Pass parameters to sbatch using --export=BATCH_SIZE=10,OVERLAP=10,RESOLUTION="32 32 32",LEARNING_RATE=0.0001
args=""
if [[ -z "${BATCH_SIZE}" ]]; then
  args="${args}"
else
  args="${args} --batch-size ${BATCH_SIZE}"
fi

if [[ -z "${OVERLAP}" ]]; then
  args="${args}"
else
  args="${args} --overlap ${OVERLAP}"
fi

if [[ -z "${RESOLUTION}" ]]; then
  args="${args}"
else
  args="${args} --resolution ${RESOLUTION}"
fi

if [[ -z "${LEARNING_RATE}" ]]; then
  args="${args}"
else
  args="${args} --learning-rate ${LEARNING_RATE}"
fi

python -u -m src.main $args >&1 | tee ~/heiRYSMA/etc/output_$ID.txt
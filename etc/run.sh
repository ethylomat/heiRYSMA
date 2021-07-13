#!/bin/bash
source ~/heiRYSMA/venv/bin/activate
cd ~/heiRYSMA

# Sending pushover message with id
../message.sh $ID

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

if [[ -z "${LOSS}" ]]; then
  args="${args}"
else
  args="${args} --loss ${LOSS}"
fi

if [[ -z "${RESIZING}" ]]; then
  args="${args}"
else
  args="${args} --resizing"
fi

python -u -m src.main $args >&1
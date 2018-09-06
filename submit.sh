#!/usr/bin/env bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q debug-flat-quad
#COBALT -A datascience
#COBALT --jobname atlas_yolo

MODELDIR=/home/parton/git/atlas-yolo

module load tensorflow
module load horovod
echo PYTHON_VERSION=$(python --version 2>&1 )

export OMP_NUM_THREADS=128
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"

export TF_INTRA_THREADS=$OMP_NUM_THREADS
export TF_INTER_THREADS=1

env | sort

echo [$SECONDS] print tensorflow version
aprun -n 1 -N 1 python -c "import tensorflow as tf;print('tensorflow version: ',tf.__version__)"
echo [$SECONDS] run job
aprun -n 1 -N 1 -cc none python $MODELDIR/atlas_yolo.py -c atlas_yolo.json -n 10 -a
echo [$SECONDS] exited $?

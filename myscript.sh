#!/bin/bash
# Request 32 gigabytes of real memory (mem)
#$ -l rmem=32G
# Request 8 processor cores
#$ -pe smp 8
# Email notifications
#$ -M utalalaj1@sheffield.ac.uk
# Email notifications if the job aborts
#$ -m ea

# load conda
module load apps/python/conda

# activate environment
source activate iqt_env

# run
python train_preprocess.py
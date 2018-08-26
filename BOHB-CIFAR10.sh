#!/bin/bash
#
# submit to the right queue
#SBATCH -p meta_gpu-ti
#SBATCH --array=1-10
#
# the execution will use the current directory for execution (important for relative paths)
#SBATCH -D .
#
# redirect the output/error to some files
#SBATCH -o ./logs/%A-%a.o
#SBATCH -e ./logs/%A-%a.e
#
#
source activate pytorch
python cifar10_master.py --array_id $SLURM_ARRAY_TASK_ID --total_num_workers 10 --num_iterations 32 --run_id $SLURM_ARRAY_JOB_ID --working_directory ./data/run_$SLURM_ARRAY_JOB_ID

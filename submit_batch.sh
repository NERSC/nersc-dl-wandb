#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH -C gpu
#SBATCH --account=<your_account>
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J dl-wandb-test
#SBATCH --module=gpu,nccl-2.15
#SBATCH --image=nersc/pytorch:ngc-22.09-v0 
#SBATCH -o job_log_%j.out

config_file=./config/default.yaml
config="test2"
run_num="0"

env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc-22.09-v0

# this is needed only for multinode jobs to avoid some hangs related
# to the nccl plugin for large scale distributed jobs
export FI_MR_CACHE_MONITOR=userfaultfd

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

# for DDP
export MASTER_ADDR=$(hostname)

# store results in some dir in your $SCRATCH
scratch="$SCRATCH/results/logging_tests/"
cmd="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$scratch"

set -x
srun -l shifter --env PYTHONUSERBASE=${env} \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    " 

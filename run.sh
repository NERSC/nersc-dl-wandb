#!/bin/bash
export MASTER_ADDR=$(hostname) # For DDP
image=nersc/pytorch:ngc-22.09-v0  # use latest nersc pytorch containers for libs

# userbase for user-specific libs
# guidelines for using a container: within any container create PYTHONUSERBASE env var to point to some directory
# and any additional libs you need on top of the ones included in the container can be installed
# with pip install --user <package_name> (for ex: pip install --user collection is needed here)
env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc-22.09-v0

ngpu=4 # number of GPUs (single node)
config_file=./config/default.yaml
config="test2"
run_num="0"

# store results in some dir in your $SCRATCH
scratch="$SCRATCH/results/logging_tests/"
# run command
cmd="python train.py --yaml_config=$config_file --config=$config --run_num=$run_num --root_dir=$scratch"

# srun to get multiple GPUs; source DDP vars to use pytorch DDP
srun -l -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --env PYTHONUSERBASE=${env} bash -c "source export_DDP_vars.sh && $cmd"

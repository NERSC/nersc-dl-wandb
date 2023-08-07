# Experiment tracking and HPO for DL applications on NERSC 
We recommend [Weights & Biases](https://wandb.ai/site) (W&B) as a ML tracking platform for general logging, tracking, and HPO. This codebase provides a PyTorch framework to build a data-parallel, multi-GPU DL application using `DistributedDataParallel` (DDP) on the Perlmutter machine, with basic W&B experiment tracking and hyperparameter optimization (HPO) capabilities.

## Layout
- Configuration files (in YAML format) are in `configs/`. An example config is in `configs/test.yaml`
- Data, trainer, and other miscellaneous utilities are in `utils/`. We use standard PyTorch dataloaders and models wrapped with DDP for distributed data-parallel training.
- A simple CNN example model is in `models/`.
- Environment variables for DDP (local rank, master port etc) are set in `export_DDP_vars.sh` to be sourced before running any distributed training. See the [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details on using DDP. Since Perlmutter uses Slurm to schedule and launch jobs, this repository configures DDP with the standard NCCL backend for GPU communication using Slurm environment variables like `SLURM_PROCID` and `SLURM_LOCALID`.
- Example running scripts are in `run.sh` (4 GPU DDP train script) and `run_sweep.sh` (for HPO with W&B). The run scripts use [shifter](https://docs.nersc.gov/development/shifter/how-to-use/) images to provide a compelte Python environment with all required PyTorch libraries and dependencies. You may also use ``module load pytorch``, or your own custom PyTorch environment, instead. Please refer to the [NERSC ML documentation](https://docs.nersc.gov/machinelearning/pytorch/#containers) for general guidelines for running PyTorch on NERSC machines. Since `run.sh` and `run_sweep.sh` use `srun` to launch tasks, they cannot be run on a login node; use either `sbatch` or `salloc` to [get a job allocation on a GPU compute node](https://docs.nersc.gov/systems/perlmutter/running-jobs/) first.
- An example sbatch script is in ``submit_batch.sh`` for submitting the training jobs to the system; you can modify it to use either `run.sh` or `run_sweep.sh` to launch training, depending on what type of training you want to do (standard or hyperparameter search).

Steps to run the minimal example:
- To get started, create a W&B account, project name (to log results), and entity (project team; if this is not needed, this can just be your username) and login to W&B on the terminal. See [Quickstart](https://docs.wandb.ai/quickstart) for details.
-  Populate your entity and project name in the configuration file `configs/test.yaml`.
-  Run using `sbatch submit_batch.sh` (or `bash run.sh` if on an interactive node). This script uses the NERSC provided NGC (NVIDIA GPU Cloud) containers for PyTorch and other libs. See comments in the respective run scripts on how to setup your environment.

## Features
Refer to [Weights & Biases docs](https://docs.wandb.ai/) for full details. Below, we outline the general features for tracking and HPO with multiGPU training that are included in this repository code. The code does the following:

- Read in hyperparameters and general configuration from the specified `yaml` config file
- Based on the config, setup a minimal `Trainer` class that contains standard training and validation code, using a dummy dataloader (providing 2D images of random noise) and simple CNN model performing a regression task (common in many SciML applications). Users should directly replace the example model and data loaders here with ones relevant to their applications.
- After initializiton, wrap the model with DDP and configure the dataloaders for data-parallel training.
- Allow model saving and checkpointing to resume training (useful for jobs which need to run longer than the 6 hour time limit on Perlmutter).
- W&B logging (with DDP) of user-defined metrics/images, and W&B HPO sweeps for grid search of hyperparameters.

We list some specific features with the corresponding code snippets below. In the `Trainer` object, all configuration fields and hyperparameters from
the `yaml` config file are stored in the `self.params` object, which exposes the hyperparameters with both a dictionary-like interface (e.g., `model = self.params['model']`) and an object-oriented interface (e.g. `model = self.params.model`). Users may use whichever style they prefer.
- Logging metrics in wandb: Initialize W&B with output directories, config and project details, and resume flag for checkpointing with:
	```
	wandb.init(dir=os.path.join(exp_dir, "wandb"),
				config=self.params.params, 
				name=self.params.name, 
				group=self.params.group, project=self.params.project,
				entity=self.params.entity, resume=self.params.resuming)
	```
	Log metrics such as loss values, hyperparameters, timings, and more with:
	
	```
	self.logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
	self.logs['time_per_epoch'] = tr_time
	wandb.log(self.logs, step=self.epoch+1)
	```
	Some logged metrics such as average loss need to be aggregated across GPUs with `torch.distributed.all_reduce`:
	
	```
	logs_to_reduce = ['train_loss', 'grad']
	if dist.is_initialized(): # reduce the logs across multiple GPUs
		for key in logs_to_reduce:
			dist.all_reduce(self.logs[key].detach()) # PyTorch distributed all-reduce (aggregates values via summation)
			self.logs[key] = float(self.logs[key]/dist.get_world_size()) # Divide by number of GPUs to get the average
	```
	User-defined matplotlib plots to track images (such as predictions) logged as follows:
	
	```
	fig = vis_fields(fields_to_plot, self.params)
	self.logs['vis'] = wandb.Image(fig)
	plt.close(fig)
	```

- HPO sweeps: W&B contains the [sweep](https://docs.wandb.ai/guides/sweeps) functionality that allows for automated search of hyperparameters. 
	An example sweep config that grid searches across different learning rates and batch sizes is in ``config/sweep_config.yaml``. First, create a sweep instance for the W&B agent to automatically sweep parameters in the config with:
	```
	shifter --image=nersc/pytorch:ngc-22.09-v0 wandb sweep config/sweep_config.yaml
	```
 	In the above line, we again use a PyTorch shifter image for the required libraries; `shifter --image=...` is not needed if using a different environment.

	Then, get the sweep ID output by the previous command and use that for the `sweep_id` in the run script ``run_sweep.sh``. When you launch the script (``bash run_sweep.sh``, the W&B agent will automatically take the base config specified in the run script and change the hyperparameters according to the sweep rule in the sweep config -- each time you run the script, a different set of hyperparameters is passed to the trainer allowing for parallel submission of several job scripts to sweep the full range of values.
	In the code: if sweep is enabled, then the run is launched using W&B agent:
	```
	wandb.agent(args.sweep_id, function=trainer.launch, count=1, entity=trainer.params.entity, project=trainer.params.project)
	```
  	For convenience, we can change the name of each sweep trial config in the W&B UI. Note this is not the default behavior in W&B (which automatically renames sweep runs with a placeholder name), but in practice it makes it much easier to analyze the final results of a large HPO sweep when there is a consistent naming scheme. The renaming is done using a context manager:
	
	```
	with wandb.init() as run:
		hpo_config = wandb.config
		self.params.update_params(hpo_config)
		# rename sweeps according to the swept parameters on wandb 
		logging.info(self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id))
		run.name = self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id)
	```
	
	For example: the ``sweep_name_suffix`` function, renames the trials based on the actual learning rate and batch size used:
	
	```
	if sweep_id in ['<your_sweep_id']:
		return 'lr%s_batchsize%d'%(format_lr(params.lr), params.batch_size)
	```
	This makes it easier to track the results and locate the trained models after the sweep has completed.

	For HPO with DDP (each HPO on multiple GPUs), we need to take additional care to ensure (1) only one of the DDP processes interacts with W&B backend to get hyperparameters for the trial, and (2) the other DDP processes are updated with the trial hyperparameters so all processes have a consistent configuration. 

    ```
    if self.sweep_id and dist.is_initialized():
        # Broadcast sweep config to other ranks
        if self.world_rank == 0: # where the wandb agent has changed params
            objects = [self.params]
        else:
            self.params = None
            objects = [None]

        dist.broadcast_object_list(objects, src=0)
        self.params = objects[0]
    ```
	Finally, we note the W&B sweeps functionality is not yet able to perform seamless checkpoint and restart of individual hyperparameter trial runs. This means that if any of your trials need to run for longer than six hours, you will have to implement a custom checkpoint and restart setup, or [get a reservation](https://docs.nersc.gov/jobs/reservations/) to run longer than the standard 6 hour time limit. Further discussion on this can be found on the `wandb` community site [in this thread](https://community.wandb.ai/t/resuming-sweep-runs-on-a-cluster-with-job-time-limits/3333).

import os, sys, time
import pickle
import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_utils import get_data_loader
from utils.optimizer_utils import set_scheduler, set_optimizer
from utils.misc_utils import compute_grad_norm, vis_fields
from utils.sweep_utils import sweep_name_suffix
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from collections import OrderedDict
import matplotlib.pyplot as plt

# models
import models.cnn

# wandb
import wandb

def set_seed(params, world_size):
    seed = None
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        ''' init vars for distributed training (ddp) and logging'''
        self.sweep_id = args.sweep_id # for wandb sweeps
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1: # multigpu, use DDP with standard NCCL backend for communication routines
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0

        params['name'] = args.config + '_' + args.run_num
        params['group'] = 'op_' + args.config

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device('cpu')
        self.params = params

        if self.world_rank==0:
            params.log()

    def init_exp_dir(self, exp_dir):
        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'wandb/'))
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False


    def launch(self):
        if self.sweep_id:
            if self.world_rank==0:
                with wandb.init() as run:
                    hpo_config = wandb.config
                    self.params.update_params(hpo_config)
                    # rename sweeps according to the swept parameters on wandb (for some reason this is not the default)
                    logging.info(self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id))
                    run.name = self.params.name+'_'+sweep_name_suffix(self.params, self.sweep_id)
                    self.name = run.name
                    self.params.name = self.name
                    exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.name])
                    self.init_exp_dir(exp_dir)
                    logging.info('HPO sweep %s, trial cfg %s'%(self.sweep_id, self.name))
                    self.build_and_run()
            else:
                self.build_and_run()
        else:
            # no sweeps, just a single run
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config, self.run_num])
            self.init_exp_dir(exp_dir)
            if self.log_to_wandb:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                           config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                           entity=self.params.entity, resume=self.params.resuming)
            self.build_and_run()


    def build_and_run(self):

        if self.sweep_id and dist.is_initialized():
            # broadcast the params to all ranks since the sweep agent has changed it
            if self.world_rank == 0: # where the wandb agent has changed params
                objects = [self.params]
            else:
                self.params = None
                objects = [None]

            dist.broadcast_object_list(objects, src=0)
            self.params = objects[0]

#            # Broadcast sweep config to other ranks using mpi4py
#            from mpi4py import MPI
#            comm = MPI.COMM_WORLD
#            rank = comm.Get_rank()
#            assert self.world_rank == rank
#            if rank != 0:
#                self.params = None
#            self.params = comm.bcast(self.params, root=0)

        if self.world_rank == 0:
            logging.info(self.params.log())

        set_seed(self.params, self.world_size)

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
        self.params['global_valid_batch_size'] = self.params.valid_batch_size
        self.params['local_valid_batch_size'] = int(self.params.valid_batch_size//self.world_size)

        # dump the yaml used
        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile )


        # get the dataloaders
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_path, dist.is_initialized(), train=True)
        self.val_data_loader, self.val_dataset, self.valid_sampler = get_data_loader(self.params, self.params.val_path, dist.is_initialized(), train=False)

        # get the model
        if self.params.model == 'cnn':
            self.model = models.cnn.simple_cnn(self.params).to(self.device)
        else:
            assert(False), "Error, model arch invalid."


        # distributed wrapper for data parallel
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank])


        # set an optimizer and learning rate scheduler
        self.optimizer = set_optimizer(self.params, self.model)
        self.scheduler = set_scheduler(self.params, self.optimizer)

        # set loss functions
        if self.params.loss_func == "mse":
            self.loss_func = torch.nn.MSELoss()
        else:
            assert(False), "Error,  loss func invalid."

        self.iters = 0
        self.startEpoch = 0

        # checkpointing
        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        self.epoch = self.startEpoch
        self.logs = {}
        self.train_loss = self.grad = 0.0
        n_params = count_parameters(self.model)
        if self.log_to_screen:
            logging.info(self.model)
            logging.info('number of model parameters: {}'.format(n_params))

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            logging.info("Starting training loop...")
        best_loss = np.inf

        best_epoch = 0
        best_err = 1
        self.logs['best_epoch'] = best_epoch

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            # training
            tr_time = self.train_one_epoch()
            # validation
            val_time, fields_to_plot = self.val_one_epoch()
            self.logs['wt_norm'] = self.get_model_wt_norm(self.model)

            # learning rate scheduler
            if self.params.scheduler == 'cosine':
                self.scheduler.step()

            # keep track of best model accorinding to validaion loss
            if self.logs['val_loss'] <= best_loss:
                is_best_loss = True
                best_loss = self.logs['val_loss']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch
            
            # log metrics, vis etc
            if self.log_to_wandb:
                # log some visualizations
                fig = vis_fields(fields_to_plot, self.params)
                self.logs['vis'] = wandb.Image(fig)
                plt.close(fig)
                # other logs
                self.logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.logs['time_per_epoch'] = tr_time
                wandb.log(self.logs, step=self.epoch+1)

            # save checkpoint (if best epoch additionally save the best epoch too)
            if self.params.save_checkpoint:
                if self.world_rank == 0:
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)

            # some print statements
            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec; with {}/{} in tr/val'.format(self.epoch+1, time.time()-start, tr_time, val_time))
                logging.info('Loss = {}, Val loss = {}'.format(self.logs['train_loss'], self.logs['val_loss']))

        if self.log_to_wandb:
            wandb.finish()
    
    def get_model_wt_norm(self, model):
        ''' some application specific function that you want to log; here it logs the l2 norm of the weight vector'''
        n = 0
        for p in model.parameters():
            p_norm = p.data.detach().norm(2)
            n += p_norm.item()**2
        n = n**0.5
        return n


    def train_one_epoch(self):
        tr_time = 0
        self.model.train()

        # buffers for logs
        logs_buff = torch.zeros((2), dtype=torch.float32, device=self.device)
        self.logs['train_loss'] = logs_buff[0].view(-1)
        self.logs['grad'] = logs_buff[1].view(-1)


        for i, (inputs, targets) in enumerate(self.train_data_loader):
            self.iters += 1
            data_start = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            tr_start = time.time()

            self.model.zero_grad()
            u = self.model(inputs)

            loss = self.loss_func(u, targets)
            loss.backward()
            self.optimizer.step()

            grad_norm = compute_grad_norm(self.model.parameters()) # some other application specific logger
    
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()
            self.logs['grad'] += grad_norm

            tr_time += time.time() - tr_start

        self.logs['train_loss'] /= len(self.train_data_loader)
        self.logs['grad'] /= len(self.train_data_loader)

        logs_to_reduce = ['train_loss', 'grad']
        if dist.is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time

    def val_one_epoch(self):
        self.model.eval()
        val_start = time.time()

        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.device)
        self.logs['val_loss'] = logs_buff[0].view(-1)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                u = self.model(inputs)
                loss = self.loss_func(u, targets)
                self.logs['val_loss'] += loss.detach()
                if i == 0:
                    target_field = targets[0,0].detach().cpu().numpy() 
                    pred_field = u[0,0].detach().cpu().numpy() 
                    fields_to_plot = [target_field, pred_field]

        self.logs['val_loss'] /= len(self.val_data_loader)
        if dist.is_initialized():
            for key in ['val_loss']:
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        val_time = time.time() - val_start

        return val_time, fields_to_plot

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_best.tar'))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank)) 
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)

        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


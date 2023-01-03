import argparse
import os
import warnings
import time
import torch
from utils.intermediate_vis import Vis_iter

from datasets.data import *
from utils.cuda import launch
from utils.multistep import get_optim
from train import Experiment

from layers.Voxel_Level.Gen_Diffusion import Diffusion
from layers.Voxel_Level.Con_Diffusion import Con_Diffusion

from layers.Latent_Level.stage1.vqvae import vqvae
from layers.Latent_Level.stage2.Gen_diffusion import latent_diffusion

from layers.Ablation.wo_diffusion import wo_diff

# environment variables
NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

def get_args():
    ###########
    ## Setup ##
    ###########
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use. If given, only the specific gpu will be used, and ddp will be disabled')
    parser.add_argument('--distribution', type=bool, default=True)
    parser.add_argument('--num_node', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29500', help='url used to set up distributed training')
    
    # Data params
    parser.add_argument('--dataset', type=str, default='carla', choices='carla')
    # Train params
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)

    # Experiemtn params
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--recon_loss', default=False)
    parser.add_argument('--mode', default='wo_diff', choices='gen, con, vis, l_vae l_gen, wo_diff')
    parser.add_argument('--l_size', default='32322', choices='882, 16162, 32322')
    parser.add_argument('--init_size', default=8)
    parser.add_argument('--l_attention', default=True)
    parser.add_argument('--vq_size', default=50)

    # Model params
    parser.add_argument('--auxiliary_loss_weight', type=int, default=0.0005)
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--diffusion_dim', type=int, default=32)
    parser.add_argument('--dp_rate', type=float, default=0.)

    # Optim params
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--momentum_sqr', type=float, default=0.999)
    parser.add_argument('--milestones', type=eval, default=[])
    parser.add_argument('--gamma', type=float, default=0.1)

    # Train params
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--resume', type=str, default=False)
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--vqvae_path', type=str, default='')

    # Logging params
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--check_every', type=int, default=5)
    parser.add_argument('--completion_epoch', type=int, default=20)
    parser.add_argument('--log_tb', type=eval, default=True)
    parser.add_argument('--log_home', type=str, default=None)
    parser.add_argument('--log_path', type=str, default='')

    args = parser.parse_args()
    return args


def main():
    print('start!')
    args = get_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    launch(start, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


def start(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    ##################
    ## Specify data ##
    ##################
    train_loader, eval_loader, test_loader, num_classes, comp_weights, seg_weights, train_sampler = get_data(args)
    args.num_classes = num_classes

    completion_criterion = torch.nn.CrossEntropyLoss(weight=comp_weights)
    seg_criterion = torch.nn.CrossEntropyLoss(weight=seg_weights, ignore_index=0)
    similarity_criterion = torch.nn.MSELoss()

    #######################
    ## Without Diffusion ##
    #######################
    if args.mode == 'wo_diff':
        model = wo_diff(args, completion_criterion).cuda()
        if args.distribution :
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    ########################
    ## Discrete Diffusion ##
    ########################
    elif args.mode == 'gen':
        model = Diffusion(args, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution :
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    elif args.mode == 'con':
        model = Con_Diffusion(args, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution :
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    
    ######################
    ## Latent Diffusion ##
    ######################
    elif args.mode == 'l_vae':
        model = vqvae(args, completion_criterion).cuda()
        if args.distribution:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    elif args.mode == 'l_gen':
        Dense = vqvae(args, completion_criterion).cuda()
        dense_check = torch.load(args.vqvae_path)
        model = latent_diffusion(args, Dense, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution:
            Dense = torch.nn.parallel.DistributedDataParallel(Dense, device_ids=[args.gpu], find_unused_parameters=False)
            Dense.module.load_state_dict(dense_check['model'])
            for p in Dense.module.parameters():
                p.requires_grad = False   
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            
    ###################
    ## Visualization ##
    ###################
    elif args.mode == 'vis':
        model = Con_Diffusion(args, completion_criterion, auxiliary_loss_weight=args.auxiliary_loss_weight).cuda()
        if args.distribution :
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
    if args.mode == 'vis':
        exp = Vis_iter(args, model, optimizer, scheduler_iter, scheduler_epoch, test_loader, args.log_path)
    
    else : 
        exp = Experiment(args, model, optimizer, scheduler_iter, scheduler_epoch,
                        train_loader, eval_loader, test_loader, train_sampler, 
                        args.log_path, args.eval_every, args.check_every)
    
    exp.run(epochs = args.epochs)

if __name__ == '__main__':
    main()

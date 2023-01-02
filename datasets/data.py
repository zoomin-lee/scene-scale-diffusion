import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.carla_dataset import *
from datasets.kitti_dataset import *

dataset_choices = {'carla', 'kitti'}


def get_data_id(args):
    return '{}'.format(args.dataset)

def get_class_weights(freq):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w))

    return weights

def get_data(args):
    assert args.dataset in dataset_choices
    if args.dataset == 'carla':
        train_dir = "/mnt/ssd1/jm/Cartesian/Train"
        val_dir = "/mnt/ssd1/jm/Cartesian/Val"
        test_dir = "/mnt/ssd1/jm/Cartesian/Test"

        x_dim = 128
        y_dim = 128
        z_dim = 8
        data_shape = [x_dim, y_dim, z_dim]
        args.data_shape= data_shape

        binary_counts = True
        transform_pose = True
        remap = True
        if remap:
            class_frequencies = remap_frequencies_cartesian
            args.num_classes = 11
        else:
            args.num_classes = 23

        comp_weights = get_class_weights(class_frequencies).to(torch.float32)
        seg_weights = get_class_weights(class_frequencies[1:]).to(torch.float32)

        train_ds = CarlaDataset(directory=train_dir, random_flips=True, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
        coor_ranges = train_ds._eval_param['min_bound'] + train_ds._eval_param['max_bound']
        voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / x_dim,
                    abs(coor_ranges[4] - coor_ranges[1]) / y_dim,
                    abs(coor_ranges[5] - coor_ranges[2]) / z_dim] # since BEV
        val_ds = CarlaDataset(directory=val_dir, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)
        test_ds = CarlaDataset(directory=test_dir, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose)

        if args is not None and args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
            train_iters = len(train_sampler) // args.batch_size
            val_iters = len(val_sampler) // args.batch_size
        else:
            train_sampler = None
            val_sampler = None
            train_iters = len(train_ds) // args.batch_size
            val_iters = len(val_ds) // args.batch_size
        
        dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=train_ds.collate_fn, num_workers=args.num_workers)
        dataloader_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, collate_fn=val_ds.collate_fn, num_workers=args.num_workers)
        dataloader_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=args.num_workers)
    
    
    return dataloader, dataloader_val, dataloader_test, args.num_classes, comp_weights, seg_weights, train_sampler

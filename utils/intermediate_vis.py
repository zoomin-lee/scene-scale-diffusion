from dataclasses import astuple
import torch
import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import yaml

from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from utils.tables import *
from utils.dicts import clean_dict
from utils.loss import lovasz_softmax


class Vis_iter(object):
    no_log_keys = ['project', 'name','log_tb', 'log_wandb','check_every', 'eval_every','device', 'parallel', 'pin_memory', 'num_workers']
                   
    def __init__(self, args, model, optimizer, scheduler_iter, scheduler_epoch, test_loader,log_path):

        # Objects
        self.model = model
        self.optimizer, self.scheduler_iter, self.scheduler_epoch= optimizer, scheduler_iter, scheduler_epoch
        # Paths
        self.log_path = log_path

        if args.dataset =='kitti':
            config_file = os.path.join('/home/jumin/multinomial_diffusion/datasets/semantic_kitti.yaml')
            kitti_config = yaml.safe_load(open(config_file, 'r'))
            self.remap = kitti_config['learning_map_inv']
            self.color_map = kitti_config["color_map"]
            label = kitti_config['labels']
            map_index = np.asarray([self.remap[i] for i in range(20)])
            self.label_to_names = np.asarray([label[map_i] for map_i in map_index])

        elif args.dataset =='carla':
            base_dir = os.path.dirname(__file__)
            config_file = os.path.join(base_dir, '../datasets/carla.yaml')
            carla_config = yaml.safe_load(open(config_file, 'r'))
            self.color_map = carla_config["remap_color_map"]
            self.remap = None
            LABEL_TO_NAMES = carla_config["label_to_names"]
            self.label_to_names = np.asarray(list(LABEL_TO_NAMES.values()))


        # Initialize
        self.current_epoch = 0
        self.train_metrics, self.eval_metrics, self.ssc_metrics, self.seg_metrics = {}, {}, {}, {}
        self.eval_epochs = []
        self.completion_epochs = []

        # Store data loaders
        self.test_loader = test_loader

        # Store args
        create_folders(args)
        save_args(args)
        self.args = args

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)

    def run(self, epochs):
        self.checkpoint_load(self.args.resume_path)
        for epoch in range(self.current_epoch, epochs): 
            self.sample()

    def sample(self):
        self.model.eval()
        with torch.no_grad():
            for iterate, (voxel_input, output, counts) in enumerate(self.test_loader):
                voxel_input = torch.from_numpy(np.asarray(voxel_input)).squeeze(1).cuda() 
                output = torch.from_numpy(np.asarray(output)).long().cuda()            
                _, intermediate = self.model.module.sample(voxel_input, intermediate=True)
                inter_vis(self.args, intermediate)
                break
                   
    def checkpoint_load(self, resume_path):
        checkpoint = torch.load(resume_path)
        
        if self.args.distribution:
            self.model.module.load_state_dict(checkpoint['model'])
        else :
            self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler_iter: self.scheduler_iter.load_state_dict(checkpoint['scheduler_iter'])
        if self.scheduler_epoch: self.scheduler_epoch.load_state_dict(checkpoint['scheduler_epoch'])

        self.current_epoch = checkpoint['current_epoch']
        self.train_metrics = checkpoint['train_metrics']
        self.eval_metrics = checkpoint['eval_metrics']
        self.eval_epochs = checkpoint['eval_epochs']

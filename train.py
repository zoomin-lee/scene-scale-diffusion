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


class Experiment(object):
    no_log_keys = ['project', 'name','log_tb', 'log_wandb','check_every', 'eval_every','device', 'parallel', 'pin_memory', 'num_workers']
                   
    def __init__(self, args, model, optimizer, scheduler_iter, scheduler_epoch,
                 train_loader, eval_loader, test_loader, train_sampler,
                 log_path, eval_every, check_every):

        # Objects
        self.model = model

        self.loss_fun = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer, self.scheduler_iter, self.scheduler_epoch= optimizer, scheduler_iter, scheduler_epoch
        # Paths
        self.log_path = log_path

        if args.dataset =='carla':
            config_file = os.path.join('./carla.yaml')
            carla_config = yaml.safe_load(open(config_file, 'r'))
            self.color_map = carla_config["remap_color_map"]
            self.remap = None
            LABEL_TO_NAMES = carla_config["label_to_names"]
            self.label_to_names = np.asarray(list(LABEL_TO_NAMES.values()))

        # Intervals
        self.eval_every, self.check_every = eval_every, check_every

        # Initialize
        self.current_epoch = 0
        self.train_metrics, self.eval_metrics, self.ssc_metrics, self.seg_metrics = {}, {}, {}, {}
        self.eval_epochs = []
        self.completion_epochs = []

        # Store data loaders
        self.train_loader, self.eval_loader, self.test_loader, self.train_sampler = train_loader, eval_loader, test_loader, train_sampler

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
        if self.args.resume: 
            self.resume()
        
        for epoch in range(self.current_epoch, epochs): 
            
            # Train
            train_dict = self.train_fn(epoch)
            self.log_metrics(train_dict, self.train_metrics)

            # Checkpoint
            self.current_epoch += 1
            if (epoch+1) % self.check_every == 0:
                self.checkpoint_save(epoch)

            # Eval
            if (epoch+1) % self.eval_every == 0:
                eval_dict = self.eval_fn(epoch)
                self.log_metrics(eval_dict, self.eval_metrics)
                self.eval_epochs.append(epoch)
            else:
                eval_dict = None

            if (epoch+1) % self.args.completion_epoch == 0:
                ssc_dict, miou, seg_dict, seg_miou = self.sample()
                self.log_metrics(ssc_dict, self.ssc_metrics)
                self.log_metrics(seg_dict, self.ssc_metrics)
                self.completion_epochs.append(epoch)
            else :
                ssc_dict, seg_dict = None, None

            # Log
            #self.save_metrics()
            if self.args.log_tb:
                for metric_name, metric_value in train_dict.items():
                    self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
                if eval_dict:
                    for metric_name, metric_value in eval_dict.items():
                        self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)
                if ssc_dict:
                    for metric_name, metric_value in ssc_dict.items():
                        self.writer.add_scalar('SSC/{}'.format(metric_name), metric_value, global_step=epoch+1)
                    self.writer.add_text("SSC_mIoU", get_miou_table(self.args, self.label_to_names, miou).get_html_string(), global_step=epoch+1)
                    for metric_name, metric_value in seg_dict.items():
                        self.writer.add_scalar('Seg/{}'.format(metric_name), metric_value, global_step=epoch+1)
                    self.writer.add_text("Seg_mIoU", get_miou_table(self.args, self.label_to_names, seg_miou).get_html_string(), global_step=epoch+1)

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        if self.args.distribution :
            self.train_sampler.set_epoch(epoch)

        for voxel_input, output, counts in self.train_loader:
            self.optimizer.zero_grad()
            voxel_input = torch.from_numpy(np.asarray(voxel_input)).long().squeeze(1).cuda() # (4,1,256,256,32)
            output = torch.from_numpy(np.asarray(output)).long().cuda()            

            loss = self.model.module(output, voxel_input)
            loss.backward()

            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)

            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(output)
            loss_count += len(output)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'loss': loss_sum/loss_count}


    def eval_fn(self, epoch):
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for voxel_input, output, counts in self.eval_loader:
                voxel_input = torch.from_numpy(np.asarray(voxel_input)).long().squeeze(1).cuda() # (4,1,256,256,32)
                output = torch.from_numpy(np.asarray(output)).long().cuda()            

                loss = self.model.module(output, voxel_input)

                loss_sum += loss.detach().cpu().item() * len(output)
                loss_count += len(output)
                print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'loss': loss_sum/loss_count}


    def sample(self):
        self.model.eval()
        with torch.no_grad():
            TP, FP, TN, FN, num_correct, num_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            s_TP, s_FP, s_TN, s_FN, s_num_correct, s_num_total = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            all_intersections, all_unions = np.zeros(self.args.num_classes), np.zeros(self.args.num_classes) + 1e-6
            s_all_intersections, s_all_unions = np.zeros(self.args.num_classes), np.zeros(self.args.num_classes) + 1e-6
            if self.args.dataset == 'carla':
                dataloader = self.test_loader
            else :
                dataloader = self.eval_loader
            for iterate, (voxel_input, output, counts) in enumerate(dataloader):
                if len(voxel_input) == self.args.batch_size :
                    voxel_input = torch.from_numpy(np.asarray(voxel_input)).long().squeeze(1).cuda() # (4,1,256,256,32)
                    output = torch.from_numpy(np.asarray(output)).long().cuda()            
                    invalid = torch.from_numpy(np.asarray(counts)).cuda()

                    if self.args.mode == 'l_vae':
                        recons = self.model.module.sample(output) 
                    else :
                        recons = self.model.module.sample(voxel_input)      

                    visualization(self.args, recons, voxel_input, output, invalid, iteration = iterate)
                    correct, total, pred_TP, pred_FP, pred_TN, pred_FN, intersection, union = get_result(self.args, invalid, output, recons)
                    all_intersections += intersection
                    all_unions += union
                    num_correct += correct
                    num_total += total
                    TP += pred_TP
                    FP += pred_FP
                    TN += pred_TN
                    FN += pred_FN

                    s_correct, s_total, s_pred_TP, s_pred_FP, s_pred_TN, s_pred_FN, s_intersection, s_union = get_result(self.args, voxel_input, output, recons, SSC=False)
                    s_all_intersections += s_intersection
                    s_all_unions += s_union
                    s_num_correct += s_correct
                    s_num_total += s_total
                    s_TP += s_pred_TP
                    s_FP += s_pred_FP
                    s_TN += s_pred_TN
                    s_FN += s_pred_FN
                   
            iou, miou = print_result(self.args, self.label_to_names, num_correct, num_total, all_intersections, all_unions, TP, FP, FN)
            s_iou, seg_miou = print_result(self.args, self.label_to_names, s_num_correct, s_num_total, s_all_intersections, s_all_unions, s_TP, s_FP, s_FN, SSC=False)
            return {"IoU" : iou, "mIoU": np.mean(miou)*100 }, miou, {"IoU" : s_iou, "mIoU": np.mean(seg_miou)*100 }, seg_miou

    def resume(self):
        self.checkpoint_load(self.args.resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]

            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: 
                eval_dict = None
            
            if epoch in self.completion_epochs:
                sample_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    sample_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: 
                sample_dict = None

            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if sample_dict:
                for metric_name, metric_value in sample_dict.items():
                    self.writer.add_scalar('sample/{}'.format(metric_name), metric_value, global_step=epoch+1)


    def log_metrics(self, dict, type):
        if len(type)==0:
            for metric_name, metric_value in dict.items():
                type[metric_name] = [metric_value]
        else:
            for metric_name, metric_value in dict.items():
                type[metric_name].append(metric_value)

    def save_metrics(self):
        # Save metrics
        with open(os.path.join(self.log_path,'metrics_train.pickle'), 'wb') as f:
            pickle.dump(self.train_metrics, f)
        with open(os.path.join(self.log_path,'metrics_eval.pickle'), 'wb') as f:
            pickle.dump(self.eval_metrics, f)

        # Save metrics table
        metric_table = get_metric_table(self.train_metrics, epochs=list(range(1, self.current_epoch+2)))
        with open(os.path.join(self.log_path,'metrics_train.txt'), "w") as f:
            f.write(str(metric_table))
        metric_table = get_metric_table(self.eval_metrics, epochs=[e+1 for e in self.eval_epochs])
        with open(os.path.join(self.log_path,'metrics_eval.txt'), "w") as f:
            f.write(str(metric_table))


    def checkpoint_save(self, epoch):        
        checkpoint = {'current_epoch': self.current_epoch,
                      'train_metrics': self.train_metrics,
                      'eval_metrics': self.eval_metrics,
                      'eval_epochs': self.eval_epochs,
                      'optimizer': self.optimizer.state_dict(),
                      'model': self.model.module.state_dict(),
                      'scheduler_iter': self.scheduler_iter.state_dict() if self.scheduler_iter else None,
                      'scheduler_epoch': self.scheduler_epoch.state_dict() if self.scheduler_epoch else None,}

        epoch_name = 'epoch{}.tar'.format(epoch)
        torch.save(checkpoint, os.path.join(self.log_path, epoch_name))

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

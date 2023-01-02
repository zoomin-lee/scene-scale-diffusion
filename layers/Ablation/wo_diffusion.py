import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from layers.Latent_Level.stage1.model import C_Encoder, C_Decoder

class wo_diff(torch.nn.Module):
    def __init__(self, args, multi_criterion) -> None:
        super(wo_diff, self).__init__()
        self.args = args

        if self.args.dataset == 'kitti':
            init_size = args.init_size
        elif self.args.dataset == 'carla':
            init_size = args.init_size
        
        self.encoder = C_Encoder(args, nclasses=self.args.num_classes, init_size=init_size, l_size=args.l_size, attention=args.l_attention)
        self.decoder = C_Decoder(args, nclasses=self.args.num_classes, init_size=init_size, l_size=args.l_size, attention=args.l_attention)
        
        self.multi_criterion = multi_criterion

    def device(self):
        return self.encoder.device

    def forward(self, x, input_ten):
        latent = self.encoder(input_ten, out_conv=False) 
        recons = self.decoder(latent, in_conv=False)
        recons_loss = self.multi_criterion(recons, x)
        return recons_loss 

    def sample(self, x):
        latent = self.encoder(x, out_conv=False) 
        recons = self.decoder(latent, in_conv=False)
        recons = recons.argmax(1)
        return recons

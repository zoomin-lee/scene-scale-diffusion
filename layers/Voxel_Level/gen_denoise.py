import math
from mimetypes import init
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn, einsum


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,padding=(0, 1, 1), bias=False)


def conv1x1x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, padding=(0, 0, 1), bias=False)


def conv1x3x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride, padding=(0, 1, 0), bias=False)


def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0), bias=False)


def conv3x1x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride, padding=(1, 0, 1), bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride)


class Asymmetric_Residual_Block(nn.Module):
    def __init__(self, in_filters, out_filters, time_filters=128):
        super(Asymmetric_Residual_Block, self).__init__()
        self.GroupNorm = nn.GroupNorm(32, in_filters)
        self.time_layers = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(time_filters, in_filters*2)
                        )

        self.conv1 = conv1x3x3(in_filters, out_filters)
        self.bn0 = nn.GroupNorm(32, out_filters)
        self.act1 = nn.LeakyReLU()
          
        self.conv1_2 = conv3x1x3(out_filters, out_filters)
        self.bn0_2 = nn.GroupNorm(32, out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1x3(in_filters, out_filters)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.GroupNorm(32, out_filters)

        self.conv3 = conv1x3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.GroupNorm(32, out_filters)


    def forward(self, x, t):
        t = self.time_layers(t)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        scale, shift = torch.chunk(t, 2, dim=1)
        
        x = self.GroupNorm(x) * (1 + scale) + shift

        shortcut = self.conv1(x) # shortcut : (2, 8, 256, 256, 32) / down1 : (2, 16, 256, 256, 32)
        shortcut = self.act1(shortcut)
        shortcut = self.bn0(shortcut)

        shortcut = self.conv1_2(shortcut) # shortcut : (2, 8, 256, 256, 32)
        shortcut = self.act1_2(shortcut)
        shortcut = self.bn0_2(shortcut)

        resA = self.conv2(x) # resA : (2, 8, 256, 256, 32) / down1 : (2, 16, 256, 256, 32)
        resA = self.act2(resA)
        resA = self.bn1(resA)

        resA = self.conv3(resA) # resA : (2, 8, 256, 256, 32) / down1 : (2, 16, 256, 256, 32)
        resA = self.act3(resA)
        resA = self.bn2(resA)
        resA += shortcut

        return resA

class DDCM(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1):
        super(DDCM, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters)
        self.bn0 = nn.GroupNorm(32, out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters)
        self.bn0_2 = nn.GroupNorm(32, out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters)
        self.bn0_3 = nn.GroupNorm(32, out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.bn0(shortcut)
        shortcut = self.act1(shortcut)

        shortcut2 = self.conv1_2(x)
        shortcut2 = self.bn0_2(shortcut2)
        shortcut2 = self.act1_2(shortcut2)

        shortcut3 = self.conv1_3(x)
        shortcut3 = self.bn0_3(shortcut3)
        shortcut3 = self.act1_3(shortcut3)
        shortcut = shortcut + shortcut2 + shortcut3

        shortcut = shortcut * x

        return shortcut

def l2norm(t):
    return F.normalize(t, dim = -1)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.to_qkv = conv1x1(dim, dim*3, stride=1)
        self.to_out = conv1x1(dim, dim, stride=1)

    def forward(self, x):
        b, c, h, w, Z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = Z)
        return self.to_out(out)

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 4, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        self.to_q = conv1x1(dim, dim, stride=1)
        self.to_k = conv1x1(dim, dim, stride=1)
        self.to_v = conv1x1(dim, dim, stride=1)

        self.to_out = conv1x1(dim, dim, stride=1)

    def forward(self, x, cond_x):
        b, c, h, w, Z = x.shape
        q = self.to_q(x)
        k = self.to_k(cond_x)
        v = self.to_v(cond_x)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h c (x y z)', h = self.heads), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = Z)
        return self.to_out(out)

class DownBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False):
        super(DownBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.residual_block = Asymmetric_Residual_Block(in_filters, out_filters)


        if pooling:
            if height_pooling:
                self.pool = nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, bias=False)
            else:
                self.pool = nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, bias=False)


    def forward(self, x, t):
        resA = self.residual_block(x, t)
        #down1 : (2, 16, 256, 256, 32) / down2 : (2, 32, 128, 128, 16) / down3 : (2, 64, 64, 64, 8) / down4 : (2, 128, 32, 32, 8)
        if self.pooling:
            resB = self.pool(resA) #down1 : (2, 16, 128, 128, 16) / down2 : (2, 32, 64, 64, 8) / down3 : (2, 64, 32, 32, 8) / down4 : (2, 128, 16, 16, 8)
            return resB, resA
        else:
            return resA

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, height_pooling, time_filters=32*4):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3x3(in_filters, in_filters)
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.GroupNorm(32, in_filters)
        self.time_layers = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(time_filters, in_filters*2)
                        )

        self.conv1 = conv1x3x3(in_filters, out_filters)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.GroupNorm(32, out_filters)

        self.conv2 = conv3x1x3(out_filters, out_filters)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.GroupNorm(32, out_filters)

        self.conv3 = conv3x3x3(out_filters, out_filters)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.GroupNorm(32, out_filters)
        
        if height_pooling :
            self.up_subm = nn.ConvTranspose3d(in_filters, in_filters, kernel_size=3, bias=False, stride=2, padding=1, output_padding=1, dilation=1)
        else : 
            self.up_subm = nn.ConvTranspose3d(in_filters, in_filters, kernel_size=(3,3,1), bias=False, stride=(2,2,1), padding=(1,1,0), output_padding=(1,1,0), dilation=1)
    

    def forward(self, x, residual, t): # x : ( 2, 128, 32, 32, 2 ), skip : ( 2, 128, 64, 64, 4 )
        upA = self.trans_dilao(x) # upA : ( 2, 128, 32, 32, 2 )
        upA = self.trans_act(upA)

        t = self.time_layers(t)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        scale, shift = torch.chunk(t, 2, dim=1)
        
        upA = self.trans_bn(upA) * (1 + scale) + shift
        ## upsample
        upA = self.up_subm(upA)
        upA += residual
        upE = self.conv1(upA)
        upE = self.act1(upE)
        upE = self.bn1(upE)

        upE = self.conv2(upE)
        upE = self.act2(upE)
        upE = self.bn2(upE)

        upE = self.conv3(upE)
        upE = self.act3(upE)
        upE = self.bn3(upE)

        return upE

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class Denoise(nn.Module):
    def __init__(self, args, num_class = 11, init_size=32, discrete=True):
        super(Denoise, self).__init__()
        self.args = args
        self.discrete = discrete
        self.num_class = num_class
        self.init_size = init_size
        self.time_size = init_size*4

        self.time_embed = nn.Sequential(
            nn.Linear(init_size, self.time_size),
            nn.SiLU(),
            nn.Linear(self.time_size, self.time_size),
        )

        self.embedding = nn.Embedding(self.num_class, init_size)
        self.conv_in = nn.Conv3d(init_size, init_size, kernel_size=1, stride=1)

        self.A = Asymmetric_Residual_Block(init_size, init_size)

        self.downBlock1 = DownBlock(init_size, 2 * init_size, 0.2, height_pooling=True)
        self.downBlock2 = DownBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True)
        if self.args.dataset=='kitti':
            self.downBlock3 = DownBlock(4 * init_size, 4 * init_size, 0.2, height_pooling=True)
        else :
            self.downBlock3 = DownBlock(4 * init_size, 8 * init_size, 0.2, height_pooling=False)
            self.downBlock4 = DownBlock(8 * init_size, 16 * init_size, 0.2, height_pooling=False)
            self.midBlock1 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size)
            self.attention = Attention(16 * init_size, 32)
            self.midBlock2 = Asymmetric_Residual_Block(16 * init_size, 16 * init_size)

        if self.args.dataset=='kitti':
            self.upBlock3 = UpBlock(4 * init_size, 4 * init_size, height_pooling=True)
            self.upBlock1 = UpBlock(2 * init_size, 1 * init_size, height_pooling=True)
            self.DDCM = DDCM(1 * init_size, 1 * init_size)
            self.logits = nn.Conv3d(2 * init_size, self.num_class, kernel_size=3, stride=1, padding=1, bias=True)
        else :        
            self.upBlock4 = UpBlock(16 * init_size, 8 * init_size, height_pooling=False)
            self.upBlock3 = UpBlock(8 * init_size, 4 * init_size, height_pooling=False)
            self.upBlock1 = UpBlock(2 * init_size, 2 * init_size, height_pooling=True)
            self.DDCM = DDCM(2 * init_size, 2 * init_size)
            self.logits = nn.Conv3d(4 * init_size, self.num_class, kernel_size=3, stride=1, padding=1, bias=True)
        self.upBlock2 = UpBlock(4 * init_size, 2 * init_size, height_pooling=True)
    

    def forward(self, x, t):
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv_in(x)

        t = self.time_embed(timestep_embedding(t, self.init_size))

        # x : (2, 8, 256, 256, 32)
        if self.args.dataset == 'carla':
            x = self.A(x, t)

        down1c, down1b = self.downBlock1(x, t) # down1c : (4, 128, 64, 64, 4) , down1b : (4, 64, 128, 128,)
        down2c, down2b = self.downBlock2(down1c, t) # down2c : (4, 128, 32, 32, 2) , down2b : (4, 128, 64, 64, 4)
        down3c, down3b = self.downBlock3(down2c, t) # down1c : (4, 128, 64, 64, 4) , down1b : (4, 64, 128, 128,)
        
        if self.args.dataset == 'carla':
            down4c, down4b = self.downBlock4(down3c, t) # down2c : (4, 128, 32, 32, 2) , down2b : (4, 128, 64, 64, 4)
            down4c = self.midBlock1(down4c, t) # (4, 128, 32, 32, 2)
            down4c = self.attention(down4c)
            down4c = self.midBlock2(down4c, t) # (4, 128, 32, 32, 2)
            up4 = self.upBlock4(down4c, down4b, t)
            up3 = self.upBlock3(up4, down3b, t)
        else :
            up3 = self.upBlock3(down3c, down3b, t)

        up2 = self.upBlock2(up3, down2b, t)
        up1 = self.upBlock1(up2, down1b, t)

        up0 = self.DDCM(up1) # up0e : (2, 16, 256, 256, 32)

        up = torch.cat((up1, up0), 1) # up0e : (2, 32, 256, 256, 32)

        logits = self.logits(up) # logits : (2, 20, 256, 256, 32)
        
        return logits
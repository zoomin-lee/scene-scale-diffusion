import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from utils.loss import lovasz_softmax
from layers.Latent_Level.stage1.model import C_Encoder, C_Decoder
from layers.Latent_Level.stage1.vector_quantizer import VectorQuantizer

class vqvae(torch.nn.Module):
    def __init__(self, args, multi_criterion) -> None:
        super(vqvae, self).__init__()
        self.args = args

        if self.args.dataset == 'kitti':
            init_size = args.init_size
        elif self.args.dataset == 'carla':
            init_size = args.init_size
        embedding_dim = self.args.num_classes
        
        self.VQ = VectorQuantizer(num_embeddings = self.args.num_classes*self.args.vq_size, embedding_dim = embedding_dim)

        self.encoder = C_Encoder(args, nclasses=self.args.num_classes, init_size=init_size, l_size=args.l_size, attention=args.l_attention)
        self.quant_conv = nn.Conv3d(self.args.num_classes, self.args.num_classes, kernel_size=1, stride=1)

        self.decoder = C_Decoder(args, nclasses=self.args.num_classes, init_size=init_size, l_size=args.l_size, attention=args.l_attention)
        self.post_quant_conv = nn.Conv3d(self.args.num_classes, self.args.num_classes, kernel_size=1, stride=1)

        self.multi_criterion = multi_criterion

    def device(self):
        return self.encoder.device

    def encode(self, x):
        latent = self.encoder(x) # latent : 8, 128, 8, 8, 2
        latent = self.quant_conv(latent)
        return latent

    def vector_quantize(self, latent):
        quantized_latent, vq_loss, quantized_latent_ind, latents_shape = self.VQ(latent)
        return quantized_latent, vq_loss, quantized_latent_ind, latents_shape

    def coodbook(self,quantized_latent_ind, latents_shape):
        quantized_latent = self.VQ.codebook_to_embedding(quantized_latent_ind.view(-1,1), latents_shape)
        return quantized_latent

    def decode(self, quantized_latent):
        quantized_latent = self.post_quant_conv(quantized_latent)
        recons = self.decoder(quantized_latent) # recons : 8, 11, 128, 128, 8
        return recons

    def forward(self, x, input_ten):
        latent = self.encode(x) # (4, 11, 32, 32, 2)
        quantized_latent, vq_loss, _, _ = self.vector_quantize(latent) 
        recons = self.decode(quantized_latent)

        recons_loss = self.multi_criterion(recons, x)
        loss = recons_loss + vq_loss 
        return loss 

    def sample(self, x):
        latent = self.encode(x)
        quantized_latent, _, _, _ = self.vector_quantize(latent)
        recons = self.decode(quantized_latent)
        recons = recons.argmax(1)
        return recons

import torch
import torch.nn.functional as F
import numpy as np
import math
from inspect import isfunction
from layers.Voxel_Level.gen_denoise import Denoise
from utils.loss import *
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)

    return alphas

class Diffusion(torch.nn.Module):
    def __init__(self, args, multi_criterion,auxiliary_loss_weight=0.05, adaptive_auxiliary_loss=True):
        super(Diffusion, self).__init__()

        #self._denoise_fn = SSCNet(num_classes=args.num_classes*50, num_steps=args.diffusion_steps)
        self.args = args
        self.num_classes = self.args.num_classes
        self.num_timesteps = self.args.diffusion_steps
        self.recon_loss = self.args.recon_loss
        self._denoise_fn = Denoise(args= self.args,  num_class = self.num_classes)
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss

        self.multi_criterion = multi_criterion

        alphas = cosine_beta_schedule(self.num_timesteps )

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps ))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps ))
    
    def device(self):
        return self.denoise_fn.device

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs

    def predict_start(self, log_x_t, t):
        x_t = log_onehot_to_index(log_x_t)

        out = self._denoise_fn(x_t, t)

        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t):
        log_x0_recon = self.predict_start(log_x, t)
        log_model_pred = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_x, t=t)
        return log_model_pred, log_x0_recon

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def forward(self, x, voxel_input):
        b, device = x.size(0), x.device
        self.shape = x.size()[1:]        
        t, pt = self.sample_time(b, device, 'importance')

        log_x_start = index_to_log_onehot(x, self.num_classes)
        log_x_t = self.q_sample(log_x_start, t) # log_x_t : (batch, #class, 128, 128, 8)

        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob, log_x0_recon = self.p_pred(log_x=log_x_t, t=t)

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        
        if self.training:
            Lt2 = kl_loss.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        kl_prior = self.kl_prior(log_x_start)

        # Upweigh loss term of the kl
        loss = kl_loss / pt + kl_prior

        kl_aux = self.multinomial_kl(log_x_start[:,:-1,:,:,:], log_x0_recon[:,:-1,:,:,:])
        kl_aux = sum_except_batch(kl_aux)
        '''if self.recon_loss : 
            kl_aux += self.multi_criterion(log_x0_recon.exp(), x)
            kl_aux += lovasz_softmax(torch.nn.functional.softmax(log_x0_recon.exp(), dim=1), x)'''

        kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
        if self.adaptive_auxiliary_loss:
            addition_loss_weight = (1-t/self.num_timesteps) + 1.0
        else:
            addition_loss_weight = 1.0

        aux_loss = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
        
        loss += aux_loss
        loss = -loss.sum() / (self.shape[0]*self.shape[1])
        #loss += seg_loss

        return -loss

    def sample(self, voxel_input):
        device = self.log_alpha.device
        self.shape = voxel_input.size()[1:]
        uniform_logits = torch.zeros((self.args.batch_size, self.num_classes) + self.shape, device=device)
        log_z = self.log_sample_categorical(uniform_logits)

        for i in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((self.args.batch_size,), i, device=device, dtype=torch.long)

            log_model_prob, log_x0_recon = self.p_pred(log_x=log_z, t=t)

            log_z = self.log_sample_categorical(log_model_prob)

        result = log_onehot_to_index(log_z)
        return result


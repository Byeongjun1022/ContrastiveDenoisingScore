# https://github.com/YSerin/ZeCon/blob/main/optimization/losses.py
# https://github.com/HyelinNAM/ContrastiveDenoisingScore/blob/main/utils/loss.py
from typing import Tuple, Union, Optional, List

from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn

class DDSLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep
    
    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):

        # latent_input = torch.cat([z_t] * 2)
        # timestep = torch.cat([timestep] * 2)
        # embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(z_t, timestep, embedd, cross_attention_kwargs=cross_attention_kwargs,).sample
        # e_t_uncond, e_t = e_t.chunk(2)
        # e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        # assert torch.isfinite(e_t).all()

        return e_t

    def __init__(self, t_min, t_max, unet, scheduler, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
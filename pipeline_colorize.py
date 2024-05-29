import torch
from typing import Any, Optional, Union, Tuple, List, Callable, Dict
import os
from PIL import Image
from torchvision import transforms as tfms
import torch.optim as optim
import numpy as np

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer

from torchvision.transforms.functional import pil_to_tensor, resize
from utils.loss_color import DDSLoss
from torchvision.transforms.functional import resize
from utils.color_util import deprocess_lab, lab_to_rgb, rgb_to_lab

class ColorPipeline(DiffusionPipeline):
    
    rgb_latent_scale_factor = 0.18215
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None
    
    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        # input_depth: Union[Image.Image, torch.Tensor, np.ndarray],
        # height: Optional[int]=None,
        # width: Optional[int]=None,
        num_inference_steps: int = 200,
        processing_res: int = 768,
        match_input_res: bool = True,
        save_path: str = 'results',
    ):
        device = self.device
        dtype = self.dtype
        
        # normalize the condition image
        input_image = input_image.convert("LAB")
        l, _, _ = input_image.split() # range of l : [0, 255]
        l = np.asarray(l) * 2.0 / 255.0 - 1.0 # range of l : [-1, 1], shape: [H, W]
        l = torch.from_numpy(l).to(dtype).cuda()
        # l_denorm = (l + 1.0) * 50.0 # range of l : [0, 100]
        ab = torch.zeros(2, l.shape[0], l.shape[1]).to(dtype).cuda()
        ab.requires_grad = True
        optimizer = optim.SGD([ab], lr=0.1)
        
        with torch.enable_grad():
            input_image = lab_to_rgb(deprocess_lab(l, ab[0], ab[1])) # range: [0, 1], shape: [H, W, 3]
            input_image = input_image.permute(2, 0, 1).unsqueeze(0) # shape: [1, 3, H, W]
            
            input_size = input_image.shape
            assert (
                4 == input_image.dim() and 3 == input_size[-3]
            ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"
            
            input_image = resize(input_image, (processing_res, processing_res), antialias=True)
            
            image_norm: torch.Tensor = input_image * 2.0 - 1.0 # range: [-1, 1]
            image_norm = image_norm.to(self.dtype).to(device)
            image_norm.clip_(-1.0, 1.0)
            
            assert image_norm.min() >= -1.0 and image_norm.max() <= 1.0
        
        z = self.encode_rgb(image_norm)
        
        # encode empty text embedding
        self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(1,1,1).to(device) # [B,2,1024]
        
        # Update latents
        # timestep ~ U(0.05, 0.95) to avoid very high/low noise level
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.05) # 50
        self.max_step = int(self.num_train_timesteps * 0.95) # 950
        
        # Define loss class
        dds_loss = DDSLoss(
            t_min=self.min_step, 
            t_max =self.max_step,
            unet = self.unet,
            scheduler = self.scheduler,
            device=self.device, 
        )
        
        # depth_latent = self.encode_rgb(depth_norm)
        # z = depth_latent.clone()
        # z.requires_grad = True
        # optimizer = optim.SGD([z], lr=0.1)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                optimizer.zero_grad()

                z_t, eps, timestep = dds_loss.noise_input(z, eps=None, timestep=None)
                
                eps_pred = self.unet(z_t, timestep, encoder_hidden_states=batch_empty_text_embed).sample
                
                grad = eps_pred - eps
                
                with torch.enable_grad():
                    loss = z * grad.clone()
                    loss = loss.mean()*1000
                    loss.backward()
                
                optimizer.step()
                
                if i == num_inference_steps - 1:
                    progress_bar.update()
            
                if (i+1)  % 50 == 0:
                    
                    output_image = lab_to_rgb(deprocess_lab(l, ab[0], ab[1])) # range: [0, 1], shape: [H, W, 3]
                    output_image = output_image.cpu().numpy()
                    output_image = Image.fromarray((output_image * 255).astype(np.uint8))
                        
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    output_image.save(os.path.join(save_path, f'{str(i).zfill(3)}.png'))
        
        result = lab_to_rgb(deprocess_lab(l, ab[0], ab[1])) # range: [0, 1], shape: [H, W, 3]
        result = result.cpu().numpy()
        result = Image.fromarray((result * 255).astype(np.uint8))
        
        return result
    
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent
    
    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode rgb latent into rgb.

        Args:
            rgb_latent (`torch.Tensor`):
                rgb latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded rgb.
        """
        # scale latent
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        stacked = self.vae.decoder(z)
        # # mean of output channels
        # rgb_mean = stacked.mean(dim=1, keepdim=True)
        return stacked
    
    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
                
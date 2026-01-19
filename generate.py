import torch
from torch import nn
from model import Dummy_DiT
from datasets import train_loader, test_loader
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import argparse
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Dummy_DiT()
    model.load_state_dict(torch.load("checkpoints/checkpoint_0.pth", map_location=torch.device('cpu')))
    model.to(device)

    is_conditional = config.train.is_conditional
    guide_scale = config.sampler.guide_scale
    cfg = True if guide_scale > 1.0 else False 
    
    imgs = []
    for num in range(10):
        noises = torch.randn((1, 28, 28)).unsqueeze(0).to(device)
        n_steps = 100
        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        label = torch.tensor([num]).to(device)
        latents = noises
    
        model.eval()
        for i in range(n_steps):
            t_start=time_steps[i]
            t_end=time_steps[i + 1]
            if not is_conditional:
                label = None
        
            velocity = model(latents, torch.tensor([t_start]).to(device), label)
            if cfg:
                negcon_velocity = model(latents, torch.tensor([t_start]).to(device), (label+1)%10)
                velocity = negcon_velocity + guide_scale * (
                    velocity - negcon_velocity)
        
            latents += velocity * (t_end - t_start)
            _latents = latents.clone().detach().squeeze()
            _latents = (_latents.clamp(min=0, max=1) * 255).to(torch.uint8)
        imgs.append(_latents.cpu().numpy())
    merged_img = np.hstack(imgs)
    img = Image.fromarray(merged_img, mode='L')
    img.save(f"outputs/final.png")
    

if __name__ == "__main__":
    main()

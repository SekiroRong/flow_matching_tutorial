import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Dummy_DiT
from datasets import train_loader, test_loader
from scheduler import Scheduler_Wrapper
from tqdm import tqdm
from PIL import Image
import os
import argparse
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from reward_models.reward_model import MNIST_CNN
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_model = MNIST_CNN()
    reward_model.load_state_dict(torch.load("./reward_models/mnist_pretrained/mnist_cnn_993.pth", map_location=device))
    reward_model.to(device)

    model = Dummy_DiT()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    loss_fn = nn.MSELoss()
    
    
    epochs = config.train.epochs
    is_conditional = config.train.is_conditional
    sampler_scheduler = Scheduler_Wrapper(config.sampler_scheduler.type, shift=3.0)
    guide_scale = config.sampler.guide_scale
    cfg = True if guide_scale > 1.0 else False 

    if config.lora.enable:
        lora_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
        state_dict = get_peft_model_state_dict(model)
        for key in state_dict.keys():
            print(key)
    
        save_file(state_dict, "checkpoints/lora.pth")
    
        state_dict = load_file("checkpoints/lora.pth")
        base_model = model.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)
    

    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        
        for i, (imgs, label) in pbar:
            imgs = imgs.to(device)
            label = label.to(device)
            noises = torch.randn_like(imgs).to(device)

            if not is_conditional:
                label = None

            timesteps = torch.arange(40)
            mid_timestep = random.randint(30, 39)
            latents = noises
    
            for i, t in enumerate(timesteps[:mid_timestep]):
                with torch.no_grad():
                    sigma = (torch.ones(imgs.size(0)) * sampler_scheduler.scheduler.sigmas[t]).to(device)
                    _sigma = sigma[:, None, None, None]

                    velocity = model(latents, sigma, label)
                    latents = sampler_scheduler.scheduler.step(latents, velocity, t)

            sigma = (torch.ones(imgs.size(0)) * sampler_scheduler.scheduler.sigmas[mid_timestep]).to(device)
            _sigma = sigma[:, None, None, None]

            velocity = model(latents, sigma, label)
            latents = sampler_scheduler.scheduler.step(latents, velocity, t)

            if not is_conditional:
                raise NotImplementError
            rewards = reward_model(latents)[:, label.to(torch.long)]
            
            optimizer.zero_grad()
            loss = F.relu(-rewards+2).mean()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
        scheduler.step()
        torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch}.pth")
    
    

if __name__ == "__main__":
    main()

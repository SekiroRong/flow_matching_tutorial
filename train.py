import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Dummy_DiT
from datasets import train_loader, test_loader
from tqdm import tqdm
from PIL import Image
import os
import argparse
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

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

    model = Dummy_DiT()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    loss_fn = nn.MSELoss()
    
    
    epochs = config.train.epochs
    is_conditional = config.train.is_conditional
    scheduler = config.scheduler.type
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
            t = torch.rand(imgs.size(0)).to(device)
            _t = t[:, None, None, None]
            
            latents = (1 - _t) * noises + _t * imgs
            velocities = imgs - noises
            
            optimizer.zero_grad()
            if not is_conditional:
                label = None
            loss = loss_fn(model(latents, t, label), velocities)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
        scheduler.step()
        torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch}.pth")

        noises = torch.randn_like(imgs[0]).unsqueeze(0).to(device)
        n_steps = 20
        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
        label = torch.tensor([0]).to(device)
        
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
        img = Image.fromarray(_latents.cpu().numpy(), mode='L')
        img.save(f"outputs/epoch_{epoch}.png")
    

if __name__ == "__main__":
    main()

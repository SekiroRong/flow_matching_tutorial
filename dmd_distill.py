import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Dummy_DiT
from datasets import train_loader, test_loader
from scheduler import Scheduler_Wrapper
from dmd_trainer import Trainer
from tqdm import tqdm
from PIL import Image
import os
import argparse
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/dmd_distill_config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = config.train.epochs
    is_conditional = config.train.is_conditional
    sampler_scheduler = Scheduler_Wrapper(config.sampler_scheduler.type, shift=3.0)
    guide_scale = config.sampler.guide_scale
    cfg = True if guide_scale > 1.0 else False 

    trainer = Trainer(config, sampler_scheduler, device)
    trainer.train()

    

if __name__ == "__main__":
    main()

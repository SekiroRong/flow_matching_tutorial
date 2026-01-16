import torch
from torch import nn
from model import Dummy_DiT
from datasets import train_loader, test_loader
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Dummy_DiT()
model.load_state_dict(torch.load("outputs/checkpoint.pth", map_location=torch.device('cpu')))
model.to(device)

def main():
    noises = torch.randn((1, 28, 28)).unsqueeze(0).to(device)
    n_steps = 100
    time_steps = torch.linspace(0, 1.0, n_steps + 1).to(device)
    label = torch.tensor([2]).to(device)
    latents = noises

    model.eval()
    for i in range(n_steps):
        t_start=time_steps[i]
        t_end=time_steps[i + 1]
    
        velocity = model(latents, torch.tensor([t_start]).to(device), label)
        latents += velocity * (t_end - t_start)
        _latents = latents.clone().detach().squeeze()
        _latents = (_latents.clamp(min=0, max=1) * 255).to(torch.uint8)
    img = Image.fromarray(_latents.cpu().numpy(), mode='L')
    img.save(f"outputs/final.png")
    

if __name__ == "__main__":
    main()

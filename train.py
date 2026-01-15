import torch
from torch import nn
from model import Dummy_DiT
from datasets import train_loader, test_loader
from tqdm import tqdm
from PIL import Image

model = Dummy_DiT()
optimizer = torch.optim.Adam(model.parameters(), 1e-2)
loss_fn = nn.MSELoss()

def main():
    epochs = 1
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for i, (imgs, label) in pbar:
            noises = torch.randn_like(imgs)
            t = torch.rand(imgs.size(0))
            _t = t[:, None, None, None]
            
            latents = (1 - _t) * noises + _t * imgs
            velocities = imgs - noises
            
            optimizer.zero_grad()
            loss = loss_fn(model(latents, t), velocities)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })

    noises = torch.randn_like(imgs[0]).unsqueeze(0)
    n_steps = 8
    time_steps = torch.linspace(0, 1.0, n_steps + 1)
    latents = noises

    for i in range(n_steps):
        t_start=time_steps[i]
        t_end=time_steps[i + 1]
    
        velocity = model(latents, torch.tensor([t_start]))
        latents += velocity * (t_end - t_start)
        _latents = latents.clone().detach().squeeze()
        _latents = (_latents * 255).to(torch.uint8)
        img = Image.fromarray(_latents.cpu().numpy(), mode='L')
        img.save(f"outputs/{i}.png")
    

if __name__ == "__main__":
    main()
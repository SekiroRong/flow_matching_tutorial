import torch
import numpy as np

class Scheduler_Wrapper:
    """
    First-order Euler method for diffusion sampling.
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """
    def __init__(self, scheduler_type, shift=1.0):
        if scheduler_type == "euler":
            self.scheduler = EulerScheduler(shift=shift)
        else:
            raise NotImplementedError

class EulerScheduler:
    def __init__(self, num_steps=1000, shift=1.0):
        self.num_steps = num_steps
        self.init_sigmas(shift)

    def init_sigmas(self, shift):
        alphas = np.linspace(1, 1 / self.num_steps, self.num_steps + 1)[::-1].copy()
        sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)

        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
    
    def step(
        self, sample: torch.Tensor, velocity: torch.Tensor, step_index: int) -> torch.Tensor:
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        dt = sigma_next - sigma

        return sample + velocity * dt

if __name__ == "__main__":
    scheduler = Scheduler_Wrapper("euler", shift=3.0)
    print(scheduler.scheduler.sigmas)
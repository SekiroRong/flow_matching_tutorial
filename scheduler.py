import torch

class Scheduler_Wrapper:
    """
    First-order Euler method for diffusion sampling.
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """
    def __init__(self, scheduler_type):
        if scheduler_type == "euler":
            self.scheduler = EulerScheduler()
        else:
            raise NotImplementedError

def EulerScheduler:
    def __init__(self, num_step=50):
        self.num_step = num_step
        self.sigmas = init_sigma()

    def init_sigma(self, ):
        return torch.linspace(1.0, 0.0, self.num_step + 1)
        
    
    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, step_index: int, eps=1e-6
    ) -> torch.Tensor:
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = (sample - denoised_sample) / (sigma + eps)

        return sample + velocity * dt
from torch import nn
import torch
from model import Dummy_DiT

class Trainer(nn.Module):
    def __init__(self, sampler_scheduler):
        super().__init__()
        self.step = 0
        self._initialize_models(sampler_scheduler)
        self.denoising_loss_func = nn.MSELoss()

    def _initialize_models(self, sampler_scheduler):
        self.generator = Dummy_DiT()
        self.real_score = Dummy_DiT()
        self.fake_score = Dummy_DiT()
        self.sampler_scheduler = sampler_scheduler

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

    def _run_generator(
        self,
        label,
        n_steps=4,
        exit_flag=None,
    ) -> torch.Tensor:
        noises = torch.randn((1, 28, 28)).unsqueeze(0).to(device)
        time_steps = torch.linspace(0, 1000, n_steps).to(device)
        sigmas = self.sampler_scheduler.scheduler.sigmas.to(device)
        label = label.to(device)
        latents = noises
    
        for i in range(n_steps):
            t_start=int(time_steps[i])
            sigma_start = sigmas[t_start]
            if not is_conditional:
                label = None

            # random sample a step to calculate gradient (Reduce Memory Usage)
            if exit_flag != i:
                with torch.no_grad():
                    velocity = model(latents, torch.tensor([sigma_start]).to(device), label)
                    latents = self.sampler_scheduler.scheduler.step(latents, velocity, t_start)
            else:
                velocity = model(latents, torch.tensor([sigma_start]).to(device), label)
                latents = self.sampler_scheduler.scheduler.step(latents, velocity, t_start)
                break

        return latents, sigma_start

    def _compute_kl_grad(
        self, noisy_latents: torch.Tensor,
        sigma: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        fake_velocity = self.fake_score(noisy_latents, sigma, label)
        real_velocity = self.real_score(noisy_latents, sigma, label)
        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (fake_velocity - real_velocity)

        return grad

    def compute_distribution_matching_loss(
        self,
        pred_image: torch.Tensor,
        label: torch.Tensor,
        sigma_start: torch.Tensor,
    ) -> torch.Tensor:
        original_latent = pred_image

        noises = torch.randn_like(original_latent)

        max_sigma = sigma_start
        min_sigma = self.sampler_scheduler.sigmas[-1]

        sigma = (torch.rand(max_sigma-min_sigma) + min_sigma).unsqueeze(0).repeat(original_latent.size(0))
        _sigma = sigma[:, None, None, None]
        
        noisy_latents = (1 - _sigma) * noises + _sigma * original_latent

        with torch.no_grad():
            grad = self._compute_kl_grad(noisy_latents, sigma, label)

        dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        
        return dmd_loss
            

    def generator_loss(
        self,
        n_steps=4,
    ) -> torch.Tensor:

        indice = torch.randint(low=0, high=n_steps, (1,))
        label = torch.randint(low=0, high=9, (1,))
        pred_image, sigma_start = self._run_generator(label, n_steps, exit_flag=indice)

        dmd_loss = self.compute_distribution_matching_loss(
            pred_image=pred_image,
            sigma_start=sigma_start,
        )

        return dmd_loss

    def critic_loss(
        self,
        n_steps=4,
    ) -> torch.Tensor:
        # Step 1: Run generator on backward simulated noisy input
        indice = torch.randint(low=0, high=n_steps, (1,))
        label = torch.randint(low=0, high=9, (1,))
        with torch.no_grad():
            pred_image, sigma_start = self._run_generator(label, n_steps, exit_flag=indice)

        # Step 3: Compute the denoising loss for the fake critic
        original_latent = pred_image
        
        noises = torch.randn_like(original_latent)

        max_sigma = sigma_start
        min_sigma = self.sampler_scheduler.sigmas[-1]

        sigma = (torch.rand(max_sigma-min_sigma) + min_sigma).unsqueeze(0).repeat(original_latent.size(0))
        _sigma = sigma[:, None, None, None]
        
        noisy_latents = (1 - _sigma) * noises + _sigma * original_latent

        velocities = original_latent - noises
        denoising_loss = self.denoising_loss_func(self.fake_score(latents, torch.tensor([sigma_start]).to(device), label), velocities)
        return denoising_loss

    def fwdbwd_one_step(self, batch_size, train_generator):
        if train_generator:
            generator_loss = self.generator_loss(batch_size=batch_size, n_steps=4)
            generator_loss.backward()
        else:
            critic_loss = self.critic_loss(batch_size=batch_size, n_steps=4)
            critic_loss.backward()

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1


            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
        
        
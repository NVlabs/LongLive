import torch


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + \
            (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / \
            (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) /
                          num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * \
                (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing


    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)


    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

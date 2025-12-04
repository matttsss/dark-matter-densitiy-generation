# generative_model/DDPM.py

import torch
import torch.nn as nn
from .DiT import DiT




class DDPM(nn.Module):
    """
    DDPM for 2D images with conditioning vectors using DiT noise predictor.
    """
    def __init__(
        self,
        img_size=100,
        in_channels=1,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        cond_dim=712,
        patch_size=10,
        hidden_size=512,
        depth=12,
        num_heads=8,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.timesteps = timesteps

        # Noise predictor eps_θ
        self.eps_model = DiT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            cond_dim=cond_dim,
        )

        # ==========================
        # Forward variance schedule
        # ==========================
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        Helper: a[t] with broadcast to x_shape.
        a: (T,)
        t: (B,)
        returns: (B, 1, ..., 1) matching x_shape
        """
        B = t.shape[0]
        out = a.gather(0, t)  # (B,)
        return out.view(B, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Forward diffusion: q(x_t | x_0)
        x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε
        """

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean(self, x_t: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Reverse mean μ_θ(x_t, t, cond):
        μ_θ = 1/sqrt(α_t) * (x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ(x_t, t, cond))
        """
        eps_pred = self.eps_model(x_t, t, conditions)

        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        mean = sqrt_recip_alphas_t * (
            x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_pred
        )
        return mean

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """
        Sample one reverse step: x_{t-1} ~ p_θ(x_{t-1} | x_t)
        """
        mean = self.p_mean(x_t, t, conditions)
        noise = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        std = self._extract(torch.sqrt(self.betas), t, x_t.shape)

        return mean + nonzero_mask * std * noise

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Full reverse process from pure noise to image.
        
        Args:
            conditions: (B, 712) - conditioning vectors
        Returns:
            (B, 1, 512, 512) - generated images
        """
        device = conditions.device
        B = conditions.shape[0]
        x = torch.randn(B, self.in_channels, self.img_size, self.img_size, device=device)
        
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((B,), t_idx, dtype=torch.long, device=device)
            x = self.p_sample(x, t, conditions)
        
        return x
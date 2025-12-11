import torch
import torch.nn as nn
from .DiT import DiT


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from 'Improved DDPM' paper"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


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
        cond_dim=768,
        patch_size=4,     
        hidden_size=512,
        depth=12,
        num_heads=8,
        schedule="cosine",  # NEW: "linear" or "cosine"
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.timesteps = timesteps

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
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
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
        B = t.shape[0]
        out = a.gather(0, t)
        return out.view(B, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_mean(self, x_t: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
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
        mean = self.p_mean(x_t, t, conditions)
        noise = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        std = self._extract(torch.sqrt(self.betas), t, x_t.shape)

        return mean + nonzero_mask * std * noise

    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, guidance_scale: float = 1.0) -> torch.Tensor:
        """
        Full reverse process with optional CFG.
        
        Args:
            conditions: (B, cond_dim) - conditioning vectors
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
        """
        device = conditions.device
        B = conditions.shape[0]
        x = torch.randn(B, self.in_channels, self.img_size, self.img_size, device=device)
        
        null_cond = torch.zeros_like(conditions)
        
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((B,), t_idx, dtype=torch.long, device=device)
            
            if guidance_scale != 1.0:
                # CFG: eps = eps_uncond + scale * (eps_cond - eps_uncond)
                eps_cond = self.eps_model(x, t, conditions)
                eps_uncond = self.eps_model(x, t, null_cond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                
                # Manual p_sample with guided eps
                sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
                betas_t = self._extract(self.betas, t, x.shape)
                sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                
                mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps)
                
                noise = torch.randn_like(x) if t_idx > 0 else 0
                std = self._extract(torch.sqrt(self.betas), t, x.shape)
                x = mean + (t_idx > 0) * std * noise
            else:
                x = self.p_sample(x, t, conditions)
        
        return x

    @torch.no_grad()
    def ddim_sample(self, conditions: torch.Tensor, steps: int = 50, eta: float = 0.0, guidance_scale: float = 1.0) -> torch.Tensor:
        """
        DDIM sampling (faster, often cleaner).
        
        Args:
            conditions: (B, cond_dim)
            steps: number of denoising steps (can be << timesteps)
            eta: stochasticity (0 = deterministic, 1 = DDPM-like)
            guidance_scale: CFG scale
        """
        device = conditions.device
        B = conditions.shape[0]
        x = torch.randn(B, self.in_channels, self.img_size, self.img_size, device=device)
        
        # Subsample timesteps
        times = torch.linspace(self.timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)
        null_cond = torch.zeros_like(conditions)
        
        for i in range(len(times) - 1):
            t = times[i]
            t_prev = times[i + 1]
            t_batch = t.expand(B)
            
            # Get eps prediction (with optional CFG)
            if guidance_scale != 1.0:
                eps_cond = self.eps_model(x, t_batch, conditions)
                eps_uncond = self.eps_model(x, t_batch, null_cond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.eps_model(x, t_batch, conditions)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)  # Optional: clamp for stability
            
            # DDIM update
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * eps
            
            noise = torch.randn_like(x) if eta > 0 and i < len(times) - 2 else 0
            x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
        
        return x
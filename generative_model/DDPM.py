import torch
import torch.nn as nn
import torch.nn.functional as F
from .DiT import DiT




class DDPM(nn.Module):

    def __init__(self, data_dim=(712,), timesteps=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.data_dim = data_dim
        self.timesteps = timesteps
        
        # Initialize noise predictor
        self.eps_model = DiT()
        
        # ===================================================================
        # VARIANCE SCHEDULE (Section 2, Algorithm 1)
        # ===================================================================
        # β_t: variance schedule, controls noise at each step
        # Paper: "We set the forward process variances to constants increasing 
        #         linearly from β_1 = 10^-4 to β_T = 0.02"
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # α_t = 1 - β_t
        # Paper Equation (4): q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
        alphas = 1.0 - betas
        
        # ᾱ_t = ∏_{s=1}^t α_s (cumulative product)
        # Paper Equation (4): Allows direct sampling of x_t from x_0
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Precompute terms for efficiency
        # Used in forward process q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # Used in reverse process (simplified posterior)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
    
    def _extract(self, a, t, x_shape):
        """Helper to extract coefficient at timestep t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_0, t, noise=None):
        """
        FORWARD DIFFUSION PROCESS: q(x_t | x_0)
        
        Paper Equation (4):
            q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
        
        Reparameterization:
            x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,  where ε ~ N(0, I)
        
        Location: Section 2, just after Equation (4)
        
        This allows us to sample x_t directly from x_0 without 
        iterating through all intermediate steps!
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # The key equation: x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_mean(self, x_t, t, conditions):
        """
        REVERSE PROCESS MEAN: μ_θ(x_t, t)
        
        Paper Algorithm 2, line 4:
            μ_θ(x_t, t) = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
        
        Location: Section 3.2, Equation (11)
        
        This is the predicted mean of p_θ(x_{t-1} | x_t).
        """
        # Predict noise
        eps_pred = self.eps_model(x_t, t, conditions)
        
        # Extract coefficients
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # Equation (11): μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ)
        mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_pred)
        
        return mean
    
    @torch.no_grad()
    def p_sample(self, x_t, t, conditions):
        """
        REVERSE PROCESS SAMPLING: Sample x_{t-1} ~ p_θ(x_{t-1} | x_t)
        
        Paper Algorithm 2, lines 4-5:
            z ~ N(0, I) if t > 1, else z = 0
            return μ_θ(x_t, t) + σ_t · z
        
        Location: Algorithm 2, Section 3.2
        
        For simplicity, we use σ_t = β_t (fixed variance).
        The paper shows this works as well as learned variance.
        """
        # Get mean
        mean = self.p_mean(x_t, t, conditions)
        
        # Sample noise
        noise = torch.randn_like(x_t)
        
        # No noise on final step (t=0)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        
        # σ_t = β_t (simplified, from Section 3.2)
        std = self._extract(torch.sqrt(self.betas), t, x_t.shape)
        
        return mean + nonzero_mask * std * noise
    
    @torch.no_grad()
    def sample(self, conditions):
        """
        FULL SAMPLING PROCESS
        
        Paper Algorithm 2:
            1. x_T ~ N(0, I)
            2. for t = T, ..., 1 do:
                  x_{t-1} ~ p_θ(x_{t-1} | x_t)
            3. return x_0
        
        Location: Algorithm 2, Section 3
        
        Start from pure noise and iteratively denoise.
        """
        # Start from pure Gaussian noise
        x = torch.randn(conditions.shape[0], self.data_dim)
        
        # Reverse diffusion
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((conditions.shape[0],), t_idx, dtype=torch.long)
            x = self.p_sample(x, t, conditions)
        
        return x
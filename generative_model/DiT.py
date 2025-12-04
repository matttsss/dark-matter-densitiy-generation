#Code adapted and sourced from : https://github.com/facebookresearch/DiT/blob/main/models.py

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: embedding dimension
        grid_size: int, the grid height and width
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Helper to convert 2D grid to sinusoidal embeddings."""
    assert embed_dim % 2 == 0
    
    # Use half dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim: output dimension for each position
        pos: positions to be encoded: size (M,)
    Returns:
        emb: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        """t: (B,) -> (B, hidden_size)"""
        return self.mlp(t.float().unsqueeze(-1))


class ConditionsEmbedder(nn.Module):
    """Embed conditioning vector into hidden space."""
    def __init__(self, hidden_size, conditions_size=712):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditions_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, conditions):
        """conditions: (B, 712) -> (B, hidden_size)"""
        return self.mlp(conditions)


class DiTBlock(nn.Module):
    """DiT transformer block with adaLN-Zero conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer to map tokens back to patch space."""
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B, N, hidden_size)
        c: (B, hidden_size)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for 2D images with conditioning.
    
    Args:
        img_size: Image size (e.g., 512 for 512x512)
        patch_size: Patch size (e.g., 16 means 16x16 patches)
        in_channels: Number of input channels (e.g., 1 for grayscale)
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension = hidden_size * mlp_ratio
        cond_dim: Conditioning vector dimension (e.g., 712)
    """
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        in_channels=1,
        hidden_size=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        cond_dim=712,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        
        # Patch embedding: split image into patches
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        
        # Time and condition embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionsEmbedder(hidden_size, conditions_size=cond_dim)
        
        # Positional embedding (fixed sin-cos)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # Final layer maps back to patch pixels
        self.final_layer = FinalLayer(hidden_size, patch_size * patch_size * self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                            int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Convert patches back to image.
        x: (B, N, patch_size**2 * C)
        returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, conditions):
        """
        Forward pass to predict noise.
        
        Args:
            x: (B, C, H, W) - noisy images
            t: (B,) - timesteps
            conditions: (B, 712) - conditioning vectors
        
        Returns:
            (B, C, H, W) - predicted noise
        """
        # Embed patches
        x = self.x_embedder(x) + self.pos_embed  # (B, N, hidden_size)
        
        # Embed time and conditions, combine them
        t_emb = self.t_embedder(t)               # (B, hidden_size)
        c_emb = self.c_embedder(conditions)      # (B, hidden_size)
        c = t_emb + c_emb                        # (B, hidden_size)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)                      # (B, N, hidden_size)
        
        # Final layer and unpatchify
        x = self.final_layer(x, c)               # (B, N, patch_size**2 * C)
        x = self.unpatchify(x)                   # (B, C, H, W)
        
        return x
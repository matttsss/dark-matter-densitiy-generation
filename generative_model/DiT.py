#Code adapted and sourced from : https://github.com/facebookresearch/DiT/blob/main/models.py

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from .DiT_utils import get_1d_sincos_pos_embed_from_grid


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):

        return self.mlp(t.unsqueeze(-1))



class ConditionsEmbedder(nn.Module):
    
    def __init__(self ,hidden_size,conditions_size = 2):
        super().__init__()
        self.conditions_size = conditions_size
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(conditions_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        

    def forward(self, conditions):
        return self.mlp(conditions)
        

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
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
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone for 1D vectors.
    """
    def __init__(
        self,
        input_size=712,
        patch_size=4,
        hidden_size=128,
        depth=3,
        num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # Calculate number of patches for 1D vector
        assert input_size % patch_size == 0, f"input_size {input_size} must be divisible by patch_size {patch_size}"
        self.num_patches = input_size // patch_size
        
        # Linear projection to embed patches
        self.x_embedder = nn.Linear(patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.num_patches))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding:
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, x):
        """
        Convert 1D vector into patches.
        x: (N, input_size) tensor
        returns: (N, num_patches, patch_size) tensor
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_patches, self.patch_size)
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to 1D vector.
        x: (N, num_patches, patch_size) tensor
        returns: (N, input_size) tensor
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.input_size)
        return x

    def forward(self, x, t, conditions):
        """
        Forward pass of DiT for 1D vectors.
        x: (N, input_size) tensor of 1D vectors (e.g., shape [batch_size, 712])
        t: (N,) tensor of diffusion timesteps
        """
        # Patchify: (N, input_size) -> (N, num_patches, patch_size)
        x = self.patchify(x)
        
        # Embed patches: (N, num_patches, patch_size) -> (N, num_patches, hidden_size)
        x = self.x_embedder(x)
        
        # Add positional embedding
        x = x + self.pos_embed  # (N, num_patches, hidden_size)
        
        # Embed timestep
        t = self.t_embedder(t)  # (N, hidden_size)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, t, conditions)  # (N, num_patches, hidden_size)
        
        # Final layer to get patches back
        x = self.final_layer(x, t, conditions)  # (N, num_patches, patch_size)
        
        # Unpatchify: (N, num_patches, patch_size) -> (N, input_size)
        x = self.unpatchify(x)
        
        return x

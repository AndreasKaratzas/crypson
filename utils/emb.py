
import math
import torch
import torch.nn as nn

from einops import repeat


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False, device='cpu'):
    """
    Create sinusoidal timestep embeddings.

    Parameters
    ----------
    timesteps : torch.Tensor
        A 1-D Tensor of N indices, one per batch element. These may be fractional.
    dim : int
        The dimension of the output.
    max_period : int, optional
        Controls the minimum frequency of the embeddings.
    repeat_only : bool, optional
        If True, only repeat the input timesteps to the desired dimension.
    device : str, optional
        The device to place the output Tensor on.
    
    Returns
    -------
    torch.Tensor
        A Tensor of shape (N, dim) of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim):
        super(TimeEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x):
        t_emb = timestep_embedding(
            x, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        return emb

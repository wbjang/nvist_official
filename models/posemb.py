
from typing import Union, Tuple
import numpy as np
import torch
from torch import Tensor
# Some of the code here is adapted from MAE, BARF 
    # https://github.com/facebookresearch/mae
    # https://github.com/chenhsuanlin/bundle-adjusting-NeRF
# then modified by the author

def positional_encoding(positions: Tensor, freqs: int) -> Tensor:
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Union[int, Tuple[int, int]], cls_token: bool=False) -> Tensor:
    """
    Generates a 2D sinusoidal positional embedding for a given grid size.

    Parameters:
    - embed_dim (int): The embedding dimension for each position.
    - grid_size (Union[int, Tuple[int, int]]): The size of the grid for which to generate
      embeddings. If an integer is provided, it's used for both height and width.
    - cls_token (bool, optional): If True, adds an additional zero embedding at the beginning
      for a class token. Default is False.
    - pose_token (bool, optional): If True, adds an additional zero embedding for a pose token,
      in addition to the class token if cls_token is also True. Default is False.

    Returns:
    - torch.Tensor: The generated 2D sinusoidal positional embeddings with shape
      [(1+pose_token+cls_token) + grid_size[0]*grid_size[1], embed_dim] if cls_token or
      pose_token is True, otherwise [grid_size[0]*grid_size[1], embed_dim].
    """


    if isinstance(grid_size, int):
        grid_size = [grid_size, grid_size]
    grid_size_h, grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32) + 0.5
    grid_w = np.arange(grid_size_w, dtype=np.float32) + 0.5
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        num_cls = 1
        pos_embed = np.concatenate([np.zeros([num_cls, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_pos_encoding(camera_params, L):
    camera_dim = camera_params.shape[-1]
    omega = torch.arange(torch.tensor(L).float()).type_as(camera_params)
    # omega /= (embed_dim - 2) / 2
    omega = 2. ** omega * np.pi
    omega = camera_params.reshape(-1,camera_dim,1) * omega.reshape(1,1,-1)
    omega = omega.reshape(omega.shape[0],-1)
    pos_embed = torch.cat([camera_params.reshape(-1,camera_dim), torch.sin(omega), torch.cos(omega)], dim=-1)
    return pos_embed

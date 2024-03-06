
# Some of the code here is adapted from from TensoRF:
    # https://github.com/apchenstu/TensoRF
# and then modified by the author.

import numpy as np
import torch
from torch import Tensor
from kornia import create_meshgrid
from typing import Union, Tuple

def get_inverse_pose(pose: np.ndarray, offset: Union[np.array, list, Tuple]) -> np.ndarray:
    """
    Computes the inverse of a given pose and applies an offset.

    Args:
        pose (np.ndarray): The pose matrix of shape [..., 4, 4].
        offset (Union[np.ndarray, list, Tuple]): The offset to be applied after inversion, should be of length 3.

    Returns:
        np.ndarray: The inverted pose with the offset applied, of the same shape as input pose.
    """
    # Ensure offset is a numpy array
    if not isinstance(offset, np.ndarray):
        offset = np.array(offset)
    
    # Validate offset size
    if offset.size != 3:
        raise ValueError("Offset must have exactly 3 elements.")

    R, T = pose[..., :3, :3], pose[..., :3, -1:]
    pose_inv = np.zeros_like(pose)
    pose_inv[..., :3, :3] = R.transpose(-1,-2)
    pose_inv[..., 3, 3] = 1.
    pose_inv[..., :3:, -1:] = -R.transpose(-1,-2)@T + offset.reshape(-1,1)

    return pose_inv


def get_rays(directions : Tensor, c2w : Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get ray origin and normalized directions in world coordinates for all pixels in one image.

    Args:
        directions (Tensor): Precomputed ray directions in camera coordinates (H, W, 3).
        c2w (Tensor): Transformation matrix from camera coordinates to world coordinates (3, 4).

    Returns:
        Tuple[Tensor, Tensor]: A Tuple containing:
            - rays_o (Tensor): The origin of the rays in world coordinates (H*W, 3).
            - rays_d (Tensor): The normalized direction of the rays in world coordinates (H*W, 3).
    """

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    # Reshape for compatibility with non-contiguous inputs
    rays_d = rays_d.reshape(-1, 3)
    rays_o = rays_o.reshape(-1, 3)

    return rays_o, rays_d

def get_ray_directions(H: int, W: int, focal: Tensor, center: Tensor = None) -> Tensor:
    """
    Get ray directions for all pixels in camera coordinate system.

    Args:
        H (int): Image height.
        W (int): Image width.
        focal (Tensor): Focal length, can be a scalar or a tensor with shape (2,) for different focal lengths along the x and y axes.
        center (Tensor, optional): The center of the image in pixels. If None, it defaults to the center of the image.

    Returns:
        Tensor: A tensor of shape (H, W, 3) representing the direction of the rays in camera coordinates.
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # Default center to image center if not provided
    if center is None:
        center = torch.tensor([W / 2, H / 2], dtype=torch.float32)

    directions = torch.stack([
        (i - center[0]) / focal if focal.dim() == 0 else (i - center[0]) / focal[0], 
        (j - center[1]) / focal if focal.dim() == 0 else (j - center[1]) / focal[1], 
        torch.ones_like(i)
    ], -1)

    return directions

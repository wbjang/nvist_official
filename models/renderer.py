

from torch import nn, Tensor
import torch
from .posemb import positional_encoding 

import torch.nn.functional as F
from torch_efficient_distloss import eff_distloss 
from easydict import EasyDict as edict

from typing import Tuple, Optional

# Some of the code here is adapted from from MAE and TensoRF:
    # https://github.com/facebookresearch/mae
    # https://github.com/apchenstu/TensoRF
# and then modified by the author.

class Renderer(nn.Module):
    def __init__(self, aabb=torch.tensor([[-0.8,-0.8,-0.8],[0.8,0.8,0.8]]), near_far=[0.0, 1.5],
                resolution=48, model='vector_matrix', # vector_matrix or triplanes 
                decoder_output_dim=32, sigma_dim=8, rgb_dim=24, # they could overlap - 
                rgb_feature_dim=64,  sigma_activation = 'ReLU', concat_viewdir=True,
                viewdir_pe = 4, using_sigmoid_at_rgb = True, apply_distloss=True,  using_sigma_mlp=False, padding_zero = True,  
                nSamples=48
                ):
        """
        Renderer class for volume rendering

        The goal is to render pixels from 3D representations given camera rays. 

        Parameters:
        - aabb (Tensor): Axis-aligned bounding box represented by its min and max corners, 
        used to define the 3D scene's spatial boundaries. Default is a box centered at the origin with side lengths 1.6.
        - near_far (list): A two-element list specifying the near and far clipping planes. Default is [0.0, 1.5].
        - resolution (int): The resolution of the output image or feature map. Default is 48.
        - model (str): The rendering model to use. Options include 'vector_matrix' and 'triplanes'. Default is 'vector_matrix'.
        - decoder_output_dim (int): The dimensionality of the decoder output, which combines sigma and RGB dimensions. Default is 32.
        - sigma_dim (int): The number of dimensions allocated for sigma (density) values. Default is 8.
        - rgb_dim (int): The number of dimensions allocated for RGB values. Default is 24.
        - rgb_feature_dim (int): The dimensionality of the features before predicting RGB values. Default is 64.
        - sigma_activation (str): The activation function to apply to sigma values. Default is 'ReLU'.
        - concat_viewdir (bool): Whether to concatenate the view direction to the input features. Default is True.
        - viewdir_pe (int): The frequency of positional encoding applied to the view direction. Default is 4.
        - using_sigmoid_at_rgb (bool): Whether to apply a sigmoid activation to the RGB output. Default is True.
        - apply_distloss (bool): Whether to apply distance-based loss during training. Default is True.
        - using_sigma_mlp (bool): Whether to use an MLP for predicting sigma values. Default is False.
        - padding_zero (bool): Whether to pad inputs with zeros. Default is True.
        - nSamples (int): The number of samples to take along each ray. Default is 48.

        This renderer is flexible and configurable to support various neural rendering techniques and optimization strategies.
        """
        
        super().__init__()
        self.aabb = aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.near_far = near_far
        self.resolution = resolution
        self.model = model


        self.sigma_dim = sigma_dim
        self.rgb_dim = rgb_dim
        self.feature_dim = decoder_output_dim
        assert self.feature_dim == self.sigma_dim + self.rgb_dim

        self.rgb_feature_dim = rgb_feature_dim

        
        self.sigma_activation = sigma_activation
        self.concat_viewdir = concat_viewdir
        self.viewdir_pe = viewdir_pe
        self.apply_distloss = apply_distloss
 
        self.using_sigma_mlp = using_sigma_mlp
        self.padding_zero = padding_zero

        self.nSamples = nSamples
        
        # the difference will be how to aggregate the features
        assert model in ['vector_matrix', 'triplanes', 'CP']
       
        # last layer activation: sigmoid (when rgb is between 0 and 1) or linear
        self.using_sigmoid_at_rgb = using_sigmoid_at_rgb

        self.initialize_mlp()
        self.matMode = [[0,1],[2,0],[1,2]]
        self.vecMode=[2,1,0]

    def initialize_mlp(self) -> None:
        """
        Initializes the multi-layer perceptron (MLP) modules for RGB values (optionally sigma) based on the rendering model.

        When we use vector_matrix, we do not have MLP for sigma as in TensoRF.
        Otherwise, we have mlp module for sigma - concatenate three sigma features then regress to single value.

        For RGB which is 4D tensor, we aggregate the RGB features and regress them into 3
        """

        # for VM/triplanes, we concatenate - so we scale 3, for CP, we scale 1
        scale = 3 if self.model in ["vector_matrix", "triplanes"] else 1

        # Initialize sigma MLP for triplanes model or vector_matrix model with sigma MLP enabled.
        if self.model == "triplanes" or (self.model == "vector_matrix" and self.using_sigma_mlp):
            self.sigma_mlp = nn.Linear(self.sigma_dim * scale, 1)

        # Determine the input dimension for the RGB MLP based on model configuration.
        rgb_input_dim = self.rgb_dim * scale
        viewdir_dim = (self.viewdir_pe * 6 + 3) if self.concat_viewdir else 0

        # Initialize the RGB MLP with an intermediate feature dimension and potentially concatenated view direction.
        self.rgb_matrix = nn.Sequential(
            nn.GELU(), nn.Linear(rgb_input_dim, self.rgb_feature_dim), nn.GELU()
        )

        self.rgb_mlp = nn.Sequential(
            nn.Linear(self.rgb_feature_dim + viewdir_dim, self.rgb_feature_dim), nn.GELU(),
            nn.Linear(self.rgb_feature_dim, 3)
        )
        

    def normalize_coord(self, xyz_sampled: Tensor) -> Tensor:
        """
        Normalize coordinates to [-1, 1] for compatibility with PyTorch's grid_sample.
        Parameters:
        - xyz_sampled (Tensor): A tensor of 3D coordinates with shape (..., 3) where '...' represents
        any number of leading dimensions (e.g., batch size).

        Returns:
        - Tensor: The normalized 3D coordinates with the same shape as `xyz_sampled`, scaled to the
        range [-1, 1] based on the AABB of the scene.
        """
        return (xyz_sampled-self.aabb[0].type_as(xyz_sampled)) * self.invaabbSize.type_as(xyz_sampled) - 1
    
    
    def sample_pts_along_rays(self, rays_o: Tensor, rays_d: Tensor, is_train: bool=True) -> Tensor:
        """
        Samples points along rays for rendering indoor scenes, with optional randomness for training.

        This method generates sample points along each ray between specified near and far planes.
        For training, randomness is introduced to the sample points. 
        For evaluation or testing, sample points are uniformly distributed along the ray.

        We sample the 'middle' points in each interval - later, we use torch.distloss based on intervals

        Parameters:
        - rays_o (Tensor): The origins of the rays with shape [num_rays, 3], where num_rays is the
        number of rays and 3 corresponds to the XYZ coordinates.
        - rays_d (Tensor): The directions of the rays with shape [num_rays, 3], normalized to unit length.
        - is_train (bool, optional): Flag to indicate if the sampling is performed for training. If True,
        randomness is added to the sampling process. Default is True.

        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing:
            - rays_pts (Tensor): The sampled points along the rays with shape [num_rays, nSamples, 3],
            where nSamples is the number of samples per ray.
            - interpx (Tensor): The interpolated sample positions along the ray in the normalized depth
            range [near, far], with shape [num_rays, nSamples + 1].
        """


        N_samples = self.nSamples
        near, far = self.near_far
        interpx = (
            torch.linspace(near, far, N_samples + 1).unsqueeze(0).to(rays_o)
        )     
        if is_train:
            interpx[:, :-1] += (
                torch.rand_like(interpx).to(rays_o)
                * ((far - near) / N_samples)
            )[:, :-1]
        interpx_mid = (interpx[:, 1:] + interpx[:, :-1]) / 2
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx_mid[..., None]

        return rays_pts, interpx


    def compute_weights(self, sigma: Tensor, dist: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the alpha/weights given density (sigma) and distances along rays.

        Parameters:
        - sigma (Tensor): A tensor of shape [N_images, N_rays, N_samples] representing the density
        of sample points along each ray.
        - dist (Tensor): A tensor of shape [N_images, N_rays, N_samples] representing the distances
        between consecutive sample points along each ray.

        Returns:
        - Tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - alpha (Tensor): The computed alpha values for each sample point, indicating the
            opacity accumulated up to that point.
            - weights (Tensor): The weights for each sample point, used for integrating the color
            along the ray.
            - T (Tensor): The accumulated transmittance to the last sample along each ray.

        """
        # Compute alpha values for each sample point
        alpha = 1. - torch.exp(-sigma*dist)
        # Initial transmittance is 1 (fully transparent) at the start of the ray
        ones_init = torch.ones(list(alpha.shape[:-1]) + [1]).to(alpha.device)
        # Compute transmittance (T) cumulatively, with a small epsilon to avoid division by zero
        T = torch.cumprod(torch.cat([ones_init, 1.0 - alpha + 1e-10], -1), -1)
         # Compute weights for each sample point
        weights = alpha * T[..., :-1]  

        return alpha, weights, T[...,-1:] 
    
    def compute_features(self, xyz_sampled: Tensor, matrixs: Tensor, vectors: Optional[Tensor]=None, viewdirs: Optional[Tensor]=None) -> Tuple[Tensor, Tensor]:
        """
        Computes sigma and RGB features from 3d grid representation.

        Parameters:
        - xyz_sampled (Tensor): Sampled 3D points with shape [B, num_pts_per_image, num_pts_along_the_ray, 3].
        - matrixs (Tensor): Feature matrices for each plane with shape [3, B, feat_dim, resolution, resolution].
        - vectors (Optional[Tensor]): Line features for vector_matrix model with shape [3, B, feat_dim, resolution, 1].
        - viewdirs (Optional[Tensor]): View directions with shape [B, num_pts_per_image * num_pts_along_the_ray, 3].

        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing:
            - sigma_features (Tensor): Density (sigma) features computed from the input.
            - rgb_features (Tensor): RGB color features computed for the sampled points.

        
        It projects `xyz_sampled` into three different planes or lines (based on the model) and uses grid sampling to extract features. 
        """

        assert ((vectors is not None) and (self.model in ['vector_matrix'])) or (vectors is None and self.model in ['triplanes', 'CP']), \
           "Vectors should be provided only for the 'vector_matrix' model or not provided for 'triplanes'/'CP' models."
    
        # xyz_sampled : B, num_pts_per_image, num_pts_along_the_ray, 3
        # matrixs: 3, B, feat_dim, triplane_res, triplane_res
        # vectors: 3, B, feat_dim, triplane_res, 1
        # to do grid_sample, we need to project xyz_sampled into three different planes
        # then turn xyz_sampled into 3, B, num_pts_per_image, num_pts_along_the_ray, 2
        padding_mode = 'zeros' if self.padding_zero else 'border'

        number_of_images = xyz_sampled.shape[0]
        number_of_points = xyz_sampled.shape[1] * xyz_sampled.shape[2]
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]])) # 3, xyz_sampled.shape[0]
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), 
            dim=-1).detach().reshape(-1,number_of_points,1,2)

        if self.model == "CP": # need to revist
            matrixs_reshaped = matrixs.reshape(-1, matrixs.shape[2], matrixs.shape[3], matrixs.shape[4])
            line_features = F.grid_sample(matrixs_reshaped, coordinate_line, mode='bilinear', padding_mode=padding_mode, align_corners=False)
            line_features = line_features.reshape(3, number_of_images, self.feature_dim, -1)
            sigma_features, rgb_features = line_features.split([self.sigma_dim, self.rgb_dim], dim=-2)
            sigma_features = sigma_features[0] * sigma_features[1] * sigma_features[2]
            sigma_features = torch.sum(sigma_features, dim=-2)  
            rgb_features = torch.cat([rgb_features[0], rgb_features[1], rgb_features[2]], dim=1)
            rgb_features = rgb_features.permute(0,2,1)

        elif self.model in ['vector_matrix', 'triplanes']:
            coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], 
                xyz_sampled[..., self.matMode[2]])).detach().reshape(-1,number_of_points,1,2)
            matrixs_reshaped = matrixs.reshape(-1, matrixs.shape[2], matrixs.shape[3], matrixs.shape[4])
            plane_features = F.grid_sample(matrixs_reshaped, coordinate_plane, mode='bilinear', padding_mode=padding_mode, align_corners=False)
            plane_features = plane_features.reshape(3, number_of_images, self.feature_dim, -1)
            plane_sigma_features, plane_rgb_features = plane_features.split([self.sigma_dim, self.rgb_dim], dim=2)

            plane_rgb_features = plane_rgb_features.split([1,1,1], dim=0)
            plane_rgb_features = torch.cat([p.squeeze(0) for p in plane_rgb_features], dim=1)

            if self.model == "vector_matrix":
                vectors_reshaped = vectors.reshape(-1, vectors.shape[2], vectors.shape[3], vectors.shape[4])
                line_features = F.grid_sample(vectors_reshaped, coordinate_line, mode='bilinear', padding_mode=padding_mode, align_corners=False)
                line_features = line_features.reshape(3, number_of_images, self.feature_dim, -1)
                line_sigma_features, line_rgb_features = line_features.split([self.sigma_dim, self.rgb_dim], dim=2)


                if not self.using_sigma_mlp:
                    sigma_features = torch.relu(torch.sum(plane_sigma_features * line_sigma_features, dim=[0,2]))
                else:
                    sigma_features = plane_sigma_features * line_sigma_features
                    sigma_features = sigma_features.split([1,1,1], dim=0)
                    sigma_features = torch.cat([s.squeeze(0) for s in sigma_features], dim=1)
                    sigma_features = sigma_features.permute(0,2,1)
                    sigma_features = self.sigma_mlp(sigma_features)
                    sigma_features = sigma_features.squeeze(-1)


                line_rgb_features = line_rgb_features.split([1,1,1], dim=0)
                line_rgb_features = torch.cat([p.squeeze(0) for p in line_rgb_features], dim=1)
                rgb_features = plane_rgb_features * line_rgb_features
            elif self.model == "triplanes": # concatenate then apply MLP
                plane_sigma_features = plane_sigma_features.split([1,1,1], dim=0)
                plane_sigma_features = torch.cat([p.squeeze(0) for p in plane_sigma_features], dim=1)
                plane_sigma_features = plane_sigma_features.permute(0,2,1)
                sigma_features = self.sigma_mlp(plane_sigma_features)
                sigma_features = torch.relu(sigma_features.squeeze(-1))
                rgb_features = plane_rgb_features
            rgb_features = rgb_features.permute(0,2,1)

        rgb_features = self.rgb_matrix(rgb_features)

        if self.concat_viewdir:
            viewdirs = viewdirs.reshape(number_of_images, number_of_points, 3)
            rgb_features = torch.cat([rgb_features, torch.cat([positional_encoding(viewdirs, self.viewdir_pe), viewdirs],dim=-1)], dim=-1)
        rgb_features = self.rgb_mlp(rgb_features)


        return sigma_features, rgb_features


    def forward(self, rays_origin: Tensor, rays_direction: Tensor, matrixs: Tensor, vectors: Optional[Tensor] = None, 
                is_train: bool = True, white_bg: bool = False) -> edict:
        """
        Performs the forward pass of the renderer for volume rendering.

        Parameters:
        - rays_origin (Tensor): The origins of the rays with shape [B, 3], where B is the batch size.
        - rays_direction (Tensor): The directions of the rays with shape [B, 3].
        - matrixs (Tensor): Feature matrices for each plane with shape [3, B, feat_dim, triplane_res, triplane_res].
        - vectors (Optional[Tensor]): Optional feature vectors for the 'vector_matrix' model with shape [3, B, feat_dim, triplane_res, 1].
        - is_train (bool): Flag indicating if the model is in training mode. Default is True.
        - white_bg (bool): Flag to use a white background, affecting the final RGB map. Default is False.

        Returns:
        - edict: An EasyDict object containing the following keys and values:
            - rgb_map (Tensor): The rendered RGB color map with shape [B, N_rays, 3].
            - depth_map (Tensor): The estimated depth map with shape [B, N_rays].
            - weight (Tensor): The weights for each sample point along the rays with shape [B, N_rays, N_pts].
            - distloss (Tensor): The effective distance loss value, useful for training.

        The method first samples points along the rays, normalizes the coordinates, computes the sigma and RGB
        features, and then calculates the final RGB color map and depth map based on the computed weights/transmittance.
        """

        # matrixs: 3, B, feat_dim, triplane_res, triplane_res
        # vectors: 3, B, feat_dim, triplane_res, 1

        viewdirs = rays_direction / torch.norm(rays_direction, dim=-1, keepdim=True)

        xyz_sampled, z_vals = self.sample_pts_along_rays(rays_origin, viewdirs, is_train=is_train)
        # xyz_sampled: B, N_rays, N_pts, 3 / z_vals : B, N_rays, N_pts
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        mid_points = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
        viewdirs = viewdirs.unsqueeze(-2).expand(xyz_sampled.shape) # need to check

        xyz_sampled = self.normalize_coord(xyz_sampled)
        sigma, rgb  = self.compute_features(xyz_sampled, matrixs, vectors, viewdirs=viewdirs)

        sigma_output = sigma.reshape(*xyz_sampled.shape[:-1])
        rgb_output = rgb.reshape((*xyz_sampled.shape[:-1], 3))

        if self.using_sigmoid_at_rgb: 
            rgb_output = torch.sigmoid(rgb_output)

        _, weight, _ = self.compute_weights(sigma_output, dists)

        rgb_map = torch.sum(weight[..., None] * rgb_output, -2)

        if white_bg:
            accumulated_transmittancy = torch.sum(weight, dim=-1)
            rgb_map = rgb_map + (1. - accumulated_transmittancy[..., None])

        with torch.no_grad():
            depth_map = torch.sum(weight * mid_points, -1)


        if self.apply_distloss: 
            distloss = eff_distloss(weight, mid_points, dists)
        else:
            with torch.no_grad(): # we still compute... just no gradient
                distloss = eff_distloss(weight, mid_points, dists)

        output = edict(rgb_map = rgb_map, depth_map = depth_map, weight=weight, distloss=distloss)
        
        return output
    

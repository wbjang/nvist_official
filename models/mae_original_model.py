
from functools import partial

from torch import nn, Tensor

from timm.models.vision_transformer import PatchEmbed, Block

import numpy as np

import torch

from utils import normalize_imgs_imagenet, unnormalize_imgs_imagenet, normalize_imgs, unnormalize_imgs

from typing import Tuple, Union, Type, List

# Some of the code here is adapted from MAE 
    # https://github.com/facebookresearch/mae
# and then modified by the author.

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Union[Tuple[int, int], List[int]], cls_token: bool=False) -> np.ndarray:
    """
    Generates a 2D sinusoidal positional embedding.

    Parameters:
    - embed_dim (int): The dimensionality of the embedding.
    - grid_size (Union[Tuple[int, int], List[int]]): A tuple or list containing the grid height and width.
    - cls_token (bool): Whether to include a positional embedding for a class token.

    Returns:
    - pos_embed (np.ndarray): The positional embedding. Shape is either [grid_size*grid_size, embed_dim] 
                              or [1+grid_size*grid_size, embed_dim] if cls_token is True.
    """

    if isinstance(grid_size, int): 
        grid_size = [grid_size, grid_size]
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Generates a 2D sinusoidal positional embedding from a given grid.

    Args:
    - embed_dim (int): The dimensionality of the embedding.
    - grid (Union[np.ndarray, List[List[float]]]): A 2D structure (list of lists or 2D numpy array)
      containing the grid coordinates.

    Returns:
    - np.ndarray: The positional embedding with shape (H*W, D), where H and W are the height and width
      of the grid respectively, and D is the embedding dimension.
    """

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Generates a 1D sinusoidal positional embedding from a list of positions.

    Args:
    - embed_dim (int): The output dimension for each position.
    - pos (Union[np.ndarray, List[float]]): A list or 1D numpy array of positions to be encoded, size (M,).

    Returns:
    - np.ndarray: The positional embedding with shape (M, D), where M is the number of positions
      and D is the embedding dimension.
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb




class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with a Vision Transformer (ViT) backbone.

    This model architecture is designed for image reconstruction tasks, utilizing a Transformer-based encoder
    for masking and encoding input images, followed by a decoder that reconstructs the image from the encoded
    representation.

    Parameters:
    - img_size (int): Size of the input images.
    - patch_size (int): Size of the patches to be extracted from input images.
    - in_chans (int): Number of input channels.
    - embed_dim (int): Dimensionality of the token/patch embeddings.
    - depth (int): Number of transformer blocks in the encoder.
    - num_heads (int): Number of attention heads within the transformer blocks.
    - decoder_embed_dim (int): Dimensionality of the decoder embeddings.
    - decoder_depth (int): Number of transformer blocks in the decoder.
    - decoder_num_heads (int): Number of attention heads within the decoder transformer blocks.
    - mlp_ratio (float): Ratio of the MLP dimension to the embedding dimension in the transformer blocks.
    - norm_layer (Type[nn.Module]): Type of normalization layer to use.
    - norm_pix_loss (bool): Whether to normalize pixel-wise loss.
    - using_residual_patch_embed (bool): Whether to use residual connections around the patch embedding.
    - apply_nonlinear_act_on_patch_embed (bool): Whether to apply a nonlinear activation on the patch embedding.
    - out_chans (int): Number of output channels.
    - using_register (bool): Whether to use register in the transformer.
    - num_registers (int): Number of registers, if using_register is True.
    - apply_minus_one_to_one_norm (bool): Whether to apply normalization to scale inputs to the range [-1, 1].
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 1024, depth: int = 24, num_heads: int = 16,
                 decoder_embed_dim: int = 512, decoder_depth: int = 8, decoder_num_heads: int = 16,
                 mlp_ratio: float = 4., norm_layer: Type[nn.Module] =nn.LayerNorm, norm_pix_loss: bool =False, using_residual_patch_embed: bool =False,
                 out_chans: int = 3, using_register: bool = False, num_registers: int = 0,
                 apply_minus_one_to_one_norm: bool = False 
                 ):
        super().__init__()
        if isinstance(img_size, int): 
            img_size = [img_size, img_size]
        self.img_size = img_size # H, W
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.using_residual_patch_embed = using_residual_patch_embed
        self.using_register = using_register
        self.num_registers = num_registers
        self.apply_minus_one_to_one_norm = apply_minus_one_to_one_norm
        if apply_minus_one_to_one_norm:
            self.unnormalize_fn = unnormalize_imgs
            self.normalize_fn = normalize_imgs
        else:
            self.unnormalize_fn = unnormalize_imgs_imagenet
            self.normalize_fn = normalize_imgs_imagenet
        

        num_patches = int((img_size[0] / patch_size) * img_size[1] / patch_size)
        self.patch_res = [int(img_size[0]/patch_size), int(img_size[1]/patch_size)]
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        patch_size = self.patch_embed.patch_size
        
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]

        self.num_patches = num_patches
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        if self.using_register:
            assert num_registers > 0
            self.register_token = nn.Parameter(torch.zeros(1,num_registers,embed_dim))
            self.register_pos_embed = nn.Parameter(torch.randn(1,num_registers,embed_dim)*1e-2)
            self.decoder_register_token = nn.Parameter(torch.zeros(1,num_registers,decoder_embed_dim))
            self.decoder_register_pos_embed = nn.Parameter(torch.randn(1,num_registers,decoder_embed_dim)*1e-2)


        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1] * out_chans, bias=True) # decoder to patch

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        """
        Initializes the weights of the Masked Autoencoder ViT model components.

        This method performs several initialization tasks:
        - Initializes positional embeddings for both the encoder and decoder using a sinusoidal pattern,
        optionally including a class token.
        - Initializes the patch embedding weights similarly to an nn.Linear layer, taking into account
        whether a nonlinear activation is applied on patch embedding.
        - Initializes class and mask tokens with a normal distribution.
        - Applies a uniform or normal initialization to all linear layers, layer normalization layers,
        and convolutional layers within the model based on their specific requirements.

        The initialization strategy is crucial for the model's convergence and overall performance.
        """

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_res, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_res, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm / nn.Conv2d
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initializes the weights of the given module based on its type.

        This method customizes weight initialization for linear layers and layer normalization
        layers as follows:
        - For `nn.Linear` layers, weights are initialized using Xavier uniform initialization.
        If the linear layer has a bias, it is initialized to zero.
        - For `nn.LayerNorm` layers, biases are initialized to zero.

        This approach to weight initialization follows the practices recommended in the official
        JAX implementation of the Vision Transformer.

        Parameters:
        - m (nn.Module): The module whose weights are to be initialized.
        """

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        Splits images into patches.

        This function takes a batch of images and splits each image into fixed-size patches.
        The images are expected to be in the format (N, C, H, W), where:
        - N is the batch size,
        - C is the number of channels (e.g., 3 for RGB),
        - H and W are the height and width of the images.

        The patches are flattened into vectors, and the function returns a tensor of these
        patch vectors with shape (N, L, patch_size**2 * C), where L is the total number of
        patches resulting from an image.

        Parameters:
        - imgs (Tensor): A tensor containing a batch of images with shape (N, C, H, W).

        Returns:
        - Tensor: A tensor containing the flattened patches with shape (N, L, patch_size[0]*patch_size[1]*C).
        """

        #p = self.patch_embed.patch_size[0]
        p1, p2 = self.patch_size[0], self.patch_size[1]
        assert imgs.shape[2] % p1 == 0 and imgs.shape[3] % p2 == 0

        h, w = imgs.shape[2] // p1, imgs.shape[3] // p2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p1, w, p2))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p1*p2*3))
        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Reconstructs images from their flattened patches.

        This function takes a batch of flattened patch vectors and reconstructs the original
        images. The patch vectors are expected to be in the format (N, L, patch_size**2 * C),
        where L is the total number of patches per image, and C is the number of channels.

        The function reconstructs images with their original height and width (H, W), resulting
        in a tensor of shape (N, C, H, W).

        Parameters:
        - x (Tensor): A tensor containing flattened patch vectors with shape (N, L, patch_size[0] * patch_size[1] * C).

        Returns:
        - Tensor: A tensor containing the reconstructed images with shape (N, C, H, W).
        """        


        #p = self.patch_embed.patch_size[0]
        p1, p2 = self.patch_size[0], self.patch_size[1]
        h, w = int(self.img_size[0] // p1), int(self.img_size[1] // p2)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p1, p2, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p1, w * p2))
        return imgs
     
    def random_masking(self, x: Tensor, masking_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies random masking to each sequence in the batch by shuffling based on random noise.

        This function randomly masks a portion of the elements in each sequence according to the
        specified masking ratio. It achieves per-sample shuffling by sorting based on random noise,
        keeping the first subset based on the masking ratio, and generating a binary mask indicating
        the kept and masked positions.

        Parameters:
        - x (Tensor): The input tensor containing sequences, with shape [N, L, D] where N is the
        batch size, L is the sequence length, and D is the dimensionality of each sequence element.
        - masking_ratio (float): The ratio of elements to mask in each sequence.

        Returns:
        - x_masked (Tensor): The tensor containing the masked sequences, with elements not selected
        for masking, having the same shape as the input tensor.
        - mask (Tensor): A binary mask indicating which elements were kept (0) and masked (1),
        with shape [N, L].
        - ids_restore (Tensor): The indices that can be used to restore the original ordering of
        elements, with shape [N, L].
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - masking_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder_only(self, x: Tensor, using_normalization: bool = True, apply_norm: bool = False) -> Tensor:
        """
        Processes the input through the MAE encoder without masking - used when we train NViST.
        using_normalization : whether we use the normalisation on the input - based on ImageNet statistics (original MAE) / -1 to 1
        apply_norm : whether we apply the layer normalisation on the output - usually False for NViST

        This method forwards the input tensor `x` through the encoder part of a Masked Autoencoder
        model. It supports optional normalization of the input before embedding and optionally applies
        a final normalization layer to the encoder output. It also includes support for class tokens
        and optionally register tokens if used.

        Parameters:
        - x (Tensor): The input tensor to the encoder, typically an image or batch of images,
        with shape [N, C, H, W] where N is the batch size, C is the number of channels,
        H is the height, and W is the width.
        - using_normalization (bool): If True, applies normalization to the input tensor before
        processing through the patch embedding layer.
        - apply_norm (bool): If True, applies a final normalization layer to the output of the
        encoder.

        Returns:
        - Tensor: The output tensor from the encoder.
        The shape of the output tensor depends on the encoder architecture and the presence
        of class and register tokens.
        """

        # run MAE encoder without masking - using the whole image
        if using_normalization:
            x = self.patch_embed(self.normalize_fn(x))
        else:
            x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        if self.using_register:
            register_tokens =self.register_token.expand(x.shape[0],-1,-1)
            register_tokens = register_tokens + self.register_pos_embed
            x = torch.cat([x, register_tokens], dim=1)

        for it, blk in enumerate(self.blocks):
            x = blk(x)
        
        if apply_norm: 
            return self.norm(x) 
        return x

    def forward_encoder(self, x: Tensor, masking_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Processes the input through the encoder,
        - positional embeddings, random masking, and Transformer blocks.

        This method applies positional embeddings, performs random masking
        based on the specified masking ratio, appends class tokens, optionally includes register
        tokens, and finally processes the data through Transformer blocks.

        Parameters:
        - x (Tensor): The input tensor with shape [N, C, H, W], where N is the batch size, C is the
        number of channels, H is the height, and W is the width.
        - masking_ratio (float): The ratio of patches to mask randomly.

        Returns:
        - Tuple[Tensor, Tensor, Tensor]: A tuple containing the encoded tensor with class and optional
        register tokens, the binary mask indicating which elements were kept (0) and masked (1),
        and the indices that can be used to restore the original ordering of elements. The shape of
        the encoded tensor is [N, L+1(+R), D], where L is the sequence length, R is the number of
        register tokens (if used), and D is the embedding dimension.
        """

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * masking_ratio
        x, mask, ids_restore = self.random_masking(x, masking_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        num_effective_tokens = x.shape[1]

        if self.using_register:
            register_tokens =self.register_token.expand(x.shape[0],-1,-1)
            register_tokens = register_tokens + self.register_pos_embed
            x = torch.cat([x, register_tokens], dim=1)
            
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.using_register:
            x = x[:,num_effective_tokens]

        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        """
        Processes the input through the decoder, handling masked and visible patches (tokens), 
        and inpainting masked patches.

        This method first embeds the input tokens, then conditionally handles register tokens if used. 
        It appends mask tokens to represent masked-out parts of the input, reorders the tokens to their 
        original positions before masking, applies positional embeddings, processes the data through 
        Transformer decoder blocks, and finally applies a prediction layer to project the token embeddings 
        back to the input space. Class tokens and optionally register tokens are removed before returning the output.

        Parameters:
        - x (Tensor): The input tensor to the decoder with shape [N, L+1(+R), D] where N is the batch size,
                    L is the sequence length, R is the number of register tokens (if used), and D is the 
                    embedding dimension.
        - ids_restore (Tensor): The indices that can be used to restore the original ordering of elements,
                                with shape [N, L].

        Returns:
        - Tensor: The output tensor from the decoder, representing the reconstructed input with the class 
                token and register tokens (if any) removed, having shape [N, L, D].
        """

        # embed tokens
        x = self.decoder_embed(x)
        if self.using_register and self.num_registers > 0:
            register_tokens = x[:, -self.num_registers:]
            n_visible_tokens = x.shape[1] - 1
        else:
            n_visible_tokens = x.shape[1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - n_visible_tokens, 1)
        if self.num_registers > 0:
            x_ = torch.cat([x[:, 1:-self.num_registers, :], mask_tokens], dim=1)  # no cls token
        else: 
            x_ = torch.cat([x[:,1:], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        if self.using_register:
            register_tokens = register_tokens + self.decoder_register_pos_embed
            x = torch.cat([x, register_tokens], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        if self.num_registers > 0:
            x = x[:, 1:-self.num_registers, :]
        else: 
            x = x[:, 1:]

        return x


    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """
        Calculates the reconstruction loss for masked patches in a batch of images.

        This method first transforms the input images into patch representations using the `patchify` method.
        It then computes the Mean Squared Error (MSE) between the predicted patches and the target patches
        derived from the input images. The loss is calculated only for the masked patches (where mask == 1),
        ignoring the unmasked (kept) patches to focus the model's learning on reconstructing the missing parts.

        Parameters:
        - imgs (Tensor): The original images with shape [N, C, H, W], where N is the batch size,
                        C is the number of channels, H is the height, and W is the width.
        - pred (Tensor): The predicted patch representations with shape [N, L, p*p*3], where L is the
                        total number of patches per image, and p is the patch size.
        - mask (Tensor): A binary mask indicating which patches were removed (1) and which were kept (0),
                        with shape [N, L]. The loss is computed only for the removed (masked) patches.

        Returns:
        - Tensor: The mean loss calculated over all masked patches in the batch.
        """

        # l2 loss
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    

    def forward_mae(self, imgs: Tensor, masking_ratio=0.75) -> Tensor:
        """
        Forward pass of the Masked Autoencoder (MAE) for image reconstruction.

        This method applies the entire MAE pipeline: it normalizes the input images, encodes them with
        a specified masking ratio, decodes the masked latent representations, and reconstructs the images
        by combining the decoded masked patches with the unmasked patches from the original images.

        Parameters:
        - imgs (Tensor): The input images with shape [N, C, H, W], where N is the batch size,
                        C is the number of channels, H is the height, and W is the width.
        - masking_ratio (float): The ratio of patches to mask randomly, defaulting to 0.75.

        Returns:
        - Tensor: The reconstructed images, having the same shape as the input images [N, C, H, W].
        """

        latent, mask, ids_restore = self.forward_encoder(self.normalize_fn(imgs), masking_ratio)
        pred = self.forward_decoder(latent, ids_restore) 
        pred_un = self.unnormalize_fn(self.unpatchify(pred))

        mask = mask.unsqueeze(-1).repeat(1,1,pred.shape[-1])
        mask = self.unpatchify(mask)
        output = imgs * (1-mask) + pred_un * mask
        return output 


    def forward(self, imgs: Tensor, masking_ratio=0.75) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the Masked Autoencoder model.

        This method encompasses the complete process of the Masked Autoencoder, including encoding
        the input images with a specified masking ratio, decoding the masked representations, and
        computing the reconstruction loss. 
        It is designed for training stage, returning the latent representations, the reconstruction loss, 
        the predicted patch representations, and the binary mask used during encoding.

        Parameters:
        - imgs (Tensor): Input images, normalized outside this method, with shape [N, C, H, W],
        where N is the batch size, C is the number of channels, H is the height, and W is the width.
        - masking_ratio (float): The ratio of patches to be masked randomly, with a default of 0.75.

        Returns:
        - Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
            - latent (Tensor): The encoded latent representations with shape [N, L, D], where L is the
            sequence length (number of patches + 1 for the class token) and D is the embedding dimension.
            - loss (Tensor): The reconstruction loss for the masked patches, a scalar Tensor.
            - pred (Tensor): The predicted patch representations with shape [N, L, p*p*3], where p is the patch size.
            - mask (Tensor): The binary mask indicating which patches were masked (1) and which were kept (0),
            with shape [N, L].
        """

        # Encode the input images and obtain latent representations, mask, and restoration indices
        latent, mask, ids_restore = self.forward_encoder(self.normalize_fn(imgs), masking_ratio)

        # Decode the latent representations to predict the masked patches
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        # Compute the reconstruction loss for the masked patches
        loss = self.forward_loss(self.normalize_fn(imgs), pred, mask)

        return latent, loss, pred, mask


def mae_vit_base(img_size: Tuple[int, int], patch_size: int, num_heads: int, using_register: bool = False, 
                 num_registers: int = 0, apply_minus_one_to_one_norm: bool = False, embed_dim: int = 768, 
                 decoder_num_heads: int = 16, **kwargs) -> MaskedAutoencoderViT:
    """
    Creates a base configuration for a Masked Autoencoder ViT (Vision Transformer) model.

    Parameters:
    - img_size (Tuple[int, int]): The size of the input images (height, width).
    - patch_size (int): The size of each image patch.
    - num_heads (int): The number of attention heads in the transformer encoder.
    - using_register (bool): If True, use register tokens in the transformer model.
    - num_registers (int): The number of register tokens to use if using_register is True.
    - apply_minus_one_to_one_norm (bool): If True, apply normalization to scale inputs to the range [-1, 1].
    - embed_dim (int): The dimensionality of the token embeddings.
    - decoder_num_heads (int): The number of attention heads in the transformer decoder.
    - **kwargs: Additional keyword arguments to be passed to the MaskedAutoencoderViT constructor.

    Returns:
    - MaskedAutoencoderViT: An instance of the MaskedAutoencoderViT model configured with the specified parameters.
    """

    model = MaskedAutoencoderViT(img_size=img_size,
        patch_size=patch_size, embed_dim=embed_dim, depth=12, num_heads=num_heads,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=decoder_num_heads,
        using_register=using_register, num_registers=num_registers,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        apply_minus_one_to_one_norm= apply_minus_one_to_one_norm, **kwargs)
    return model 

def mae_vit_base_patch16_dec512d8b(**kwargs) -> MaskedAutoencoderViT:
    """
    Creates a 'base' configuration of the Masked Autoencoder ViT with a specific decoder configuration.

    The model uses a patch size of 16, an embedding dimension of 768, a transformer depth of 12,
    and 12 attention heads. The decoder is configured with an embedding dimension of 512, a depth
    of 8 blocks, and 16 attention heads.

    Additional keyword arguments can be passed to further customize the model.

    Returns:
        MaskedAutoencoderViT: An instance of the MaskedAutoencoderViT model.
    """

    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs) -> MaskedAutoencoderViT:
    """
    Creates a 'large' configuration of the Masked Autoencoder ViT with a specific decoder configuration.

    This configuration uses a patch size of 16, an embedding dimension of 1024, a transformer depth
    of 24, and 16 attention heads. The decoder is configured with an embedding dimension of 512, a depth
    of 8 blocks, and 16 attention heads.

    Additional keyword arguments can be passed to further customize the model.

    Returns:
        MaskedAutoencoderViT: An instance of the MaskedAutoencoderViT model.
    """

    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b(**kwargs) -> MaskedAutoencoderViT:
    """
    Creates a 'huge' configuration of the Masked Autoencoder ViT with a specific decoder configuration.

    This setup uses a smaller patch size of 14, a larger embedding dimension of 1280, a transformer
    depth of 32, and 16 attention heads. The decoder maintains an embedding dimension of 512, a depth
    of 8 blocks, and 16 attention heads.

    Additional keyword arguments can be passed to further customize the model.

    Returns:
        MaskedAutoencoderViT: An instance of the MaskedAutoencoderViT model.
    """

    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

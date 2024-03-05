

from .blocks import Block
from .posemb import get_2d_sincos_pos_embed, get_pos_encoding
import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from utils import turn_int_into_list

# Some of the code here is adapted from from MAE and TensoRF:
    # https://github.com/facebookresearch/mae
    # https://github.com/apchenstu/TensoRF
# and then modified by the author.

class Decoder(nn.Module):
    """
    A decoder module for Vision Transformer architectures, designed for 3d representation
    The default 3d representation is 'vector_matrix'

    Parameters:
    - resolution (int): The resolution of the output feature map.
    - decoder_patch_size (int): The size of each patch in the decoder.
    - input_feature_dim (int): The dimensionality of the input feature vector.
    - decoder_embed_dim (int): The dimensionality of the token embeddings in the decoder.
    - decoder_depth (int): The number of transformer blocks in the decoder.
    - decoder_num_heads (int): The number of attention heads in each transformer block of the decoder.
    - decoder_output_dim (int): The dimensionality of the decoder output vector for each patch.
    - model (str): The model type for the decoder, supporting 'vector_matrix', 'triplanes', and 'CP' configurations.
    - mlp_ratio (float): Ratio of the MLP hidden dimension to the input dimension within each transformer block.
    - norm_layer_name (str): The name of the normalization layer to use within transformer blocks.
    - using_learnable_pos_embed (bool): If True, uses learnable positional embeddings.
    - using_encoded_visible_patches (bool): If True, uses encoded visible patches for cross-attention.
    - num_encoded_visible_patches (int): The number of encoded visible patches.
    - using_cross_attention (bool): If True, enables cross-attention mechanisms in the decoder.
    - kv_separate_norm (bool): If True, applies separate normalization to key and value vectors in the attention mechanism.
    - apply_decoder_embed (bool): If True, applies an embedding layer to match dimensions of visible encoded patches.
    - using_augmented_self_attention (bool): If True, augments the self-attention mechanism with additional features.
    - apply_cam_dist_condition (bool): If True, conditions the decoding process on camera distance.
    - apply_focal_condition (bool): If True, conditions the decoding process on camera focal length.
    - using_which_method_for_camera_parameters (str): Specifies the method for incorporating camera parameters ('adaln' or 'concat').
    - camera_params_pe_freq (int): The frequency of positional encoding for camera parameters.
    - use_adaln_zero (bool): If True, initializes AdaLN parameters to zero.
    - add_pe_on_y (bool): If True, adds positional embeddings on the y-axis for encoded visible patches.

    The forward pass of the decoder takes input feature vectors and optionally camera parameters, processes
    them through a series of transformer blocks, and outputs a 3d representation.

    """
    def __init__(self, resolution: int = 48, decoder_patch_size: int = 3, input_feature_dim: int = 768,
                decoder_embed_dim: int = 768, decoder_depth: int = 12, decoder_num_heads: int = 12, decoder_output_dim: int = 32,
                model: str = 'vector_matrix', mlp_ratio: float = 4., norm_layer_name: str ='LayerNorm', using_learnable_pos_embed: bool = False,  
                using_encoded_visible_patches: bool = True, num_encoded_visible_patches: int =574, using_cross_attention: bool = True, 
                kv_separate_norm: bool = True, apply_decoder_embed: bool = False, using_augmented_self_attention: bool = False,
                apply_cam_dist_condition: bool = True, apply_focal_condition: bool = True, using_which_method_for_camera_parameters: str = "adaln",
                camera_params_pe_freq: int =4, use_adaln_zero: bool =False, add_pe_on_y: bool = True,
                ):

        super().__init__()

        if isinstance(resolution, int): 
            resolution = turn_int_into_list(resolution)
        if isinstance(decoder_patch_size, int): 
            patch_size = turn_int_into_list(decoder_patch_size)
        else: 
            patch_size = decoder_patch_size
        
        assert model in ['vector_matrix', 'triplanes', 'CP']
        self.model = model

        if using_encoded_visible_patches: # cross attention
            assert num_encoded_visible_patches > 0

        # apply camera_params_condition
        assert using_which_method_for_camera_parameters in ['adaln', 'concat']
        if apply_cam_dist_condition:
            camera_input_dimensions = int(apply_cam_dist_condition) + int(apply_focal_condition) 
            input_dim = camera_input_dimensions + 2 * camera_input_dimensions * camera_params_pe_freq 
            self.camera_params_embed = nn.Sequential(nn.Linear(input_dim, decoder_embed_dim), nn.ReLU(), nn.Linear(decoder_embed_dim, decoder_embed_dim))
            if using_which_method_for_camera_parameters == "adaln": 
                self.using_adaln = True
            elif using_which_method_for_camera_parameters == "concat":
                 self.using_adaln = False
        else:
            self.using_adaln = False
            
        # three matrices
        if self.model in ['vector_matrix', 'triplanes']:
            self.num_patches_hw = [3 * resolution[0] // patch_size[0], resolution[1] //patch_size[1]]
            self.num_patches = self.num_patches_hw[0] * self.num_patches_hw[1]
            if self.model in ['vector_matrix'] : 
                self.num_patches += (self.num_patches_hw[0]) # three vectors
        elif self.model == "CP":
            self.num_patches_hw = [3*resolution[0] // patch_size[0], 1]
            self.num_patches = self.num_patches_hw[0] * self.num_patches_hw[1]

        # positional embeddings - first one cls
        if using_learnable_pos_embed: 
            self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches+1, decoder_embed_dim) * 1e-2, requires_grad=True)
        else: 
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False) # will be initialized

        if using_encoded_visible_patches:
            if apply_decoder_embed: # match dimensions of visible encoded patches
                self.encoded_patches_embed = nn.Linear(input_feature_dim, decoder_embed_dim, bias=True)
            if add_pe_on_y:
                self.encoded_patches_pos_embed = nn.Parameter(torch.randn(1, num_encoded_visible_patches, decoder_embed_dim) * 1e-2, 
                                                          requires_grad=True)
        # normalization layer
        norm_layer = getattr(torch.nn, norm_layer_name)

        # output_token
        self.output_token = nn.Parameter(torch.zeros(1,1,decoder_embed_dim), requires_grad =True)

        # decoding block : but i modify the name so that i can easily take the ViT pretrained weights
        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, kv_separate_norm=kv_separate_norm, qkv_bias=True, norm_layer=norm_layer, use_adaln = self.using_adaln)
            for i in range(decoder_depth)]) 


        # apply the layer normalization on decoder blocks
        self.decoder_out_norm = nn.Sequential(norm_layer(decoder_embed_dim, eps=1e-6), nn.GELU())
        if self.model in ["vector_matrix"]:
            self.decoder_vector_norm = nn.Sequential(norm_layer(decoder_embed_dim, eps=1e-6), nn.GELU())
            self.decoder_vector_pred = nn.ModuleList(nn.Linear(decoder_embed_dim, patch_size[0] * 1 * decoder_output_dim) for _ in range(3))
        self.decoder_pred = nn.ModuleList(nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * decoder_output_dim) for _ in range(3))


        self.patch_size = patch_size

        self.using_learnable_pos_embed = using_learnable_pos_embed
        self.using_encoded_visible_patches = using_encoded_visible_patches

        self.using_cross_attention = using_cross_attention # it is for triplane generator

        self.apply_decoder_embed = apply_decoder_embed
        
        self.using_augmented_self_attention = using_augmented_self_attention
        self.apply_cam_dist_condition = apply_cam_dist_condition
        self.output_dim = decoder_output_dim

        self.camera_params_pe_freq = camera_params_pe_freq
        self.using_which_method_for_camera_parameters = using_which_method_for_camera_parameters
        self.using_adaln_zero = use_adaln_zero
        self.add_pe_on_y = add_pe_on_y
        self.initialize_weights()   


    def initialize_weights(self) -> None:
        """
        Initializes the weights of the Decoder module.

        """
        if not self.using_learnable_pos_embed: 
            if self.model in ["vector_matrix"]:
                decoder_pos_embed =get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], torch.tensor([self.num_patches_hw[0], self.num_patches_hw[1]+1]),
                                                           cls_token=True)
            else:
                decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches_hw, cls_token=True)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))            

        torch.nn.init.normal_(self.output_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """
        Initializes weights for the given module `m` according to specific rules.

        For `nn.Linear` layers, it applies Xavier uniform initialization to the weights and sets biases to 0.
        For `nn.LayerNorm` layers with `elementwise_affine` enabled, it sets biases to 0 and weights to 1.
        If `using_adaln_zero` is True, it initializes the last layer weights and biases of adaptive layer
        normalization (adaLN) modules in all blocks to 0.

        Parameters:
        - m (nn.Module): The module to be weight-initialized. Supported modules are `nn.Linear` and `nn.LayerNorm`.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # AdaLN-zero
        if self.using_adaln_zero:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Reconstructs the matrices from flattened patch representations.

        This method reshapes and reorders the encoded patch representations back into
        the original spatial arrangement of the matrices.

        Here, we do not have 'patchify' as 'patch_embed' will do the same operation.
        

        Parameters:
        - x (Tensor): The flattened patch representations with shape 
                    (N, L, patch_size[0] * patch_size[1] * output_dim), where
                    N is the batch size, L is the total number of patches, and
                    output_dim is the dimensionality of the output channel.

        Returns:
        - Tensor: The reconstructed matrices with shape (N, output_dim, H, W),
                where H and W are the height and width of the reconstructed image.
        """
        p1, p2 = self.patch_size
        # h = w = int(x.shape[1]**.5)
        hh, ww = self.num_patches_hw
        assert hh * ww == x.shape[1]
        final_dim = self.output_dim 
        x = x.reshape(shape=(x.shape[0], hh, ww, p1, p2, final_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        matrices = x.reshape(shape=(x.shape[0], final_dim, hh * p1, ww * p2))
        return matrices  

    def unpatchify_vector(self, vector: Tensor) -> Tensor:
        """
        Reconstructs the vectors from flattened vector representations.
        We use (hh, 1) for vector representations.

        Parameters:
        - vector (Tensor): The flattened vector representations with shape 
                            (N, hh, patch_size[0] * output_dim), where
                            N is the batch size, hh is the height dimension in terms of patches,
                            and output_dim is the dimensionality of the output channel.

        Returns:
        - Tensor: The reconstructed image vectors with shape (N, output_dim, hh * patch_size[0], 1),
                effectively representing a vertical strip or column of the image.
        """
        p1, p2 = self.patch_size
        hh, ww = self.num_patches_hw
        final_dim = self.output_dim 
        x = vector.reshape(shape=(vector.shape[0],hh,1,p1,1,final_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        vectors = x.reshape(shape=(x.shape[0],final_dim,hh*p1,1))
        return vectors


    def forward(self, latent: Tensor, camera_params: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Processes the latent representations through the decoder to reconstruct the grid representation.

        Parameters:
        - latent (Tensor): The latent representations with shape [N, L, D], where N is the batch size,
        L is the total number of image patches (plus one for the class token in some configurations), and D
        is the dimensionality of each representation.
        - camera_params (Optional[Tensor]): Optional tensor containing camera parameters with shape
        [N, num_camera_params], used for conditional reconstruction based on camera settings.

        Returns:
        - Tuple[Tensor, Optional[Tensor]]: A tuple containing the reconstructed image matrix with shape
        [N, C, H, W] (where C is the output channel dimension, H and W are height and width of the matrices),
        and an optional tensor for vector representations if the model is configured as 'vector_matrix'.
        The vector tensor, if returned, will have shape of [N, C, H, 1]

        """



        if self.using_encoded_visible_patches: # we divide cls / image features
            x = latent[:, :1]
            y = latent[:, 1:]
            if self.apply_decoder_embed:
                y = self.encoded_patches_embed(y)
            if self.add_pe_on_y:
                y = y + self.encoded_patches_pos_embed # add pe on image features
        else:
            x = latent
        

        if len(x.shape) == 2: 
            x = x.unsqueeze(1)

        if self.apply_decoder_embed:
            x = self.decoder_embed(x)

        
        tokens = self.output_token.repeat(x.shape[0], self.num_patches,1)
        x = torch.cat([x, tokens], dim=1) # we concat cls and masked tokens
        x = x + self.decoder_pos_embed

        if self.using_encoded_visible_patches and not self.using_cross_attention:
            x = torch.cat([x, y], dim=1)
            y = None
        elif self.using_encoded_visible_patches and self.using_cross_attention:
            x, y = x, y

        if self.apply_cam_dist_condition: 
            # camera_params include cam_dist, normalized_focal, scale
            assert camera_params is not None
            if self.camera_params_pe_freq > 0 : 
                camera_params = get_pos_encoding(camera_params, self.camera_params_pe_freq)
            
            camera_params = camera_params.unsqueeze(1)
            camera_params = self.camera_params_embed(camera_params) 
            if self.using_which_method_for_camera_parameters == "concat":
                y, c = torch.cat([y, camera_params], dim=1), None
            elif self.using_which_method_for_camera_parameters == "adaln":
                y, c = y, camera_params[:,0]
        else:
            c = None


        # self.num_patches_hw : h, w for 'matrix' 
        # self.num_patches : total number of patches ; including 'vectors' in vector-matrix case
        number_of_patches_for_matrix = self.num_patches_hw[0] * self.num_patches_hw[1]

        for it, blk in enumerate(self.blocks):
            if not self.using_cross_attention: 
                x = blk(x, c=c) # use self.attention for all
            elif self.using_cross_attention and self.using_encoded_visible_patches:
                if it % 2 == 1: # self-attention
                    if self.using_augmented_self_attention:
                        x = blk(x, k=torch.cat([x,y], dim=1), v=torch.cat([x,y], dim=1), c=c)
                    else:
                        x = blk(x,c=c)
                else: # cross-attention
                    x = blk(x, k=y, v=y, c=c) 

        matrix = self.decoder_out_norm(x[:,1:(number_of_patches_for_matrix+1)])
        num_matrix_patches = matrix.shape[-2] // 3
        matrix_intermediates =[self.decoder_pred[j](matrix[:,j*num_matrix_patches:(j+1)*num_matrix_patches]) for j in range(3)]
        matrix = torch.cat(matrix_intermediates, dim=1)
        matrix = self.unpatchify(matrix)
        H = matrix.shape[-2]
        matrix = matrix.split([H//3, H//3, H//3], dim=-2)
        matrix = torch.cat([m.unsqueeze(0) for m in matrix])


        if self.model in ['vector_matrix']:
            vector = self.decoder_vector_norm(x[:,(number_of_patches_for_matrix+1):(self.num_patches+1)])
            num_vector_patches = vector.shape[-2] // 3 # per plane
            vector_intermediates =[self.decoder_vector_pred[j](vector[:,j*num_vector_patches:(j+1)*num_vector_patches]) for j in range(3)]
            vector = torch.cat(vector_intermediates, dim=1)
            vector = self.unpatchify_vector(vector)
            vector = vector.split([H//3, H//3, H//3], dim=-2)
            vector = torch.cat([v.unsqueeze(0) for v in vector])
            return matrix, vector

        return matrix, None

# if __name__ == "__main__":
#     dec = Decoder(apply_cam_dist_condition=False)
#     out = dec(torch.randn(1,575,768))
#     import pdb; pdb.set_trace()
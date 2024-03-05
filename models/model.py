

from .mae_original_model import mae_vit_base, mae_vit_base_patch16, mae_vit_large_patch16
from .decoder import Decoder
from .renderer import Renderer

from utils import load_pretrained_when_shape_matches, load_pretrained_weights

from torch import nn, Tensor
import torch
import os
from easydict import EasyDict as edict
import inspect
from typing import List

class Model(nn.Module):
    """
    A Model class that encapsulates the encoder, decoder, and renderer components for a novel view synthesis task.

    Attributes:
        args (EasyDict): Configuration arguments for the model components.
        start_iteration (int): Starting iteration number for training, useful for resuming training.
        encoder (nn.Module): The encoder part of the model, responsible for feature extraction.
        decoder (nn.Module): The decoder part, responsible for generating matrices and vectors from latent vectors.
        renderer (nn.Module): The renderer component, which takes ray origins and directions along with matrices and vectors to render the final output.
    """

    def __init__(self, args: dict, aabb:Tensor, near_far:List[float]):
        """
        Initializes the Model with encoder, decoder, and renderer components.

        Args:
            args (dict): Configuration arguments passed to the model components.
            aabb (torch.Tensor): Axis-aligned bounding box for the scene, used in the renderer.
            near_far (List[float]): Near and far clipping distances for rays, used in the renderer.
        """

        super().__init__()
        self.args = edict(vars(args))
        self.start_iteration = 0
        self.define_encoder()
        self.define_decoder()
        self.define_renderer(aabb, near_far)

    def define_encoder(self):
        """
        Initializes the encoder component based on the configuration arguments.
        We initialize the encoder with the fine-tuned or pre-trained weights.
        """
        args = self.args
        self.encoder = mae_vit_base(img_size=args.img_size, patch_size=args.encoder_patch_size, 
                num_heads=args.encoder_num_heads, apply_minus_one_to_one_norm=args.apply_minus_one_to_one_norm,
                embed_dim = args.encoder_embed_dim
        )

        if args.using_mae_finetuned and not args.ckpt:
            print('we are training using the fine-tuned MAE')
            ckpt = torch.load(os.path.join(args.finetuned_mae_dir, 'model.pth'), map_location=torch.device('cpu'))
            self.encoder = load_pretrained_weights(ckpt, self.encoder)
            print('Loading finetuned MAE weights from ' + str(args.finetuned_mae_dir))

        elif args.using_mae_pretrained and not args.ckpt:
            print("we are training using the pre-trained MAE")
            if args.encoder_embed_dim == 1024:
                pretrained = torch.load('pretrained/mae_visualize_vit_large.pth', map_location = torch.device('cpu')) 
                encoder_p16 = mae_vit_large_patch16()  
            elif args.encoder_embed_dim == 768:
                pretrained = torch.load('pretrained/mae_visualize_vit_base.pth', map_location = torch.device('cpu'))
                encoder_p16 = mae_vit_base_patch16()
            else:
                ValueError("embed_dim does not match any known configuration")
            encoder_p16.load_state_dict(pretrained['model'])
            self.encoder = load_pretrained_when_shape_matches(self.encoder, encoder_p16)
            self.start_iteration = 0

        self.delete_mae_inpainting_decoder()

    def define_decoder(self):
        """
        Initializes the decoder component based on the configuration arguments.
        We can initialize the decoder with pretrained weights.
        """
        args = self.args
        decoder_init_args = inspect.signature(Decoder.__init__).parameters
        filtered_args = {k: v for k, v in args.items() if k in decoder_init_args}
        self.decoder = Decoder(**filtered_args)
        if args.initialize_decoder_with_pretrained:
            pretrained = torch.load('pretrained/mae_visualize_vit_base.pth', map_location = torch.device('cpu'))
            encoder_p16 = mae_vit_base_patch16()
            encoder_p16.load_state_dict(pretrained['model'])
            self.decoder = load_pretrained_when_shape_matches(self.decoder, encoder_p16)
            print('initalize the decoder with pretrained MAE for matrices that match shapes')

    def define_renderer(self, aabb, near_far):
        """Initializes the renderer component based on the configuration arguments."""
        args = self.args
        renderer_init_args = inspect.signature(Renderer.__init__).parameters
        filtered_args = {k: v for k, v in args.items() if k in renderer_init_args}
        filtered_args['aabb'] = aabb
        filtered_args['near_far'] = near_far
        self.renderer = Renderer(**filtered_args)
    
    def get_start_iteration(self):
        return self.start_iteration
    
    def delete_mae_inpainting_decoder(self):
        """Deletes MAE's inpainting decoder components from the encoder if they exist."""

        layers_to_delete = ['decoder_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred']
        for layer in layers_to_delete:
            if hasattr(self.encoder, layer): 
                delattr(self.encoder, layer)
            else: 
                print(f"Layer {layer} not found in encoder")

    def forward(self, input_images, camera_params, ro_train, rd_train):
        """
        Forwards input through the model to generate output images.

        Args:
            input_images (torch.Tensor): Input images for the encoder.
            camera_params (torch.Tensor): Camera parameters for conditional rendering.
            ro_train (torch.Tensor): Ray origins for training samples.
            rd_train (torch.Tensor): Ray directions for training samples.

        Returns:
            dict: Rendered output images and potentially additional data like depth maps.
        """

        latent_vectors =self.encoder.forward_encoder_only(input_images)
        matrices, vectors = self.decoder(latent_vectors, camera_params)
        output = self.renderer(ro_train, rd_train, matrices, vectors)
        return output




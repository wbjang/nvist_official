

import configargparse
from typing import List, Optional, Any  # Assuming you add type hints
from utils import str2bool

def config_parser(cmd: Optional[List[str]] = None) -> Any:
    arg_parser = configargparse.ArgumentParser()
    arg_parser.add_argument('--config', is_config_file=True, help='config file path')
    
    arg_parser.add_argument("--use_ema", default=True, type=str2bool)
    arg_parser.add_argument("--ema_beta", default=0.995, type=float)


    # batchsize
    arg_parser.add_argument("--batch_size", default=4, type=int)
    arg_parser.add_argument(
        "--batch_pixel_size", default=60000, type=int, 
        help=("total number of pixels we sample - across all 'B' images")
    )
    arg_parser.add_argument("--split_batches", default=True, type=str2bool)

    # visualize
    arg_parser.add_argument("--n_vis_scenes", default=5, type=int, help='how many scenes you are going to visualize')
    arg_parser.add_argument("--n_vis_images", default=5, type=int, help='how many images you visualize for each scene')


    # dataset
    arg_parser.add_argument("--dataset_name", type=str, default="mvimgnet")
    arg_parser.add_argument("--data_dir", default='../../data/mvimgnet', type=str)
    arg_parser.add_argument("--img_size", type=int, nargs='+', default=[160,90])
    arg_parser.add_argument("--number_of_imgs_from_the_same_scene", type=int, default=5)

    # loss_weights
    arg_parser.add_argument("--weight_lpips", default=0.1, type=float)
    arg_parser.add_argument("--weight_dist", default=0.001, type=float)

    # saving directories
    arg_parser.add_argument("--base_dir", default="../../output/mae_finetuned")
    arg_parser.add_argument("--expname", default="mvimgnet", type=str)

    # number of iters / checkpoints / fid / visualization
    arg_parser.add_argument("--n_iters", default=100000, type=int)
    arg_parser.add_argument("--ckpt", default=False, type=str2bool)
    arg_parser.add_argument("--vis_every", default=10000, type=int)
    arg_parser.add_argument("--render_test_all", default=150000, type=int)

    # ckpt for eval
    arg_parser.add_argument("--ckpt_base", default='../../output/bask_nvist/', type=str)
    arg_parser.add_argument("--ckpt_dir", default="0125/0125_normal_2gpus", type=str)

    # learning rates
    arg_parser.add_argument("--lr_encoder_init", default=0.0001, type=float)
    arg_parser.add_argument("--lr_decoder_init", default=0.0001, type=float)
    arg_parser.add_argument("--lr_renderer_init", default=0.0001, type=float)
    arg_parser.add_argument("--lr_minimum", default=0.0, type=float)
    arg_parser.add_argument("--encoder_warmup_iters", default=0, type=int)
    arg_parser.add_argument("--decoder_warmup_iters", default=0, type=int)
    arg_parser.add_argument("--renderer_warmup_iters", default=0, type=int)


    # MAE parameters - encoder
    arg_parser.add_argument("--encoder_patch_size", default=8, type=int)
    arg_parser.add_argument("--encoder_depth", default=12, type=int)
    arg_parser.add_argument("--apply_minus_one_to_one_norm", default=False, type=str2bool, help='if it is False, it uses imagenet norm')
    arg_parser.add_argument("--encoder_num_heads", default=12, type=int)
    arg_parser.add_argument("--encoder_embed_dim", default=768, type=int)

    # MAE parameters - mae decoder (inpainting)
    arg_parser.add_argument("--mae_decoder_embed_dim", default=512, type=int)
    arg_parser.add_argument("--mae_decoder_num_heads", default=16, type=int)
    arg_parser.add_argument("--mae_decoder_depth", default=8, type=int)
    arg_parser.add_argument("--masking_ratio", default=0.75, type=float)

    # MAE parameters - initialization: load from the pre-trained one or finetuned MAE
    arg_parser.add_argument("--using_mae_finetuned", default=True, type=str2bool)
    arg_parser.add_argument("--finetuned_mae_dir", default="../../output/mae_finetuned/mae_mvimgnet_imgnet", type=str)
    arg_parser.add_argument("--using_mae_pretrained", default=True, type=str2bool,
                            help="it only matters when using_mae_finetuned=False")
    arg_parser.add_argument("--initialize_decoder_with_pretrained", default=True, type=str2bool)

    # Decoder
    arg_parser.add_argument("--resolution", default=48, type=int)
    arg_parser.add_argument("--decoder_patch_size", default=3, type=int)
    arg_parser.add_argument("--input_feature_dim", default=768, type=int)
    arg_parser.add_argument("--decoder_embed_dim", default=768, type=int)
    arg_parser.add_argument("--decoder_depth", default=12, type=int)
    arg_parser.add_argument("--decoder_num_heads", default=12, type=int)
    arg_parser.add_argument("--decoder_output_dim", default=32, type=int)
    arg_parser.add_argument("--decoder_model", default="vector_matrix", type=str)
    arg_parser.add_argument("--using_learnable_pos_embed", default=False, type=str2bool)
    arg_parser.add_argument("--using_encoded_visible_patches", default=True, type=str2bool)
    arg_parser.add_argument("--num_encoded_visible_patches", default=574, type=int)
    arg_parser.add_argument("--using_cross_attention", default=True, type=str2bool, 
                            help="if False, we concatenate then do self-attention")
    arg_parser.add_argument("--kv_separate_norm", default=True, type=str2bool,
                            help="whether we use the separate Norm for k/v in attention blocks")
    arg_parser.add_argument("--apply_decoder_embed", default=False, type=str2bool,
                            help="whether we use additional MLP to match dimensions")
    arg_parser.add_argument("--apply_cam_dist_condition", default=True, type=str2bool,
                            help="whether we condition on camera distance")
    arg_parser.add_argument("--apply_focal_condition", default=True, type=str2bool,
                            help="whether we also condition on focal when apply_cam_dist_condition==True")
    arg_parser.add_argument("--using_which_method_for_camera_parameters", default="adaln", type=str,
                            help="we can sue adaln or concat")
    arg_parser.add_argument("--camera_params_pe_freq", default=4, type=int)
    arg_parser.add_argument("--use_adaln_zero", default=False, type=str2bool,
                            help="whether we initialize adaln as zero")
    arg_parser.add_argument("--add_pe_on_y", default=True, type=str2bool, 
                            help="whether we add positional embedding on encoded visible patches")

    # Renderer
    arg_parser.add_argument("--sigma_dim", default=8, type=int)
    arg_parser.add_argument("--rgb_dim", default=24, type=int)
    arg_parser.add_argument("--rgb_feature_dim", default=64, type=int)
    arg_parser.add_argument("--sigma_activation", default="ReLU", type=str)
    arg_parser.add_argument("--concat_viewdir", default=True, type=str2bool)
    arg_parser.add_argument("--viewdir_pe", default=4, type=int)
    arg_parser.add_argument("--using_sigmoid_at_rgb", default=True, type=str2bool)
    arg_parser.add_argument("--apply_distloss", default=True, type=str2bool)
    arg_parser.add_argument("--using_sigma_mlp", default=False, type=str2bool,
                            help="when decoder_model=triplane, then it is always True")
    arg_parser.add_argument("--padding_zero", default=True, type=str2bool)
    arg_parser.add_argument("--nSamples", default=48, type=int)
    


    
    if cmd is not None: 
        return arg_parser.parse_args(cmd)
    else: 
        return arg_parser.parse_args()
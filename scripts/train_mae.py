
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import json
import numpy as np
import imageio
from tqdm.auto import tqdm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dataLoader import dataset_dict
import lpips
from accelerate import Accelerator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from models.mae_original_model import mae_vit_base, mae_vit_base_patch16, mae_vit_large_patch16
from opt import config_parser
from utils import adjust_learning_rate, cycle, visualize_mae
from utils import load_pretrained_when_shape_matches, put_gt_output_side_by_side





def main():
    """
    Executes the main training and evaluation loop for a MAE-based model.

    This script performs several key operations:
    - Parses command-line arguments for configuration settings.
    - Initializes datasets for training and testing.
    - Configures the Accelerator for potentially utilizing mixed precision and multi-GPU setups.
    - Prepares the model, data loaders, and optimizers for training.
    - Loads pretrained weights if specified, or initializes the model for training from scratch.
    - Enters the main training loop, periodically evaluating the model on the test dataset.
    - Visualizes and saves generated images and metrics during training and testing.
    - Outputs training progress to the console and logs to TensorBoard.

    """

    args = config_parser()

    print('The batchsize is '+ str(args.batch_size))
    if args.split_batches:
        print('We split the above batch into ' + str(torch.cuda.device_count()) + ' gpus')
    else:
        print('each gpu will process ' + str(torch.cuda.device.count()))
    print('We are going to use this GPU - ' + torch.cuda.get_device_name(0))

    accelerator = Accelerator(split_batches = args.split_batches)
    print('whether accelerator use fp16 - ' + str(accelerator.state.use_fp16))


    dataset = dataset_dict[args.dataset_name]

    train_dataset = dataset(args.data_dir, "train")
    test_dataset = dataset(args.data_dir, "test")

    output_dir = os.path.join(args.base_dir, args.expname)
    os.makedirs(os.path.join(output_dir, 'imgs_vis_train'), exist_ok = True)
    os.makedirs(os.path.join(output_dir, 'imgs_vis_test'), exist_ok = True)
    os.makedirs(os.path.join(output_dir, 'imgs_train'), exist_ok = True)
    os.makedirs(os.path.join(output_dir, 'imgs_test'), exist_ok = True)
    summary_writer = SummaryWriter(output_dir)

    args_dict = vars(args)
    with open(os.path.join(output_dir, 'hpams.txt'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=4, shuffle=True)

    encoder = mae_vit_base(img_size=args.img_size, patch_size=args.encoder_patch_size, 
                           num_heads=args.encoder_num_heads, apply_minus_one_to_one_norm=args.apply_minus_one_to_one_norm,
                           embed_dim = args.encoder_embed_dim
                           )
    
    # when we resume our training from the checkpoint
    if args.ckpt:
        if args.ckpt_path is None:
            print('we resume training')
            ckpt = torch.load(os.path.join(output_dir,'model.th'), map_location=torch.device('cpu'))
        
        else:
            print('we will save at ' + str(output_dir))
            print('we will use the ckpt saved at ' + str(args.ckpt_path))
            ckpt = torch.load(os.path.join(args.ckpt_path,'model.th'), map_location = torch.device('cpu'))

        encoder.load_state_dict(ckpt['state_dict'], strict=True)
        start_iteration = ckpt['iteration']
        print('Loading pretrained MAE only - starting from ' + str(start_iteration))
    
    elif args.using_mae_pretrained:
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
        encoder = load_pretrained_when_shape_matches(encoder, encoder_p16)
        start_iteration = 0
    else: 
        start_iteration = 0

    encoder_grad_vars =  [{'params':encoder.parameters(), 'lr': args.lr_encoder_init}]
    encoder_opt = torch.optim.AdamW(encoder_grad_vars)

    train_loader, encoder, encoder_opt = accelerator.prepare(train_loader, encoder, encoder_opt)
    dl = cycle(train_loader)

    pbar = tqdm(range(start_iteration, args.n_iters, 1), miniters = 10)

    lr_encoder_original = args.lr_encoder_init
    lr_min = args.lr_minimum
    warmup_iters = args.encoder_warmup_iters
    total_iters = args.n_iters

    pbar = tqdm(range(start_iteration, total_iters, 1), miniters = 10)

    loss_fn_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)

    psnr_train, psnr_test, lpips_train, lpips_test = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

    # evaluate_mae -> can we change it into a prettier function?

    for iteration in pbar:
        samples = next(dl)

        train_images = samples['images']

        train_images = train_images.permute(0,3,1,2).float()  # NHWC -> NCHW

        if iteration == 0:
            print('train_images_shape : ' + (str(train_images.shape)))

        with accelerator.autocast():
            latent_vectors, loss, pred_mask, mask = encoder(train_images)
            
        accelerator.backward(loss)

        accelerator.wait_for_everyone()
        accelerator.clip_grad_norm_(encoder.parameters(), 1.)
        
        encoder_opt.step()
        encoder_opt.zero_grad()
        accelerator.wait_for_everyone()


        summary_writer.add_scalar('train/loss_l2', loss.item(), global_step=iteration)

        adjust_learning_rate(iteration, warmup_iters, total_iters, lr_encoder_original, lr_min, encoder_opt)


        prtx = f'{iteration:05d}'

        if iteration % 10 == 0:
            memory_cached = torch.cuda.memory_reserved()
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' l2_train={float(loss.item()):.2f}'
                + f' lpips_train={float(lpips_train.item()):.2f}'
                + f' psnr_train={float(psnr_train.item()):.2f}'
                + f' psnr_test={float(psnr_test.item()):.2f}'
                + f' lpips_test={float(lpips_test):.2f}'
                + f' GPU: {memory_cached / (1024 ** 3):.1f} GB'
            )

        if (iteration % 1000 == 0 and iteration <= 5000) or iteration % args.vis_every == 0:
            # visualize images - train
            # train_grid = visualize_images(train_images)
            images_filled = visualize_mae(encoder, pred_mask, mask, train_images)
            with torch.no_grad():
                l2_train = torch.mean((images_filled - train_images) ** 2)
                psnr_train = -10 * np.log(l2_train.item()) / np.log(10)
                lpips_train = loss_fn_lpips(train_images, images_filled).mean()

            train_vis = put_gt_output_side_by_side(train_images, images_filled)
            imageio.imwrite(os.path.join(output_dir, 'imgs_vis_train', prtx + '.png'), train_vis)

            # visualize images - test
            jump_idx = (len(test_dataset.num_objs) - 1) // (args.n_vis_scenes - 1)
            loss_test_l2, loss_test_lpips = [], []
            # for each scene
            for scene_idx in range(0, len(test_dataset.num_objs), jump_idx):
                start_idx = test_dataset.starting_idxs[scene_idx]
                end_idx = test_dataset.ending_idxs[scene_idx]
                jump_file_idx = (end_idx - start_idx - 1) // (args.n_vis_images - 1)
                file_idxs = [idx for idx in range(start_idx, end_idx, jump_file_idx)]
                test_images = test_dataset.load_images(file_idxs, scene_idx)
                test_images = torch.from_numpy(test_images).permute(0,3,1,2).float() # nchw
                
                with torch.no_grad():
                    _, loss, pred_mask, mask = encoder(test_images.to(accelerator.device))
                
                test_filled = visualize_mae(encoder, pred_mask, mask, test_images)
                test_vis = put_gt_output_side_by_side(test_images, test_filled)
                scene_prtx = f'{scene_idx:02d}'
                imageio.imwrite(os.path.join(output_dir, 'imgs_vis_test', prtx + '_' + scene_prtx + '.png'), test_vis)
                
                with torch.no_grad():
                    l2_test = torch.mean((test_filled - test_images.to(accelerator.device)) ** 2)
                    psnr_test = -10 * np.log(l2_test.item()) / np.log(10)
                    lpips_test = loss_fn_lpips(test_filled, test_images.to(accelerator.device)).mean()
                
                loss_test_l2.append(psnr_test.item())
                loss_test_lpips.append(lpips_test.item())
            loss_test_l2, loss_test_lpips = np.mean(loss_test_l2).item(), np.mean(loss_test_lpips).item()

            summary_writer.add_scalar('test/psnr', loss_test_l2, global_step=iteration)
            summary_writer.add_scalar('test/lpips', loss_test_lpips, global_step=iteration)

            # saving
            ckpt = {'state_dict':encoder.state_dict(), 'iteration':iteration} # chech duplicates
            torch.save(ckpt, os.path.join(output_dir, 'model.pth'))


    ckpt = {'state_dict':encoder.state_dict(), 'iteration':iteration}
    torch.save(ckpt, os.path.join(output_dir, 'model.pth'))

        
if __name__ == "__main__":
    main()
 
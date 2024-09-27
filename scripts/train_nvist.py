
import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
import imageio
import json
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import lpips

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dataLoader import dataset_dict
from models.model import Model
from opt import config_parser
from utils import adjust_learning_rate, visualize_images, cycle, SimpleSampler
from utils import visualize_input_novel_view_gt, visualize_depth, visualize_depths
from utils import load_pretrained_weights

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def main():
    """
    Main function for training and evaluating NViST.

    This function executes the following steps:
    1. Parses arguments for model configuration and training settings.
    2. Initializes the distributed data parallel settings and accelerator for efficient training.
    3. Loads the training and testing datasets, including specific configurations such as the number of images from the same scene.
    4. Prepares the output directory for saving visualizations, checkpoints, and training logs.
    5. Initializes the model, optimizers, and loads a checkpoint if available. Otherwise, it initializes the model with pretrained weights or from scratch.
    6. Sets up the training loop, including data loading, model training, loss calculation, and gradient updates.
    7. Performs periodic evaluations on the test dataset to compute metrics such as PSNR and LPIPS, and generates visualizations of the model outputs.
    8. Saves model checkpoints and training metrics.
    """

    args = config_parser()

    print('The batchsize is '+ str(args.batch_size))
    if args.split_batches:
        print('We split the above batch into ' + str(torch.cuda.device_count()) + ' gpu (s)')
    else:
        print('each gpu will process ' + str(torch.cuda.device_count()))
    print('We are going to use this GPU - ' + torch.cuda.get_device_name(0))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches = args.split_batches, kwargs_handlers=[ddp_kwargs])
    print('whether accelerator use fp16 - ' + str(accelerator.state.use_fp16))


    dataset = dataset_dict[args.dataset_name]

    train_dataset = dataset(args.data_dir, "train", number_of_imgs_from_the_same_scene=args.number_of_imgs_from_the_same_scene)
    test_dataset = dataset(args.data_dir, "test",
                    number_of_imgs_from_the_same_scene=args.number_of_imgs_from_the_same_scene)

    aabb = test_dataset.scene_bbox
    near_far = test_dataset.near_far

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
    #test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    W, H = train_dataset.img_wh

    model = Model(args, aabb, near_far)
    if args.ckpt:
        ckpt = torch.load(os.path.join(output_dir,'nvist.pth'), map_location=torch.device('cpu'))
        model = load_pretrained_weights(ckpt, model)

        start_iteration = ckpt['iteration']
        print('starting from the previous checkpoint - ' + str(start_iteration))
    else:
        start_iteration = 0

    encoder_grad_vars = [{'params': model.encoder.parameters(), 'lr': args.lr_encoder_init}]
    decoder_grad_vars = [{'params': model.decoder.parameters(), 'lr': args.lr_decoder_init}]
    renderer_grad_vars = [{'params': model.renderer.parameters(), 'lr': args.lr_renderer_init}]

    encoder_opt = torch.optim.AdamW(encoder_grad_vars)
    decoder_opt = torch.optim.AdamW(decoder_grad_vars)
    renderer_opt = torch.optim.AdamW(renderer_grad_vars)

    train_loader, model, encoder_opt, decoder_opt, renderer_opt = accelerator.prepare(train_loader, model, encoder_opt, decoder_opt, renderer_opt)
    dl = cycle(train_loader)

    loss_fn_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)

    pbar = tqdm(range(start_iteration, args.n_iters, 1), miniters = 10)

    # to use lpips loss
    assert args.batch_pixel_size >= H * W * args.batch_size

    psnr_test, lpips_test = [0.],[0.]

    batch_per_gpu = int(args.batch_size / torch.cuda.device_count())
    batch_pixels_per_gpu = int(args.batch_pixel_size /torch.cuda.device_count())

    for iteration in pbar:
        samples = next(dl)

        imgs, ros_all, rds_all, _, original_img_hws = samples['images'], samples['ray_origins'], samples['ray_directions'], samples['scene_idx'], samples['original_img_hws']
        focals = samples['normalized_focals']
        rgbs_train_all = torch.cat([i.unsqueeze(0) for i in samples['images']])
        
        original_img_hws = samples['original_img_hws']

        input_images = rgbs_train_all[0].permute(0,3,1,2).float()

        rgbs_gt_all = []
        for idx_tmp, (H_tmp, W_tmp) in enumerate(original_img_hws): # for each batch
            rgbs_gt_all.append(rgbs_train_all[:,idx_tmp,:H_tmp,:W_tmp].reshape(1,args.number_of_imgs_from_the_same_scene,-1,3))
        
        rgbs_gt_all = torch.cat(rgbs_gt_all)
        rgbs_gt_all = rgbs_gt_all.reshape(batch_per_gpu,-1,3)
        focals_all = torch.cat([f.unsqueeze(0) for f in focals]).permute(1,0)
        ros_all = torch.cat([ro.unsqueeze(0) for ro in ros_all]).permute(1,0,2,3).reshape(batch_per_gpu,-1,3)
        rds_all = torch.cat([rd.unsqueeze(0) for rd in rds_all]).permute(1,0,2,3).reshape(batch_per_gpu,-1,3)
        assert rgbs_gt_all.shape == ros_all.shape

        if batch_per_gpu * args.number_of_imgs_from_the_same_scene * 3 * H * W != np.prod(torch.cat(imgs).shape): 
            continue

        idx = torch.randint(low=1, high=args.number_of_imgs_from_the_same_scene, size=(1,))
        pixel_idxs_lpips = torch.arange((idx*H*W).item(),((1+idx)*H*W).item())

        if idx == args.number_of_imgs_from_the_same_scene-1:
            training_sampler = SimpleSampler(H*W*idx.item(), batch_pixels_per_gpu // batch_per_gpu - H*W)
            sample_idxs = training_sampler.nextids()
            pixel_idxs = torch.cat([pixel_idxs_lpips, sample_idxs])
        else:
            training_sampler_1 = SimpleSampler(H*W*idx.item(), (batch_pixels_per_gpu // batch_per_gpu - H*W) // 2)
            sample_idxs_1 = training_sampler_1.nextids()
            training_sampler_2 = SimpleSampler(H*W*(args.number_of_imgs_from_the_same_scene - idx.item() - 1), (batch_pixels_per_gpu // batch_per_gpu - H*W) // 2)
            sample_idxs_2 = training_sampler_2.nextids() + (1+idx.item())*H*W
            pixel_idxs = torch.cat([pixel_idxs_lpips, sample_idxs_1, sample_idxs_2])

        rgb_gt = rgbs_gt_all[:,pixel_idxs].to(accelerator.device)
        ro_train = ros_all[:,pixel_idxs].to(accelerator.device)
        rd_train = rds_all[:,pixel_idxs].to(accelerator.device)

        loss = torch.tensor(0.)

        camdist = -ros_all[:,0,2]
        camera_params = camdist.reshape(-1,1).float()

        if args.apply_focal_condition:
            camera_params = torch.cat([camera_params, focals_all[:,0].reshape(-1,1).float()], dim=-1)

        if iteration == 0:
            print('input_images_shape : ' + str(input_images.shape))
            print('ro_train_shape : ' + str(ro_train.shape))
            print('rd_train_shape : ' + str(rd_train.shape))

        with accelerator.autocast():
            output = model(input_images, camera_params, ro_train, rd_train) 
            loss_l2 = torch.mean((rgb_gt - output['rgb_map']) ** 2)

            output_images = output['rgb_map'][:,:(H*W)].permute(0,2,1) # nc(h*w)
            gt_images = rgb_gt[:,:(H*W)].permute(0,2,1).float()
            loss_lpips = loss_fn_lpips(output_images.reshape(-1,3,H,W), gt_images.reshape(-1,3,H,W)).mean()
            loss_dist = output['distloss']

            loss = loss_l2 + args.weight_lpips * loss_lpips + args.weight_dist * loss_dist

            
        accelerator.backward(loss)

        accelerator.wait_for_everyone()
        accelerator.clip_grad_norm_(model.parameters(), 1.)
        
        encoder_opt.step()
        decoder_opt.step()
        renderer_opt.step()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        renderer_opt.zero_grad()
        accelerator.wait_for_everyone()


        summary_writer.add_scalar('train/loss_l2', loss_l2.item(), global_step=iteration)
        summary_writer.add_scalar('train/loss_lpips', loss_lpips.item(), global_step=iteration)
        summary_writer.add_scalar('train/loss_dist', loss_dist.item(), global_step=iteration)

        adjust_learning_rate(iteration, args.encoder_warmup_iters, args.n_iters, args.lr_encoder_init, args.lr_minimum, encoder_opt)
        adjust_learning_rate(iteration, args.decoder_warmup_iters, args.n_iters, args.lr_decoder_init, args.lr_minimum, decoder_opt)
        adjust_learning_rate(iteration, args.renderer_warmup_iters, args.n_iters, args.lr_renderer_init, args.lr_minimum, renderer_opt)

        prtx = f'{iteration:07d}'

        with torch.no_grad():
            psnr_train = -10 * np.log(loss_l2.item()) / np.log(10)

        if iteration % 10 == 0:
            memory_cached = torch.cuda.memory_reserved()
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' l2_train={float(loss.item()):.2f}'
                + f' lpips_train={float(loss_lpips.item()):.2f}'
                + f' psnr_train={float(psnr_train):.2f}'
                + f' psnr_test={float(np.mean(psnr_test).item()):.2f}'
                + f' lpips_test={float(np.mean(lpips_test).item()):.2f}'
                + f' GPU: {memory_cached / (1024 ** 3):.1f} GB'
            )

        if iteration % args.vis_every == 0:
            output_depth = output['depth_map'][:,:(H*W)].reshape(-1,H,W)
            output_depth = visualize_depths(output_depth)
            train_grid = visualize_input_novel_view_gt(input_images, output_images.reshape(-1,3,H,W), gt_images.reshape(-1,3,H,W), output_depth)

            imageio.imwrite(os.path.join(output_dir, 'imgs_vis_train', prtx + '.png'), train_grid)

            jump_idx = (len(test_dataset.num_objs) - 1) // (args.n_vis_scenes - 1)
            psnr_test, lpips_test = [], []
            # for each scene
            with torch.no_grad():
                for scene_idx in range(0, len(test_dataset.num_objs), jump_idx):
                    test_images, ros, rds, normalized_focals, test_input = test_dataset.get_data(scene_idx)
                    camdist = -ros[len(ros) // 2][0,0,2]
                    camera_params = camdist.reshape(-1,1).repeat(len(test_images),1).float()
                    if args.apply_focal_condition:
                        camera_params = torch.cat([camera_params, torch.tensor(normalized_focals).reshape(-1,1).float()], dim=-1)
                    vis_images = [test_input]
                    test_input = test_input.to(accelerator.device)
                    psnr_tmp, lpips_tmp = [], []
                    for test_image, ro, rd, camera_param in zip(test_images, ros, rds, camera_params):                
                        test_image, ro, rd, camera_param = test_image.to(accelerator.device), ro.to(accelerator.device), rd.to(accelerator.device), camera_param.to(accelerator.device)
                        output = model(test_input, camera_param,ro, rd)
                        loss_l2 = torch.mean((output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2) - test_image)**2)
                        psnr_tmp.append(-10 * np.log(loss_l2.item()) / np.log(10))
                        lpips_tmp.append(loss_fn_lpips(test_image, output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2)).item())
                        vis_images.append(output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2).detach().cpu())
                        vis_images.append(test_image.detach().cpu())
                        depth = visualize_depth(output['depth_map'].reshape(H,W))
                        vis_images.append(depth.detach().cpu())
                    scenertx = f'{scene_idx:03d}'
                    test_vis = visualize_images(torch.cat(vis_images))
                    imageio.imwrite(os.path.join(output_dir, 'imgs_vis_test', prtx + '_' + scenertx + '.png'), test_vis)
                    psnr_test.append(np.mean(psnr_tmp))
                    lpips_test.append(np.mean(lpips_tmp))
                np.savetxt(os.path.join(output_dir, 'imgs_vis_test', prtx + '.txt'), np.asarray([np.mean(psnr_test), np.mean(lpips_test)]))
                summary_writer.add_scalar('test/psnr', np.mean(psnr_test), global_step=iteration)
                summary_writer.add_scalar('test/lpips', np.mean(lpips_test), global_step=iteration)

        if iteration > 0 and iteration % args.vis_every == 0:
            ckpt = {'state_dict':model.state_dict(), 'iteration':iteration}
            torch.save(ckpt, os.path.join(output_dir, 'nvist.pth'))

        # if iteration > 0 and iteration % args.render_test_all == 0:
        #     jump_idx = 1
        #     with torch.no_grad():
        #         for scene_idx in range(0, len(test_dataset.num_objs), jump_idx):
        #             test_images, ros, rds, normalized_focals, test_input = test_dataset.get_data(scene_idx)
        #             camdist = -ros[len(ros) // 2][0,0,2]
        #             camera_params = camdist.reshape(-1,1).repeat(len(test_images),1).float()
        #             if args.apply_focal_condition:
        #                 camera_params = torch.cat([camera_params, torch.tensor(normalized_focals).reshape(-1,1).float()], dim=-1)
        #             vis_images = [test_input]
        #             test_input = test_input.to(accelerator.device)
        #             psnr_tmp, lpips_tmp = [], []
        #             for test_image, ro, rd, camera_param in zip(test_images, ros, rds, camera_params):                
        #                 test_image, ro, rd, camera_param = test_image.to(accelerator.device), ro.to(accelerator.device), rd.to(accelerator.device), camera_param.to(accelerator.device)
        #                 output = model(test_input, camera_param,ro, rd)
        #                 loss_l2 = torch.mean((output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2) - test_image)**2)
        #                 psnr_tmp.append(-10 * np.log(loss_l2.item()) / np.log(10))
        #                 lpips_tmp.append(loss_fn_lpips(test_image, output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2)).item())
        #                 vis_images.append(output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2).detach().cpu())
        #                 vis_images.append(test_image.detach().cpu())
        #                 depth = visualize_depth(output['depth_map'].reshape(H,W))
        #                 vis_images.append(depth.detach().cpu())
        #             scenertx = f'{scene_idx:03d}'
        #             test_vis = visualize_images(torch.cat(vis_images))
        #             imageio.imwrite(os.path.join(output_dir, 'imgs_test', prtx + '_' + scenertx + '.png'), test_vis)
        #             psnr_test.append(np.mean(psnr_tmp))
        #             lpips_test.append(np.mean(lpips_tmp))
        #         np.savetxt(os.path.join(output_dir, 'imgs_test', prtx + '.txt'), np.asarray([np.mean(psnr_test), np.mean(lpips_test)]))
        #         summary_writer.add_scalar('test/psnr_all', np.mean(psnr_test), global_step=iteration)
        #         summary_writer.add_scalar('test/lpips_all', np.mean(lpips_test), global_step=iteration)        

        
if __name__ == "__main__":
    main()
 

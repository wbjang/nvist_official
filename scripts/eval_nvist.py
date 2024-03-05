
import os
import sys
import torch


import json
import numpy as np
import imageio

from accelerate import Accelerator
import lpips
from easydict import EasyDict as edict

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from dataLoader import dataset_dict
from models.model import Model
from opt import config_parser
from utils import visualize_images, visualize_depth, load_pretrained_weights

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """
    Main function for evaluating a neural rendering model.

    This script performs the following operations:
    1. Loads configuration parameters and hyperparameters from the saved checkpoint.
    2. Initializes and loads the dataset for testing.
    3. Sets up the model and loads pretrained weights.
    4. Evaluates the model on the test dataset to compute metrics like PSNR and LPIPS.
    5. Generates visualizations for the test scenes, including rendered images and depth maps, and compiles them into videos.
    6. Saves the evaluation metrics and visualizations to specified directories.

    """

    # load the hpams
    args = config_parser()

    hpam_path = os.path.join(args.ckpt_base, args.ckpt_dir, 'hpams.txt')
    with open(hpam_path, 'r') as f:
        hpams = json.load(f)
    hpams = edict(hpams)

    # load dataset
    dataset = dataset_dict[hpams.dataset_name]
    test_dataset = dataset(hpams.data_dir, "test", 
                           number_of_imgs_from_the_same_scene=hpams.number_of_imgs_from_the_same_scene)
    aabb = test_dataset.scene_bbox
    near_far = test_dataset.near_far

    # load accelerator
    accelerator = Accelerator()

    # load ckpt
    model = Model(hpams, aabb, near_far).to(accelerator.device)

    output_dir = os.path.join(args.ckpt_base, args.ckpt_dir)
    os.makedirs(os.path.join(output_dir, 'imgs_test'), exist_ok = True)
    os.makedirs(os.path.join(output_dir, 'videos_test'), exist_ok = True)
    W, H = test_dataset.img_wh

    ckpt = torch.load(os.path.join(output_dir,'nvist.pth'), map_location=torch.device('cpu'))
    model = load_pretrained_weights(ckpt, model)
    finish_iteration = ckpt['iteration']
    print('the model has been trained for - ' + str(finish_iteration))

    loss_fn_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)
    
    eval_test = {}
    eval_test['psnr'] = []
    eval_test['lpips'] = []
    eval_test['scene_names'] = []
    # let's make videos for each scene
    # also save numbers for each scene
    with torch.no_grad():
        for scene_idx in range(0, len(test_dataset.num_objs)-1, 1):
            test_images, ros, rds, normalized_focals, test_input, scene_name = test_dataset.get_data(scene_idx, return_scene_name=True)
            camdist = -ros[len(ros) // 2][0,0,2]
            camera_params = camdist.reshape(-1,1).repeat(len(test_images),1).float()
            if hpams.apply_focal_condition:
                camera_params = torch.cat([camera_params, torch.tensor(normalized_focals).reshape(-1,1).float()], dim=-1)
            video_scene = []
            vis_image = []
            test_input = test_input.to(accelerator.device)
            psnr_tmp, lpips_tmp = [], []

            for test_image, ro, rd, camera_param in zip(test_images, ros, rds, camera_params):                
                test_image, ro, rd, camera_param = test_image.to(accelerator.device), ro.to(accelerator.device), rd.to(accelerator.device), camera_param.to(accelerator.device)
                output = model(test_input, camera_param,ro, rd)
                loss_l2 = torch.mean((output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2) - test_image)**2)
                # compute loss
                psnr_tmp.append(-10 * np.log(loss_l2.item()) / np.log(10))
                lpips_tmp.append(loss_fn_lpips(test_image, output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2)).item())
                # save images
                vis_output = output['rgb_map'].reshape(1,H,W,3).permute(0,3,1,2).detach().cpu()
                vis_depth = visualize_depth(output['depth_map'].reshape(H,W))
                vis_frame = torch.cat([test_input.detach().cpu(), vis_output, test_image.detach().cpu(), vis_depth], dim=-1)
                vis_image.append(vis_frame[0])
                vis_frame = vis_frame[0] * 255
                video_scene.append(vis_frame.permute(1,2,0).to(torch.uint8))
                

            imageio.mimwrite(os.path.join(output_dir, 'videos_test', scene_name + '.mp4'), video_scene)
            test_vis = visualize_images(torch.stack(vis_image))
            imageio.imwrite(os.path.join(output_dir, 'imgs_test', scene_name + '.png'), test_vis)
            eval_test['psnr'].append(np.mean(psnr_tmp))
            eval_test['lpips'].append(np.mean(lpips_tmp))
            eval_test['scene_names'].append(scene_name)
    eval_test['final_psnr'] = np.mean(eval_test['psnr'])
    eval_test['final_lpips'] = np.mean(eval_test['lpips'])

    save_eval_path = os.path.join(output_dir, 'eval.txt')
    with open(save_eval_path, 'w') as f:
        json.dump(save_eval_path, f, indent=2) 
    print('final evals - ' + str(eval_test['final_psnr']) + ' ' + str(eval_test['final_lpips']))       

if __name__ == "__main__":
    main()
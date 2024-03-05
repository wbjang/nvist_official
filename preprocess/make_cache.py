import os
import json
import numpy as np
import glob
import imageio
import torch
import argparse
from typing import Tuple

def convert_3_4_to_4_4(pose: np.ndarray) -> np.ndarray:
    """
    Converts a 3x4 pose matrix to a 4x4 pose matrix by appending a row [0, 0, 0, 1].

    Args:
        pose (np.ndarray): The 3x4 pose matrix.

    Returns:
        np.ndarray: The converted 4x4 pose matrix.
    """
    assert pose.shape == (3, 4), "Input pose must be a 3x4 matrix."
    pose_new = np.zeros((4,4))
    pose_new[:3] = pose
    pose_new[3,3] = 1.
    return pose_new

def load_camera(campath: str, scenepath: str) -> Tuple[np.ndarray, float, float, float, np.ndarray, int, int]:
    """
    Loads camera parameters from a JSON file and applies scene transformations.

    These parameters are for 160x90 resolution images - images_12

    Args:
        campath (str): Path to the camera JSON file.
        scenepath (str): Path to the scene JSON file.

    Returns:
        Tuple containing:
        - extr (np.ndarray): The extrinsic camera parameters as a 4x4 matrix.
        - focal (float): The scaled focal length.
        - near (float): The near clipping distance.
        - far (float): The far clipping distance.
        - scene_bbox (np.ndarray): The scene bounding box.
        - W (int): The resized image width.
        - H (int): The resized image height.

    Raises:
        json.JSONDecodeError: If there is an error decoding the JSON from the file.
    """

    cam = json.load(open(campath))
    
    try: 
        scene_json = json.load(open(scenepath))
        scene_center = np.array(scene_json['center'])
        scene_scale = scene_json['scale']
        scene_bbox = np.array(scene_json['bbox']) * 1.6
        near = scene_json['near'] 
        far = scene_json['far'] 

        # self.scale_factor : additional scale_factor to scale down images
        cam['position'] = cam['position'] - scene_center
        cam['position'] = cam['position'] * scene_scale
        cam['position'] = cam['position']

        W, H = cam['image_size']
        

        extr = np.concatenate([np.array(cam['orientation']).transpose(-1,-2), np.array(cam['position']).reshape(-1,1)], axis=-1)
        extr = convert_3_4_to_4_4(extr)

        if W > H : 
            resize_scale = 160 / W
        else: 
            resize_scale = 160 / H
        focal = cam['focal_length'] * resize_scale
        W = W * resize_scale
        H = H * resize_scale

        return extr, focal, near, far, scene_bbox,W,H
    
    except json.JSONDecodeError:
        print(f'Error decoding JSON from file: {scenepath}')


def main():
    """
    Generates cache files containing scene information for a specified dataset split.

    This script processes a dataset to accumulate information about images, camera parameters,
    and scene configurations. 

    The script distinguishes between 'portrait' and 'landscape' images based on their dimensions
    and counts the total number of each type. It supports processing either 'train' or 'test' splits
    of the dataset, as specified by the user.

    Usage:
        python -m preprocess.make._cache --data_dir <path_to_data_directory> --split <train/test>

    Arguments:
        --data_dir: Path to the root directory of the dataset.
        --split: The dataset split to process ('train' or 'test').
    """

    parser = argparse.ArgumentParser(description="make cache files")
    parser.add_argument("--data_dir", type=str, default="../../data/mvimgnet_original")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    split = args.split

    catdirs = sorted(os.listdir(args.data_dir))
    catdirs = [os.path.join(args.data_dir, catdir) for catdir in catdirs if os.path.isdir(os.path.join(args.data_dir, catdir))]
    img_files_total = []
    cam_files_total = []
    read_cams = []
    number_of_imgs_per_scene = []
    scenedirs_all = []
    img_sizes_hw = []
    focals = []
    num_portraits = 0

    for catdir in catdirs:
        scenedirs = sorted(os.listdir(catdir))
        scenedirs = [os.path.join(catdir, scenedir) for scenedir in scenedirs if os.path.isdir(os.path.join(catdir,scenedir))]
        
        for it, scenedir in enumerate(scenedirs):
            if (split == 'train' and it % 100 != 0) or (split=='test' and it % 100 == 0):
                scenepath = os.path.join(scenedir, 'scene_new.json')
                camfiles = sorted(glob.glob(os.path.join(scenedir, 'camera_new/*.json')))
                imgfiles = sorted(glob.glob(os.path.join(scenedir, 'images_12/*.jpg')))
                if len(camfiles) > 0 and len(imgfiles) > 0:
                    img_check = imageio.imread(imgfiles[0])
                    H, W = img_check.shape[:2]

                    # check if it is portrait
                    if H * W == 14400 and H == 160:
                        cam_files_total.extend(camfiles)
                        img_files_total.extend(imgfiles)
                        read_cams.extend([load_camera(camfile, scenepath) for camfile in camfiles])
                        focals.append(load_camera(camfiles[0], scenepath)[1])
                        number_of_imgs_per_scene.append(len(imgfiles))
                        img_sizes_hw.extend([[H, W]])
                        scenedirs_all.append(scenedir.split(args.data_dir)[1])
                        num_portraits += 1

    c2ws = [read_cam[0] for read_cam in read_cams]
    nears = [read_cam[2] for read_cam in read_cams]
    fars = [read_cam[3] for read_cam in read_cams]
    near_fars = [[read_cam[2], read_cam[3]] for read_cam in read_cams]


    cache = dict()
    cache['img_files'] = img_files_total
    cache['c2ws'] = c2ws
    cache['focals'] = focals
    cache['number_of_imgs_per_scene'] = number_of_imgs_per_scene
    cache['near_far'] = [min(nears), max(fars)]
    cache['near_fars'] = near_fars

    cache['img_sizes_hw'] = img_sizes_hw
    cache['subdirs'] = scenedirs_all

    print('img_files_total ', len(img_files_total))
    print('c2ws ', len(c2ws))
    print('focals (number of scenes) ', len(focals))
    print('img_sizes_hw ', len(img_sizes_hw))
    print('number_of_portraits ', num_portraits)

    cache_file_name = os.path.join(args.data_dir, 'cache_portrait_{}.th'.format(split))

    print('cache file will be saved at ' + cache_file_name)
    torch.save(cache, cache_file_name)
    print('we save cache file ', cache_file_name)

if __name__ == "__main__":
    main()
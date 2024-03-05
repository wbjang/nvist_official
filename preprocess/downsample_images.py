


import os
from PIL import Image
import glob
import argparse
from opt import str2bool

def main():
    """
    Downsamples images in a dataset to multiple resolutions and saves them in separate directories.

    This script iterates through a dataset directory, downsampling images to specified resolutions
    and saving them in corresponding subdirectories. By default, it resizes images to have a maximum
    dimension of 160 pixels. If the `multi_resolution` flag is set, it also creates downscaled images
    at resolutions of 3, 4, 6 times the original size, and an additional resolution where the minimum
    dimension is 128 pixels.

    Usage:
        python downsample_images.py --data_dir <path_to_dataset> --multi_resolution <True/False> 

    Arguments:
        -d, --data_dir: Path to the dataset directory.
        -r, --multi_resolution: Whether to also downsample images by 3 and 6 times the original size.
    """
    parser = argparse.ArgumentParser(description='Downsample images')
    parser.add_argument("-d", "--data_dir", type=str, default="../../data/mvimgnet_original", help="where the dataset is")
    parser.add_argument("-r", "--multi_resolution", type=str2bool, default=False, help="if yes, we also downsample images by 3 and 6")

    args = parser.parse_args()

    dirnames = os.listdir(args.data_dir)[::-1]
    dirnames = [dirname for dirname in dirnames if os.path.isdir(os.path.join(args.data_dir, dirname))]

    for dirname in dirnames:
        scenenames = sorted(os.listdir(os.path.join(args.data_dir, dirname)))
        cat_name = {}
        for scenename in scenenames:
            imgfiles = sorted(glob.glob(os.path.join(args.data_dir,dirname,scenename,'images/*.jpg')))
            savedir_12 = os.path.join(args.data_dir,dirname,scenename,'images_12')
            os.makedirs(savedir_12, exist_ok=True)
            if args.multi_resolution:
                savedir_3 = os.path.join(args.data_dir, dirname, scenename, 'images_3')
                savedir_4 = os.path.join(args.data_dir, dirname, scenename, 'images_4')
                savedir_6 = os.path.join(args.data_dir, dirname, scenename, 'images_6')
                savedir_128 = os.path.join(args.data_dir, dirname, scenename, 'images_128')
                os.makedirs(savedir_3, exist_ok=True)
                os.makedirs(savedir_4, exist_ok =True)
                os.makedirs(savedir_6, exist_ok=True)
                os.makedirs(savedir_128, exist_ok = True)
            scale = 0.0
            if len(imgfiles) > 0:
                print(dirname, scenename, savedir_12, scale)
                for imgfile in imgfiles:
                    if imgfile.endswith(".jpg") or imgfile.endswith(".png"):
                        fname = imgfile.split('/')[-1]
                        savepath_12 = os.path.join(savedir_12, fname)
                        if not os.path.isfile(savepath_12):
                            with Image.open(imgfile) as img:
                                width, height = img.size
                                if width > height : 
                                    scale = 160 / width
                                else: 
                                    scale = 160 / height
                                new_width, new_height = int(width * scale), int(height * scale)
                                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                resized_img.save(savepath_12)
                        else: 
                            scale = 0.0
                        if args.multi_resolution:
                            savepath_3 = os.path.join(savedir_3, fname)
                            savepath_4 = os.path.join(savedir_4, fname)
                            savepath_6 = os.path.join(savedir_6, fname)
                            savepath_128 = os.path.join(savedir_128, fname)
                            if not os.path.isfile(savepath_3):
                                with Image.open(imgfile) as img:
                                    width, height = img.size
                                    if width > height: 
                                        scale = 640 / width
                                    else: 
                                        scale = 640 / height
                                    new_width, new_height = int(width * scale), int(height * scale)
                                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                    resized_img.save(savepath_3)
                            if not os.path.isfile(savepath_4):
                                with Image.open(imgfile) as img:
                                    width, height = img.size
                                    if width > height : 
                                        scale = 480 / width
                                    else: 
                                        scale = 480 / height
                                    new_width, new_height = int(width * scale), int(height * scale)
                                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                    resized_img.save(savepath_4)
                            if not os.path.isfile(savepath_6):
                                with Image.open(imgfile) as img:
                                    width, height = img.size
                                    if width > height : 
                                        scale = 320 / width
                                    else: 
                                        scale = 320 / height
                                    new_width, new_height = int(width * scale), int(height * scale)
                                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                    resized_img.save(savepath_6)
                            if not os.path.isfile(savepath_128):
                                with Image.open(imgfile) as img:
                                    width, height = img.size
                                    if width < height : # opposite sign : minimum res will have 128
                                        scale = 128 / width
                                    else: 
                                        scale = 128 / height
                                    new_width, new_height = int(width * scale), int(height * scale)
                                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                    resized_img.save(savepath_128)

                        else:
                            scale = 0.0
                    else: 
                        scale = 0.0
            else: 
                print(dirname, scenename, savedir_12)
if __name__ == "__main__":
    main()
        

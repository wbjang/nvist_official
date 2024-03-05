

import subprocess
import os
import glob
import argparse

def main():
    """
    Processes scene directories in the MVImgNet dataset by reading COLMAP results where applicable.

    This script iterates over category directories within the base MVImgNet directory. For each scene
    directory that contains a 'sparse/0' directory but no JSON files in 'camera_new/', it triggers
    the `preprocess.read_colmap_results` module to process the COLMAP results for that scene.

    Usage:
        python -m preprocess.read_colmap_results_mvimgnet.py --base_dir <path_to_mvimgnet_directory>

    Arguments:
        --base_dir: The root directory of the MVImgNet dataset where category directories are located.
    """

    arg_parser = argparse.ArgumentParser(description="Wrapper for MVImgNet")
    arg_parser.add_argument("--base_dir", default="../../data/mvimgnet", help="category")
    args = arg_parser.parse_args()
    base_dir = args.base_dir

    catdirs = os.listdir(base_dir)
    catdirs = [catdir for catdir in catdirs if os.path.isdir(os.path.join(base_dir,catdir))]

    scenes = {}
    for catdir in catdirs:
        scenedirs = os.listdir(os.path.join(base_dir, catdir))
        scenedirs = [scenedir for scenedir in scenedirs if os.path.isdir(os.path.join(base_dir,catdir,scenedir))]
        scenes[catdir] = []
        for scenedir in scenedirs:
            if os.path.isdir(os.path.join(base_dir, catdir, scenedir)):
                num_json_files = len(glob.glob(os.path.join(base_dir, catdir, scenedir, 'camera_new/*.json')))
                if os.path.exists(os.path.join(base_dir, catdir, scenedir, 'sparse/0')) and num_json_files == 0:
                    print('we will process ' + catdir + "  " + scenedir)
                    cmd = ['python', '-m' 'preprocess.read_colmap_results', '--base_dir', os.path.join(base_dir, catdir), '--capture_name', scenedir]
                    subprocess.run(cmd)
                else: 
                    print('we already processed '+catdir + ' ' + scenedir)
            else: 
                print('we already processed '+catdir + ' ' + scenedir)

if __name__ == "__main__":
    main()

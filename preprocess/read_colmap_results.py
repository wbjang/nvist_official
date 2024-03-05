
import argparse
import pycolmap
import os
import numpy as np
import json
from typing import Dict
import imageio
import pandas as pd
import logging
from nerfies.camera import Camera

# Code is adapted from Nerfies
    # https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/camera.py#L107
# and modified by the author

def convert_colmap_camera(colmap_camera, colmap_image):
    """Converts a pycolmap `image` to an SFM camera."""
    camera_rotation = colmap_image.R()
    camera_position = -(colmap_image.t @ camera_rotation)
    new_camera = Camera(
        orientation=camera_rotation,
        position=camera_position,
        focal_length=colmap_camera.fx,
        pixel_aspect_ratio=colmap_camera.fx / colmap_camera.fx,
        principal_point=np.array([colmap_camera.cx, colmap_camera.cy]),
        # radial_distortion=np.array([colmap_camera.k1, colmap_camera.k2, 0.0]),
        # tangential_distortion=np.array([colmap_camera.p1, colmap_camera.p2]),
        skew=0.0,
        image_size=np.array([colmap_camera.width, colmap_camera.height])
    )
    return new_camera

class SceneManager:
    """A thin wrapper around pycolmap."""

    @classmethod
    def from_pycolmap(cls, colmap_path, image_path, min_track_length=10):
        """Create a scene manager using pycolmap."""
        manager = pycolmap.SceneManager(str(colmap_path))
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        manager.filter_points3D(min_track_len=min_track_length)
        sfm_cameras = _pycolmap_to_sfm_cameras(manager)
        return cls(sfm_cameras, manager.get_filtered_points3D(), image_path)

    def __init__(self, cameras, points, image_path):
        #self.image_path = Path(image_path)
        self.image_path = image_path
        self.camera_dict = cameras
        self.points = points

        logging.info('Created scene manager with %d cameras', len(self.camera_dict))

    def __len__(self):
        return len(self.camera_dict)

    @property
    def image_ids(self):
        return sorted(self.camera_dict.keys())

    @property
    def camera_list(self):
        return [self.camera_dict[i] for i in self.image_ids]

    @property
    def camera_positions(self):
        """Returns an array of camera positions."""
        return np.stack([camera.position for camera in self.camera_list])

    def load_image(self, image_id):
        """Loads the image with the specified image_id."""
        #path = self.image_path / f'{image_id}.png'
        path = os.path.join(self.image_path, f'{image_id}.png')
        with open(path, 'rb') as f:
          return imageio.imread(f)

    def change_basis(self, axes, center):
      """Change the basis of the scene.

      Args:
        axes: the axes of the new coordinate frame.
        center: the center of the new coordinate frame.

      Returns:
        A new SceneManager with transformed points and cameras.
      """
      transform_mat = np.zeros((3, 4))
      transform_mat[:3, :3] = axes.T
      transform_mat[:, 3] = -(center @ axes)
      return self.transform(transform_mat)

 
    def filter_images(self, image_ids):
      num_filtered = 0
      for image_id in image_ids:
        if self.camera_dict.pop(image_id, None) is not None:
          num_filtered += 1

      return num_filtered
    
def estimate_near_far(scene_manager):
    """Estimate near/far plane for a set of randomly-chosen images."""
    # image_ids = sorted(scene_manager.images.keys())
    image_ids = scene_manager.image_ids
    rng = np.random.RandomState(0)
    image_ids = rng.choice(
        image_ids, size=len(scene_manager.camera_list), replace=False)
    
    result = []
    for image_id in image_ids:
        near, far = estimate_near_far_for_image(scene_manager, image_id)
        result.append({'image_id': image_id, 'near': near, 'far': far})
    result = pd.DataFrame.from_records(result)
    return result

def filter_outlier_points(points, inner_percentile):
    """Filters outlier points."""
    outer = 1.0 - inner_percentile
    lower = outer / 2.0
    upper = 1.0 - lower
    centers_min = np.quantile(points, lower, axis=0)
    centers_max = np.quantile(points, upper, axis=0)
    result = points.copy()

    too_near = np.any(result < centers_min[None, :], axis=1)
    too_far = np.any(result > centers_max[None, :], axis=1)

    return result[~(too_near | too_far)]

def get_bbox_corners(points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.stack([lower, upper])

def estimate_near_far_for_image(scene_manager, image_id):
    """Estimate near/far plane for a single image based via point cloud."""
    points = filter_outlier_points(scene_manager.points, 0.95)
    points = np.concatenate([
        points,
        scene_manager.camera_positions,
    ], axis=0)
    camera = scene_manager.camera_dict[image_id]
    pixels = camera.project(points)
    depths = camera.points_to_local_points(points)[..., 2]

    # in_frustum = camera.ArePixelsInFrustum(pixels)
    in_frustum = (
        (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] <= camera.image_size_x)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] <= camera.image_size_y))
    depths = depths[in_frustum]

    in_front_of_camera = depths > 0
    depths = depths[in_front_of_camera]

    near = np.quantile(depths, 0.001)
    far = np.quantile(depths, 0.999)

    return near, far

def _pycolmap_to_sfm_cameras(manager: SceneManager) -> Dict[int, Camera]:
    """Creates SFM cameras."""
    # Use the original filenames as indices.
    # This mapping necessary since COLMAP uses arbitrary numbers for the
    # image_id.
    image_id_to_colmap_id = {
        image.name.split('.')[0]: image_id
        for image_id, image in manager.images.items()
    }

    sfm_cameras = {}
    for image_id in image_id_to_colmap_id:
        colmap_id = image_id_to_colmap_id[image_id]
        image = manager.images[colmap_id]
        camera = manager.cameras[image.camera_id]
        sfm_cameras[image_id] = convert_colmap_camera(camera, image)

    return sfm_cameras


def main():
    """
    Processes a capture from the MVImgNet dataset to compute and save camera and scene information.

    This script reads COLMAP reconstruction results, computes near/far values for the scene,
    filters outlier points, and estimates the bounding box. It then saves the scene's metadata
    and each camera's information in a new format.

    Usage:
        python -m preprocess.read_colmap_results --base_dir <base_directory_path> --capture_name <capture_name> --outlier_percentage <outlier_filter_threshold>
    
    Arguments:
        --base_dir: Directory containing the capture folders.
        --capture_name: Name of the capture folder to process.
        --outlier_percentage: Percentage of points to consider as outliers based on depth values.
    """
    
    arg_parser = argparse.ArgumentParser(description="ReadingColmap")

    arg_parser.add_argument("--base_dir", default="../../data/mvimgnet/0", help="category")
    arg_parser.add_argument("--capture_name",dest="capture_name",default="3a004e55")
    arg_parser.add_argument("--outlier_percentage", dest="outlier_percentage", default=0.95)
    args = arg_parser.parse_args()

    colmap_dir = os.path.join(args.base_dir, args.capture_name, 'sparse/0')
    img_dir = os.path.join(args.base_dir, args.capture_name, 'images')

    scene_manager = SceneManager.from_pycolmap(colmap_dir, img_dir)
    new_scene_manager = scene_manager
    near_far = estimate_near_far(new_scene_manager)
    print('Statistics for near/far computation:')
    print(near_far.describe())

    near = near_far['near'].quantile(0.001) * 0.9
    far = near_far['far'].quantile(0.999) * 1.1
    print('Selected near/far values:')
    print(f'Near = {near:.04f}')
    print(f'Far = {far:.04f}')

    points = filter_outlier_points(new_scene_manager.points, args.outlier_percentage)
    bbox_corners = get_bbox_corners(
        np.concatenate([points, new_scene_manager.camera_positions], axis=0)) # lowest/highest
    scene_center = np.mean(bbox_corners, axis=0)

    # rescale the point clouds into a unit cube
    scene_scale = 1.0 / np.max(bbox_corners[1] - bbox_corners[0])
    print(f'Scene Center: {scene_center}')
    print(f'Scene Scale: {scene_scale}')

    bbox_corners_centered = bbox_corners - scene_center
    bbox_corners_scaled = bbox_corners_centered * scene_scale # bbox : computing based on point clouds
    scene_json_path = os.path.join(args.base_dir, args.capture_name, 'scene_new.json')

    with open(scene_json_path,'w') as f:
        json.dump({
            'scale': scene_scale,
            'center': scene_center.tolist(),
            'bbox_unscaled': bbox_corners.tolist(),
            'bbox': bbox_corners_scaled.tolist(),
            'near': near * scene_scale,
            'far': far * scene_scale,
        }, f, indent=2)
    
    print(f'Saved scene information to {scene_json_path}')

    all_ids = scene_manager.image_ids

    camera_dir = os.path.join(args.base_dir, args.capture_name, 'camera_new')
    os.makedirs(camera_dir, exist_ok = True)
    for item_id, camera in new_scene_manager.camera_dict.items():
        camera_path = os.path.join(camera_dir, f'{item_id}.json')
        print(f'Saving camera to {camera_path!s}')
        with open(camera_path, 'w') as f:
            json.dump(camera.to_json(), f, indent=2)


if __name__ == "__main__":
    main()
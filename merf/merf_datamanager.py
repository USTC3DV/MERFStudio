# 作者 任晨曲
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig,Nerfstudio
from dataclasses import field,dataclass
from typing import Type
from jaxtyping import Float
from torch import Tensor

from dataclasses import dataclass, field
from pathlib import Path
from typing import  Type
from typing import List, Tuple
import numpy as np
import torch

from merf.camera_utils import auto_orient_and_center_poses
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    get_train_eval_split_all,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

from merf.point_util.ransc import get_normal
from merf.point_util.read_bin import read_bin_output_ply
from merf.point_util.read_ply import get_center_and_scale
from merf.point_util.rotation import get_rotation_matrix,initialize_points
from merf.point_util.visualize_pointcloud_and_camera import visualize_pointcloud_and_cameras,visualize_camera_poses_plotly
import os
import warnings
import open3d as o3d

MAX_AUTO_RESOLUTION = 1600

def rot_matrix_angle(poses,angle=np.pi/10):
    rot_matrix = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    aligned_poses = np.copy(poses)
    aligned_poses[:, :3, :3] = np.einsum('ij,njk->nik', rot_matrix, poses[:, :3, :3])
    aligned_poses[:, :3, 3] = np.einsum('ij,nj->ni', rot_matrix, poses[:, :3, 3])
    return torch.tensor(aligned_poses)

def align_rectangle_with_axes(input_poses):
    """
    Aligns a set of camera poses such that the rectangle is aligned with the xy axes.

    Parameters:
        input_poses (numpy array): A Nx4x4 numpy array, where N is the number of poses.

    Returns:
        numpy array: A Nx4x4 numpy array of aligned poses.
    """
    # Extract camera positions
    camera_positions = input_poses[:, :3, 3]
    
    # Step 1: Find the corner points of the rectangle
    x_min_index = np.argmin(camera_positions[:, 0])
    x_max_index = np.argmax(camera_positions[:, 0])
    y_min_index = np.argmin(camera_positions[:, 1])
    y_max_index = np.argmax(camera_positions[:, 1])
    
    corner_indices = [x_min_index, x_max_index, y_min_index, y_max_index]
    rectangle_corners = camera_positions[corner_indices,:]

    
    # Step 2: Determine the base vectors
    vec_x = np.array([])  # Vector along the length
    vec_y = rectangle_corners[3,:] - rectangle_corners[0,:]  # Vector along the width
    
     # Normalize the vectors
    vec_x /= np.linalg.norm(vec_x)
    vec_y /= np.linalg.norm(vec_y)

    # Find the third base vector
    vec_z = np.cross(vec_x, vec_y)
    vec_z /= np.linalg.norm(vec_z)

    # Construct the rotation matrix
    rot_matrix = np.column_stack([vec_x, vec_y, vec_z]).transpose()
    
    # Check the orientation of the rotation matrix
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] = -rot_matrix[:, 2]


    # Step 4: Apply the rotation to the entire pose matrix using batch processing
    aligned_poses = np.copy(input_poses)
    aligned_poses[:, :3, :3] = np.einsum('ij,njk->nik', rot_matrix, aligned_poses[:, :3, :3])
    aligned_poses[:, :3, 3] = np.einsum('ij,nj->ni', rot_matrix, aligned_poses[:, :3, 3])
    
    return torch.tensor(aligned_poses)


def get_train_eval_split_blender(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [str(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    # print(basenames)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "test" in basename or "eval" in basename:
            i_eval.append(idx)
        else:
            i_train.append(idx)
            #raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


@dataclass
class MerfDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: MerfDataParser)
    scene_scale: float = 1.0
    is_transform : bool = False
    z_limit_max: float = 1.0
    z_limit_min: float = -1.0
    scale_1: float = 1.0
    angle_rot: float = 0.0


@dataclass
class MerfDataParser(Nerfstudio):
    config: MerfDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
            path = "colmap/sparse/0/points3D.bin"
            in_path  = os.path.join(data_dir,path)
            if not os.path.exists(in_path):
                warnings.warn(f"{in_path} not found",category=UserWarning)
                self.config.is_transform=False
            out_path = data_dir
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data
            path = "colmap/sparse/0/points3D.bin"
            in_path  = os.path.join(self.config.data,path)
            if not os.path.exists(in_path):
                warnings.warn(f"{in_path} not found",category=UserWarning)
                self.config.is_transform=False
            out_path = self.config.data



        image_filenames = []
        image_filenames_eval = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )
            if self.config.eval_mode == "blender":
                image_filenames.append(fname if "_eval"  not in str(fname) else Path(str(fname).replace("_eval", "")))
                image_filenames_eval.append(fname)
            else:
                image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = Path(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
                # i_eval, i_train = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
                # i_eval, i_train = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "blender":
                i_train, i_eval = get_train_eval_split_blender(image_filenames_eval)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor /= self.config.scale_1

        poses[:, :3, 3] *= scale_factor
        
        maxz = self.config.z_limit_max
        minz = self.config.z_limit_min
        if self.config.is_transform:
            print(f"  ===  processing point ======>............ ")

            # assert os.path.exists(in_path)
            self.padding = torch.tensor([[0.,0.,0.,1.]],dtype=torch.float32)
            transform_final = torch.diag(torch.tensor([scale_factor] * 3 + [1])) @ torch.cat([transform_matrix,self.padding],dim=0)
            out_path_ = os.path.join(out_path,"out_put.ply")
            read_bin_output_ply(in_path,out_path_)
            # point_cloud = o3d.io.read_point_cloud(out_path_)
            points = initialize_points(out_path_,transform_final.numpy())
            # visualize_pointcloud_and_cameras(points,poses[:,:3,3])
            # A1,B1,t1 = get_normal(out_path_)
            # out_path_rotation = os.path.join(out_path,"out_put_rotation.ply")
            # self.rotation_matrix = get_rotation_matrix(A1,B1,t1,out_path_,out_path_rotation)
            self.center,self.scene_scale_t, maxz, minz = get_center_and_scale(out_path_)
            self.padding = self.padding.repeat(poses.shape[0],1,1)
            print(f" transform_final = {transform_final} ")
            poses_ = torch.cat([poses,self.padding],dim=1)
            # poses_ = torch.matmul(torch.tensor(self.rotation_matrix,dtype=torch.float32),poses_)  # rotation
            # points = torch.matmul(torch.tensor(self.rotation_matrix,dtype=torch.float64),torch.tensor(np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)).unsqueeze(-1)) 
            points -= self.center
            points *= 1./self.scene_scale_t
            poses_[:,:3,3] -= torch.tensor(self.center,dtype=torch.float32)  #transform
            poses_[:,:3,3] *= torch.tensor(1./self.scene_scale_t,dtype=torch.float32)  #scale
            poses = poses_[:,:3,:]
            
        maxz = self.config.z_limit_max
        minz = self.config.z_limit_min
        
        
        # if not self.config.is_transform:           
        #     self.padding = torch.tensor([[0.,0.,0.,1.]],dtype=torch.float32)
        #     transform_final = torch.diag(torch.tensor([scale_factor] * 3 + [1])) @ torch.cat([transform_matrix,self.padding],dim=0)
        #     out_path_ = os.path.join(out_path,"out_put.ply")
        #     read_bin_output_ply(in_path,out_path_)
        #     # point_cloud = o3d.io.read_point_cloud(out_path_)
        #     points = initialize_points(out_path_,transform_final.numpy())
        #    # poses = align_rectangle_with_axes(poses)
        #     # poses = rot_matrix_angle(poses,angle=-np.pi/30)
        #     self.center,self.scene_scale_t, maxz, minz = get_center_and_scale(out_path_)
        #     self.padding = self.padding.repeat(poses.shape[0],1,1)
        #     print(f" transform_final = {transform_final} ")
        #     self.center = np.array([0,0,self.center[2]])
        #     poses_ = torch.cat([poses,self.padding],dim=1)
        #     # poses_ = torch.matmul(torch.tensor(self.rotation_matrix,dtype=torch.float32),poses_)  # rotation
        #     # points = torch.matmul(torch.tensor(self.rotation_matrix,dtype=torch.float64),torch.tensor(np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)).unsqueeze(-1)) 
        #     points -= self.center
        #     # points *= 1./self.scene_scale_t
        #     poses_[:,:3,3] -= torch.tensor(self.center,dtype=torch.float32)  #transform
        #     # poses_[:,:3,3] *= torch.tensor(1./self.scene_scale_t,dtype=torch.float32)  #scale
        #     poses = poses_[:,:3,:]
        #     print("!!!!!!!!!!!!no point clouds")
        #print(points.shape)

        
        poses = rot_matrix_angle(poses,angle=1.0*np.pi*self.config.angle_rot)
        #visualize_pointcloud_and_cameras(points[np.random.choice(points.shape[0],10000),:],poses[:,:3,3].cpu().numpy())
        visualize_pointcloud_and_cameras(np.array([[0.0,0.0,0.0]]),poses[:,:3,3].cpu().numpy())
        # poses[:, :3, 3] *= self.config.scale_factor
        # poses[:, 2, 3] -= 0.32
        visualize_camera_poses_plotly(poses)
        
        # meta_new = meta.copy()
        # del meta_new['frames']
        # frames_new = []
        # for i in range(poses.shape[0]):
        #     fname_new = os.path.join('images',image_filenames[i])
        #     pose_new = (torch.cat([poses[i], torch.tensor([[0,0,0,1]])], dim=0)).numpy()
        #     frame_new = {
        #         'file_path': os.path.join(fname_new.split('/')[-2],fname_new.split('/')[-1]),
        #         'transform_matrix': pose_new.tolist(),
        #     }
        #     frames_new.append(frame_new)
        # meta_new['frames'] = frames_new
        # meta_new['aabb'] = (torch.tensor(
        #         [[-self.config.scene_scale, -self.config.scene_scale, minz], [self.config.scene_scale, self.config.scene_scale, maxz]], dtype=torch.float32
        #     )).tolist()
        # import json
        # with open(os.path.join(self.config.data, 'transforms_new.json'), 'w') as f:
        #     json.dump(meta_new,f,indent=2)
        
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        print("minz",minz)
        print("maxz",maxz)
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, minz], [aabb_scale, aabb_scale, maxz]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:

            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            },
        )
        return dataparser_outputs
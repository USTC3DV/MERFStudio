
from __future__ import annotations

import gc
import itertools
import math
import os
import types
from dataclasses import dataclass, field
from os import path
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import nerfacc
import numpy as np
import scipy
import skimage.measure
import torch
import tqdm
import yaml
from jaxtyping import Float, Int, Shaped
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from skimage.metrics import structural_similarity
from torch import Tensor, nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from merf.coord import contract, stepsize_in_squash
from merf.grid_utils import (WORLD_MAX, WORLD_MIN, calculate_grid_config,
                             grid_to_world, world_to_grid)
from merf.merf_field import (NUM_CHANNELS, MERFactoField, MERFContraction,
                             MERFViewEncoding)
from merf.merf_model import FeatureRenderer, MERFModel, MERFModelConfig
from merf.quantize import map_quantize_tuple
from merf.stepfun_torch import max_dilate_weights, sample_intervals
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (Encoding, HashEncoding,
                                                   NeRFEncoding, SHEncoding)
from nerfstudio.field_components.field_heads import (FieldHeadNames,
                                                     PredNormalsFieldHead,
                                                     SemanticFieldHead,
                                                     TransientDensityFieldHead,
                                                     TransientRGBFieldHead,
                                                     UncertaintyFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (SceneContraction,
                                                             SpatialDistortion)
from nerfstudio.fields.base_field import Field
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss, distortion_loss, interlevel_loss, orientation_loss,
    pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.model_components.ray_samplers import (
    PDFSampler, ProposalNetworkSampler, Sampler,
    UniformLinDispPiecewiseSampler, UniformSampler)
from nerfstudio.model_components.renderers import (BACKGROUND_COLOR_OVERRIDE,
                                                   AccumulationRenderer,
                                                   BackgroundColor,
                                                   DepthRenderer,
                                                   NormalsRenderer,
                                                   RGBRenderer)
from nerfstudio.model_components.scene_colliders import (AABBBoxCollider,
                                                         NearFarCollider)
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors
from merf.loss.s3im import S3IM

def skimage_ssim(image, rgb):
    # Scikit implementation used in PointNeRF
    values = [
        structural_similarity(gt, img, win_size=11, multichannel=True, channel_axis=2, data_range=1.0)
        for gt, img in zip(image.cpu().permute(0, 2, 3, 1).numpy(), rgb.cpu().permute(0, 2, 3, 1).numpy())
    ]
    return sum(values) / len(values)


from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from merf.baking.baking_config import BakingConfig
from merf.baking.baking_utils import (
    as_mib, get_atlas_block_size, parallel_write_images,
    reshape_into_3d_atlas_and_compute_indirection_grid, save_8bit_png,
    save_json)
from merf.robust_loss_pytorch import lossfun
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


def constructor_merf_model_config(loader, node):

    return MERFModelConfig()


def contains_infinity(lst):
    return any(x == float('inf') or x == float('-inf') for x in lst)


def get_chunk_idx(points, chunks_boundary):

    if points.dim == 2:
        sh = (points.shape,)
    else:
        sh = (points.shape[0],points.shape[1])

    points = points.view(-1, 3)  
    
    N, _ = points.size()
    num_lods = len(chunks_boundary)

    lod_indices = torch.full((N, num_lods), -1, dtype=torch.int32).to(points.device)

    for lod_idx, lod in enumerate(chunks_boundary):
        for chunk_idx, boundary in enumerate(lod):
            x_min, y_min, x_max, y_max = boundary

            boundary_tensor = torch.tensor([x_min, y_min, x_max, y_max]).view(1, 4).expand(N, 4).to(points.device)

            mask_min = (points[:, :2] >= boundary_tensor[:, :2]) | (boundary_tensor[:, :2] == float('-inf'))
            mask_max = (points[:, :2] <= boundary_tensor[:, 2:]) | (boundary_tensor[:, 2:] == float('inf'))
            mask = mask_min & mask_max
            mask = mask.all(dim=-1)

            lod_indices[mask, lod_idx] = chunk_idx




    return lod_indices.view(sh+(num_lods,))


def get_chunk_idx_certain(points, chunks_boundary,chunk_id):

    if points.dim == 2:
        sh = (points.shape,)
    else:
        sh = (points.shape[0],points.shape[1])

    points = points.view(-1, 3)  
    
    N, _ = points.size()
    num_lods = len(chunks_boundary)

    lod_indices = torch.full((N, num_lods), -1, dtype=torch.int32).to(points.device)

    for lod_idx, lod in enumerate(chunks_boundary):
        for chunk_idx, boundary in enumerate(lod):
            if chunk_idx != chunk_id:
                continue
            else:
                x_min, y_min, x_max, y_max = boundary

                boundary_tensor = torch.tensor([x_min, y_min, x_max, y_max]).view(1, 4).expand(N, 4).to(points.device)

                mask_min = (points[:, :2] >= boundary_tensor[:, :2]) | (boundary_tensor[:, :2] == float('-inf'))
                mask_max = (points[:, :2] <= boundary_tensor[:, 2:]) | (boundary_tensor[:, 2:] == float('inf'))
                mask = mask_min & mask_max
                mask = mask.all(dim=-1)

                lod_indices[mask, lod_idx] = chunk_idx




    return lod_indices.view(sh+(num_lods,))

def get_chunk_boundary_and_transform(num_chunks:List[int]=[1,4],global_boundary=None,num_xy:List[List[int]]=None) -> Tuple(List[List[List[float]]],List[List[Tensor]]):
    """
    initialization block representation
    input: num chunks
    """
    chunks_boundary = []
    chunks_transform = []
    
    for lod in range(len(num_chunks)):
        sub_chunks_boundary = []
        sub_chunks_transform = []
        
        # side length per chunk in lod

        
        for i in range(num_xy[lod][0]):
            for j in range(num_xy[lod][1]):


                side_length_x = (global_boundary[0][1] - global_boundary[0][0]) / num_xy[lod][0]  #  [-1, 1] length is 2
                side_length_y = (global_boundary[1][1] - global_boundary[1][0]) / num_xy[lod][1]  #  [-1, 1] length is 2
                x_min = (global_boundary[0][0] + i * side_length_x) if i > 0 else float('-inf')
                x_max =  (global_boundary[0][0] + (i+1) * side_length_x) if i < int(num_xy[lod][0]) - 1 else float('inf')
                y_min = global_boundary[1][0] + j * side_length_y if j > 0 else float('-inf')
                y_max = global_boundary[1][0] + (j+1) * side_length_y if j < int(num_xy[lod][1]) - 1 else float('inf')
                
                real_x_min = global_boundary[0][0] + i * side_length_x
                real_x_max = global_boundary[0][0] + (i+1) * side_length_x
                real_y_min = global_boundary[1][0] + j * side_length_y
                real_y_max = global_boundary[1][0] + (j+1) * side_length_y 

                boundary  = [x_min, y_min,x_max,y_max]
                   
                scale_factor = 2.0/side_length_y if side_length_x < side_length_y else 2.0/side_length_x
                translation_x = (real_x_min + real_x_max) / 2 
                translation_y = (real_y_min + real_y_max) / 2 
                
                sub_chunks_boundary.append(boundary)


                # # Calculate the scale factor and translation vector
                # scale_factor = 2 / side_length  # The ratio of the side length of the sub-chunk to the main chunk
                # translation_x = (real_x_min + real_x_max) / 2  # The x-coordinate of the center of the sub-chunk
                # translation_y = (real_y_min + real_y_max) / 2  # The y-coordinate of the center of the sub-chunk
                
                # Construct the 3D transformation matrix; the z-coordinate remains unchanged
                transform = torch.tensor([
                    [scale_factor, 0, 0, -1.0*scale_factor*translation_x],
                    [0, scale_factor, 0, -1.0*scale_factor*translation_y],
                    [0, 0, scale_factor, 0.0],
                    [0, 0, 0, 1]
                ])
                
                # Append the transformation matrix to the list
                sub_chunks_transform.append(transform)
                
        chunks_boundary.append(sub_chunks_boundary)
        chunks_transform.append(sub_chunks_transform)
    
    return chunks_boundary, chunks_transform
        




@dataclass
class BlockModelConfig(ModelConfig):
    """Nerfacto Model Config"""
    
    _target: Type = field(default_factory=lambda: BlockModel)
    save_chunk_config_path: Path = Path("outputs/chunk_config.yml")
    """Path for saving chunk config"""
    near_plane: float = 0.02
    """How far along the ray to start sampling."""
    far_plane: float = 100.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "random"
    """Whether to randomize the background color."""
    proposal_update_every: int = 1
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (128, 64)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 32
    """Number of samples per ray for the nerf network."""
    use_proposal_weight_anneal:bool = True
    
    global_boundary: List[List[float]] = field(default_factory=lambda:[[-0.95,0.95],[-0.95,0.95]])
    
    """  x_min y min z min x max y max zmax not real boundary just core reigion"""
    """""""""""""""""
    BLOCK ARUGUMENTS
    """""""""""""""""
    num_chunks: List[int] =  field(default_factory=lambda:[1])
    num_xy: List[List[int]] =  field(default_factory=lambda:[[1,1]])
    chunks_model_config: List[List[ModelConfig]] = None
    """Model config of chunk """
    chunks_boundary: List[List[List[float]]] = field(default_factory=list)
    """The range represented by each chunk ...... [...,0] x_min [...,1] y_min [...,2]x_max [...,3]y_max could be float("inf") """
    chunks_optimized: List[List[bool]] = field(default_factory=list)
    """Whether each chunk should be optimized"""
    chunks_transform: List[List[Tensor]] = field(default_factory=list)
    """Transform glocal coordiante to  each chunk's local coordinate, if not represent unbounded scene , scene box after transformed should be [-1,1]"""
    chunks_bounded: List[List[bool]] = field(default_factory=list)
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    num_proposal_iterations:int = 2
    alpha_threshold_param:Tuple[int, int, float, float] = field(
      default_factory=lambda: (10000, 3000, 5e-4, 1e-2, 20000)
    ) 
    appearance_dim: int = 16
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding for training."""
    multi_scale_training: bool = True
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 1e-2
    """Distortion loss multiplier."""
    sparsity_loss_mult: float = 0.1
    """Sparsity loss multiplier."""
    regularize_loss_mult: float = 0.0
    specular_mult: float = 0.0
    specular_reg_mult: float = 0.0
    num_random_samples : int = 2**12
    spatial_distortion : bool = True
    rgb_loss_method:Literal["mse", "charb"] = "charb"
    bounded_scene: bool = True
    accumulation_loss_mult: float = 0.05
    gaussian_samples_in_prop:bool = True
    s3im_loss_mult: float = 1.0
    charb_eps : float = 5e-3
    patch_h: int = 200
    patch_w: int = 200
    unique_deferred: bool = False
    global_deferred: bool = True
    
    def __post_init__(self):
        # yaml.add_constructor("tag:yaml.org,2002:python/object:merf.merf_model.MERFModelConfig", constructor_merf_model_config, Loader=yaml.Loader)
        self._target = BlockModel
        self.save_chunk_config_path.parent.mkdir(parents=True,exist_ok=True)
        self.init_chunk_config()
        # else:
        #     self.load_chunk_config()
            
        assert self.chunks_boundary is not None
        self.chunks_bounded = []
            
        for lod in self.chunks_boundary:
            chunk_bounded = []
            for chunk in lod:
                if contains_infinity(chunk):
                    chunk_bounded.append(False)
                else:
                    chunk_bounded.append(True)
            self.chunks_bounded.append(chunk_bounded)
            
        print(self.chunks_boundary)
        import numpy as np
        import json
        print(self.chunks_transform)
        print(self.chunks_transform[0])
        with open('data.json', 'w') as json_file:
            json.dump([self.chunks_transform[0][i].tolist() for i in range(len(self.chunks_transform[0]))], json_file)
        print(self.chunks_bounded)
            
    def init_chunk_config(self):
        
        block_config = {"num_chunks":self.num_chunks}
        self.num_lods = len(self.num_chunks)
        self.chunks_boundary,self.chunks_transform = get_chunk_boundary_and_transform(self.num_chunks,global_boundary=self.global_boundary,num_xy=self.num_xy)
        self.chunks_optimized = [[True for i in range(num_chunk)] for num_chunk in self.num_chunks]
        if self.chunks_model_config is None:
            self.chunks_model_config = []
            for num_chunk in self.num_chunks:
                self.chunks_model_config.append([MERFModelConfig(use_appearance_embedding=self.use_appearance_embedding) for i in range(num_chunk)])
                
        block_config.update({"chunks_boundary":self.chunks_boundary,
                             "chunks_transform":[[chunk_transform.tolist() for chunk_transform in lod] for lod in self.chunks_transform],
                             "chunks_model_config":self.chunks_model_config,      
        })
        
        # self.block_config.write_text(yaml.dump(self), "utf8")
        
        
    
    
    # def load_chunk_config(self):

    #     if not self.save_chunk_config_path.exists():
    #         raise FileNotFoundError(f"{self.save_chunk_config_path} does not exist!")
    #     block_config =  yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
    #     with open(self.save_chunk_config_path, 'r') as file:
    #         block_config = yaml.load(file, Loader=yaml.FullLoader)
            
    #     self.num_chunks = block_config.get("num_chunks", self.num_chunks)
    #     self.num_lods = len(self.num_chunks)
    #     self.chunks_boundary = block_config.get("chunks_boundary", self.chunks_boundary)
    #     chunks_transform_list = block_config.get("chunks_transform", self.chunks_transform)
    #     self.chunks_transform  = [[torch.tensor(chunk_transform) for chunk_transform in lod] for lod in chunks_transform_list] #### TODO:DEVICE
    #     self.chunks_model_config = block_config.get("chunks_model_config", self.chunks_model_config)
    
    
def sparsity_loss(random_positions, random_viewdirs, density, voxel_size):
  step_size = stepsize_in_squash(
      random_positions, random_viewdirs, voxel_size
  )
  return (1.0 - torch.exp(-step_size.unsqueeze(-1) * density)).mean()

class CharbonnierLoss(nn.Module):
    def __init__(self, charb_padding=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = charb_padding

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)



class BlockMERFProposalNetworkSampler(Sampler):
    """Sampler that uses a proposal network to generate samples.

    Args:
        num_proposal_samples_per_ray: Number of samples to generate per ray for each proposal step.
        num_nerf_samples_per_ray: Number of samples to generate per ray for the NERF model.
        num_proposal_network_iterations: Number of proposal network iterations to run.
        single_jitter: Use a same random jitter for all samples along a ray.
        update_sched: A function that takes the iteration number of steps between updates.
        initial_sampler: Sampler to use for the first iteration. Uses UniformLinDispPiecewise if not set.
    """

    def __init__(
        self,
        num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 3,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
        initial_sampler: Optional[Sampler] = None,
    ) -> None:
        super().__init__()
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")
        
        

        
        # samplers
        if initial_sampler is None:
            self.initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            self.initial_sampler = initial_sampler
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        density_fns: Optional[List[Callable]] = None,
        chunks_transform: Optional[List[List[Tensor]]] = None,
        chunks_boundary: Optional[List[List[List]]] = None,
        num_lods: int = None,
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None



        n = self.num_proposal_network_iterations
        weights_list = [[None for _ in range(n)] for _ in range(num_lods)]
        ray_samples_list = [[None for _ in range(n)] for _ in range(num_lods)]
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)

            
            if is_prop:
                samples_chunk_idx = get_chunk_idx(ray_samples.frustums.get_gaussian_samples(), chunks_boundary)
                density = torch.zeros(ray_samples.shape).to(ray_samples.deltas)[...,None]
                for lod in range(num_lods):
                    for j,chunk_density_fn in enumerate(density_fns[lod]): 
                        chunk_mask = (samples_chunk_idx[...,lod] == j)
                        sub_samples = ray_samples[chunk_mask]
                        if sub_samples.shape[0] > 0:
                            if updated:
                                # always update on the first step or the inf check in grad scaling crashes
                                # density = density_fns[i_level](ray_samples.frustums.get_positions())
                                sub_density,_ = chunk_density_fn[i_level](sub_samples,samples_transform=chunks_transform[lod][j])
                            else:
                                with torch.no_grad():
                                    # density = density_fns[i_level](ray_samples.frustums.get_positions())
                                    sub_density,_ = chunk_density_fn[i_level](sub_samples,samples_transform=chunks_transform[lod][j])
                            density[chunk_mask] = sub_density

                    weights = ray_samples.get_weights(density)
                    weights_list[lod][i_level] = weights  # (num_rays, num_samples)
                    ray_samples_list[lod][i_level] = ray_samples
                
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
    
    
class BlockModel(Model):
    
    """block model
    
    Args:
        config: merf configuration to instantiate model
    """
    config : BlockModelConfig
    

    def populate_modules(self):
        """
        Set models per chunk.
        """
        """Set the fields and modules."""
        # torch.autograd.set_detect_anomaly(True)
        super().populate_modules()
        
        
        
        self._model = torch.nn.ModuleList()
        for lod_idx in range(self.config.num_lods):
            chunks = torch.nn.ModuleList()
            for i in range(self.config.num_chunks[lod_idx]):
                self.config.chunks_model_config[lod_idx][i].appearance_dim = self.config.appearance_dim
                chunk_model = self.config.chunks_model_config[lod_idx][i].setup(
                    scene_box=self.scene_box,
                    num_train_data=1,
                    metadata=self.kwargs['metadata'],
                    device=self.kwargs['device'],
                    grad_scaler=self.kwargs['grad_scaler'],
                )
                chunks.append(chunk_model)
            self._model.append(chunks)
        if self.config.use_appearance_embedding:
            self.appearance_embedding = Embedding(self.num_train_data, self.config.appearance_dim)
        else:
            self.appearance_embedding = None
        self.direction_encoding  = MERFViewEncoding(in_dim=3,deg_enc=4,include_input=True)
        self.chunks_transform = [[chunks.to(self.kwargs['device']) for chunks in lod]for lod in self.config.chunks_transform]
                # Collider
                
        for lod_idx in range(self.config.num_lods):
            chunks = torch.nn.ModuleList()
            for i in range(self.config.num_chunks[lod_idx]):
                if not self.config.chunks_optimized[lod_idx][i]:
                    
                    for param in self._model[lod_idx][i].parameters():
                        param.requires_grad = False
                        
                       
        if self.config.bounded_scene:
            self.collider = AABBBoxCollider(scene_box=self.scene_box,near_plane=self.config.near_plane)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        
        
        # initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        initial_sampler = None
        self.renderer_features = FeatureRenderer(background_color=self.config.background_color)
        # self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()
        # for i in range(self.config.num_lods):
        #     for j,chunk_density_fn in enumerate(density_fns[i]): 
        #         chunk_mask = (samples_chunk_idx[i] == j)
        #         sub_samples = ray_samples(chunk_mask)
        # shaders
        self.normals_shader = NormalsShader()
        
                # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        for lod_idx in range(self.config.num_lods):
            self.proposal_networks.append( torch.nn.ModuleList(self._model[lod_idx][i].proposal_networks for i in range(self.config.num_chunks[lod_idx])))
      
        self.density_fns.extend([[[network.get_density for network in chunks_network] for chunks_network in lod_network ] for lod_network in self.proposal_networks])
        if self.config.global_deferred:
            self.global_deferred_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim() + 7,
                num_layers=4,
                layer_width=128,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="torch",
            ).to(self.kwargs['device'])
            
        self.proposal_sampler = BlockMERFProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # losses
        if self.config.rgb_loss_method == "charb":
            self.rgb_loss = CharbonnierLoss(charb_padding=1e-3)
        elif self.config.rgb_loss_method == "mse":
            self.rgb_loss = MSELoss()
        else:
            raise NotImplementedError("Not implemented rgb loss type")
        self.rgb_loss = CharbonnierLoss(charb_padding=self.config.charb_eps)
        self.acc_loss = torch.nn.BCELoss()
        self.s3im_loss = S3IM(patch_height=self.config.patch_h,patch_width=self.config.patch_w)
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        
    def get_param_groups(self) -> Dict[str, List[Parameter]]:

        param_group = {}
        param_group["proposal_networks"] = []
        param_group["deferred_mlp"] = []
        param_group["fields"] = []
        if self.config.global_deferred:
            param_group["deferred_mlp"] = list(self.global_deferred_mlp.parameters())
        for lod_idx in range(self.config.num_lods):
            for chunk_idx in range(self.config.num_chunks[lod_idx]):
                params = self._model[lod_idx][chunk_idx].get_param_groups()
                param_group["proposal_networks"].extend(params["proposal_networks"])
                param_group["deferred_mlp"].extend(params["deferred_mlp"])
                param_group["fields"].extend(params["fields"])
        
        if self.config.use_appearance_embedding:
            param_group["apperance"] = list(self.appearance_embedding.parameters())
        return param_group
    
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        def set_alpha_culling_value(step):
                
            def log_lerp(t, v0, v1):
                """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
                if v0 <= 0 or v1 <= 0:
                    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
                lv0 = np.log(v0)
                lv1 = np.log(v1)
                return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)
            
            if step < self.config.alpha_threshold_param[0]:
                alpha_thres = 0.0
            elif step > self.config.alpha_threshold_param[1]:
                alpha_thres = self.config.alpha_threshold_param[3]
            elif step > self.config.alpha_threshold_param[4]: 
                t = (step - self.config.alpha_threshold_param[4]) / (self.config.alpha_threshold_param[1] - self.config.alpha_threshold_param[4])
                alpha_thres = log_lerp(t, self.config.alpha_threshold_param[2], self.config.alpha_threshold_param[3])
            else:
                t =  (step - self.config.alpha_threshold_param[0]) / (self.config.alpha_threshold_param[4] - self.config.alpha_threshold_param[0])
                alpha_thres = t * self.config.alpha_threshold_param[2]
            for lod_idx in range(self.config.num_lods):
                for chunk_idx in range(self.config.num_chunks[lod_idx]):
                    self._model[lod_idx][chunk_idx].field.set_alpha_threshold(alpha_thres)
        
        callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_alpha_culling_value,
                )
            )
                
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            # N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = 1.0

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)
                
                
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks
    
    
    def get_baking_outputs_for_camera_ray_bundle(self,ray_bundle:RayBundle,lod_idx=None,chunk_idx=None):
        ray_samples: RaySamples
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns, chunks_transform=self.chunks_transform,chunks_boundary=self.config.chunks_boundary,num_lods=self.config.num_lods)
        # ray_samples, weights_list, ray_samples_list = self.merf_sampler(ray_bundle, density_fns=self.density_fns)
        # ray samples in actual world space 

        chunk_weights_list = []
        rgb_list = []
        depth_list = []
        accumulation_list = []
        
        samples_chunk_idx = get_chunk_idx(ray_samples.frustums.get_gaussian_samples(),self.config.chunks_boundary,chunk_idx)
        # dir_encode = self.direction_encoding(ray_bundle.directions).to(self.device)
        
        # for i in range(self.config.num_lods):
        density = torch.zeros(ray_samples.shape+(1,),device=self.device, dtype=torch.float32)
        # features = torch.zeros(ray_samples.shape+(7,),device=self.device, dtype=torch.float32)
        #     chunk_rgb = torch.zeros((ray_samples.shape[0],self.config.num_chunks[i],3),device=self.device, dtype=torch.float32)
        #     chunk_weights = torch.zeros((ray_samples.shape[0],self.config.num_chunks[i],1),dtype=torch.float32,device=self.device)
        #     for j,chunk_model in enumerate(self._model[i]): 
        
        chunk_model = self._model[lod_idx][chunk_idx]
        chunk_mask = (samples_chunk_idx[...,lod_idx] == chunk_idx)
        
        
        sub_samples = ray_samples[chunk_mask]
        if sub_samples.shape[0] > 0:
            sub_field_outputs = chunk_model.field.get_outputs(sub_samples,sub_samples.frustums.directions,samples_transform=self.chunks_transform[lod_idx][chunk_idx],appearance_embedding=self.appearance_embedding)
            density[chunk_mask] = sub_field_outputs[FieldHeadNames.DENSITY]
            for j,chunk_model_j in enumerate(self._model[lod_idx]):
                if j != chunk_idx:
                    chunk_mask_j = (samples_chunk_idx[...,lod_idx] == j)
                    sub_samples_j = ray_samples[chunk_mask_j]
                    
                    if sub_samples_j.shape[0] > 0:
                        sub_field_outputs = chunk_model_j.field.get_outputs(sub_samples_j,sub_samples_j.frustums.directions,samples_transform=self.chunks_transform[lod_idx][j],appearance_embedding=self.appearance_embedding)
                        density[chunk_mask_j] = sub_field_outputs[FieldHeadNames.DENSITY]

                
        # simulate volume rendering inside chunk
        # chunk_samples_density = torch.where(chunk_mask.unsqueeze(-1).expand_as(density),density,torch.zeros_like(density))
        # chunk_samples_features = torch.where(chunk_mask.unsqueeze(-1).expand_as(features),features,torch.zeros_like(features))
        chunk_samples_weights = ray_samples.get_weights(density)
        # chunk_ray_features = self.renderer_features(features=chunk_samples_features, weights=chunk_samples_weights) # BxC
        # chunk_ray_specular = chunk_model.deferred_mlp(torch.cat([chunk_ray_features,dir_encode],dim=-1))
        # chunk_ray_rgb = chunk_ray_features[...,0:3] + chunk_ray_specular
        # chunk_rgb[:,j,:] = chunk_ray_rgb # Bx3
              


        # weights = ray_samples.get_weights(chunk_samples_density)
        outputs = {
            "density": density[chunk_mask],
            "weights": chunk_samples_weights[chunk_mask],
            "ray_samples": sub_samples,
        }

        return outputs
    
    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns, chunks_transform=self.chunks_transform,chunks_boundary=self.config.chunks_boundary,num_lods=self.config.num_lods)
        # print(ray_samples.camera_indices)
        chunk_weights_list = []
        rgb_list = []
        
        depth_list = []
        accumulation_list = []
        if self.config.global_deferred:
            rgb_global_list = []
        loss_specular_sum = 0.
        loss_specular_min = 0.
        # specular_list = [] # 2-level lod/chunk
        # feature_list = []
        # dir_list = []
        samples_pos = ray_samples.frustums.get_gaussian_samples()
        
        samples_chunk_idx = get_chunk_idx(samples_pos,self.config.chunks_boundary)
        dir_encode = self.direction_encoding(ray_bundle.directions).to(samples_pos.device)
        
        for i in range(self.config.num_lods):
            density = torch.zeros(ray_samples.shape+(1,),device=samples_pos.device, dtype=torch.float32)
            features = torch.zeros(ray_samples.shape+(7,),device=samples_pos.device, dtype=torch.float32)
            chunk_rgb = torch.zeros((ray_samples.shape[0],self.config.num_chunks[i],3),device=samples_pos.device, dtype=torch.float32)
            chunk_rgb_global = torch.zeros((ray_samples.shape[0],self.config.num_chunks[i],3),device=samples_pos.device, dtype=torch.float32)
            chunk_weights = torch.zeros((ray_samples.shape[0],self.config.num_chunks[i],1),dtype=torch.float32,device=samples_pos.device)
            # specular_chunk_list = []
            for j,chunk_model in enumerate(self._model[i]): 
                chunk_mask = (samples_chunk_idx[...,i] == j)
                sub_samples = ray_samples[chunk_mask]
                if sub_samples.shape[0] > 0:
                    sub_field_outputs = chunk_model.field.get_outputs(sub_samples,sub_samples.frustums.directions,samples_transform=self.chunks_transform[i][j],appearance_embedding=self.appearance_embedding)
                    density[chunk_mask] = sub_field_outputs[FieldHeadNames.DENSITY]
                    features[chunk_mask] = sub_field_outputs['features']
                
                # simulate volume rendering inside chunk
                chunk_samples_density = torch.where(chunk_mask.unsqueeze(-1).expand_as(density),density,torch.zeros_like(density))
                chunk_samples_features = torch.where(chunk_mask.unsqueeze(-1).expand_as(features),features,torch.zeros_like(features))
                chunk_samples_weights = ray_samples.get_weights(chunk_samples_density)
                chunk_ray_features = self.renderer_features(features=chunk_samples_features, weights=chunk_samples_weights) # BxC
                
                if self.config.unique_deferred:
                    chunk_ray_specular = self._model[i][0].deferred_mlp(torch.cat([chunk_ray_features,dir_encode],dim=-1))
                else:
                    chunk_ray_specular = chunk_model.deferred_mlp(torch.cat([chunk_ray_features,dir_encode],dim=-1))

                chunk_ray_rgb = chunk_ray_features[...,0:3] + chunk_ray_specular
                chunk_rgb[:,j,:] = chunk_ray_rgb # Bx3
                if self.config.global_deferred:
                    chunk_ray_specular_global = self.global_deferred_mlp(torch.cat([chunk_ray_features,dir_encode],dim=-1))
                    chunk_ray_rgb_global = chunk_ray_features[...,0:3] + chunk_ray_specular_global
                    #print(chunk_ray_rgb_global.shape)
                    chunk_rgb_global[:,j,:] = chunk_ray_rgb_global # Bx3
                if self.training and self.config.specular_mult > 0:
                    dir_ori = ray_bundle.directions.repeat(10,1).to(samples_pos.device)
                    chunk_ray_features = chunk_ray_features.repeat(10,1)
                    rgb_specular_ori = chunk_ray_specular.repeat(10,1)
                    # eps_dir = 0.3*torch.randn_like(ray_bundle.directions)
                    eps_dir = 0.3*torch.randn_like(dir_ori)
                    # eps_dir = eps_dir + ray_bundle.directions
                    eps_dir = eps_dir + dir_ori
                    eps_dir /= eps_dir.norm(dim=-1, keepdim=True)
                    dir_encode_ = self.direction_encoding(eps_dir)

                    rgb_specular_ = chunk_model.deferred_mlp(torch.cat([chunk_ray_features,dir_encode_],dim=-1))
                    # print(rgb_specular_.shape)
                    # loss_specular = (torch.cosine_similarity(eps_dir, ray_bundle.directions, dim=-1).unsqueeze(-1) * (rgb_specular_-rgb_specular)**2).mean()
                    loss_specular = (torch.cosine_similarity(eps_dir, dir_ori, dim=-1).unsqueeze(-1) * (rgb_specular_-rgb_specular_ori)**2).mean()
                    loss_specular_sum += loss_specular
                    #if self.training and self.config.reg_specular_mult > 0.0:
                    #    loss_specular += self.config.reg_specular_mult * torch.abs(rgb_specular).mean()
                    
            # specular_list.append(specular_chunk_list)
             # simulate volume rendering across chunk and record weights

            weights_real = ray_samples.get_weights(density)
            weights_list[i].append(weights_real)  # (num_rays, num_samples)
            ray_samples_list[i].append(ray_samples)
            # print("Max index:", samples_chunk_idx[..., i].max())
            # print("Min index:", samples_chunk_idx[..., i].min())
            # print("shape of samples_chunk",samples_chunk_idx.shape)
            # print("shape of chunk_weights",chunk_weights.shape)
            # print("shape of weights_real",weights_real.shape)
            chunk_weights = chunk_weights.scatter_add(1,samples_chunk_idx[...,i].unsqueeze(-1).to(torch.int64),weights_real) 
            chunk_weights_list.append(chunk_weights)
            rgb = (chunk_weights * chunk_rgb).sum(dim=-2)

            depth = self.renderer_depth(weights=weights_real, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights_real)
            rgb_list.append(rgb)
            if self.config.global_deferred:
                #print("chunk_weights")
                #print(chunk_weights.shape)
                #print(chunk_rgb.shape)
               # print(chunk_rgb_global.shape)
                rgb_global = (chunk_weights * chunk_rgb_global).sum(dim=-2)
               # print(rgb_global.shape)
                rgb_global_list.append(rgb_global)
            depth_list.append(depth)
            accumulation_list.append(accumulation)

        outputs = {
            "rgb": rgb_list,
            "accumulation": accumulation_list,
            "depth": depth_list,
            "rgb_global": rgb_global_list if self.config.global_deferred else None,
            "loss_specular": loss_specular_sum
        }

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
            outputs["chunk_weights_list"] = chunk_weights_list

        for i in range(self.config.num_proposal_iterations):
            for lod in range(self.config.num_lods):
                outputs[f"prop_depth_{lod}_{i}"] = self.renderer_depth(weights=weights_list[lod][i], ray_samples=ray_samples_list[lod][i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        if batch['image'].shape[-1] == 4:
            batch['image'] = batch['image'][...,0:3]
        metrics_dict = {}
        image = batch["image"]
        for lod in range(self.config.num_lods):
            # print(outputs["rgb"][lod].shape)
            # print(outputs["rgb_global"][lod].shape)
            # print(image.shape)
            # metrics_dict[f"psnr_{lod}"] = self.psnr(outputs["rgb"][lod], image.to(outputs["rgb"][lod].device))
            
            metrics_dict[f"psnr_global_{lod}"] = self.psnr(outputs["rgb_global"][lod], image.to(outputs["rgb_global"][lod].device)) if self.config.global_deferred else 0.0
            if self.training:
                metrics_dict[f"distortion_{lod}"] = distortion_loss(outputs["weights_list"][lod], outputs["ray_samples_list"][lod])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if batch['image'].shape[-1] == 4:
            batch['image'] = batch['image'][...,0:3]
        loss_dict = {}
        image = batch["image"]
        loss_dict[f"specular"] = self.config.specular_mult * outputs['loss_specular']
        for lod in range(self.config.num_lods):
            loss_dict[f"rgb_loss_{lod}"] = self.rgb_loss(image.to(outputs["rgb"][lod].device), outputs["rgb"][lod])
            # print(outputs["rgb"][lod].shape)
            loss_dict[f"s3im_loss_{lod}"] = self.s3im_loss(outputs["rgb"][lod],image.to(outputs["rgb"][lod].device))
            #print(outputs["rgb"][lod].shape)
            loss_dict[f"rgb_global_loss_{lod}"] = self.rgb_loss(image.to(outputs["rgb_global"][lod].device), outputs["rgb_global"][lod]) if self.config.global_deferred else 0.0
            #print("outputs")
            #print(outputs["rgb_global"][lod].shape)
            #print("image")
            #print(image.shape)
            #loss_dict[f"s3im_global_loss_{lod}"] = self.s3im_loss(outputs["rgb_global"][lod],image.to(outputs["rgb_global"][lod].device)) if self.config.global_deferred else 0.0
            
        if self.training:
            for lod in range(self.config.num_lods):
                loss_dict[f"interlevel_loss_{lod}"] = self.config.interlevel_loss_mult * interlevel_loss(
                    outputs["weights_list"][lod], outputs["ray_samples_list"][lod]
                )

                assert metrics_dict is not None and f"distortion_{lod}" in metrics_dict
                loss_dict[f"distortion_loss_{lod}"] = self.config.distortion_loss_mult * metrics_dict[f"distortion_{lod}"]

            if self.config.sparsity_loss_mult > 0.0 :
                # sample points
                num_random_samples = self.config.num_random_samples 

                random_positions = -1.0 + 2.0 * torch.rand(num_random_samples, 3).to(self.device)

                # random_positions = WORLD_MIN + (WORLD_MAX - WORLD_MIN) * torch.rand(num_random_samples, 3).to(self.device)
                random_viewdirs = torch.normal(mean=0, std=1, size=(num_random_samples, 3)).to(self.device)
                random_viewdirs /= torch.norm(random_viewdirs, dim=-1, keepdim=True)
                # print(random_viewdirs.shape)
                # print(random_positions.shape)
                for lod in range(self.config.num_lods):
                    loss_dict[f"sparsity_loss_{lod}"] = 0.0
                    for chunk_model in self._model[lod]:
                        density = chunk_model.field(
                            None,
                            random_viewdirs,
                            random_positions
                        )[FieldHeadNames.DENSITY]
                        loss_dict[f"sparsity_loss_{lod}"] += self.config.sparsity_loss_mult * sparsity_loss(random_positions, random_viewdirs, density, chunk_model.voxel_size_to_use)
        
            if self.config.accumulation_loss_mult > 0.0:
                for lod in range(self.config.num_lods):
                    loss_dict[f"acc_loss_{lod}"] = self.config.accumulation_loss_mult * self.acc_loss(outputs["accumulation"][lod].clip(1e-5, 1.0 - 1e-5),torch.ones_like(outputs["accumulation"][lod]))
        
        return loss_dict

    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        if batch['image'].shape[-1] == 4:
            batch['image'] = batch['image'][...,0:3]
        image = batch["image"]
        images_dict = {}
        metrics_dict = {}
        for lod in range(self.config.num_lods):
            rgb = outputs[f"rgb_{lod}"]
            acc = colormaps.apply_colormap(outputs[f"accumulation_{lod}"])
            depth = colormaps.apply_depth_colormap(
                outputs[f"depth_{lod}"],
                accumulation=outputs[f"accumulation_{lod}"],
            )
            rgb = torch.clamp(rgb,0.0,1.0)
            combined_rgb = torch.cat([image.to(outputs[f"rgb_{lod}"].device), rgb], dim=1)
            combined_acc = torch.cat([acc], dim=1)
            combined_depth = torch.cat([depth], dim=1)

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
            
            psnr = self.psnr(image.to(outputs[f"rgb_{lod}"].device), rgb)
            ssim = self.ssim(image.to(outputs[f"rgb_{lod}"].device), rgb)
            lpips = self.lpips(image.to(outputs[f"rgb_{lod}"].device), rgb)

            # all of these metrics will be logged as scalars
            metrics_dict.update({f"psnr_{lod}": float(psnr.item()), f"ssim_{lod}": float(ssim), f"lpips_{lod}":float(lpips)}) # type: ignore
            # metrics_dict[f"lpips_{lod}"] = float(lpips)

            images_dict.update({f"img_{lod}": combined_rgb, f"accumulation_{lod}": combined_acc, f"depth_{lod}": combined_depth})

            for i in range(self.config.num_proposal_iterations):
                key = f"prop_depth_{lod}_{i}"
                prop_depth_i = colormaps.apply_depth_colormap(
                    outputs[key],
                    accumulation=outputs[f"accumulation_{lod}"],
                )
                images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
    
    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for lod in range(self.config.num_lods):
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                outputs = self.forward(ray_bundle=ray_bundle)
                for output_name, output in outputs.items():  # type: ignore
                    if output_name == "rgb" or output_name == "depth" or output_name == "accumulation" or output_name == "rgb_global":
                        outputs_lists[output_name].append(output[lod])
                    else:
                        outputs_lists[output_name].append(output)
                    if self.config.global_deferred and  output_name == "rgb_global":
                        outputs_lists[output_name].append(output[lod])
            outputs = {}
            
            for output_name, outputs_list in outputs_lists.items():
 
                if self.config.global_deferred:
                    if output_name == "rgb" or output_name == "depth" or output_name == "accumulation" or output_name == "rgb_global":
                        outputs[output_name+f"_{lod}"] = torch.cat(outputs_list).view(image_height, image_width, -1)   # type: ignore
                    elif output_name == "loss_specular":
                        outputs["loss_specular"] = outputs_list
                    else:
                        outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
                else:
                    if output_name == "rgb" or output_name == "depth" or output_name == "accumulation":
                            outputs[output_name+f"_{lod}"] = torch.cat(outputs_list).view(image_height, image_width, -1)   # type: ignore
                    elif output_name == "loss_specular":
                        outputs["loss_specular"] = outputs_list
                    else:
                        outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
        

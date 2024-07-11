from __future__ import annotations

import gc
import itertools
import os
import types
from dataclasses import dataclass, field
from os import path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import nerfacc
import numpy as np
import scipy
import skimage.measure
import torch
import tqdm
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
from merf.quantize import map_quantize_tuple, map_quantize_tuple_list
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
from merf.loss.s3im import S3IM

class CharbonnierLoss(nn.Module):
    def __init__(self, charb_padding=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = charb_padding

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)


class MERFProposalNetworkSampler(Sampler):
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
        num_proposal_network_iterations: int = 2,
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
    ) -> Tuple[RaySamples, List, List]:
        assert ray_bundle is not None
        assert density_fns is not None

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
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
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    # density = density_fns[i_level](ray_samples.frustums.get_positions())
                    density,_ = density_fns[i_level](ray_samples)
                else:
                    with torch.no_grad():
                        # density = density_fns[i_level](ray_samples.frustums.get_positions())
                        density,_ = density_fns[i_level](ray_samples)
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list

class MERFHashMLPDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 10,
        max_res: int = 1024,
        base_res: int = 32,
        log2_hashmap_size: int = 16,
        features_per_level: int = 2,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        gaussian_samples_in_prop: bool = False
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        self.gaussian_samples_in_prop = gaussian_samples_in_prop
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        implementation = "tcnn"
        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        implementation = "torch"
        if not self.use_linear:
            self.mlp_network = MLP(
                in_dim=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            # self.mlp_base = torch.nn.Sequential(self.encoding, self.mlp_network)
        else:
            self.linear = torch.nn.Linear(self.encoding.get_out_dim(), 1)

    def get_density(self, ray_samples: RaySamples, samples_transform:Tensor = None) -> Tuple[Tensor, None]:
        
        
        if self.spatial_distortion is not None:
            if self.gaussian_samples_in_prop:
                positions = ray_samples.frustums.get_gaussian_samples()
            else:
                positions = ray_samples.frustums.get_positions()
            if samples_transform is not None:
                positions =  torch.matmul(samples_transform,torch.cat([positions, torch.ones(*positions.shape[:-1], 1).to(positions)], dim=-1).T).T[:,0:3]
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            if self.gaussian_samples_in_prop:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_gaussian_samples(), self.aabb)
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_gaussian_samples(), self.aabb)
    
        # Make sure the tcnn gets inputs between 0 and 1.
        

        positions_flat = positions.view(-1, 3)
        

        if not self.use_linear:
            hash_features = self.encoding(positions_flat).to(positions)
            
            density_before_activation = (
                self.mlp_network(hash_features).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation - 1.0)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}



class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, sigmas, ray_dist):
        ctx.save_for_backward(ray_dist)
        return colors, sigmas, ray_dist

    @staticmethod
    def backward(ctx, grad_output_colors, grad_output_sigmas, grad_output_ray_dist):
        (ray_dist,) = ctx.saved_tensors
        scaling = torch.square(ray_dist).clamp(0, 1)
        return grad_output_colors * scaling, grad_output_sigmas * scaling, grad_output_ray_dist

    
# pylint: disable=attribute-defined-outside-init


def conical_frustum_to_gaussian_mean(
    origins: Float[Tensor, "*batch 3"],
    directions: Float[Tensor, "*batch 3"],
    starts: Float[Tensor, "*batch 1"],
    ends: Float[Tensor, "*batch 1"],
    radius: Float[Tensor, "*batch 1"],
):
    """Approximates conical frustums with a Gaussian distributions.

    Uses stable parameterization described in mip-NeRF publication.

    Args:
        origins: Origins of cones.
        directions: Direction (axis) of frustums.
        starts: Start of conical frustums.
        ends: End of conical frustums.
        radius: Radii of cone a distance of 1 from the origin.

    Returns:
        Gaussians: Approximation of conical frustums
    """
    mu = (starts + ends) / 2.0
    hw = (ends - starts) / 2.0 

    return origins + directions * (mu + (2.0 * mu * hw * hw) / (3.0 * mu * mu + hw * hw))

def get_gaussian_samples(self):
        
    """Calculates guassian approximation of conical frustum.

    Returns:
        Conical frustums approximated by gaussian distribution.
    """
    # Cone radius is set such that the square pixel_area matches the cone area.
    cone_radius = torch.sqrt(self.pixel_area) / 1.7724538509055159  # r = sqrt(pixel_area / pi)
    if self.offsets is not None:
        raise NotImplementedError()
    return conical_frustum_to_gaussian_mean(
        origins=self.origins,
        directions=self.directions,
        starts=self.starts,
        ends=self.ends,
        radius=cone_radius,
    )


def sparsity_loss(random_positions, random_viewdirs, density, voxel_size):
  step_size = stepsize_in_squash(
      random_positions, random_viewdirs, voxel_size
  )
  return 1.0 - torch.exp(-step_size.unsqueeze(-1) * density).mean()


class FeatureRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color: BackgroundColor = "random") -> None:
        super().__init__()
        self.background_color: BackgroundColor = background_color

    @classmethod
    def combine_rgb(
        cls,
        features: Float[Tensor, "*bs num_samples n"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_features = nerfacc.accumulate_along_rays(
                weights[..., 0], values=features, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_features = torch.sum(weights * features, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "last_sample":
            background_color = features[..., -1, 0:3]
        if background_color == "random":
            background_color = torch.rand_like(comp_features[Ellipsis,0:3]).to(features.device)
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color].to(features.device)

        assert isinstance(background_color, torch.Tensor)
        comp_diffuse_rgb = comp_features[Ellipsis,0:3] + background_color.to(weights.device) * (1.0 - accumulated_weight)
        comp_features[...,0:3] = comp_diffuse_rgb
        return comp_features

    def forward(
        self,
        features: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of rgb values.
        """

        if not self.training:
            features = torch.nan_to_num(features)
        features = self.combine_rgb(
            features, weights, background_color=self.background_color, ray_indices=ray_indices, num_rays=num_rays
        )
     
        
        # if not self.training:
        #     torch.clamp_(diffuse_rgb, min=0.0, max=1.0)
        
        return features
    
@dataclass
class MERFModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: MERFModel)
    near_plane: float = 0.001
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "random"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_specular: int = 16
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 22
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 32
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 1
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 64, "log2_hashmap_size": 16, "num_levels": 10, "max_res": 512, "use_linear": False},
            {"hidden_dim": 64, "log2_hashmap_size": 16, "num_levels": 10, "max_res": 1024, "use_linear": False},
        ]   
    )
    param_regularizers:Dict = field(default_factory=lambda:{
        "proposal_networks": (0.03, torch.mean, 2.0, 1.0),
        "field": (0.03, torch.mean, 2.0, 1.0),
    }
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 1e-2
    """Distortion loss multiplier."""
    sparsity_loss_mult:Tuple[int, int, float, float] = field(
      default_factory=lambda: (500, 1000, 10, 0.05)
    ) 
    """Sparsity loss multiplier."""
    regularize_loss_mult: float = 0.0
    s3im_loss_mult: float = 1.0
    charb_eps : float = 5e-3
    appearance_dim : int = 16
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding for training."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    # Representation.
    triplane_resolution: int = 4096  # Planes will have dimensions (T, T) where
    # T = triplane_resolution.
    sparse_grid_resolution: int = 1024  # Vox el grid will have dimensions (S, S, S)
    # where S = sparse_grid_resolution.
    range_features: Tuple[float, float] = field(
    default_factory=lambda: (-7.0, 7.0)
    )  # Value range for appearance features.
    range_density: Tuple[float, float] = field(
      default_factory=lambda: (-14.0, 14.0)
    )  # Value range for density features.
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    alpha_threshold_param:Tuple[int, int, float, float] = field(
      default_factory=lambda: (10000, 30000, 5e-4, 5e-2, 20000)
    ) 
    """alpha culling in MERF:(start iters, end iters, start culling value, end culling value)"""

    num_random_samples : int = 2**14
    spatial_distortion : bool = True
    rgb_loss_method:Literal["mse", "charb"] = "charb"
    bounded_scene: bool = True
    accumulation_loss_mult: float = 0.05
    gaussian_samples_in_prop:bool = True
    virtual_grid_corner: bool = True


class MERFModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: MERFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        print("succesfully load!!!!!")
        if self.config.spatial_distortion:
            scene_contraction = MERFContraction()

        Frustums.get_gaussian_samples = get_gaussian_samples
        
        # Fields
        self.grid_config = calculate_grid_config(self.config)
        self.voxel_size_to_use = self.grid_config['voxel_size_to_use']
        
        self.field = MERFactoField(
            merf_config=self.config,
            aabb=self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_appearance_embedding=self.config.use_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            implementation=self.config.implementation,
        )
        self.direction_encoding  = MERFViewEncoding(in_dim=3,deg_enc=4,include_input=True)
        self.deferred_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.field.geo_feat_dim,
            num_layers=3,
            layer_width=16,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation="torch",
        ).to(self.kwargs['device'])
        self.sparsity_loss_mult= 10.0
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = MERFHashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
                gaussian_samples_in_prop=self.config.gaussian_samples_in_prop
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = MERFHashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                    gaussian_samples_in_prop=self.config.gaussian_samples_in_prop
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])


        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform" or (not self.config.spatial_distortion) or (self.config.bounded_scene):
            # initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter, spacing_fn=piecewise_warp_fwd, spacing_fn_inv=piecewise_warp_inv)
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)
        self.proposal_sampler = MERFProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        if self.config.bounded_scene:
            self.collider = AABBBoxCollider(scene_box=self.scene_box,near_plane=self.config.near_plane)
        else:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        
        # renderers
        self.renderer_features = FeatureRenderer(background_color=self.config.background_color)
        # self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        if self.config.rgb_loss_method == "charb":
            self.rgb_loss = CharbonnierLoss(charb_padding=self.config.charb_eps)
        elif self.config.rgb_loss_method == "mse":
            self.rgb_loss = MSELoss()
        else:
            raise NotImplementedError("Not implemented rgb loss type")

        self.acc_loss = torch.nn.BCELoss()
        self.s3im_loss = S3IM()
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["deferred_mlp"] = list(self.deferred_mlp.parameters())
        param_groups["fields"] = list(self.field.parameters())
        # print(param_groups["proposal_networks"])
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        def set_sparsity_loss_mult(step):
            def exp_lerp(t,v0,v1):
                if v0 <= 0 or v1 <= 0:      
                    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
                gamma = v1/v0
                return gamma ** t * v0
            if step > self.config.sparsity_loss_mult[1] + 1:
                return None 
            if step < self.config.sparsity_loss_mult[0]:            
                self.sparsity_loss_mult = self.config.sparsity_loss_mult[2]
                print(self.config.sparsity_loss_mult[2])
            elif step > self.config.sparsity_loss_mult[1]:
                self.sparsity_loss_mult = self.config.sparsity_loss_mult[3]
                print(self.config.sparsity_loss_mult[3])
            else:
                t = (step - self.config.sparsity_loss_mult[0]) / (self.config.sparsity_loss_mult[1] - self.config.sparsity_loss_mult[0])
                print(exp_lerp(t,self.config.sparsity_loss_mult[2],self.config.sparsity_loss_mult[3]))
                self.sparsity_loss_mult = exp_lerp(t,self.config.sparsity_loss_mult[2],self.config.sparsity_loss_mult[3])
            
        callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_sparsity_loss_mult,
                )
            )
        def set_alpha_culling_value(step):
                
            def log_lerp(t, v0, v1):
                """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
                if v0 <= 0 or v1 <= 0:
                    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
                lv0 = np.log(v0)
                lv1 = np.log(v1)
                return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)
            if step > self.config.alpha_threshold_param[4] + 1: 
                return None 
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
            
            self.field.set_alpha_threshold(alpha_thres)
        
        callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=10,
                    func=set_alpha_culling_value,
                )
            )
                
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

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

    def get_baking_outputs_for_camera_ray_bundle(self,ray_bundle: RayBundle):
        ray_samples: RaySamples
        
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        # ray_samples, weights_list, ray_samples_list = self.merf_sampler(ray_bundle, density_fns=self.density_fns)
        # ray samples in actual world space 
        field_outputs = self.field.get_outputs(ray_samples,ray_samples.frustums.directions)


        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        outputs = {
            "density": field_outputs[FieldHeadNames.DENSITY],
            "weights": weights,
            "ray_samples": ray_samples,
        }

        return outputs
    
    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        # ray_samples, weights_list, ray_samples_list = self.merf_sampler(ray_bundle, density_fns=self.density_fns)
        # ray samples in actual world space 
        # print("ray samples shape is :",ray_samples.shape)
        # print("ray samples frustums :",ray_samples.frustums.directions.shape)
        field_outputs = self.field.get_outputs(ray_samples,ray_samples.frustums.directions)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        dir_encode = self.direction_encoding(ray_bundle.directions)
        features = self.renderer_features(features=field_outputs['features'], weights=weights)
        rgb_specular = self.deferred_mlp(torch.cat([features,dir_encode],dim=-1))
        rgb = features[...,0:3] + rgb_specular
        # rgb = torch.clamp(rgb,0.0,1.0)
        # rgb = features[...,0:3]
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }
        
        # outputs = {
        #     "rgb": rgb,
        #     "accumulation": accumulation,
        #     "depth": depth,
        # }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def regularize_params_loss(self):
        """Computes regularizer loss(es) over optimized parameters for L2 loss."""
        
        loss_reg = 0.0
        
        # Iterate through the model's named parameters
        for name, param in self.named_parameters():
            # Check if this parameter has a regularizer in the config
            for key in self.config.param_regularizers:
                if key in name:
                    # Extract regularizer settings
                    mult, acc_fn, alpha, scale = self.config.param_regularizers[key]
                    
                    # Check if we match the conditions for L2 loss
                   
                    loss_reg += mult * acc_fn(lossfun(param, torch.tensor(alpha), torch.tensor(scale)))
                    
                    break
                
        return loss_reg
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        loss_dict["s3im_loss"] = self.config.s3im_loss_mult * self.s3im_loss( outputs["rgb"],image)
        # loss_dict["specular_loss"] = self.rgb_loss(outputs["distill_specular"], outputs["specular"].detach())
        # loss_dict["diffuse_loss"] = self.rgb_loss(image, outputs["diffuse"])
        if self.training:
            if self.config.regularize_loss_mult > 0.0:
                loss_dict["regularize_loss"] = self.regularize_params_loss()
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.sparsity_loss_mult > 0.0 :
                # sample points
                num_random_samples = self.config.num_random_samples 
                if self.config.bounded_scene:
                    random_positions = self.scene_box.aabb[0].min() + (self.scene_box.aabb[1].max() - self.scene_box.aabb[0].min()) * torch.rand(num_random_samples, 3).to(self.device)
                else:
                    random_positions = WORLD_MIN + (WORLD_MAX - WORLD_MIN) * torch.rand(num_random_samples, 3).to(self.device)
                random_viewdirs = torch.normal(mean=0, std=1, size=(num_random_samples, 3)).to(self.device)
                random_viewdirs /= torch.norm(random_viewdirs, dim=-1, keepdim=True)
                # print(random_viewdirs.shape)
                # print(random_positions.shape)
                density = self.field(
                    None,
                    random_viewdirs,
                    random_positions
                )[FieldHeadNames.DENSITY]
                loss_dict["sparsity_loss"] = self.sparsity_loss_mult * sparsity_loss(random_positions, random_viewdirs, density, self.voxel_size_to_use)
        
            if self.config.accumulation_loss_mult > 0.0:
                loss_dict["acc_loss"] = self.config.accumulation_loss_mult * self.acc_loss(outputs["accumulation"].clip(1e-5, 1.0 - 1e-5),torch.ones_like(outputs["accumulation"]))
        return loss_dict

    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = 0.0

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
    
    """
    Baking Functions
    """
   
    
    def compute_alive_voxels(self,chunks_model=None,lod_idx=None,chunk_idx=None):
        """Performs visiblity culling.

        Returns:
        A grid that indicates which voxels are alive.

        We only consider a voxel as occupied if it contributed siginficantly to the
        rendering of any training image. We therefore render the entire training set
        and record the sampling positions predicted by the ProposalMLP. For any
        sampled point with a sufficently high volume rendering weight and alpha value
        we mark the 8 adjacent voxels as occupied.

        We are able to instantiate a full-res occupancy grid that is stored on the
        CPU.
        """

        
        voxel_size = self.grid_config['voxel_size_to_use']
        grid_size = [self.grid_config['resolution_to_use']] * 3
        alive_voxels = torch.zeros(grid_size, dtype=bool)

        self.eval()
        with torch.no_grad():
            num_images = len(self.datamanager.fixed_indices_eval_dataloader)
            eval_num = 0
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("[green]Evaluating all images for baking...", total=num_images)
                
                print(f"{eval_num} images have benn tested")
                
                for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                    # time this the following line
                    print(f"{eval_num} images have benn tested",flush=True)
                    eval_num += 1
                    height, width = camera_ray_bundle.shape
                    
                    # get per image rays' samples and their weights,density(for culling)
                    camera_ray_bundle = camera_ray_bundle.reshape(-1)
                    num_batch = camera_ray_bundle.shape[0] // self.config.baking_config.eval_chunk
                    for i in range(num_batch):
                        if i < num_batch - 1:
                            batch_camera_ray_bundle = camera_ray_bundle[i*self.config.baking_config.eval_chunk:i*self.config.baking_config.eval_chunk+self.config.baking_config.eval_chunk]
                        else:
                            batch_camera_ray_bundle = camera_ray_bundle[(num_batch - 1)*self.config.baking_config.eval_chunk:]
                        if chunks_model is not None:
                            outputs = chunks_model.get_baking_outputs_for_camera_ray_bundle(self.collider(batch_camera_ray_bundle),lod_idx=lod_idx,chunk_idx=chunk_idx) ##TODO collider fixed
                        else:
                            outputs = self.get_baking_outputs_for_camera_ray_bundle(self.collider(batch_camera_ray_bundle))
                        if outputs is None:
                            continue
                        density = outputs['density'].cpu()
                        ray_samples = outputs['ray_samples']
                        weights = outputs['weights'].cpu()
                        
                    
                        # delete unnecessary variables to save memory
                        del outputs
                        gc.collect()
                        
                        # get ray samples' positions
                        positions = ray_samples.frustums.get_gaussian_samples()
                        if chunks_model is not None:
                            positions =  torch.matmul(chunks_model.chunks_transform[lod_idx][chunk_idx],torch.cat([positions, torch.ones(*positions.shape[:-1], 1).to(positions)], dim=-1).T).T[:,0:3]
                        num_samples = positions.shape[-2]
                        
                        # culling samples by weights
                        weight_threshold = self.config.baking_config.weight_threshold
                        alive_samples = (weights > weight_threshold).squeeze(-1)  # BxS.``
                        positions_alive = positions[alive_samples].view(-1, 3)  # Ax3.
                        
                    
                        # culling samples by alpha
                        alpha_threshold = self.config.baking_config.alpha_threshold
                        viewdirs_alive = ray_samples.frustums.directions[alive_samples].reshape(-1, 3)  # Ax3.
                        ### TODO:check viewdirs_alive and num_samples
                        if self.config.baking_config.use_alpha_culling:
                            step_sizes = stepsize_in_squash(
                                positions_alive, viewdirs_alive, voxel_size)  # A

                        positions_alive = contract(positions_alive)
                        # if chunks_model is not None:
                        #     positions_alive =  torch.matmul(chunks_model.chunks_transform[lod_idx][chunk_idx],torch.cat([positions_alive, torch.ones(*positions_alive.shape[:-1], 1).to(positions_alive)], dim=-1).T).T[:,0:3]
                        
                        if self.config.baking_config.use_alpha_culling:
                            density_alive = density[alive_samples].reshape(-1)
                            alpha = 1.0 - trunc_exp(- density_alive.to(self.device)  * step_sizes)
                            positions_alive = positions_alive[alpha > alpha_threshold]
                        
                
                        # convert world space to grid space
                        positions_alive = world_to_grid(positions_alive, voxel_size)
                        positions_alive_0 = torch.floor(positions_alive).to(torch.int32)
                        positions_alive_1 = torch.ceil(positions_alive).to(torch.int32)
                            
                        def remove_out_of_bound_points(x):
                            mask = (x >= 0).all(dim=1) & (x < self.grid_config['resolution_to_use']).all(dim=1)
                            return x[mask]
                        
                        # Splat to eight voxel corners.
                        corner_coords = [[False, True] for _ in range(3)]
                        for z in itertools.product(*corner_coords):
                            l = []
                            for i, b in enumerate(z):
                                l.append(positions_alive_1[..., i] if b else positions_alive_0[..., i])
                            positions_corner = torch.stack(l, dim=-1)
                        
                            positions_corner = remove_out_of_bound_points(positions_corner)
                            
                            alive_voxels[
                                positions_corner[:, 0], positions_corner[:, 1], positions_corner[:, 2]
                            ] = True
                        
                    progress.advance(task)
                
        return alive_voxels

    def bake_sparse_grid(self,alive_macroblocks):
        
        # Pre computing some attributes
        sparse_grid_voxel_size = self.grid_config['sparse_grid_voxel_size']
        data_block_size = self.config.baking_config.data_block_size
        atlas_block_size = get_atlas_block_size(data_block_size)
        num_occupied_blocks = alive_macroblocks.sum()
        batch_size_in_blocks = self.config.baking_config.batch_size_in_blocks
        
        
        def create_atlas_1d(n):
            sh = (
                num_occupied_blocks,
                atlas_block_size,
                atlas_block_size,
                atlas_block_size,
                n,
            )
            return np.zeros(sh, dtype=np.float32)
        
        sparse_grid_features_1d = create_atlas_1d(NUM_CHANNELS)
        sparse_grid_density_1d = create_atlas_1d(1)
        
        min_voxel = (
            np.stack(np.nonzero(alive_macroblocks), axis=-1).astype(np.float32)
            * data_block_size
        )
        
        for block_start in tqdm.tqdm(list(range(0, num_occupied_blocks, batch_size_in_blocks))):
            
            block_end = min(block_start + batch_size_in_blocks, num_occupied_blocks)

            # Build AxAxA grid for each block.
            min_voxel_batch = min_voxel[block_start:block_end]  # Bx3.
            span = np.arange(atlas_block_size)  # A.
            x_grid = np.stack(
                np.meshgrid(span, span, span, indexing='ij'), axis=-1
            )  # AxAxAx3.
            x_grid = (
                min_voxel_batch[:, np.newaxis, np.newaxis, np.newaxis, :]
                + x_grid[np.newaxis]
            )  # Bx1x1x1x3 + 1xAxAxAx3 = BxAxAxAx3.

            x_grid_tensor = torch.from_numpy(x_grid).cuda()
            # Evaluate MLP.
            positions = grid_to_world(
                x_grid_tensor, sparse_grid_voxel_size
            )  # BxAxAxAx3.
            features, density = self.field.evaluate_to_baking(positions)  # BxAxAxAx7, BxAxAxAx3.
            sparse_grid_features_1d[block_start:block_end] = features.cpu().numpy()
            sparse_grid_density_1d[block_start:block_end] = density.cpu().numpy()
            del features, density, min_voxel_batch, positions
            gc.collect()
            
        return sparse_grid_features_1d, sparse_grid_density_1d
                
                
    
    def bake_triplane(self,masks_2d_list):
        
        """Bakes triplanes."""
        
        triplane_voxel_size = self.grid_config['triplane_voxel_size']
        triplane_resolution = self.config.triplane_resolution
        batch_size_triplane = self.config.baking_config.batch_size_triplane
        
        planes_features = []
        planes_density = []
        
        # Setup MLP (doing this inside the loop may cause re-jitting).

        for plane_idx in range(3):
            
            print('baking plane', plane_idx)
            spans = [
                np.arange(triplane_resolution) if plane_idx != c else np.array([0.0])
                for c in range(3)
            ]
            positions_grid = np.stack(np.meshgrid(*spans, indexing='ij'), axis=-1)
            positions_grid_tensor = torch.from_numpy(positions_grid).cuda()
            positions = grid_to_world(positions_grid_tensor, triplane_voxel_size).reshape(-1, 3)
            
            num_texels = positions.shape[0]
            plane = np.zeros((num_texels, NUM_CHANNELS + 1), dtype=np.float32)
            for batch_start in tqdm.tqdm(list(range(0, num_texels, batch_size_triplane))):
                batch_end = min(batch_start + batch_size_triplane, num_texels)
                features, density = self.field.evaluate_to_baking(positions[batch_start:batch_end])
                plane[batch_start:batch_end, : NUM_CHANNELS] = features.cpu().numpy()
                plane[batch_start:batch_end, [-1]] = density.cpu().numpy()
                del features, density
                gc.collect()
            plane = plane.reshape((triplane_resolution, triplane_resolution, -1))
            # Project alive voxels on plane and set dead texels to zero. Otherwise
            # dead texels would contain random values that cannot be compressed well
            # with PNG.
            # mask_2d = alive_voxels.any(axis=plane_idx)
            mask_2d = masks_2d_list[plane_idx]
            mask_2d = ~scipy.ndimage.maximum_filter(mask_2d, size=25)  # dilate heavily
            plane[mask_2d] = -1000000.0
            planes_features.append(plane[Ellipsis, : NUM_CHANNELS])
            planes_density.append(plane[Ellipsis, [-1]])
            
        return planes_features, planes_density
    
    
    def export_scene(
        self,
        sparse_grid_features,
        sparse_grid_density,
        sparse_grid_block_indices,
        planes_features,
        planes_density,
        occupancy_grids,
    ):
        """Exports the baked repr. into a format that can be read by the webviewer."""

        # Store one lossless copy of the scene as PNGs.
        baked_dir = path.join(self.config.baking_config.baking_path, 'baked')
        os.makedirs(baked_dir, exist_ok=True)

        output_paths = []
        output_images = []

        export_scene_params = {
            'triplane_resolution': self.config.triplane_resolution,
            'triplane_voxel_size': self.grid_config['triplane_voxel_size'],
            'sparse_grid_resolution': self.config.sparse_grid_resolution,
            'sparse_grid_voxel_size': self.grid_config['sparse_grid_voxel_size'],
            'range_features': self.config.range_features,
            'range_density': self.config.range_density,
        }

        # export Tri-Planes
        for plane_idx, (plane_features, plane_density) in enumerate(
            zip(planes_features, planes_density)
        ):
            plane_rgb_and_density = np.concatenate(
                [plane_features[Ellipsis, :3], plane_density], axis=-1
            ).transpose([1, 0, 2])
            output_images.append(plane_rgb_and_density)
            output_paths.append(
                path.join(baked_dir, f'plane_rgb_and_density_{plane_idx}.png')
            )
            plane_features = plane_features[Ellipsis, 3:].transpose([1, 0, 2])
            output_images.append(plane_features)
            output_paths.append(
                path.join(baked_dir, f'plane_features_{plane_idx}.png')
            )

         # export sparse grid
        sparse_grid_rgb_and_density = np.copy(
            np.concatenate(
                [sparse_grid_features[Ellipsis, :3], sparse_grid_density], axis=-1
            )
        )  # [..., 4]
        sparse_grid_features = sparse_grid_features[Ellipsis, 3:]  # [..., 4]

        # Subdivide volume into slices and store each slice in a seperate png
        # file. This is nessecary since the browser crashes when tryining to
        # decode a large png that holds the entire volume. Also multiple slices
        # enable parallel decoding in the browser.
        slice_depth = 1
        num_slices = sparse_grid_rgb_and_density.shape[2] // slice_depth

        def write_slices(x, prefix):
            for i in range(0, x.shape[2], slice_depth):
                stack = []
                for j in range(slice_depth):
                    plane_index = i + j
                    stack.append(x[:, :, plane_index, :].transpose([1, 0, 2]))
                output_path = path.join(baked_dir, f'{prefix}_{i:03d}.png')
                output_images.append(np.concatenate(stack, axis=0))
                output_paths.append(output_path)

        write_slices(sparse_grid_rgb_and_density, 'sparse_grid_rgb_and_density')
        write_slices(sparse_grid_features, 'sparse_grid_features')

        sparse_grid_block_indices_path = path.join(
            baked_dir, 'sparse_grid_block_indices.png'
        )
        sparse_grid_block_indices_image = (
            np.transpose(sparse_grid_block_indices, [2, 1, 0, 3])
            .reshape((-1, sparse_grid_block_indices.shape[0], 3))
            .astype(np.uint8)
        )
        output_paths.append(sparse_grid_block_indices_path)
        output_images.append(sparse_grid_block_indices_image)

        export_scene_params.update({
            'data_block_size': self.config.baking_config.data_block_size,
            'atlas_width': sparse_grid_features.shape[0],
            'atlas_height': sparse_grid_features.shape[1],
            'atlas_depth': sparse_grid_features.shape[2],
            'num_slices': num_slices,
            'slice_depth': slice_depth,
        })

        for occupancy_grid_factor, occupancy_grid in occupancy_grids:
            
            occupancy_grid_path = path.join(
                baked_dir, f'occupancy_grid_{occupancy_grid_factor}.png'
            )
            # Occupancy grid only holds a single channel, so we create a RGBA image by
            # repeating the channel 4 times over.
            occupany_grid_image = (
                np.transpose(np.repeat(occupancy_grid[Ellipsis, None], 4, -1), [2, 1, 0, 3])
                .reshape((-1, occupancy_grid.shape[0], 4))
                .astype(np.uint8)
            )
            output_images.append(occupany_grid_image)
            output_paths.append(occupancy_grid_path)

        # TODO check mat shape correction  
        export_scene_params.update({
            '0_weights': self.deferred_mlp.layers[0].weight.data.cpu().transpose(0,1).tolist(),
            '1_weights': self.deferred_mlp.layers[1].weight.data.cpu().transpose(0,1).tolist(),
            '2_weights': self.deferred_mlp.layers[2].weight.data.cpu().transpose(0,1).tolist(),
            '0_bias': self.deferred_mlp.layers[0].bias.data.cpu().tolist(),
            '1_bias': self.deferred_mlp.layers[1].bias.data.cpu().tolist(),
            '2_bias': self.deferred_mlp.layers[2].bias.data.cpu().tolist(),
        })

        parallel_write_images(save_8bit_png, list(zip(output_images, output_paths)))

        scene_params_path = path.join(baked_dir, 'scene_params.json')
        save_json(export_scene_params, scene_params_path)

        # Calculate disk consumption.
        total_storage_in_mib = 0
        for output_path in output_paths:
            if 'occupancy_grid' not in output_path:
                storage_in_mib = os.stat(output_path).st_size / (1024**2)
                total_storage_in_mib += storage_in_mib
        storage_path = path.join(baked_dir, 'storage.json')
        print(f'Disk consumption: {total_storage_in_mib:.2f} MiB')
        save_json(total_storage_in_mib, storage_path)

        print(f'Exported scene to {baked_dir}')

    def baking_merf_model(self,datamanager,chunks_model=None,lod_idx=None,chunk_idx=None):
        assert self.config.baking_config is not None
        with torch.no_grad():
            gc.collect()
            planes_features = planes_density = sparse_grid_features = sparse_grid_density = sparse_grid_block_indices = None
            self.grid_config = calculate_grid_config(self.config)
            self.datamanager = datamanager
            
            # alive voxels save path
            occupancy_grids_path=[]
            masks_2d_list_path=[]
            alive_voxels_path = path.join(self.config.baking_config.baking_path, 'alive_voxels.npy')
            for i in range(5):
                occupancy_grids_path.append(path.join(self.config.baking_config.baking_path, f'occupancy_grids_{i}.npy'))
            for i in range(3):
                masks_2d_list_path.append(path.join(self.config.baking_config.baking_path, f'masks_2d_list_{i}.npy'))
            
            alive_voxels_3d_grid_path = path.join(self.config.baking_config.baking_path, 'alive_voxels_3d_grid.npy')
            
            
            # load  occupancy_grids/alive_voxels_3d_grid/masks_2d_list
            occupancy_grid_factors = [8, 16, 32, 64, 128]
            if self.config.baking_config.load_alive_voxels_from_disk:
                if self.config.baking_config.load_occ_grid:
                    occupancy_grids = []
                    masks_2d_list = []
                    for i in range(5):
                        occupancy_grids.append((occupancy_grid_factors[i],np.load(occupancy_grids_path[i])))
                    for i in range(3):
                        masks_2d_list.append(np.load(masks_2d_list_path[i]))
                    alive_voxels_3d_grid = np.load(alive_voxels_3d_grid_path)
                else:
                    alive_voxels = np.load(alive_voxels_path,mmap_mode='r')
            else:
                # Compute alive voxels by sample points per ray in datamanager
                alive_voxels = self.compute_alive_voxels(chunks_model=chunks_model,lod_idx=lod_idx,chunk_idx=chunk_idx)
                alive_voxels = alive_voxels.cpu().numpy()

                
                
                print(
                    '{:.1f}% voxels are occupied.'.format(
                        100 * alive_voxels.sum() / alive_voxels.size
                    )
                )
                
            if self.config.baking_config.save_alive_voxels_to_disk and not self.config.baking_config.load_alive_voxels_from_disk :
                
                np.save(alive_voxels_path,alive_voxels)
                print(f"save alive voxels in {alive_voxels_path}")
                
            if not self.config.baking_config.load_occ_grid:
                # Compute occupancy grids based on downsampling factors.
                print(f"start baking occ grids")
                occupancy_grid_factors = [8, 16, 32, 64, 128]
                occupancy_grids = []
                
                pre_factor = 1
                occupancy_grid = alive_voxels
                for i, occupancy_grid_factor in enumerate(occupancy_grid_factors):
                    occupancy_grid = skimage.measure.block_reduce(
                        occupancy_grid,
                        (occupancy_grid_factor//pre_factor, occupancy_grid_factor//pre_factor, occupancy_grid_factor//pre_factor),
                        np.max,
                    )
                    pre_factor = occupancy_grid_factor
                    print(f'occupancy_grid_factor_{occupancy_grid_factor} has been computed')
                    occupancy_grids.append((occupancy_grid_factor, occupancy_grid))
                occ_shape = [occ_grid[1].shape for occ_grid in occupancy_grids]
                print(f"occ shape : {occ_shape}")
                
                # # Specfiy finest occupancy grid used during rendering.
                # occupancy_grid_index = 1  # downsampling factor = 16.
                # occupancy_grid_factor = occupancy_grids[occupancy_grid_index][0]
                # occupancy_grid = occupancy_grids[occupancy_grid_index][1]
                
                # compute   alive_voxels for 3d_grid
                
                downsampling_ratio_3d_grid = int(
                    self.config.triplane_resolution / self.config.sparse_grid_resolution
                )
                
                alive_voxels_3d_grid = skimage.measure.block_reduce(
                    alive_voxels,
                    (
                        downsampling_ratio_3d_grid,
                        downsampling_ratio_3d_grid,
                        downsampling_ratio_3d_grid,
                    ),
                    np.max,
                )
                print(f'alive_voxels_3d_grid has been computed')
                # compute alive mask for triplane 
                
                masks_2d_list = []
                for plane_idx in range(3):
                    masks_2d_list.append(alive_voxels.any(axis=plane_idx))

                print(f'masks_2d_list has been computed')
                del alive_voxels
                del occupancy_grid
                
                # save necessary for succesory computing occupancy_grids/alive_voxels_3d_grid/masks_2d_list
                np.save(alive_voxels_3d_grid_path, alive_voxels_3d_grid)
                for i in range(5):
                    np.save(occupancy_grids_path[i], occupancy_grids[i][1])
                for i in range(3):
                    np.save(masks_2d_list_path[i], masks_2d_list[i])


            data_block_size = self.config.baking_config.data_block_size
            
            # compute alive voxels 3d grid
            

            alive_macroblocks = skimage.measure.block_reduce(
            alive_voxels_3d_grid,
            (data_block_size, data_block_size, data_block_size),
            np.max,
            )
            num_alive_macroblocks = alive_macroblocks.sum()
            
            print('Sparse grid:')
            print(
            '{} out of {} ({:.1f}%) macroblocks are occupied.'.format(
                num_alive_macroblocks,
                alive_macroblocks.size,
                100 * num_alive_macroblocks / alive_macroblocks.size,
            )
        )
            # Baking sparse grid
            batch_size_in_blocks = self.config.baking_config.batch_size_in_blocks
            sparse_grid_features_1d, sparse_grid_density_1d = self.bake_sparse_grid(alive_macroblocks)
            sparse_grid_features, sparse_grid_density, sparse_grid_block_indices = reshape_into_3d_atlas_and_compute_indirection_grid(
                sparse_grid_features_1d,
                sparse_grid_density_1d,
                data_block_size,
                alive_macroblocks,
            )
            
            # Baking triplane
            planes_features, planes_density = self.bake_triplane(masks_2d_list)
            
            
            # Compute VRAM consumption.
            vram_consumption = {}

            vram_consumption = {
                'sparse_3d_grid': as_mib(sparse_grid_features) + as_mib(
                    sparse_grid_density
                ),
                'indirection_grid': as_mib(sparse_grid_block_indices),
            }

                # Assume that all three planes have the same size.
            vram_consumption['triplanes'] = 3 * (
                as_mib(planes_features[0]) + as_mib(planes_density[0])
            )
            vram_consumption['total'] = sum(vram_consumption.values())
            print('VRAM consumption:')
            for k in vram_consumption:
                print(f'{k}: {vram_consumption[k]:.2f} MiB')
            save_json(vram_consumption, os.path.join(self.config.baking_config.baking_path, 'vram.json'))

            planes_features, planes_density = (
                map_quantize_tuple_list(
                    planes_features,
                    planes_density,
                )
            )
            
            sparse_grid_features, sparse_grid_density = (
                map_quantize_tuple(
                    sparse_grid_features,
                    sparse_grid_density,
                )
            )
            
            


            self.export_scene(
                sparse_grid_features,
                sparse_grid_density,
                sparse_grid_block_indices,
                planes_features,
                planes_density,
                occupancy_grids
                )
        
        



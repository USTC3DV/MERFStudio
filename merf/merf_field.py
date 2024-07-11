from typing import Dict, Literal, Optional, Tuple

import torch
from jaxtyping import Shaped
from torch import Tensor, nn

from merf.coord import contract, pos_enc, stepsize_in_squash
from merf.grid_utils import (calculate_grid_config,
                             get_eval_positions_and_local_coordinates,
                             interpolate_based_on_local_coordinates)
from merf.quantize import simulate_quantization
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding
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

NUM_CHANNELS = 7




def simulate_alpha_culling(
    density, positions, viewdirs, alpha_threshold, voxel_size_to_use
):
  """Computes the alpha value based on a constant step size."""
  # During real-time rendering a constant step size (i.e. voxel size) is used.
  # During training a variable step size is used that can be vastly different
  # from the voxel size. When baking we discard voxels that would only
  # contribute neglible alpha values in the real-time renderer. To make this
  # lossless we already simulate the behaviour of the real-time renderer during
  # training by ignoring alpha values below the threshold.
#   print(viewdirs.shape)
#   print(positions.shape)
  def zero_density_below_threshold(density):
    if viewdirs.shape != positions.shape:
        viewdirs_b = torch.broadcast_to(
            viewdirs, positions.shape
        ).reshape(-1, 3)
    else:
        viewdirs_b = viewdirs.reshape(-1, 3)
    positions_b = positions.view(-1, 3)
    step_size_uncontracted = stepsize_in_squash(
        positions_b, viewdirs_b, voxel_size_to_use
    )
    step_size_uncontracted = step_size_uncontracted.reshape(density.shape)
    alpha =  1.0 - trunc_exp(- density  * step_size_uncontracted)
    return torch.where(
        alpha >= alpha_threshold, density, 0.0
    )  # density = 0 <=> alpha = 0

  return zero_density_below_threshold(density) if alpha_threshold > 0.0 else density



class MERFContraction(SpatialDistortion):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, positions):
        return contract(positions)

class MERFViewEncoding(Encoding):
    
    def __init__(
        self,
        in_dim: int,
        deg_enc: int = 4,
        include_input: bool = True,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__(in_dim)
        self.deg_enc = deg_enc
        self.include_input = include_input
        
    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim * self.deg_enc * 2 + self.in_dim if self.include_input else self.in_dim * self.deg_enc * 2

    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        return pos_enc(in_tensor,min_deg=0,max_deg=self.deg_enc,append_identity=self.include_input)

class MERFactoField(Field):
    """Compound MERF Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        merf_config,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 3,
        hidden_dim: int = 256,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        features_per_level: int = 2,
        log2_hashmap_size: int = 20,
        num_layers_specular: int = 2,
        num_layers_transient: int = 2,
        hidden_dim_specular: int = 16,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 16,
        transient_embedding_dim: int = 16,
        use_appearance_embedding: bool = False,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = MERFContraction,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        
        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.alpha_threshold = 0.0
        self.use_appearance_embedding = use_appearance_embedding
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.use_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients

        self.base_res = base_res
        self.features_per_level = features_per_level
        self.geo_feat_dim = NUM_CHANNELS
        self.merf_model_config = merf_config
        implementation = "tcnn"
        encoder = HashEncoding(
            num_levels=num_levels,
            min_res=self.base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        if self.use_appearance_embedding:    
            density_network = MLP(
                in_dim=encoder.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=64,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            
            feature_network = MLP(      
                in_dim = 64 + self.appearance_embedding_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.mlp_density = density_network
            self.mlp_base_feature = feature_network
            self.hash_encoder = encoder
        else:
            self.density_and_feature_network = MLP(
                in_dim=encoder.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1 + self.geo_feat_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.hash_encoder = encoder

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_gaussian_samples()
        return self.get_outputs(ray_samples=None, viewdirs=None, positions=positions)

    def evaluate_to_baking(self, positions: Tensor, appearance_embedding = None) -> Tuple[Tensor, Tensor]:
        # positiosn : (...,3)  range:[-2,2]
    
        sh = positions.shape[:-1]
        positions = positions.reshape((-1, 3))
        actual_num_inputs = positions.shape[0]
        
        # normalize positions
        tgpositions = (positions + 2.0) / 4.0
        if self.use_appearance_embedding and appearance_embedding is None:
            appearance_embedding = self.embedding_appearance
        if self.use_appearance_embedding or appearance_embedding is not None:
            embedded_appearance = torch.ones(
                    (*tgpositions.shape[:-1], self.appearance_embedding_dim), device=tgpositions.device
                ) * appearance_embedding.mean(dim=0)
            hash_features = self.hash_encoder(tgpositions)
            h = self.mlp_density(hash_features) # U*Kx64
            density, _ = torch.split(h, [1, 63], dim=-1)
            features = self.mlp_base_feature(
                torch.cat([h,embedded_appearance],dim=-1)
            )    
        else:
            hash_features = self.hash_encoder(tgpositions)
            h = self.density_and_feature_network(hash_features)
            density, features = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        
        features = features.view(sh+(features.shape[-1],))
        density = density.view(sh+(density.shape[-1],))
        
        return features,density
    
    def set_alpha_threshold(self,alpha_threshold):
        self.alpha_threshold = alpha_threshold  
        
    def get_outputs(
        self, ray_samples: RaySamples, viewdirs: Optional[Tensor]=None, positions=None, samples_transform=None, appearance_embedding=None,
    ) -> Dict[FieldHeadNames, Tensor]:
        
        outputs = {}
        grid_config = calculate_grid_config(self.merf_model_config)
        only_return_density = positions is not None
        if only_return_density:
            # print(positions.max())
            # print(positions.min())
            camera_indices = None
        else:
            assert viewdirs is not None
            
            if ray_samples.camera_indices is None:
                raise AttributeError("Camera indices are not provided.")
            camera_indices = ray_samples.camera_indices.squeeze()
            
            
            
            if self.spatial_distortion is not None:
                positions = ray_samples.frustums.get_gaussian_samples()
                if samples_transform is not None:
                    positions =  torch.matmul(samples_transform,torch.cat([positions, torch.ones(*positions.shape[:-1], 1).to(positions)], dim=-1).T).T[:,0:3]
                positions = self.spatial_distortion(positions)
 
                # normalize to [0,1]
                # positions = (positions + 2.0) / 4.0
                # print(positions.max())
                # print(positions.min())
            else:
                positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_gaussian_samples(), self.aabb)
        # print(positions.shape)
        batch_shape = positions.shape[:-1]
        # Make sure the tcnn gets inputs between 0 and 1.
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        
        tgpositions = positions.view(-1, 3) # U*Kx3

        tgpositions, triplane_positions_local, sparse_grid_positions_local, camera_indices = (
        get_eval_positions_and_local_coordinates(
        tgpositions, None, self.merf_model_config, grid_config
    )
)
            # triplane_positions_local, sparse_grid_positions_local: U*Kx3
        tgpositions = (tgpositions + 2.0)/4.0
        # print(tgpositions.max())
        # print(tgpositions.min())
  
        if self.use_appearance_embedding and appearance_embedding is None:
            appearance_embedding = self.embedding_appearance
        if self.use_appearance_embedding or appearance_embedding is not None:
            if self.training and not only_return_density :
                embedded_appearance = appearance_embedding(camera_indices).view(-1,self.appearance_embedding_dim) # U*KxD
            else:      
                embedded_appearance = torch.ones(
                    (*tgpositions.shape[:-1], self.appearance_embedding_dim), device=tgpositions.device
                ) * appearance_embedding.mean(dim=0)
            # print(embedded_appearance.shape)
            hash_features = self.hash_encoder(tgpositions).to()
            h = self.mlp_density(hash_features) # U*Kx64
            density, _ = torch.split(h, [1, 63], dim=-1)
            features = self.mlp_base_feature(
                torch.cat([h,embedded_appearance],dim=-1)
            )
            density = simulate_quantization(
                density, self.merf_model_config.range_density[0], self.merf_model_config.range_density[1]
            )
            
            density = interpolate_based_on_local_coordinates(
                density, triplane_positions_local, sparse_grid_positions_local, self.merf_model_config
            )  # U*Kx1.
            
            density = trunc_exp(density - 1.0)
            
            density = density.reshape(*batch_shape,-1)
            
            outputs.update({FieldHeadNames.DENSITY: density})
            
            if only_return_density:
            
                return outputs
        


            features = simulate_quantization(
                features, self.merf_model_config.range_features[0], self.merf_model_config.range_features[1]
            )

            # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
            features = interpolate_based_on_local_coordinates(
                features, triplane_positions_local, sparse_grid_positions_local, self.merf_model_config
            )  # U*Kx7.

            features = torch.sigmoid(features)

            features = features.reshape(*batch_shape,-1)
            

            # check *batch_shape == *ray_samples.frustums.shape
        
        else:
            # tgpositions = positions.view(-1,3) #TODO
            # print(tgpositions.shape)
            hash_features = self.hash_encoder(tgpositions)
            h = self.density_and_feature_network(hash_features)
            density, features = torch.split(h, [1, self.geo_feat_dim], dim=-1)
           
            

            density = simulate_quantization(
                density, self.merf_model_config.range_density[0], self.merf_model_config.range_density[1]
            )
            
            density = interpolate_based_on_local_coordinates(
                density, triplane_positions_local, sparse_grid_positions_local, self.merf_model_config
            )  # U*Kx1.
            density = trunc_exp(density - 1.0)
            
            
            # print(density.shape)
            density = density.reshape(*batch_shape,-1)
            
            density = simulate_alpha_culling(density, positions, viewdirs, self.alpha_threshold, grid_config['voxel_size_to_use'])

            
            outputs.update({FieldHeadNames.DENSITY: density})
                                    
            if only_return_density:
            
                return outputs
            # Grid simulation: bi-lineary and/or tri-linearly interpolate outputs.
            features = simulate_quantization(
                features, self.merf_model_config.range_features[0], self.merf_model_config.range_features[1]
            )
            
            features = interpolate_based_on_local_coordinates(
                features, triplane_positions_local, sparse_grid_positions_local, self.merf_model_config
            )  # U*Kx7.

            features = torch.sigmoid(features)

            features = features.reshape(*batch_shape,-1)
            
        

        positions = positions.reshape(*batch_shape,-1)

        

        outputs.update({"features":features})               # # transients #TODO
        # if self.use_transient_embedding and self.training:
        #     embedded_transient = self.embedding_transient(camera_indices)
        #     transient_input = torch.cat(
        #         [
        #             density_embedding.view(-1, self.geo_feat_dim),
        #             embedded_transient.view(-1, self.transient_embedding_dim),
        #         ],
        #         dim=-1,
        #     )
        #     x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
        #     outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
        #     outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
        #     outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # # semantics #TODO
        # if self.use_semantics:
        #     semantics_input = density_embedding.view(-1, self.geo_feat_dim)
        #     if not self.pass_semantic_gradients:
        #         semantics_input = semantics_input.detach()

        #     x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
        #     outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # # predicted normals #TODO
        # if self.use_pred_normals:
        #     positions = ray_samples.frustums.get_gaussian_samples()

        #     positions_flat = self.position_encoding(positions.view(-1, 3))
        #     pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        #     x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
        #     outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        return outputs

    def forward(self, ray_samples: RaySamples, viewdirs: Optional[Tensor]=None, positions=None) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
      
        field_outputs = self.get_outputs(ray_samples, viewdirs=viewdirs, positions=positions)

        return field_outputs

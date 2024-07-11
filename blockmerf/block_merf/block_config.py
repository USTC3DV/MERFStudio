import dataclasses
from dataclasses import dataclass, field
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig,VanillaPipeline
from nerfstudio.plugins.types import MethodSpecification
from merf.merf_model import MERFModel, MERFModelConfig
from block_merf.block_model import BlockModelConfig
from merf.engine.merf_trainer import MERFTrainerConfig
from merf.merf_datamanager import MerfDataParser,MerfDataParserConfig
from block_merf.pipeline.merf_pipeline import BlockMERFPipeline, BlockMERFPipelineConfig

block_config = MERFTrainerConfig(
    method_name="block-merf",
    viewer=ViewerConfig(
        relative_log_filename='viewer_log_filename.txt',
        websocket_port=None,
        websocket_port_default=7007,
        websocket_host='0.0.0.0',
        num_rays_per_chunk=4,
        max_num_display_images=512,
        quit_on_train_completion=False,
        image_format='jpeg',
        jpeg_quality=90
    ),
    pipeline=BlockMERFPipelineConfig(
        _target=BlockMERFPipeline,
        datamanager=VanillaDataManagerConfig(
            # _target=RayPruningDataManager,
            dataparser=MerfDataParserConfig(
            is_transform=False,
            downscale_factor=1,
            train_split_fraction=0.99,
            orientation_method='vertical',
            eval_mode="blender",
            center_method='focus',
            scene_scale=64.0,
            z_limit_min = -64.0,
            z_limit_max = 64.0, 
            angle_rot = 1.0/6.0,       
            ),
            eval_num_rays_per_batch=40000, #50176
            train_num_rays_per_batch=40000,#36864
            patch_size=1,
        ),
        model=BlockModelConfig(_target=MERFModel,
                               alpha_threshold_param=(10000, 25000, 5e-4, 1e-2, 15000) 
                              ),
    ),
    max_num_iterations=40000,
    steps_per_save=10000,
    steps_per_eval_batch=5000,
    steps_per_eval_image=5000,
    steps_per_eval_all_images=39999,
       optimizers={ 
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        },
        "deferred_mlp": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        },
        "apperance":{
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        }
    },
    gradient_accumulation_steps = 1,
)
BLOCKMERF = MethodSpecification(
    config=block_config, description="Implementation of Large-scale MERF "
)

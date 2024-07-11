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
from merf.engine.merf_trainer import MERFTrainerConfig
from merf.merf_datamanager import MerfDataParser,MerfDataParserConfig
from merf.pipeline.merf_pipeline import MERFPipeline, MERFPipelineConfig

merf_config = MERFTrainerConfig(
    method_name="merf-ns",
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
    pipeline=MERFPipelineConfig(
        _target=MERFPipeline,
        datamanager=VanillaDataManagerConfig(
            # _target=RayPruningDataManager,
            dataparser=MerfDataParserConfig(
            is_transform = True,
            downscale_factor=1,
            train_split_fraction=0.95,
            orientation_method='vertical',
            center_method='poses',
            scene_scale=2.0,
            scale_factor=1.5,
            z_limit_min=-0.18,
            z_limit_max=0.18,
            ),
            eval_num_rays_per_batch=4096,
            train_num_rays_per_batch=4096,
        ),
        model=MERFModelConfig(_target=MERFModel,
                               alpha_threshold_param=(10000, 30000, 5e-4, 1e-2, 20000) 
                              ),
    ),
    max_num_iterations=50000,
    steps_per_save=2000,
    steps_per_eval_batch=10000,
    steps_per_eval_image=500,
    steps_per_eval_all_images=28000,
       optimizers={ 
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        },
        "deferred_mlp": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=50000),
        }
    },
    gradient_accumulation_steps = 4,
)
MERFNS = MethodSpecification(
    config=merf_config, description="Unofficial implementation of MERF paper"
)

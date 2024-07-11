"""
MERF Pipeline class.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
import gc

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)

from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from nerfstudio.models.base_model import Model, ModelConfig
import numpy as np
import torch
from merf.coord import stepsize_in_squash, contract
from merf.grid_utils import calculate_grid_config, world_to_grid
from merf.merf_model import MERFModel,MERFModelConfig

@dataclass
class BlockMERFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: BlockMERFPipeline)
    """target class to instantiate"""


class BlockMERFPipeline(VanillaPipeline):
 
    def baking_merf(self,lod,chunk_idx):
        self.model._model[lod][chunk_idx].baking_merf_model(self.datamanager,self.model,lod,chunk_idx)

   
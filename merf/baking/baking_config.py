from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

@dataclass
class BakingConfig:
    baking_path: Path = Path("outputs/baking")
    # using sub sampling factor and downscale factor when loading dataset to speed up computing alive voxels
    sub_sampling_factor: int = 4
    downscale_factor: int = 8
    eval_chunk:int = 32768
    # whether to use alpha culling
    use_alpha_culling: bool = True
    # alpha threshold and weight_threshold for culling when use alpha culling is True
    alpha_threshold: float = 0.01
    weight_threshold: float = 0.005
    save_alive_voxels_to_disk: bool = False
    load_alive_voxels_from_disk: bool = False
    load_occ_grid: bool = False
    # use disk not memory to load numpy array
    mmap_mode = False
    data_block_size: int = 8
    batch_size_in_blocks: int = 2**9
    batch_size_triplane: int = 2**22
    DEBUG: bool = False

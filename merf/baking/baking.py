#!/usr/bin/env python
"""
baking.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from merf.baking.utils import baking_setup
from merf.baking.baking_config import BakingConfig
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class MERFBaking:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    # Path to save baking result
    baking_config: BakingConfig = BakingConfig()
    
    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = baking_setup(self.load_config, baking_config=self.baking_config)
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True,exist_ok=True)
        assert self.baking_config.baking_path is not None
        self.baking_config.baking_path.mkdir(parents=True,exist_ok=True)
        metrics_dict = pipeline.baking_merf()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MERFBaking).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(MERFBaking)  # noqa

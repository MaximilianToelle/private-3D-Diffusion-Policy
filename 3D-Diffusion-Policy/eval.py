if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
from hydra.core.hydra_config import HydraConfig

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    checkpoint_tag = getattr(cfg.checkpoint, 'checkpoint_tag', 'latest')
    
    # Detect if we should use dataset by checking if dataset path was overridden
    overrides = HydraConfig.get().overrides.task
    use_dataset = any(o.startswith('task.dataset.zarr_path=') for o in overrides)

    # Resolve checkpoint path from Hydra output dir without constructing the full workspace
    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    ckpt_path = output_dir / 'checkpoints' / f'{checkpoint_tag}.ckpt'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build workspace from the checkpoint's own config so the model architecture
    # always matches the trained weights (critical for ablations).
    workspace = TrainDP3Workspace.create_from_checkpoint(path=str(ckpt_path))
    
    if use_dataset:
        workspace.cfg.task.dataset.zarr_path = cfg.task.dataset.zarr_path
        
    # Override output_dir so eval artifacts go to the right place
    workspace._output_dir = str(output_dir)
    workspace.eval(checkpoint_tag, use_dataset=use_dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dataset', type=str, default=None, help='Path to dataset for in-distribution evaluation')
    args, unknown = parser.parse_known_args()
    
    # Reconstruct sys.argv for Hydra, removing the user-facing --use_dataset option
    sys.argv = [sys.argv[0]] + unknown
    
    # If path is provided, append it as a native Hydra config override
    if args.use_dataset is not None:
        sys.argv.append(f'task.dataset.zarr_path={args.use_dataset}')
        
    main()

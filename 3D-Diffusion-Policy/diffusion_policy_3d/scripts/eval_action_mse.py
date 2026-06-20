"""
Evaluate action MSE of a trained policy on a batch from the training distribution.

Usage (from 3D-Diffusion-Policy/3D-Diffusion-Policy/):
    python diffusion_policy_3d/scripts/eval_action_mse.py \
        --checkpoint path/to/checkpoint.ckpt \
        --dataset path/to/dataset.zarr \
        --batch-size 128 \
        --num-batches 10 \
        --seed 42
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    # Script: .../3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/scripts/eval_action_mse.py
    #   parent^1 = scripts/
    #   parent^2 = diffusion_policy_3d/
    #   parent^3 = 3D-Diffusion-Policy/ (inner, where train.py lives)
    #   parent^5 = submodules/ (ROOT_DIR, matching train.py/eval.py convention)
    DP3_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent.parent)
    sys.path.insert(0, DP3_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import copy
import torch
import numpy as np
import random
import hydra
from omegaconf import OmegaConf
from termcolor import cprint

from train import TrainDP3Workspace
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.dataset.data_augmentations import GaussianCompose
from diffusion_policy_3d.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)


def evaluate_action_mse(checkpoint_path: str, dataset_path: str, 
                        batch_size: int = 128, num_batches: int = 10, seed: int = 42):
    """
    Load a trained policy from checkpoint and evaluate action MSE on
    batches sampled from the training distribution.
    """
    # ── Load workspace from checkpoint ──
    checkpoint_path = str(pathlib.Path(checkpoint_path).resolve())
    cprint(f"Loading checkpoint: {checkpoint_path}", "cyan")
    workspace = TrainDP3Workspace.create_from_checkpoint(path=checkpoint_path)
    cfg = copy.deepcopy(workspace.cfg)

    # ── Seed everything ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Configure dataset ──
    if dataset_path is not None:
        cfg.task.dataset.zarr_path = dataset_path
    dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseDataset)
    normalizer = dataset.get_normalizer()
    cprint(f"Dataset loaded: {len(dataset)} samples from {cfg.task.dataset.zarr_path}", "cyan")
    cprint(f"Training episodes: {np.nonzero(dataset.global_train_mask)[0].tolist()}", "cyan")

    # ── Configure augmentations (train augmentations to model training distribution) ──
    if 'train_augmentations' in cfg.task and cfg.task.train_augmentations is not None:
        augmentations = [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.train_augmentations]
        train_augmentations = GaussianCompose(augmentations)
    else:
        raise ValueError("Train data augmentations must be explicitly provided in the dataset config!")

    # ── Configure policy ──
    device = torch.device(cfg.training.device)
    policy = workspace.model
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    cprint(f"Policy loaded (EMA={cfg.training.use_ema}, epoch={workspace.epoch}, "
           f"global_step={workspace.global_step})", "cyan")

    # ── Evaluate action MSE over multiple batches ──
    all_mse = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Sample a batch from the training distribution
            sample_indices = torch.randint(len(dataset), (batch_size,)).tolist()
            batch = torch.utils.data.default_collate([dataset[i] for i in sample_indices])
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            batch = train_augmentations(batch)

            obs_dict = batch['obs']
            gt_action = batch['action']

            result = policy.predict_action(obs_dict)
            pred_action = result['action_pred']
            mse = torch.nn.functional.mse_loss(pred_action, gt_action).item()
            all_mse.append(mse)

            cprint(f"  Batch {batch_idx + 1}/{num_batches}: MSE = {mse:.6f}", "yellow")

            del batch, obs_dict, gt_action, result, pred_action

    # ── Report results ──
    mean_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    min_mse = np.min(all_mse)
    max_mse = np.max(all_mse)

    cprint("=" * 50, "green")
    cprint(f"Action MSE Results ({num_batches} batches, {batch_size} samples each)", "green")
    cprint(f"  Mean:  {mean_mse:.6f}", "green")
    cprint(f"  Std:   {std_mse:.6f}", "green")
    cprint(f"  Min:   {min_mse:.6f}", "green")
    cprint(f"  Max:   {max_mse:.6f}", "green")
    cprint(f"  Epoch: {workspace.epoch}", "green")
    cprint("=" * 50, "green")

    return {
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'min_mse': min_mse,
        'max_mse': max_mse,
        'all_mse': all_mse,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate action MSE of a trained policy")
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to checkpoint file (.ckpt)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to dataset (.zarr). If not provided, uses the path from the checkpoint config.')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='Batch size for evaluation (default: 128)')
    parser.add_argument('--num-batches', '-n', type=int, default=10,
                        help='Number of batches to evaluate (default: 10)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    evaluate_action_mse(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
    )

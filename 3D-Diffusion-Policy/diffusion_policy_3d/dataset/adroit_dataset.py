from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class AdroitDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == "__main__":
    import hydra
    from torch.utils.data import DataLoader
    import time
    import tqdm
    from diffusion_policy_3d.common.pytorch_util import dict_apply
    from diffusion_policy_3d.dataset.data_augmentations import GaussianCompose
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    @hydra.main(
        version_base=None,
        config_path="../config",
        config_name="dp3"
    )
    def main(cfg):
        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"
        print(f"Dataset instantiated. Length: {len(dataset)}")
        
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        print("DataLoader instantiated.")

        # Test loading for 3 epochs
        num_epochs = 3
        print(f"Testing batch loading for {num_epochs} epochs...")
        
        device = torch.device(cfg.training.device)

        if 'train_augmentations' in cfg.task and cfg.task.train_augmentations is not None:
            augmentations = [hydra.utils.instantiate(t_cfg) for t_cfg in cfg.task.train_augmentations]
            train_augmentations = GaussianCompose(augmentations)

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                t_before_next_batch = time.time()
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                   
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    t1_1 = time.time()

                    # augmentations
                    if 'train_augmentations' in cfg.task and cfg.task.train_augmentations is not None:
                        batch = train_augmentations(batch)
                    t1_2 = time.time()

                    print(f" batch generation time {t1-t_before_next_batch:.3f}")
                    print(f" device transfer time: {t1_1-t1:.3f}")
                    print(f" augmentations time: {t1_2-t1_1:.3f}")

                    t_before_next_batch = time.time()

    main()

import torch
from pytorch3d.ops import sample_farthest_points, ball_query


class GaussianCompose:
    """Composes several transforms together sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch


class GaussianFPS:
    """
    Standard Farthest Point Sampling on the GPU.
    Used for evaluation and baseline models to reduce the number of Gaussians.
    """
    def __init__(self, num_samples=1024):
        self.num_samples = num_samples

    def __call__(self, batch):
        obs = batch['obs']
        B, T, N, _ = obs['gs_positions'].shape
        
        # Grab t=0 positions and the lengths tensor
        points_t0 = obs['gs_positions'][:, 0, :, :].to(torch.float32)
        lengths = obs.get('gs_length', None)
        if 'gs_length' in obs:
            del batch['obs']['gs_length']       # not needed afterwards
        
        # Batched GPU FPS with strict boundary enforcement
        _, sampled_indices = sample_farthest_points(
            points_t0, 
            lengths=lengths, 
            K=self.num_samples
        )
        
        keys_to_subsample = [k for k in obs.keys() if k not in ['agent_pos', 'gs_length']]
            
        # Apply indices across all features and time steps
        for key in keys_to_subsample:
            tensor = obs[key]       # (B, T, 32768, D)
            D = tensor.shape[-1]
            
            # Expand indices from (B, num_samples) to (B, T, num_samples, D)
            gather_idx = sampled_indices.view(B, 1, self.num_samples, 1).expand(-1, T, -1, D)
            
            # Subsample to final target size
            obs[key] = torch.gather(tensor, dim=2, index=gather_idx).to(torch.float32)
            
        return batch


class GaussianFPSAndBallQuery:
    """
    Augmentation that runs FPS, then for each centroid, randomly selects
    one neighboring Gaussian within a specified radius using a ball query.
    """
    def __init__(self, num_samples=1024, radius=0.02, max_neighbors=16):
        self.num_samples = num_samples
        self.radius = radius
        self.max_neighbors = max_neighbors

    def __call__(self, batch):
        obs = batch['obs']
        B, T, N, _ = obs['gs_positions'].shape
        
        # Grab t=0 positions and the lengths tensor
        points_t0 = obs['gs_positions'][:, 0, :, :].to(torch.float32)
        lengths = obs.get('gs_length', None)
        if 'gs_length' in obs:
            del batch['obs']['gs_length']
            
        # 1. FPS to get centroids
        _, centroid_idx = sample_farthest_points(
            points_t0, 
            lengths=lengths, 
            K=self.num_samples
        )
        
        # Extract the centroid coordinates
        # Expand centroid_idx to (B, num_samples, 3) to gather from points_t0
        centroids = torch.gather(points_t0, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, 3))
        
        # 2. Ball query around centroids
        # How many non-padded Gaussians are in the query space for each batch -> no padding
        lengths1 = torch.full((B,), self.num_samples, dtype=torch.long, device=points_t0.device)
        dists, idxs, nn = ball_query(
            p1=centroids, 
            p2=points_t0, 
            lengths1=lengths1,     
            lengths2=lengths,      # how many non-padded Gaussians are in the search space for each batch
            K=self.max_neighbors, 
            radius=self.radius
        )
        
        # 3. Randomly select one valid neighbor for each centroid
        # idxs has shape (B, num_samples, max_neighbors). Padded entries have value -1.
        valid_mask = idxs != -1
        valid_counts = valid_mask.sum(dim=-1) # shape: (B, num_samples)
        
        # Random integers between 0 and max_neighbors
        rand_cols = torch.randint(0, self.max_neighbors, (B, self.num_samples), device=points_t0.device)
        
        # Clamp rand_cols to strictly be within the valid count for each point
        # Modulo ensures it wraps into a valid column index
        rand_cols = torch.remainder(rand_cols, torch.clamp(valid_counts, min=1))
        
        # Extract the final indices
        # We gather from the max_neighbors dimension using rand_cols
        sampled_indices = torch.gather(idxs, 2, rand_cols.unsqueeze(-1)).squeeze(-1)
        
        keys_to_subsample = [k for k in obs.keys() if k not in ['agent_pos', 'gs_length']]
        
        # Apply indices across all features and time steps
        for key in keys_to_subsample:
            tensor = obs[key]
            D = tensor.shape[-1]
            
            # Expand indices from (B, num_samples) to (B, T, num_samples, D)
            # This ensures EXACTLY the same Gaussians are selected across all timesteps T
            gather_idx = sampled_indices.view(B, 1, self.num_samples, 1).expand(-1, T, -1, D)
            
            # Subsample to final target size
            obs[key] = torch.gather(tensor, dim=2, index=gather_idx).to(torch.float32)
            
        return batch

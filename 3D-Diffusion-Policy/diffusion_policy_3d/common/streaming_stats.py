"""Streaming statistics over datasets too large to retain in memory."""

import torch


class StreamingPercentileHistogram:
    """Constant-memory percentile estimation over streamed values, one histogram per slot
    (e.g. one per (timestep, dim) pair of a per-timestep normalizer).

    Bin counts are exact integers, so the bin holding the target rank is located exactly;
    the returned bin center deviates from the exact order statistic by at most one bin
    width (2 * value_range / num_bins). The center is clamped into the slot's observed
    [min, max] -- the true percentile always lies there, so this only tightens the
    estimate, and for point-mass slots (e.g. the anchor timestep of relative proprio,
    identically 0) it becomes exact instead of half-a-bin off, which the degenerate-slot
    scale 2/1e-6 would otherwise amplify. Single pass, deterministic, and the observed
    range is asserted at read time, so an insufficient value_range fails loudly instead
    of silently clipping the percentile.
    """

    def __init__(self, num_slots, value_range, num_bins):
        self.num_slots = num_slots
        self.value_range = value_range
        self.num_bins = num_bins
        self.bin_width = 2 * value_range / num_bins
        # One flat counter tensor, so a single index_add_ per update covers all slots.
        # int64 because one gs slot accumulates billions of values at large trajectory counts.
        self._counts = torch.zeros(num_slots * num_bins, dtype=torch.int64)
        self._slot_offsets = torch.arange(num_slots).unsqueeze(-1) * num_bins    # (num_slots, 1)
        self._values_per_slot = 0
        self._slot_min = torch.full((num_slots,), float('inf'))
        self._slot_max = torch.full((num_slots,), float('-inf'))

    def update(self, values):
        """values: (num_slots, num_values) -- every slot receives num_values values."""
        assert values.shape[0] == self.num_slots, \
            f"expected (num_slots={self.num_slots}, num_values), got {tuple(values.shape)}"
        self._slot_min = torch.minimum(self._slot_min, values.min(dim=1).values)
        self._slot_max = torch.maximum(self._slot_max, values.max(dim=1).values)
        bin_indices = ((values + self.value_range) / self.bin_width).long()
        bin_indices.clamp_(0, self.num_bins - 1)
        flat_indices = (bin_indices + self._slot_offsets).flatten()
        self._counts.index_add_(0, flat_indices, torch.ones_like(flat_indices))
        self._values_per_slot += values.shape[1]

    def percentile(self, percentile):
        """(num_slots,) float32 percentile values at rank int(p/100 * (count-1)) per slot."""
        assert self._values_per_slot > 0, "percentile of an empty histogram"
        observed_min, observed_max = self._slot_min.min().item(), self._slot_max.max().item()
        assert -self.value_range <= observed_min and observed_max <= self.value_range, (
            f"streamed values span [{observed_min:.3f}, {observed_max:.3f}] and exceed the "
            f"+-{self.value_range} histogram range -- raise value_range (config: "
            "percentile_histogram_range).")
        cumulative_counts = torch.cumsum(self._counts.view(self.num_slots, self.num_bins), dim=1)
        target_rank = int(percentile / 100.0 * (self._values_per_slot - 1))
        ranks = torch.full((self.num_slots, 1), target_rank, dtype=torch.int64)
        bin_indices = torch.searchsorted(cumulative_counts, ranks, right=True).squeeze(1)
        bin_centers = ((bin_indices.to(torch.float64) + 0.5) * self.bin_width - self.value_range).float()
        return torch.clamp(bin_centers, self._slot_min, self._slot_max)

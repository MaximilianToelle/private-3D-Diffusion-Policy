import torch.distributed as dist


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def print_dist(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

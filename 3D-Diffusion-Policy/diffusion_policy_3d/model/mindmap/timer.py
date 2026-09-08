class Timer:
    """No-op stand-in for nvblox_torch.timer.Timer so the vendored model runs
    without nvblox installed (training needs no reconstruction)."""

    def __init__(self, name=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stop(self):
        pass

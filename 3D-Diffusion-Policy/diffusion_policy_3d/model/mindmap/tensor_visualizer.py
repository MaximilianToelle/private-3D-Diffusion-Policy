class TensorVisualizer:
    """No-op stand-in for mindmap.visualization.tensor_visualizer.TensorVisualizer
    (wandb/debug tensor plotting stripped from the vendored copy). Accepts any
    method call and does nothing."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop

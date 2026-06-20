"""Runtime defaults for GPU-oriented runs."""

DEFAULT_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
}


def gpu_runtime_env(base=None):
    """Return environment variables that expose the primary CUDA device."""
    env = dict(base or {})
    env.update(DEFAULT_ENV)
    return env


def gpu_device() -> str:
    """Return the best available torch device string.

    Queries ``torch.cuda.current_device()`` rather than hard-coding GPU 0, so
    the function respects ``CUDA_VISIBLE_DEVICES`` and multi-GPU environments.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except ImportError:
        pass
    return "cpu"

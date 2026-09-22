"""GuardLens NAACL causal-localization package."""

from guardlens.config import GuardLensConfig

__all__ = [
    "GuardLensConfig",
    "GuardLens",
    "GuardLensDataset",
    "GuardLensCollator",
    "GuardLensLoss",
]


def __getattr__(name):
    # Data audits and evaluation reporting must not require a GPU runtime.
    from importlib import import_module
    modules = {
        "GuardLens": "guardlens.models.guardlens",
        "GuardLensDataset": "guardlens.data.dataset",
        "GuardLensCollator": "guardlens.data.dataset",
        "GuardLensLoss": "guardlens.training.loss",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value

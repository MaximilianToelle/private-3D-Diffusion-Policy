"""The OmegaConf resolvers the configs use (config/base.yaml). Every entry point that composes a
config calls register_config_resolvers() first."""
import pathlib

from omegaconf import OmegaConf

# the gsplat_policy repository this fork is a submodule of: ${repo_root:}/datasets holds the datasets
REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[5])


def merge_disjoint(*parts):
    """${merge:a,b}: one dictionary from several, each key from exactly one of them, so no part
    silently overrides another's value."""
    containers = [OmegaConf.to_container(part, resolve=True, throw_on_missing=True) for part in parts]
    keys = [key for container in containers for key in container]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    assert not duplicates, f"merge: {duplicates} defined in more than one part"
    return OmegaConf.create({key: value for container in containers for key, value in container.items()})


def register_config_resolvers():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("merge", merge_disjoint, replace=True)
    OmegaConf.register_new_resolver("repo_root", lambda: REPO_ROOT, replace=True)

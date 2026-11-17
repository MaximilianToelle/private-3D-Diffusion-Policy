"""The OmegaConf resolvers the configs use (config/base.yaml). Every entry point that composes a
config calls register_config_resolvers() first."""
from omegaconf import OmegaConf


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

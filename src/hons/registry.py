import importlib
import pkgutil
from typing import Any

from hons.config import Config, StageConfig

# stage name -> implementation name -> class
REGISTRY: dict[str, dict[str, type]] = {}


def register(stage: str, name: str):
    # Class decorator: @register('s4_novelty', 'exact_hash')
    def decorator(cls: type) -> type:
        impls = REGISTRY.setdefault(stage, {})
        if name in impls:
            raise ValueError(f'{stage} already has an implementation called {name!r}')
        impls[name] = cls
        return cls
    return decorator


def build(stage: str, section: StageConfig) -> Any:
    # A disabled stage gets its `off` implementation, and its params are ignored so a stage
    # can be switched off without deleting them
    name = section.impl if section.enabled else 'off'
    if name is None:
        raise ValueError(f'[{stage}] is enabled but names no impl')
    impls = REGISTRY.get(stage, {})
    if name not in impls:
        known = ', '.join(sorted(impls)) or 'none registered'
        raise ValueError(f'[{stage}] has no implementation {name!r} (known: {known})')
    params = section.params if section.enabled else {}
    try:
        return impls[name](**params)
    except TypeError as e:
        raise TypeError(f'[{stage}] impl {name!r}: {e}') from e


def build_stages(config: Config) -> dict[str, Any]:
    # Implementations register themselves when their module is imported, so import everything
    # in hons/stages first. New stage modules are picked up without being listed anywhere
    import hons.stages
    for module in pkgutil.iter_modules(hons.stages.__path__):
        importlib.import_module(f'hons.stages.{module.name}')
    return {stage: build(stage, section) for stage, section in config.stages.items()}

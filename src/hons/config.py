import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# The server's rule for RunMetadata.runtag, checked here so a bad tag fails before day 0 rather than at start_run
RUNTAG_PATTERN = re.compile(r'^[a-zA-Z0-9_-][a-zA-Z0-9_.-]{1,19}$')

# Stages 2, 3 and 5 have no `off`, so a run must name an implementation for each. Stage 6 has no options
REQUIRED_STAGES = ('s2_score', 's3_tail', 's5_gate')
# A missing section means the stage's `off` implementation
OPTIONAL_STAGES = ('s0_fragment', 's1_questions', 's4_novelty', 's7_propose', 's8_state')


@dataclass(frozen=True)
class RunConfig:
    # The [run] section. Paths are relative to the directory the run is launched from
    runtag: str
    description: str
    topics: Path
    mode: Literal['replay', 'live'] = 'replay'
    days: tuple[int, int] | None = None     # sequence numbers [start, end), so [0, 3] is the first three days
    seed: int = 0
    db: Path = Path('docs.db')              # replay source
    base_url: str = 'http://127.0.0.1:8000' # live server

    def __post_init__(self):
        if not RUNTAG_PATTERN.match(self.runtag):
            raise ValueError(f'runtag must be 2-20 letters, digits, ".", "-" or "_", got {self.runtag!r}')
        if self.mode not in ('replay', 'live'):
            raise ValueError(f'mode must be "replay" or "live", got {self.mode!r}')
        object.__setattr__(self, 'topics', Path(self.topics))
        object.__setattr__(self, 'db', Path(self.db))
        if self.days is not None:
            start, end = self.days
            if not 0 <= start < end:
                raise ValueError(f'days must be [start, end) with 0 <= start < end, got {list(self.days)}')
            if self.mode == 'live' and start != 0:
                raise ValueError('a live run always starts at day 0; use replay for a later start')
            object.__setattr__(self, 'days', (start, end))


@dataclass(frozen=True)
class StageConfig:
    # params is the rest of the section. The registry passes it to the implementation's
    # constructor, so each implementation declares (and rejects) its own keys
    enabled: bool = False
    impl: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, section: dict[str, Any]) -> 'StageConfig':
        params = dict(section)
        enabled = params.pop('enabled', True)   # writing the section at all means "on" unless it says otherwise
        impl = params.pop('impl', None)
        if not isinstance(enabled, bool):
            raise ValueError(f'enabled must be true or false, got {enabled!r}')
        return cls(enabled, impl, params)


@dataclass(frozen=True)
class Config:
    run: RunConfig
    stages: dict[str, StageConfig]
    models: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'Config':
        unknown = raw.keys() - {'run', 'models', *REQUIRED_STAGES, *OPTIONAL_STAGES}
        if unknown:
            raise ValueError(f'unknown config sections: {sorted(unknown)}')
        if 'run' not in raw:
            raise ValueError('config needs a [run] section')

        stages = {}
        for name in REQUIRED_STAGES:
            if name not in raw:
                raise ValueError(f'[{name}] is required, it has no off implementation')
            stage = StageConfig.from_dict(raw[name])
            if not stage.enabled or stage.impl is None:
                raise ValueError(f'[{name}] cannot be off and needs an impl')
            stages[name] = stage
        for name in OPTIONAL_STAGES:
            stages[name] = StageConfig.from_dict(raw[name]) if name in raw else StageConfig()

        return cls(
            run=RunConfig(**raw['run']),        # unknown or missing keys raise TypeError here
            stages=stages,
            models=tuple(raw.get('models', {}).get('used', [])),
            raw=raw,
        )

    def digest(self) -> str:
        # Goes in the run directory name (runs/<runtag>-<digest>/) so results trace back to the exact toggles
        canonical = json.dumps(self.raw, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def load_config(path: Path) -> Config:
    with Path(path).open('rb') as f:
        return Config.from_dict(tomllib.load(f))

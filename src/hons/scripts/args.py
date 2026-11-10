import argparse
from pathlib import Path

from hons.config import Config, load_config


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('config', type=Path)
    parser.add_argument('--days', type=int, nargs=2, metavar=('START', 'END'),
                        help='sequence numbers [START, END) to run, so "--days 0 30" is the first 30 days. '
                             'Omit to use the config, which runs every day unless it sets days. '
                             'Pass the same values to run and evaluate, as they set the run directory')


def load_with_days(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if args.days is None:
        return config
    raw = {**config.raw, 'run': {**config.raw['run'], 'days': list(args.days)}}
    return Config.from_dict(raw)

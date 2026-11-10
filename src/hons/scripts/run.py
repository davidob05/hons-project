import argparse
from pathlib import Path

from hons.config import load_config
from hons.runner import run


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline from a TOML config; writes runs/<runtag>-<digest>/run.json')
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == '__main__':
    main()

import argparse

from hons.runner import run
from hons.scripts.args import add_config_arguments, load_with_days


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline from a TOML config; writes runs/<runtag>-<digest>/run.json')
    add_config_arguments(parser)
    args = parser.parse_args()
    print(run(load_with_days(args)))


if __name__ == '__main__':
    main()

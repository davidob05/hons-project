import argparse
from pathlib import Path

from hons.config import load_config
from hons.metrics.recall import evaluate, report
from hons.runner import run_dir_for


def main():
    parser = argparse.ArgumentParser(description='Report recall of a finished run against the expected documents')
    parser.add_argument('config', type=Path, help='the config the run used; it names the run directory and the database')
    parser.add_argument('--run-dir', type=Path, help='defaults to runs/<runtag>-<digest> from the config')
    parser.add_argument('--expected', type=Path, default=Path('data/dev-topics-all-docs.jsonl'))
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.run_dir or run_dir_for(config)
    report(evaluate(run_dir / 'run.json', args.expected, config.run.db), run_dir / 'eval.json')


if __name__ == '__main__':
    main()

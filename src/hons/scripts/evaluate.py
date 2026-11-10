import argparse
from pathlib import Path

from hons.metrics.evaluation import evaluate, report
from hons.runner import run_dir_for
from hons.scripts.args import add_config_arguments, load_with_days


def main():
    parser = argparse.ArgumentParser(description='Report recall of a finished run against the expected documents')
    add_config_arguments(parser)
    parser.add_argument('--run-dir', type=Path, help='defaults to runs/<runtag>-<digest> from the config and --days')
    parser.add_argument('--expected', type=Path, default=Path('data/dev-topics-all-docs.jsonl'))
    args = parser.parse_args()

    config = load_with_days(args)
    run_dir = args.run_dir or run_dir_for(config)
    report(evaluate(run_dir / 'run.json', args.expected, config.run.db), run_dir / 'eval.json')


if __name__ == '__main__':
    main()

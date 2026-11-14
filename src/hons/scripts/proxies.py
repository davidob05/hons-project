import argparse
from pathlib import Path

from hons.metrics.proxies import evaluate_proxies, report
from hons.runner import run_dir_for
from hons.scripts.args import add_config_arguments, load_with_days


def main():
    parser = argparse.ArgumentParser(description='Report the proxy metrics P1 to P5 for a finished run')
    add_config_arguments(parser)
    parser.add_argument('--run-dir', type=Path, help='defaults to runs/<runtag>-<digest> from the config and --days')
    parser.add_argument('--expected', type=Path, default=Path('data/dev-topics-all-docs.jsonl'))
    parser.add_argument('--empty', action='store_true', help='score an empty run over the same days instead of the run')
    args = parser.parse_args()

    config = load_with_days(args)
    run_dir = args.run_dir or run_dir_for(config)
    result = evaluate_proxies(run_dir / 'run.json', args.expected, config.run.db, empty=args.empty)
    report(result, run_dir / ('proxies_empty.json' if args.empty else 'proxies.json'))


if __name__ == '__main__':
    main()

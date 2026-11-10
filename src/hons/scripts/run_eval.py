import argparse
import subprocess
import sys

from hons.scripts.args import add_config_arguments


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline, then evaluate the finished run')
    add_config_arguments(parser)
    parser.add_argument('--expected', help='passed to the evaluation; its default is used if omitted')
    args = parser.parse_args()

    shared = [str(args.config)]
    if args.days is not None:
        shared += ['--days', *map(str, args.days)]

    subprocess.run([sys.executable, '-m', 'hons.scripts.run', *shared], check=True)
    evaluation = [sys.executable, '-m', 'hons.scripts.evaluate', *shared]
    if args.expected is not None:
        evaluation += ['--expected', args.expected]
    subprocess.run(evaluation, check=True)


if __name__ == '__main__':
    main()

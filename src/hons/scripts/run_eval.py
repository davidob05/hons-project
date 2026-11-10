import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline, then evaluate the finished run')
    parser.add_argument('config', type=Path)
    parser.add_argument('--expected', type=Path, help='passed to the evaluation; its default is used if omitted')
    args = parser.parse_args()

    # separate processes, so the evaluation reads only what the run wrote to disk, exactly as a later standalone evaluation would
    subprocess.run([sys.executable, '-m', 'hons.scripts.run', str(args.config)], check=True)
    command = [sys.executable, '-m', 'hons.scripts.evaluate', str(args.config)]
    if args.expected is not None:
        command += ['--expected', str(args.expected)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

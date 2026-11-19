import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from hons.config import Config, load_config
from hons.runner import run
from hons.stages import s2_bm25, s2_scores

SCORING = {'dense', 'fused', 'rerank'}
RELEVANT = {'awareness', 'assessment', 'action'}


class Meter:
    def __init__(self):
        self.bm25: list[float] = []
        self.dense: list[float] = []
        self.fuse: list[float] = []
        self.fragments: list[int] = []
        self.documents: list[int] = []

    def reset(self):
        self.__init__()


def instrument(meter: Meter) -> None:
    original_bm25 = s2_scores.BM25Day.run
    original_python = s2_bm25.BM25Accumulated.run
    original_dense = s2_scores.DenseEncoder.run
    original_fuse = s2_scores.FusedScorer.fuse

    def timed_bm25(self, fragments, questions):
        start = time.perf_counter()
        result = original_bm25(self, fragments, questions)
        meter.bm25.append(time.perf_counter() - start)
        return result

    def timed_dense(self, fragments, questions):
        meter.documents.append(len(fragments.fragments))
        meter.fragments.append(sum(len(by_index) for by_index in fragments.fragments.values()))
        start = time.perf_counter()
        result = original_dense(self, fragments, questions)
        meter.dense.append(time.perf_counter() - start)
        return result

    def timed_fuse(self, lexical, semantic, questions):
        start = time.perf_counter()
        result = original_fuse(self, lexical, semantic, questions)
        meter.fuse.append(time.perf_counter() - start)
        return result

    def timed_python(self, fragments, questions):
        start = time.perf_counter()
        result = original_python(self, fragments, questions)
        meter.bm25.append(time.perf_counter() - start)
        return result

    s2_scores.BM25Day.run = timed_bm25
    s2_bm25.BM25Accumulated.run = timed_python
    s2_scores.DenseEncoder.run = timed_dense
    s2_scores.FusedScorer.fuse = timed_fuse


def steady(values: list[float]) -> float | None:
    usable = values[1:] if len(values) > 1 else values
    return sum(usable) / len(usable) if usable else None


def load_rankings(run_file: Path) -> dict[tuple[str, str, str], list[str]]:
    rankings = {}
    with open(run_file) as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for day, questions in record['results'].items():
                for question in questions:
                    rankings[(record['topic'], question['qid'], day)] = [hit['doc_id'] for hit in question['doc_ranking']]
    return rankings


def load_qrels(path: Path) -> dict[tuple[str, str, str], set[str]]:
    relevant: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for line in open(path):
        parts = line.split()
        if len(parts) == 5 and parts[4] in RELEVANT:
            relevant[(parts[0], parts[1], parts[2])].add(parts[3])
    return relevant


def quality(rankings, baseline, relevant, days: set[str]) -> dict:
    keys = [key for key in baseline if key[2] in days]
    both = [key for key in keys if baseline[key] and rankings.get(key)]
    out = {
        'top1_same': sum(1 for key in both if baseline[key][0] == rankings[key][0]) / len(both) if both else None,
        'baseline_top20_kept': (sum(len(set(baseline[key]) & set(rankings[key])) for key in both) / sum(len(baseline[key]) for key in both)) if both else None,
    }
    if relevant is not None:
        cells = [key for key in relevant if key[1] in days]
        hit = reciprocal = 0.0
        for tid, day, qid in cells:
            ranking = rankings.get((tid, qid, day), [])
            positions = [i for i, doc_id in enumerate(ranking, start=1) if doc_id in relevant[(tid, day, qid)]]
            hit += 1 if positions else 0
            reciprocal += 1 / positions[0] if positions else 0.0
        out.update({'answerable': len(cells), 'found_in_top20': hit / len(cells) if cells else None,
                    'rr': reciprocal / len(cells) if cells else None})
    return out


def main():
    parser = argparse.ArgumentParser(description='Time the Stage 2 scorer of one or more configs on the same days and compare their output; the first config is the baseline')
    parser.add_argument('configs', nargs='+', type=Path)
    parser.add_argument('--topics', type=Path, help='override the topics file, for example the 665-question test topics')
    parser.add_argument('--days', type=int, nargs=2, default=[0, 20], metavar=('START', 'END'))
    parser.add_argument('--qrels', type=Path, help='a qrels file, to report how often a relevant document reaches the top 20')
    parser.add_argument('--keep-cache', action='store_true', help='leave the score caches on; the timings then measure the cache, not the model')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    meter = Meter()
    instrument(meter)
    relevant = load_qrels(args.qrels) if args.qrels else None
    days_run = args.days[1] - args.days[0]
    results, baseline = [], None
    for path in args.configs:
        config = load_config(path)
        raw = json.loads(json.dumps(config.raw, default=str))
        raw['run']['days'] = list(args.days)
        if args.topics:
            raw['run']['topics'] = str(args.topics)
        scorer = raw.get('s2_score', {})
        if scorer.get('impl') in SCORING and not args.keep_cache:
            scorer['cache_dir'] = ''
        meter.reset()
        started = time.perf_counter()
        run_dir = run(Config.from_dict(raw))
        total = time.perf_counter() - started
        rankings = load_rankings(run_dir / 'run.json')
        days = {key[2] for key in rankings}
        if baseline is None:
            baseline = rankings
        row = {
            'config': path.stem, 'run_dir': str(run_dir), 'seconds': total, 'seconds_per_day': total / days_run,
            'bm25_per_day': steady(meter.bm25), 'dense_per_day': steady(meter.dense), 'fuse_per_day': steady(meter.fuse),
            'documents_encoded_per_day': sum(meter.documents) / len(meter.documents) if meter.documents else None,
            'fragments_encoded_per_day': sum(meter.fragments) / len(meter.fragments) if meter.fragments else None,
            **quality(rankings, baseline, relevant, days),
        }
        results.append(row)
        print(f'finished {path.stem}: {total:.1f}s', flush=True)

    base = results[0]
    print(f'\n{days_run} days from day {args.days[0]}; per-day times exclude the first day (start-up); the first config is the baseline')
    header = f'{"config":36}{"s/day":>8}{"bm25":>7}{"dense":>7}{"docs enc.":>10}{"frag enc.":>10}{"speedup":>9}{"top1 same":>11}{"top20 kept":>11}'
    if relevant is not None:
        header += f'{"found@20":>10}{"RR":>7}'
    print(header)
    for row in results:
        speed = (base['dense_per_day'] / row['dense_per_day']) if row['dense_per_day'] and base['dense_per_day'] else None
        line = (f'{row["config"]:36}{row["seconds_per_day"]:8.2f}{(row["bm25_per_day"] or 0):7.2f}{(row["dense_per_day"] or 0):7.2f}'
                f'{(row["documents_encoded_per_day"] or 0):10.0f}{(row["fragments_encoded_per_day"] or 0):10.0f}'
                f'{(speed if speed else 0):8.2f}x{(row["top1_same"] if row["top1_same"] is not None else 0):11.3f}{(row["baseline_top20_kept"] if row["baseline_top20_kept"] is not None else 0):11.3f}')
        if relevant is not None:
            line += f'{(row.get("found_in_top20") or 0):10.3f}{(row.get("rr") or 0):7.3f}'
        print(line)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

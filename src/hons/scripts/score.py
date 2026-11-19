import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SCORER = Path('ignored/cdet-scorer-1.1.0/src')
HEADLINE = ("recip_rank'", 'ap_plain', "ap'", "ndcg'", "rbp'(p=0.8)", 'abstention', 'noise')
COUNTS = ('num_days', 'num_qs', 'num_ret', 'num_rel', 'num_rel_ret', 'unjudged')


def load_judgments(path: Path) -> dict[tuple[str, str], set[str]]:
    judged = {}
    for line in open(path):
        if not line.strip():
            continue
        topic = json.loads(line)
        for question in topic['questions']:
            judged[(topic['tid'], question['qid'])] = set(question.get('rel_docs', []))
    return judged


def days_of(db_path: Path, doc_ids: set[str]) -> dict[str, str]:
    db = sqlite3.connect(str(db_path))
    found = {}
    wanted = list(doc_ids)
    for start in range(0, len(wanted), 500):
        chunk = wanted[start:start + 500]
        query = f'select id, day from documents where id in ({",".join("?" * len(chunk))})'
        found.update(dict(db.execute(query, chunk).fetchall()))
    return found


def write_qrels(judged: dict[tuple[str, str], set[str]], days: dict[str, str], path: Path, level: str) -> int:
    lines = []
    for (tid, qid), docs in sorted(judged.items()):
        for doc_id in sorted(docs):
            if doc_id in days:
                lines.append(f'{tid} {days[doc_id]} {qid} {doc_id} {level}')
    path.write_text('\n'.join(lines) + '\n')
    return len(lines)


def score(scorer_src: Path, qrels: Path, topics: Path, run: Path, min_rel_level: str, max_depth: int) -> dict[tuple[str, str, str, str], str]:
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(scorer_src), os.environ.get('PYTHONPATH', '')]).rstrip(os.pathsep)}
    result = subprocess.run(
        [sys.executable, '-m', 'cdet_scorer.cli', '-l', min_rel_level, '-M', str(max_depth), str(qrels), str(topics), str(run)],
        capture_output=True, text=True, env=env)
    if result.returncode != 0:
        sys.exit(f'the scorer failed:\n{result.stderr.strip() or result.stdout.strip()}')
    values = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 5:
            values[tuple(parts[:4])] = parts[4]
    return values


def plain_ap(run: Path, judged: dict[tuple[str, str], set[str]], days: dict[str, str], max_depth: int) -> float | None:
    relevant: dict[tuple[str, str, str], set[str]] = {}
    for (tid, qid), docs in judged.items():
        for doc_id in docs:
            if doc_id in days:
                relevant.setdefault((tid, qid, days[doc_id]), set()).add(doc_id)
    rankings: dict[tuple[str, str, str], list[str]] = {}
    with open(run) as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for day, questions in record['results'].items():
                for question in questions:
                    hits = sorted(question['doc_ranking'], key=lambda hit: hit['score'], reverse=True)
                    rankings[(record['topic'], question['qid'], day)] = [hit['doc_id'] for hit in hits[:max_depth]]
    values = []
    for key, docs in relevant.items():
        found = 0
        total = 0.0
        for rank, doc_id in enumerate(rankings.get(key, []), start=1):
            if doc_id in docs:
                found += 1
                total += found / rank
        values.append(total / len(docs))
    return sum(values) / len(values) if values else None


def number(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def summarise(values: dict[tuple[str, str, str, str], str]) -> dict:
    overall = {key[3]: number(value) for key, value in values.items() if key[:3] == ('all', 'all', 'all') and number(value) is not None}
    topics = {}
    for key, value in values.items():
        topic, day, qid, measure = key
        if topic != 'all' and day == 'all' and qid == 'all' and number(value) is not None:
            topics.setdefault(topic, {})[measure] = number(value)
    return {'overall': overall, 'topics': topics}


def fmt(value: float | None, name: str) -> str:
    if value is None:
        return '      -'
    return f'{value:7.0f}' if name in COUNTS or name == 'noise' and value >= 100 else f'{value:7.3f}'


def print_table(results: dict[str, dict], per_topic: bool) -> None:
    sets = list(results)
    print(f'{"":14}' + ''.join(f'{name:>10}' for name in sets))
    for measure in (*HEADLINE, *COUNTS):
        print(f'{measure:14}' + ''.join(f'{fmt(results[s]["overall"].get(measure), measure):>10}' for s in sets))
    if per_topic:
        topics = sorted({topic for s in sets for topic in results[s]['topics']})
        for topic in topics:
            print(f'\n{topic}')
            for measure in HEADLINE:
                print(f'  {measure:12}' + ''.join(f'{fmt(results[s]["topics"].get(topic, {}).get(measure), measure):>10}' for s in sets))


def main():
    parser = argparse.ArgumentParser(description="Score a finished run with the track's official scorer, using the dev judgments (and the silver judgments if present) as the qrels")
    parser.add_argument('run', type=Path, help='a run.json file, or a run folder containing one')
    parser.add_argument('--scorer', type=Path, default=Path(os.environ.get('CDET_SCORER_SRC', DEFAULT_SCORER)),
                        help='the scorer\'s src folder, the one containing cdet_scorer/ (env CDET_SCORER_SRC)')
    parser.add_argument('--judgments', type=Path, default=Path('data/dev-topics-all-docs.jsonl'),
                        help='also used as the topics file, so it must list every question')
    parser.add_argument('--silver', type=Path, default=Path('data/silver-judgments.jsonl'))
    parser.add_argument('--no-silver', action='store_true')
    parser.add_argument('--db', type=Path, default=Path('docs.db'), help='used to find the day each judged document appears on')
    parser.add_argument('--min-rel-level', default='awareness', choices=['awareness', 'assessment', 'action'])
    parser.add_argument('--level', default='awareness', choices=['awareness', 'assessment', 'action'],
                        help='the relevance level written into the qrels for every judged document; the dev judgments are binary')
    parser.add_argument('--max-depth', type=int, default=100)
    parser.add_argument('--per-topic', action='store_true')
    parser.add_argument('--out', type=Path, help='defaults to official.json next to run.json')
    args = parser.parse_args()

    run = args.run / 'run.json' if args.run.is_dir() else args.run
    if not run.exists():
        sys.exit(f'no run file at {run}')
    if not (args.scorer / 'cdet_scorer').is_dir():
        sys.exit(f'the scorer was not found at {args.scorer}; pass --scorer or set CDET_SCORER_SRC')

    dev = load_judgments(args.judgments)
    sets = {'dev': dev}
    if not args.no_silver and args.silver.exists():
        silver = load_judgments(args.silver)
        sets['combined'] = {key: dev.get(key, set()) | silver.get(key, set()) for key in dev.keys() | silver.keys()}
    days = days_of(args.db, set().union(*(docs for judged in sets.values() for docs in judged.values())))

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, judged in sets.items():
            qrels = Path(tmp) / f'qrels_{name}'
            write_qrels(judged, days, qrels, args.level)
            results[name] = summarise(score(args.scorer, qrels, args.judgments, run, args.min_rel_level, args.max_depth))
            results[name]['overall']['ap_plain'] = plain_ap(run, judged, days, args.max_depth)

    print(f'official scorer, run {run}')
    print('prime measures are scored on question-days with a relevant document; abstention and noise on the other question-days of assessed days')
    print_table(results, args.per_topic)
    out = args.out or run.parent / 'official.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

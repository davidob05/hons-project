import argparse
import json
import os
import pickle
import sqlite3
import sys
from hashlib import sha1
from pathlib import Path

from hons.scripts.score import days_of, load_judgments
from hons.stages.s2_rerank import load_scorer


def load_topics(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    topics = {}
    for line in open(path):
        if not line.strip():
            continue
        topic = json.loads(line)
        for question in topic['questions']:
            topics[(topic['tid'], question['qid'])] = {
                'label': topic['label'].replace('-', ' ').replace('_', ' '),
                'narrative': topic['narrative'],
                'question': question['question'],
            }
    return topics


def assessed_days(judgment_files: list[Path], db_path: Path) -> dict[str, set[str]]:
    judged = {}
    for path in judgment_files:
        for key, docs in load_judgments(path).items():
            judged.setdefault(key, set()).update(docs)
    days = days_of(db_path, set().union(*judged.values()))
    by_topic: dict[str, set[str]] = {}
    for (tid, _), docs in judged.items():
        by_topic.setdefault(tid, set()).update(days[doc] for doc in docs if doc in days)
    return by_topic


def lead_text(db: sqlite3.Connection, doc_id: str, words: int, memo: dict[str, str]) -> str:
    if doc_id not in memo:
        found = db.execute('select text from documents where id=?', (doc_id,)).fetchone()
        memo[doc_id] = ' '.join(found[0].split()[:words]) if found and found[0] else ''
    return memo[doc_id]


def window_texts(db: sqlite3.Connection, doc_id: str, size: int, stride: int, memo: dict[str, list[str]]) -> list[str]:
    if doc_id not in memo:
        found = db.execute('select text from documents where id=?', (doc_id,)).fetchone()
        words = found[0].split() if found and found[0] else []
        if len(words) <= size:
            memo[doc_id] = [' '.join(words)] if words else []
        else:
            out = []
            for start in range(0, len(words), stride):
                out.append(' '.join(words[start:start + size]))
                if start + size >= len(words):
                    break
            memo[doc_id] = out
    return memo[doc_id]


def aggregate(scores: list[float], how: str) -> float:
    if not scores:
        return -100.0
    ordered = sorted(scores, reverse=True)
    if how == 'top2':
        return sum(ordered[:2]) / len(ordered[:2])
    return ordered[0]


def pair_key(query: str, passage: str) -> str:
    return sha1(f'{query}\x00{passage}'.encode()).hexdigest()


def load_cache(path: Path) -> dict[str, float]:
    try:
        with path.open('rb') as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return {}


def save_cache(path: Path, cached: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        pickle.dump(cached, f)
    os.replace(tmp, path)


def rrf(ranks: list[int], k: int) -> float:
    return sum(1 / (k + rank) for rank in ranks)


def main():
    parser = argparse.ArgumentParser(description='Rerank the saved ranking of a finished run with a cross-encoder, writing a replace run and a blend run that score.py can evaluate')
    parser.add_argument('run', type=Path, help='a run folder or run.json')
    parser.add_argument('--model', default='cross-encoder/ms-marco-MiniLM-L-12-v2')
    parser.add_argument('--template', default='{label}: {question}')
    parser.add_argument('--words', type=int, default=100, help='words of the lead passage when --passage is lead')
    parser.add_argument('--passage', choices=['lead', 'windows'], default='lead', help='lead scores the first --words words; windows scores every window and aggregates')
    parser.add_argument('--window-words', type=int, default=100)
    parser.add_argument('--window-stride', type=int, default=50)
    parser.add_argument('--aggregate', choices=['max', 'top2'], default='max', help='how the window scores of one document are combined')
    parser.add_argument('--depth', type=int, default=20, help='how many of the saved top documents to rerank')
    parser.add_argument('--blend-k', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default=None)
    parser.add_argument('--tag', required=True, help='short name for the output runs; the run tags are <tag>_rep and <tag>_rrf and must stay within 20 characters')
    parser.add_argument('--judgments', type=Path, nargs='*', default=[Path('data/dev-topics-all-docs.jsonl'), Path('data/silver-judgments.jsonl')],
                        help='the days carrying a judged document are the days reranked; every other day keeps its original ranking')
    parser.add_argument('--all-days', action='store_true', help='rerank every day in the run')
    parser.add_argument('--topics', type=Path, help='defaults to the topics file named in the run\'s config.json')
    parser.add_argument('--db', type=Path, default=Path('docs.db'))
    parser.add_argument('--cache-dir', type=Path, default=Path('cache/rerank_offline'))
    parser.add_argument('--out-dir', type=Path, default=Path('runs'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    for suffix in ('_rep', '_rrf'):
        if len(args.tag + suffix) > 20:
            sys.exit(f'run tag {args.tag + suffix!r} is longer than 20 characters')

    run_file = args.run / 'run.json' if args.run.is_dir() else args.run
    config = json.loads((run_file.parent / 'config.json').read_text())
    topics = load_topics(args.topics or Path(config['run']['topics']))
    wanted_days = None if args.all_days else assessed_days(args.judgments, args.db)

    lines = open(run_file).read().splitlines()
    header = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:] if line.strip()]

    db = sqlite3.connect(str(args.db))
    memo: dict[str, str] = {}
    window_memo: dict[str, list[str]] = {}
    jobs = []
    pairs: dict[str, tuple[str, str]] = {}
    for record in records:
        tid = record['topic']
        for day, questions in record['results'].items():
            if wanted_days is not None and day not in wanted_days.get(tid, set()):
                continue
            for question in questions:
                key = (tid, question['qid'])
                query = args.template.format(**topics[key])
                entries = []
                for hit in question['doc_ranking'][:args.depth]:
                    if args.passage == 'windows':
                        passages = window_texts(db, hit['doc_id'], args.window_words, args.window_stride, window_memo)
                    else:
                        passages = [lead_text(db, hit['doc_id'], args.words, memo)]
                    pks = []
                    for passage in passages:
                        pk = pair_key(query, passage)
                        pairs[pk] = (query, passage)
                        pks.append(pk)
                    entries.append((hit['doc_id'], pks))
                jobs.append((question, entries))

    cache_path = args.cache_dir / (args.model.replace('/', '__') + '.pkl')
    cached = load_cache(cache_path)
    missing = [pk for pk in pairs if pk not in cached]
    print(f'{len(jobs)} question-days, {len(pairs)} distinct pairs, {len(missing)} not cached, {sum(1 for _, p in pairs.values() if not p)} with no text')
    if args.dry_run:
        return

    if missing:
        model = load_scorer(args.model, args.device)
        missing.sort(key=lambda pk: len(pairs[pk][1]))
        chunk = args.batch_size * 40
        for start in range(0, len(missing), chunk):
            part = missing[start:start + chunk]
            predicted = model.predict([pairs[pk] for pk in part], batch_size=args.batch_size, show_progress_bar=False)
            for pk, score in zip(part, predicted):
                cached[pk] = float(score)
            if (start // chunk) % 25 == 24 or start + chunk >= len(missing):
                save_cache(cache_path, cached)
            print(f'scored {min(start + chunk, len(missing))}/{len(missing)}', flush=True)

    reordered = {'rep': {}, 'rrf': {}}
    for question, entries in jobs:
        fused_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(entries, start=1)}
        raw = {doc_id: aggregate([cached[pk] for pk in pks], args.aggregate) for doc_id, pks in entries}
        ce_order = sorted(raw, key=lambda doc_id: raw[doc_id], reverse=True)
        ce_rank = {doc_id: rank for rank, doc_id in enumerate(ce_order, start=1)}
        blended = {doc_id: rrf([ce_rank[doc_id], fused_rank[doc_id]], args.blend_k) for doc_id in raw}
        reordered['rep'][id(question)] = [{'doc_id': doc_id, 'score': raw[doc_id]} for doc_id in ce_order]
        reordered['rrf'][id(question)] = [{'doc_id': doc_id, 'score': blended[doc_id]} for doc_id in sorted(blended, key=lambda doc_id: blended[doc_id], reverse=True)]

    labels = {'rep': 'replaced by', 'rrf': 'blended by reciprocal rank fusion with'}
    for suffix, verb in labels.items():
        for question, entries in jobs:
            question['doc_ranking'] = reordered[suffix][id(question)]
        out_header = dict(header)
        out_header['runtag'] = args.tag + '_' + suffix
        out_header['description'] = f'{header["description"]} Top {args.depth} {verb} the cross-encoder {args.model} (may be contaminated) on days with judged documents.'
        out_dir = args.out_dir / out_header['runtag']
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'run.json', 'w') as f:
            f.write(json.dumps(out_header) + '\n')
            for record in records:
                f.write(json.dumps(record) + '\n')
        (out_dir / 'config.json').write_text(json.dumps({**config, 'rerank': {'source_run': str(run_file), 'model': args.model, 'template': args.template, 'words': args.words, 'passage': args.passage, 'window_words': args.window_words, 'window_stride': args.window_stride, 'aggregate': args.aggregate, 'depth': args.depth, 'mode': suffix, 'blend_k': args.blend_k}}, indent=2))
        print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()

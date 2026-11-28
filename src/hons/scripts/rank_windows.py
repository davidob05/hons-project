import argparse
import collections
import json
import sqlite3
import sys
import tomllib
from pathlib import Path

from hons.scripts.topic_system import DEFAULTS, DayIndex, LiveScorer

MODELS = {'monot5_win': 'castorini/monot5-base-msmarco', 'minilm_win': 'cross-encoder/ms-marco-MiniLM-L-12-v2'}
SYSTEMS = ('bm25_top20', 'monot5_win', 'minilm_win', 'fused_win')


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank each question\'s BM25 candidates on the judged days with monoT5 and MiniLM over overlapping windows (best window), and fuse the three rankings')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    raw = tomllib.loads(args.config.read_text())
    config = {**DEFAULTS, 'candidates': 20, **raw['run']}
    topics = {}
    for line in open(config['topics']):
        if line.strip():
            topic = json.loads(line)
            if topic['tid'] in config['topics_only']:
                topics[topic['tid']] = topic
    missing = sorted(set(config['topics_only']) - set(topics))
    if missing:
        sys.exit(f'topics missing from {config["topics"]}: {missing}')
    judged = collections.defaultdict(list)
    for line in open(config['judged_days']):
        tid, day = line.split()[:2]
        if tid in topics:
            judged[day].append(tid)
    db = sqlite3.connect(config['db'])
    days = [row[0] for row in db.execute('select distinct day from documents order by day')]
    scorers = {name: LiveScorer({**config, 'model': model}, topics, db) for name, model in MODELS.items()}
    results = {name: {tid: {day: [] for day in days} for tid in topics} for name in SYSTEMS}
    labels = {tid: topic['label'].replace('-', ' ').replace('_', ' ') for tid, topic in topics.items()}
    k = config['rrf_k']
    todo = sorted(judged)
    print(f'{config["tag"]}: {len(topics)} topics, {len(todo)} judged days, top {config["candidates"]} BM25 candidates, windows of {config["window_words"]} words every {config["window_stride"]}', flush=True)
    for number, day in enumerate(todo):
        rows = db.execute('select id, text from documents where day=?', (day,)).fetchall()
        if not rows:
            continue
        index = DayIndex(rows)
        first: dict[tuple[str, str], list[tuple[float, str]]] = {}
        for tid in judged[day]:
            for q in topics[tid]['questions']:
                scored = index.scores(f"{labels[tid]}: {q['question']} {topics[tid]['narrative']}")
                first[(tid, q['qid'])] = sorted((item for item in scored if item[0] > 0), key=lambda item: -item[0])[:config['candidates']]
        requests = [(tid, doc, qid) for (tid, qid), found in first.items() for _, doc in found]
        model_scores = {name: scorer.score(day, requests, 'order') if requests else {} for name, scorer in scorers.items()}
        for scorer in scorers.values():
            scorer.memo.clear()
        for tid in judged[day]:
            entries: dict[str, list] = {name: [] for name in SYSTEMS}
            for q in topics[tid]['questions']:
                qid = q['qid']
                found = first[(tid, qid)]
                if not found:
                    continue
                docs = [doc for _, doc in found]
                values = {'bm25_top20': {doc: score for score, doc in found}}
                for name in MODELS:
                    values[name] = {doc: model_scores[name][(tid, doc, qid)] for doc in docs}
                ranks = [{doc: i for i, doc in enumerate(sorted(docs, key=lambda doc: -values[name][doc]), 1)} for name in ('bm25_top20', *MODELS)]
                values['fused_win'] = {doc: sum(1 / (k + rank[doc]) for rank in ranks) for doc in docs}
                for name in SYSTEMS:
                    ordered = sorted(docs, key=lambda doc: -values[name][doc])
                    entries[name].append((values[name][ordered[0]], qid, q['question'], [{'doc_id': doc, 'score': values[name][doc]} for doc in ordered]))
            for name in SYSTEMS:
                chosen = sorted(entries[name], key=lambda item: -item[0])
                results[name][tid][day] = [{'qid': qid, 'question_rank': rank, 'question_text': text, 'doc_ranking': hits, 'extra': None}
                                           for rank, (_, qid, text, hits) in enumerate(chosen, 1)]
        if (number + 1) % 20 == 0 or number + 1 == len(todo):
            print(f'judged day {number + 1}/{len(todo)} {day}', flush=True)
    for scorer in scorers.values():
        scorer.finish()
    folder = Path(config['out_dir']) / config['tag']
    folder.mkdir(parents=True, exist_ok=True)
    for name in SYSTEMS:
        with open(folder / f'{name}.json', 'w') as f:
            f.write(json.dumps({'runtag': name, 'description': f'BM25 top {config["candidates"]} per question on each judged day, system {name}: monoT5 and MiniLM score {config["window_words"]}-word windows every {config["window_stride"]} words and keep the best; fused_win is reciprocal rank fusion of BM25, monoT5 and MiniLM. No gate.',
                                'run_type': 'automatic', 'models': list(MODELS.values()), 'extern': config['extern'], 'extra': None}) + '\n')
            for tid in sorted(topics):
                f.write(json.dumps({'topic': tid, 'results': results[name][tid], 'extra': None}) + '\n')
        print(f'wrote {folder / f"{name}.json"}', flush=True)


if __name__ == '__main__':
    main()

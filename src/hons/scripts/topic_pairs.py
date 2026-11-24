import argparse
import collections
import copy
import json
import math
import sqlite3
import sys
from pathlib import Path

from hons.scripts.rerank_run import aggregate, assessed_days, load_cache, pair_key, save_cache, window_texts
from hons.stages.s2_bm25 import tokenise
from hons.stages.s2_rerank import load_scorer

K1, B, K3 = 1.2, 0.75, 8.0


def topic_query(topic: dict, kind: str) -> str:
    label = topic['label'].replace('-', ' ').replace('_', ' ')
    if kind == 'label':
        return label
    if kind == 'label_narrative':
        return f"{label}: {topic['narrative']}"
    return f"{label}: {topic['narrative']} " + ' '.join(q['question'] for q in topic['questions'])


def day_rankings(db: sqlite3.Connection, day: str, queries: dict[str, str], keep: int) -> dict[str, tuple[list[str], list[float]]]:
    rows = db.execute('select id, text from documents where day=?', (day,)).fetchall()
    if not rows:
        return {tid: ([], []) for tid in queries}
    tokens = [tokenise(text, True) for _, text in rows]
    count = len(rows)
    average = sum(len(t) for t in tokens) / count
    frequency = []
    document_frequency: collections.Counter = collections.Counter()
    for t in tokens:
        c = collections.Counter(t)
        frequency.append(c)
        for term in c:
            document_frequency[term] += 1
    idf = {term: math.log((count - n + 0.5) / (n + 0.5) + 1) for term, n in document_frequency.items()}
    out = {}
    for tid, query in queries.items():
        query_terms = collections.Counter(tokenise(query, True))
        scored = []
        for index, (doc_id, _) in enumerate(rows):
            score = 0.0
            for term, qtf in query_terms.items():
                f = frequency[index].get(term, 0)
                if not f:
                    continue
                score += idf[term] * (f * (K1 + 1) / (f + K1 * (1 - B + B * len(tokens[index]) / average))) * ((K3 + 1) * qtf / (K3 + qtf))
            scored.append((score, doc_id))
        best = sorted(scored, key=lambda item: -item[0])[:max(keep, 20)]
        out[tid] = ([doc_id for _, doc_id in best], [score for score, _ in best])
    return out


def loose_gate_days(days: list[str], best: list[float], quantile: float, window: int, warmup: int) -> set[str]:
    history: collections.deque = collections.deque(maxlen=window)
    chosen = set()
    for day, value in zip(days, best):
        if len(history) >= warmup:
            ordered = sorted(history)
            if sum(1 for x in ordered if x < value) / len(ordered) >= quantile:
                chosen.add(day)
        history.append(value)
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Search BM25 once per topic per judged day, shortlist the top documents, and score every shortlisted document against every question of its topic with a reranker, windowed by maximum')
    parser.add_argument('--config', type=Path, required=True, help='a TOML file with a [run] table and one [[cell]] table per scoring run')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import tomllib

    config = tomllib.loads(args.config.read_text())
    shared = {'model': 'castorini/monot5-base-msmarco', 'shortlist': 10, 'query': 'label_narrative_questions', 'days': 'judged',
              'loose_quantile': 0.6, 'loose_window': 60, 'loose_warmup': 30,
              'window_words': 200, 'window_stride': 100, 'batch_size': 32, 'aggregate': 'max', 'device': None,
              'topics': 'data/dev-topics-all-docs.jsonl', 'db': 'docs.db', 'cache_dir': 'cache/rerank_offline', 'out_dir': 'runs',
              'judgments': ['data/dev-topics-all-docs.jsonl', 'data/silver-judgments.jsonl'], **config.get('run', {})}
    topics = {}
    for line in open(shared['topics']):
        topic = json.loads(line)
        if shared.get('topics_only') and topic['tid'] not in shared['topics_only']:
            continue
        topics[topic['tid']] = topic
    db = sqlite3.connect(shared['db'])
    judged_days = assessed_days([Path(p) for p in shared['judgments']], Path(shared['db']))
    keep = shared['shortlist']
    queries = {tid: topic_query(topic, shared['query']) for tid, topic in topics.items()}
    all_days = [row[0] for row in db.execute('select distinct day from documents order by day')]
    series: dict[str, dict[str, list[float]]] = {tid: {} for tid in topics}
    top_ids: dict[tuple[str, str], list[str]] = {}
    if shared['days'] == 'gate':
        for day in all_days:
            for tid, (ids, scores) in day_rankings(db, day, queries, keep).items():
                series[tid][day] = scores[:20]
                top_ids[(tid, day)] = ids
    selected: dict[str, set[str]] = {}
    for tid in topics:
        wanted_days = set(judged_days.get(tid, set()))
        if shared['days'] == 'gate':
            best = [series[tid][d][0] if d in series[tid] and series[tid][d] else 0.0 for d in all_days]
            wanted_days |= loose_gate_days(all_days, best, shared['loose_quantile'], shared['loose_window'], shared['loose_warmup'])
        selected[tid] = wanted_days
    topic_days = [(tid, day) for tid in sorted(topics) for day in sorted(selected[tid])]
    ranking: dict[tuple[str, str], list[str]] = {}
    for tid, day in topic_days:
        if (tid, day) in top_ids:
            ranking[(tid, day)] = top_ids[(tid, day)][:keep]
        else:
            ranking[(tid, day)] = day_rankings(db, day, {tid: queries[tid]}, keep)[tid][0][:keep]
    print(f'topics {sorted(topics)}; days mode {shared["days"]}; selected topic-days per topic {{{", ".join(f"{t}: {len(selected[t])}" for t in sorted(selected))}}}; shortlist of {keep}; query {shared["query"]}', flush=True)

    models: dict = {}
    window_memo: dict[str, list[str]] = {}
    for cell in config['cell']:
        settings = {**shared, **cell}
        template = settings['template']
        pairs: dict[str, tuple[str, str]] = {}
        index: list[tuple[str, str, str, str, list[str]]] = []
        for tid, day in topic_days:
            topic = topics[tid]
            label = topic['label'].replace('-', ' ').replace('_', ' ')
            for doc_id in ranking[(tid, day)]:
                passages = window_texts(db, doc_id, settings['window_words'], settings['window_stride'], window_memo)
                for question in topic['questions']:
                    query = template.format(label=label, narrative=topic['narrative'], question=question['question'])
                    keys = []
                    for passage in passages:
                        key = pair_key(query, passage)
                        pairs[key] = (query, passage)
                        keys.append(key)
                    index.append((tid, day, doc_id, question['qid'], keys))
        cache_path = Path(settings['cache_dir']) / (settings['model'].replace('/', '__') + '.pkl')
        cached = load_cache(cache_path)
        missing = [key for key in pairs if key not in cached]
        print(f'== {settings["tag"]}: {len(index)} document-question pairs, {len(pairs)} distinct window pairs, {len(missing)} not cached', flush=True)
        if args.dry_run:
            continue
        if missing:
            if settings['model'] not in models:
                models[settings['model']] = load_scorer(settings['model'], settings['device'])
            model = models[settings['model']]
            missing.sort(key=lambda key: len(pairs[key][1]))
            chunk = settings['batch_size'] * 40
            for start in range(0, len(missing), chunk):
                part = missing[start:start + chunk]
                predicted = model.predict([pairs[key] for key in part], batch_size=settings['batch_size'], show_progress_bar=False)
                for key, score in zip(part, predicted):
                    cached[key] = float(score)
                if (start // chunk) % 25 == 24 or start + chunk >= len(missing):
                    save_cache(cache_path, cached)
                print(f'scored {min(start + chunk, len(missing))}/{len(missing)}', flush=True)
        scores = {f'{tid}|{day}|{doc_id}|{qid}': aggregate([cached[key] for key in keys], settings['aggregate'])
                  for tid, day, doc_id, qid, keys in index}
        out_dir = Path(settings['out_dir']) / settings['tag']
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {'tag': settings['tag'], 'model': settings['model'], 'template': template, 'query': settings['query'],
                  'shortlist_size': settings['shortlist'], 'window_words': settings['window_words'], 'window_stride': settings['window_stride'],
                  'shortlist': {f'{tid}|{day}': docs for (tid, day), docs in ranking.items()}, 'scores': scores,
                  'selected_days': {tid: sorted(days) for tid, days in selected.items()},
                  'series': series, 'top_ids': {f'{tid}|{day}': ids for (tid, day), ids in top_ids.items()}}
        (out_dir / 'pair_scores.json').write_text(json.dumps(result))
        print(f'wrote {out_dir / "pair_scores.json"}', flush=True)


if __name__ == '__main__':
    main()

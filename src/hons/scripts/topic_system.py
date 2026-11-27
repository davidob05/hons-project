import argparse
import collections
import json
import math
import re
import sqlite3
import sys
import tempfile
import tomllib
from pathlib import Path

from hons.scripts.rerank_run import aggregate, load_cache, pair_key, save_cache, window_texts
from hons.scripts.score import DEFAULT_SCORER, days_of, load_judgments, plain_ap, score, summarise
from hons.scripts.topic_pairs import B, K1, K3, topic_query
from hons.stages.s2_bm25 import tokenise
from hons.stages.s2_rerank import load_scorer

RUNTAG = re.compile(r'^[a-zA-Z0-9_-][a-zA-Z0-9_.-]{1,19}$')
MISSING = -30.0

DEFAULTS = {
    'tag': 'topic_system',
    'description': 'Topic-first change detection: a BM25 topic gate on a rolling history, a shortlist of the five best documents, per-question ordering by fusing BM25 with monoT5, and a per-question history gate on the monoT5 question score.',
    'extern': 'No data beyond the topics and the RAGTIME1 English dataset. Corpus statistics come only from the documents of each day. The monoT5 checkpoint (castorini/monot5-base-msmarco) was fine-tuned on MS MARCO and its release date may overlap the start of the corpus period.',
    'models': ['castorini/monot5-base-msmarco'],
    'topics': 'data/dev-topics-all-docs.jsonl',
    'db': 'docs.db',
    'out_dir': 'runs',
    'cache_dir': 'cache/rerank_offline',
    'scorer': 'live',
    'reuse': 'runs/gx_tg_{short}_{cell}',
    'model': 'castorini/monot5-base-msmarco',
    'device': None,
    'batch_size': 32,
    'window_words': 200,
    'window_stride': 100,
    'topic_quantile': 0.8,
    'topic_window': 60,
    'topic_warmup': 30,
    'loose_quantile': 0.6,
    'shortlist': 5,
    'depth': 3,
    'question_quantile': 0.5,
    'question_window': 60,
    'question_warmup': 20,
    'rrf_k': 60,
    'first_day': None,
    'last_day': None,
    'topics_only': None,
    'template_gate': '{question}',
    'template_order': '{label}: {question}',
}


class DayIndex:
    def __init__(self, rows: list[tuple[str, str]]):
        self.ids = [doc_id for doc_id, _ in rows]
        self.position = {doc_id: i for i, doc_id in enumerate(self.ids)}
        tokens = [tokenise(text, True) for _, text in rows]
        self.lengths = [len(t) for t in tokens]
        self.average = sum(self.lengths) / len(rows)
        self.frequency = [collections.Counter(t) for t in tokens]
        document_frequency: collections.Counter = collections.Counter()
        for counts in self.frequency:
            for term in counts:
                document_frequency[term] += 1
        count = len(rows)
        self.idf = {term: math.log((count - n + 0.5) / (n + 0.5) + 1) for term, n in document_frequency.items()}

    def scores(self, query: str, subset: list[str] | None = None) -> list[tuple[float, str]]:
        terms = collections.Counter(tokenise(query, True))
        indices = range(len(self.ids)) if subset is None else [self.position[d] for d in subset if d in self.position]
        out = []
        for i in indices:
            total = 0.0
            for term, qtf in terms.items():
                f = self.frequency[i].get(term, 0)
                if f:
                    total += self.idf[term] * (f * (K1 + 1) / (f + K1 * (1 - B + B * self.lengths[i] / self.average))) * ((K3 + 1) * qtf / (K3 + qtf))
            out.append((total, self.ids[i]))
        return out


class ReuseScorer:
    def __init__(self, pattern: str, tids: list[str]):
        self.tables = {}
        for tid in tids:
            for cell in ('q', 'lq'):
                path = Path(pattern.format(short=tid[-3:], cell=cell)) / 'pair_scores.json'
                self.tables[(tid, cell)] = json.load(open(path))['scores']

    def score(self, day: str, requests: list[tuple[str, str, str]], kind: str) -> dict[tuple[str, str, str], float]:
        cell = 'q' if kind == 'gate' else 'lq'
        return {(tid, doc, qid): self.tables[(tid, cell)].get(f'{tid}|{day}|{doc}|{qid}', MISSING) for tid, doc, qid in requests}

    def finish(self) -> None:
        return None


class LiveScorer:
    def __init__(self, config: dict, topics: dict[str, dict], db: sqlite3.Connection):
        self.config = config
        self.topics = topics
        self.db = db
        self.model = None
        self.memo: dict[str, list[str]] = {}
        self.cache_path = Path(config['cache_dir']) / (config['model'].replace('/', '__') + '.pkl')
        self.cache = load_cache(self.cache_path)
        self.fresh = 0
        self.questions = {tid: {q['qid']: q['question'] for q in topic['questions']} for tid, topic in topics.items()}

    def score(self, day: str, requests: list[tuple[str, str, str]], kind: str) -> dict[tuple[str, str, str], float]:
        template = self.config['template_gate' if kind == 'gate' else 'template_order']
        wanted: dict[tuple[str, str, str], list[str]] = {}
        pairs: dict[str, tuple[str, str]] = {}
        for tid, doc, qid in requests:
            label = self.topics[tid]['label'].replace('-', ' ').replace('_', ' ')
            query = template.format(label=label, narrative=self.topics[tid]['narrative'], question=self.questions[tid][qid])
            keys = []
            for passage in window_texts(self.db, doc, self.config['window_words'], self.config['window_stride'], self.memo):
                key = pair_key(query, passage)
                pairs[key] = (query, passage)
                keys.append(key)
            wanted[(tid, doc, qid)] = keys
        missing = [key for key in pairs if key not in self.cache]
        if missing:
            if self.model is None:
                self.model = load_scorer(self.config['model'], self.config['device'])
            missing.sort(key=lambda key: len(pairs[key][1]))
            predicted = self.model.predict([pairs[key] for key in missing], batch_size=self.config['batch_size'], show_progress_bar=False)
            for key, value in zip(missing, predicted):
                self.cache[key] = float(value)
            self.fresh += len(missing)
            if self.fresh >= 5000:
                save_cache(self.cache_path, self.cache)
                self.fresh = 0
        return {key: aggregate([self.cache[k] for k in keys], 'max') for key, keys in wanted.items()}

    def finish(self) -> None:
        save_cache(self.cache_path, self.cache)


def percentile(history, value: float, warmup: int) -> float | None:
    if len(history) < warmup:
        return None
    return sum(1 for x in history if x < value) / len(history)


def run_system(config: dict, topics: dict[str, dict], db: sqlite3.Connection, days: list[str]) -> dict[str, dict[str, list[dict]]]:
    scorer = ReuseScorer(config['reuse'], sorted(topics)) if config['scorer'] == 'reuse' else LiveScorer(config, topics, db)
    labels = {tid: topic['label'].replace('-', ' ').replace('_', ' ') for tid, topic in topics.items()}
    topic_queries = {tid: topic_query(topic, 'label_narrative_questions') for tid, topic in topics.items()}
    topic_history = {tid: collections.deque(maxlen=config['topic_window']) for tid in topics}
    question_history = {(tid, q['qid']): collections.deque(maxlen=config['question_window']) for tid, topic in topics.items() for q in topic['questions']}
    results: dict[str, dict[str, list[dict]]] = {tid: {} for tid in topics}
    stats = collections.Counter()
    k = config['rrf_k']
    for number, day in enumerate(days):
        rows = db.execute('select id, text from documents where day=?', (day,)).fetchall()
        index = DayIndex(rows) if rows else None
        active: dict[str, tuple[bool, list[str]]] = {}
        for tid in topics:
            top = sorted(index.scores(topic_queries[tid]), key=lambda item: -item[0])[:config['shortlist']] if index else []
            best = top[0][0] if top else 0.0
            pct = percentile(topic_history[tid], best, config['topic_warmup'])
            topic_history[tid].append(best)
            if pct is not None and pct >= config['loose_quantile'] and top:
                active[tid] = (pct > config['topic_quantile'], [doc for _, doc in top])
        requests = [(tid, doc, q['qid']) for tid, (_, docs) in active.items() for doc in docs for q in topics[tid]['questions']]
        gate_scores = scorer.score(day, requests, 'gate') if requests else {}
        fired_requests = [(tid, doc, q['qid']) for tid, (fired, docs) in active.items() if fired for doc in docs for q in topics[tid]['questions']]
        order_scores = scorer.score(day, fired_requests, 'order') if fired_requests else {}
        for tid in topics:
            results[tid][day] = []
        for tid, (fired, docs) in active.items():
            stats['loose topic-days'] += 1
            entries = []
            for q in topics[tid]['questions']:
                qid = q['qid']
                best = max(gate_scores[(tid, doc, qid)] for doc in docs)
                pct = percentile(question_history[(tid, qid)], best, config['question_warmup'])
                question_history[(tid, qid)].append(best)
                if not fired or pct is None or pct <= config['question_quantile']:
                    continue
                own = dict((doc, value) for value, doc in index.scores(f"{labels[tid]}: {q['question']} {topics[tid]['narrative']}", docs))
                by_bm25 = sorted(docs, key=lambda doc: -own.get(doc, -1e9))
                by_model = sorted(docs, key=lambda doc: -order_scores[(tid, doc, qid)])
                fused = {doc: 1 / (k + by_bm25.index(doc) + 1) + 1 / (k + by_model.index(doc) + 1) for doc in docs}
                keep = sorted(docs, key=lambda doc: -fused[doc])[:config['depth']]
                entries.append((best, qid, q['question'], [{'doc_id': doc, 'score': fused[doc]} for doc in keep]))
            if fired:
                stats['fired topic-days'] += 1
            entries.sort(key=lambda item: -item[0])
            results[tid][day] = [{'qid': qid, 'question_rank': rank, 'question_text': text, 'doc_ranking': hits, 'extra': None}
                                 for rank, (_, qid, text, hits) in enumerate(entries, 1)]
            stats['question-days returned'] += len(entries)
        if (number + 1) % 20 == 0 or number + 1 == len(days):
            print(f'day {number + 1}/{len(days)} {day}: {dict(stats)}', flush=True)
    scorer.finish()
    return results


def write_run(path: Path, config: dict, topics: dict[str, dict], results: dict[str, dict[str, list[dict]]]) -> None:
    if not RUNTAG.match(config['tag']):
        sys.exit(f'bad runtag {config["tag"]!r}: at most 20 characters of letters, digits, period, hyphen, underscore, not starting with a period')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(json.dumps({'runtag': config['tag'], 'description': config['description'], 'run_type': 'automatic',
                            'models': config['models'], 'extern': config['extern'], 'extra': None}) + '\n')
        for tid in sorted(topics):
            f.write(json.dumps({'topic': tid, 'results': results[tid], 'extra': None}) + '\n')


def cut_run(run: Path, depth: int) -> Path:
    out = run.parent / f'{run.stem}_d{depth}.json'
    with open(run) as f, open(out, 'w') as g:
        g.write(f.readline())
        for line in f:
            record = json.loads(line)
            for questions in record['results'].values():
                for question in questions:
                    question['doc_ranking'] = sorted(question['doc_ranking'], key=lambda hit: -hit['score'])[:depth]
            g.write(json.dumps(record) + '\n')
    return out


def merge_runs(parts: list[Path], out: Path, tag: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(out, 'w') as f:
        for i, part in enumerate(parts):
            lines = open(part).read().splitlines()
            if i == 0:
                f.write(json.dumps({**json.loads(lines[0]), 'runtag': tag}) + '\n')
            for line in lines[1:]:
                tid = json.loads(line)['topic']
                if tid not in seen:
                    seen.add(tid)
                    f.write(line + '\n')
    print(f'merged {len(parts)} runs, {len(seen)} topics, into {out}')


def evaluate(config: dict, run: Path, topics_file: Path, days: list[str]) -> None:
    section = config.get('score', {})
    mode = section.get('mode', 'dev')
    scorer_src = Path(section.get('scorer', DEFAULT_SCORER))
    if mode == 'none':
        return
    tids = [json.loads(line)['tid'] for line in open(topics_file) if line.strip()]
    plain = None
    if mode == 'dev':
        files = [Path(p) for p in section.get('judgments', ['data/dev-topics-all-docs.jsonl', 'data/silver-judgments.jsonl'])]
        judged: dict[tuple[str, str], set[str]] = {}
        for path in files:
            for key, docs in load_judgments(path).items():
                judged.setdefault(key, set()).update(docs)
        doc_day = days_of(Path(config['db']), set().union(*judged.values()))
        relq: dict[tuple[str, str, str], set[str]] = {}
        for (tid, qid), docs in judged.items():
            for doc in docs:
                if doc in doc_day:
                    relq.setdefault((tid, qid, doc_day[doc]), set()).add(doc)
        active = collections.defaultdict(set)
        for tid, _, day in relq:
            active[tid].add(day)
        lines = [f'{tid} {day} {qid} {doc} awareness' for (tid, qid, day), docs in sorted(relq.items()) for doc in sorted(docs)]
        lines += [f'{tid} {day} * dummy_doc not-relevant' for tid in tids for day in days if day not in active[tid]]
        qrels = run.parent / 'qrels_assumed_quiet.txt'
        qrels.write_text('\n'.join(lines) + '\n')
        plain = plain_ap(run, judged, doc_day, 100)
        print('scoring on the dev judgments; days with no relevant document are assumed quiet (a not-relevant line is added for them)')
    else:
        qrels = Path(section['qrels'])
        print(f'scoring against {qrels}')
    raw = score(scorer_src, qrels, topics_file, run, 'awareness', 100)
    values = summarise(raw)
    overall = values['overall']
    for key, value in raw.items():
        if key[0] == 'all' and key[-1].startswith('rbo'):
            overall['rbo'] = float(value)
    if plain is not None:
        overall['ap_plain'] = plain
    names = ('rbo', "recip_rank'", "ap'", "ndcg'", "rbp'(p=0.8)", 'ap_plain', 'abstention', 'noise', 'num_days', 'num_qs', 'num_ret', 'num_rel', 'num_rel_ret', 'unjudged')
    print(f'official scorer, run {run}')
    for name in names:
        if name in overall:
            value = overall[name]
            print(f'  {name:14}{value:10.3f}' if value < 1000 and not float(value).is_integer() or name in ('abstention', 'noise') else f'  {name:14}{value:10.0f}')
    (run.parent / f'{run.stem}_official.json').write_text(json.dumps(values, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the topic-first change detection system from a TOML config: write a track-format run file and score it with the official scorer')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--score-only', type=Path, help='skip the system and score this existing run file with the config\'s [score] table')
    parser.add_argument('--merge', type=Path, nargs='+', help='merge shard run files into --out and then score it')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--depth', type=int, nargs='+', help='with --score-only or --merge, score the run cut to each of these depths instead of as stored')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw = tomllib.loads(args.config.read_text()) if args.config else {}
    config = {**DEFAULTS, **raw.get('run', {}), 'score': raw.get('score', {})}
    topics_file = Path(config['topics'])
    db = sqlite3.connect(config['db'])
    days = [row[0] for row in db.execute('select distinct day from documents order by day')]
    if config['first_day']:
        days = [d for d in days if d >= config['first_day']]
    if config['last_day']:
        days = [d for d in days if d <= config['last_day']]
    if args.merge or args.score_only:
        if args.merge:
            merge_runs(args.merge, args.out, config['tag'])
        run = args.out if args.merge else args.score_only
        for depth in args.depth or [None]:
            evaluate(config, cut_run(run, depth) if depth else run, topics_file, days)
        return
    topics = {}
    for line in open(topics_file):
        if line.strip():
            topic = json.loads(line)
            if not config['topics_only'] or topic['tid'] in config['topics_only']:
                topics[topic['tid']] = topic
    print(f'{config["tag"]}: {len(topics)} topics, {len(days)} days, scorer {config["scorer"]}, topic quantile {config["topic_quantile"]}, question quantile {config["question_quantile"]}, depth {config["depth"]}', flush=True)
    results = run_system(config, topics, db, days)
    run = Path(config['out_dir']) / config['tag'] / 'run.json'
    write_run(run, config, topics, results)
    print(f'wrote {run}', flush=True)
    if not config['topics_only']:
        evaluate(config, run, topics_file, days)


if __name__ == '__main__':
    main()

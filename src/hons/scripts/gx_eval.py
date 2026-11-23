import argparse
import collections
import glob
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

from hons.scripts.score import days_of, load_judgments


def auc(pos, neg) -> float:
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def load_run(path: str) -> dict[tuple[str, str, str], dict[str, float]]:
    f = open(path)
    next(f)
    out = {}
    for line in f:
        record = json.loads(line)
        for day, questions in record['results'].items():
            for question in questions:
                out[(record['topic'], question['qid'], day)] = {hit['doc_id']: hit['score'] for hit in question['doc_ranking']}
    return out


def main():
    parser = argparse.ArgumentParser(description='Compare the reranker grid cells (runs/gx_<model>_<query>_<passage>_rep) on ranking, separation from sibling-only and unjudged documents, the day-level signal and length bias')
    parser.add_argument('--base', default='bm25_doc_both', help='run folder prefix of the BM25 candidates')
    parser.add_argument('--judgments', type=Path, nargs='*', default=[Path('data/dev-topics-all-docs.jsonl'), Path('data/silver-judgments.jsonl')])
    parser.add_argument('--topics', type=Path, default=Path('data/dev-topics-all-docs.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('docs.db'))
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--models', nargs='*', default=['l12', 't5'])
    args = parser.parse_args()

    judged: dict[tuple[str, str], set[str]] = {}
    for path in args.judgments:
        for key, docs in load_judgments(path).items():
            judged.setdefault(key, set()).update(docs)
    days = days_of(args.db, set().union(*judged.values()))
    relq: dict[tuple[str, str, str], set[str]] = {}
    for (tid, qid), docs in judged.items():
        for doc_id in docs:
            if doc_id in days:
                relq.setdefault((tid, qid, days[doc_id]), set()).add(doc_id)
    assessed: dict[str, set[str]] = {}
    for tid, _, day in relq:
        assessed.setdefault(tid, set()).add(day)
    questions = {}
    for line in open(args.topics):
        topic = json.loads(line)
        questions[topic['tid']] = [q['qid'] for q in topic['questions']]
    points = [(tid, qid, day) for tid, ds in assessed.items() for day in ds for qid in questions[tid]]
    answerable = [p for p in points if p in relq]
    nil = [p for p in points if p not in relq]
    sibling = collections.defaultdict(set)
    for (tid, qid, day), docs in relq.items():
        for doc_id in docs:
            sibling[(tid, day, doc_id)].add(qid)

    base = load_run(glob.glob(str(args.runs / f'{args.base}-*' / 'run.json'))[0])
    bm_rank = {p: {d: i for i, d in enumerate([d for d, _ in sorted(base[p].items(), key=lambda x: -x[1])][:20], 1)} for p in points}
    db = sqlite3.connect(str(args.db))
    docs_all = sorted({d for p in points for d in base.get(p, {})})
    words = {}
    for start in range(0, len(docs_all), 500):
        chunk = docs_all[start:start + 500]
        query = f'select id, text from documents where id in ({",".join("?" * len(chunk))})'
        for doc_id, text in db.execute(query, chunk):
            words[doc_id] = len(text.split())

    def rr_of(lists) -> float:
        total = 0.0
        for p in answerable:
            for i, d in enumerate(lists.get(p, []), 1):
                if d in relq[p]:
                    total += 1 / i
                    break
        return total / len(answerable)

    names = {'q': 'question', 'lq': 'label+question', 'lqn': 'label+q+narrative'}
    passages = {'ld': 'first 100w', 'w1': '100w windows', 'w2': '200w windows'}
    print(f"{'model':>5} {'query':18}{'passage':>14}{'RR′':>7}{'AP′':>7}{'NDCG′':>7}{'RR blend':>9}{'AUCsib':>7}{'AUCunj':>7}{'dayAUC':>7}{'lenCorr':>8}{'top=long':>9}")
    for model in args.models:
        for template in ('q', 'lq', 'lqn'):
            for passage in ('ld', 'w1', 'w2'):
                name = f'gx_{model}_{template}_{passage}'
                path = glob.glob(str(args.runs / f'{name}_rep' / 'run.json'))
                if not path:
                    continue
                run = load_run(path[0])
                official = args.runs / f'{name}_rep' / 'official.json'
                overall = json.load(open(official))['combined']['overall'] if official.exists() else None
                blended = {}
                for p in points:
                    scores = run[p]
                    order = sorted(scores, key=lambda d: -scores[d])
                    rank = {d: i for i, d in enumerate(order, 1)}
                    blended[p] = sorted(scores, key=lambda d: -(1 / (60 + rank[d]) + 1 / (60 + bm_rank[p].get(d, 25))))
                pos, neg, unj = [], [], []
                for p in points:
                    for d, s in run[p].items():
                        if d in relq.get(p, set()):
                            pos.append(s)
                        elif sibling.get((p[0], p[2], d)):
                            neg.append(s)
                        else:
                            unj.append(s)
                top = np.array([max(run[p].values()) for p in answerable + nil])
                label = np.array([True] * len(answerable) + [False] * len(nil))
                day_auc = auc(top[label], top[~label])
                corrs, longest = [], 0
                for p in points:
                    ids = list(run[p])
                    if len(ids) < 5:
                        continue
                    s = np.array([run[p][d] for d in ids])
                    length = np.array([words.get(d, 0) for d in ids])
                    if s.std() > 0 and length.std() > 0:
                        corrs.append(np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(length)))[0, 1])
                    if ids[int(np.argmax(s))] == ids[int(np.argmax(length))]:
                        longest += 1
                rr = overall["recip_rank'"] if overall else float('nan')
                ap = overall["ap'"] if overall else float('nan')
                nd = overall["ndcg'"] if overall else float('nan')
                print(f"{model:>5} {names[template]:18}{passages[passage]:>14}{rr:7.3f}{ap:7.3f}{nd:7.3f}{rr_of(blended):9.3f}{auc(pos, neg):7.3f}{auc(pos, unj):7.3f}{day_auc:7.3f}{float(np.mean(corrs)):8.2f}{100 * longest / len(points):8.0f}%")

    print('\nofficial scores of blend runs:')
    for path in sorted(glob.glob(str(args.runs / 'gx_*_rrf' / 'official.json'))):
        overall = json.load(open(path))['combined']['overall']
        print(f"  {Path(path).parent.name}: RR′ {overall[chr(114) + 'ecip_rank' + chr(39)]:.3f} AP {overall['ap_plain']:.3f} AP′ {overall['ap' + chr(39)]:.3f} NDCG′ {overall['ndcg' + chr(39)]:.3f}")


if __name__ == '__main__':
    main()

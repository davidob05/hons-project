import argparse
import collections
import glob
import json
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser(description='Learn a combination of several scorers on the same (question, document) pairs, trained on one topic and tested on the other, and compare it with the best single score')
    parser.add_argument('--cells', nargs='+', required=True, help='name=cell pairs; the first is the main scorer, for example t5=gx_t5_q_w2 t5lq=gx_t5_lq_w2 qa=gx_qa_el_q_w2')
    parser.add_argument('--base', default='bm25_doc_both')
    parser.add_argument('--gate-quantile', type=float, default=0.8)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--db', type=Path, default=Path('docs.db'))
    parser.add_argument('--topics', type=Path, default=Path('data/dev-topics-all-docs.jsonl'))
    parser.add_argument('--judgments', type=Path, nargs='*', default=[Path('data/dev-topics-all-docs.jsonl'), Path('data/silver-judgments.jsonl')])
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
    point_set = set(points)
    answerable = [p for p in points if p in relq]
    nil = [p for p in points if p not in relq]
    sibling = collections.defaultdict(set)
    for (tid, qid, day), docs in relq.items():
        for doc_id in docs:
            sibling[(tid, day, doc_id)].add(qid)
    total_relevant = sum(len(v) for v in relq.values())

    cells = dict(item.split('=', 1) for item in args.cells)
    names = list(cells)
    runs = {n: {p: v for p, v in load_run(str(args.runs / f'{c}_rep' / 'run.json')).items() if p in point_set} for n, c in cells.items()}
    main_name = names[0]
    pairs = [(p, d) for p in points for d in runs[main_name][p] if all(d in runs[n].get(p, {}) for n in names)]
    index = {pair: i for i, pair in enumerate(pairs)}
    base = load_run(glob.glob(str(args.runs / f'{args.base}-*' / 'run.json'))[0])
    base_rank = {p: {d: i for i, d in enumerate([d for d, _ in sorted(base[p].items(), key=lambda x: -x[1])][:20], 1)} for p in points}
    base_scores = np.array([base[p][d] for p, d in pairs])
    base_z = (base_scores - base_scores.mean()) / base_scores.std()

    per_doc = collections.defaultdict(list)
    per_point = collections.defaultdict(list)
    for i, (p, d) in enumerate(pairs):
        per_doc[(p[0], p[2], d)].append(i)
        per_point[p].append(i)

    z = {}
    for n in names:
        raw = np.array([runs[n][p][d] for p, d in pairs])
        z[n] = (raw - raw.mean()) / raw.std()

    feat_cols = {}
    for n in names:
        zn = z[n]
        rel = np.zeros(len(pairs))
        rank = np.zeros(len(pairs))
        dayrel = np.zeros(len(pairs))
        for key, idx in per_doc.items():
            best = max(zn[i] for i in idx)
            order = sorted(idx, key=lambda i: -zn[i])
            for r, i in enumerate(order):
                rel[i] = zn[i] - best
                rank[i] = r / max(len(idx) - 1, 1)
        for p, idx in per_point.items():
            best = max(zn[i] for i in idx)
            for i in idx:
                dayrel[i] = zn[i] - best
        feat_cols[n] = {'z': zn, 'rel': rel, 'rank': rank, 'dayrel': dayrel}
    n_questions = np.zeros(len(pairs))
    for idx in per_doc.values():
        for i in idx:
            n_questions[i] = len(idx) / 15
    bm_rank = np.array([base_rank[p][d] for p, d in pairs]) / 20

    label_rel = np.array([d in relq.get(p, set()) for p, d in pairs])
    label_sib = np.array([(d not in relq.get(p, set())) and bool(sibling.get((p[0], p[2], d))) for p, d in pairs])
    core = label_rel | label_sib
    topic_of = np.array([p[0] for p, _ in pairs])

    def matrix(spec: list[str]):
        cols = []
        for item in spec:
            if item == 'bm25':
                cols += [base_z, bm_rank]
            elif item == 'nq':
                cols += [n_questions]
            else:
                for part in ('z', 'rel', 'rank', 'dayrel'):
                    cols.append(feat_cols[item][part])
        return np.column_stack(cols)

    specs = {f'{main_name} alone': [main_name]}
    if len(names) > 1:
        for other in names[1:]:
            specs[f'{main_name} + {other}'] = [main_name, other]
        specs['all scorers'] = list(names)
        specs['all scorers + BM25 + n questions'] = list(names) + ['bm25', 'nq']

    def fit_predict(model_kind: str, X, y, train_mask, test_mask):
        scaler = StandardScaler().fit(X[train_mask])
        if model_kind == 'lr':
            model = LogisticRegression(C=1.0, max_iter=2000)
        else:
            model = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0)
        model.fit(scaler.transform(X[train_mask]), y[train_mask])
        return model.predict_proba(scaler.transform(X[test_mask]))[:, 1]

    topics_list = sorted(assessed)
    print(f'{len(pairs)} pairs; {int(label_rel.sum())} relevant to the question, {int(label_sib.sum())} relevant only to a sibling; scorers: {", ".join(f"{n}={c}" for n, c in cells.items())}')
    print(f'single score {main_name} on the same pairs (no training): AUC vs sibling-only {auc(z[main_name][label_rel], z[main_name][label_sib]):.3f}')

    def top1_accuracy(prob_by_index: dict[int, float]) -> tuple[float, float]:
        hits, n = 0, 0
        for key, idx in per_doc.items():
            relset = sibling.get(key, set())
            scored = {pairs[i][0][1] for i in idx}
            r = len(relset & scored)
            if r == 0 or len(scored) == r or any(i not in prob_by_index for i in idx):
                continue
            best = max(idx, key=lambda i: prob_by_index[i])
            hits += pairs[best][0][1] in relset
            n += 1
        return (hits / n if n else float('nan')), n

    results = {}
    print('\nTASK 1, which question does a document belong to: trained on the pairs of one topic, tested on the other (relevant to this question against relevant only to a sibling)')
    print(f"{'model':>5} {'scorers':36}{'AUC (mean)':>11}{'AUC per topic':>20}{'top-1 right':>13}")
    for kind in ('lr', 'gb'):
        for label, spec in specs.items():
            X = matrix(spec)
            prob_all: dict[int, float] = {}
            aucs = []
            for t in topics_list:
                test = core & (topic_of == t)
                train = core & (topic_of != t)
                prob = fit_predict(kind, X, label_rel, train, test)
                for i, pr in zip(np.where(test)[0], prob):
                    prob_all[int(i)] = pr
                aucs.append(auc(prob[label_rel[test]], prob[~label_rel[test]]))
            acc, n = top1_accuracy(prob_all)
            print(f"{kind:>5} {label:36}{np.mean(aucs):11.3f}{'  '.join(f'{t[-3:]}:{a:.3f}' for t, a in zip(topics_list, aucs)):>20}{100 * acc:12.0f}%")

    per_q: dict[tuple[str, str], list] = collections.defaultdict(list)
    for (tid, qid, day), by in base.items():
        per_q[(tid, qid)].append((day, by))
    gate = {}
    for key, rows in per_q.items():
        history: deque[float] = deque(maxlen=60)
        for day, by in sorted(rows, key=lambda x: x[0]):
            docs = []
            if by and len(history) >= 30:
                ordered = sorted(history)
                bar = ordered[min(int(args.gate_quantile * len(ordered)), len(ordered) - 1)]
                docs = [d for d, s in sorted(by.items(), key=lambda t: -t[1])[:20] if s > bar]
            gate[(key[0], key[1], day)] = docs
            if by:
                history.append(max(by.values()))

    def metrics(lists):
        returned = sum(len(lists.get(p, [])) for p in points)
        relevant = sum(len([d for d in lists.get(p, []) if d in relq[p]]) for p in answerable)
        abstain = sum(1 for p in nil if not lists.get(p)) / len(nil)
        rr = 0.0
        for p in answerable:
            for i, d in enumerate(lists.get(p, []), 1):
                if d in relq[p]:
                    rr += 1 / i
                    break
        return returned, relevant, abstain, rr / len(answerable)

    g = metrics(gate)
    print(f'\nTASK 2, the gated result: BM25 quantile {args.gate_quantile} gate, then keep pairs whose predicted probability of being relevant to the question is at least a cut (trained on the other topic)')
    print(f"{'scorers':36}{'cut':>6}{'returned':>9}{'relevant':>9}{'found':>7}{'docs/rel':>9}{'abstain':>8}{'RR':>7}")
    print(f"{'(gate alone)':36}{'':>6}{g[0]:9d}{g[1]:9d}{100 * g[1] / total_relevant:6.0f}%{g[0] / g[1]:9.1f}{g[2]:8.2f}{g[3]:7.3f}")
    for label, spec in specs.items():
        X = matrix(spec)
        prob_full = np.zeros(len(pairs))
        for t in topics_list:
            test = topic_of == t
            prob_full[test] = fit_predict('lr', X, label_rel, ~test, test)
        for cut in (0.5, 0.3, 0.2, 0.1):
            lists = {}
            for p in points:
                lists[p] = sorted([d for d in gate.get(p, []) if (p, d) in index and prob_full[index[(p, d)]] >= cut], key=lambda d: -prob_full[index[(p, d)]])
            m = metrics(lists)
            print(f"{label:36}{cut:6.2f}{m[0]:9d}{m[1]:9d}{100 * m[1] / total_relevant:6.0f}%{(m[0] / m[1] if m[1] else float('nan')):9.1f}{m[2]:8.2f}{m[3]:7.3f}")


if __name__ == '__main__':
    main()

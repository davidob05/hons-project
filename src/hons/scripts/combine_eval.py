import argparse
import collections
import copy
import glob
import json
import statistics
from collections import deque
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
    parser = argparse.ArgumentParser(description='Combine the scores of two grid cells on the same (question, document) pairs and compare the combinations on sibling separation, assignment and the gated trade-off')
    parser.add_argument('--a', required=True, help='first cell, for example gx_t5_q_w2')
    parser.add_argument('--b', required=True, help='second cell, for example gx_qa_rb_q_w2')
    parser.add_argument('--base', default='bm25_doc_both')
    parser.add_argument('--gate-quantile', type=float, default=0.8)
    parser.add_argument('--margins', type=float, nargs='*', default=[0.5, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument('--write-tag', help='also write each chosen combination as a ranking run and a gated run (<tag>_<combo>_rk and _gt) that score.py can evaluate; the tag must stay within 12 characters')
    parser.add_argument('--write-combos', nargs='*', default=['zs', 'zm'], choices=['a', 'b', 'zs', 'zm', 'rm'], help='which combinations to write: a, b (each alone), zs (z-sum), zm (z-min), rm (rank mean; ranking run only)')
    parser.add_argument('--write-margin', type=float, default=2.0, help='the margin used for the gated runs')
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

    run_a = {p: v for p, v in load_run(str(args.runs / f'{args.a}_rep' / 'run.json')).items() if p in point_set}
    run_b = {p: v for p, v in load_run(str(args.runs / f'{args.b}_rep' / 'run.json')).items() if p in point_set}
    pairs = [(p, d) for p in points for d in run_a[p] if d in run_b.get(p, {})]
    a_all = np.array([run_a[p][d] for p, d in pairs])
    b_all = np.array([run_b[p][d] for p, d in pairs])
    za = (a_all - a_all.mean()) / a_all.std()
    zb = (b_all - b_all.mean()) / b_all.std()
    ra = np.argsort(np.argsort(a_all)) / len(a_all)
    rb = np.argsort(np.argsort(b_all)) / len(b_all)
    combos = {
        'A alone': za,
        'B alone': zb,
        'z-sum (A + B)': (za + zb) / 2,
        'z-min (both must agree)': np.minimum(za, zb),
        'rank mean': (ra + rb) / 2,
    }
    index = {pair: i for i, pair in enumerate(pairs)}
    kind = []
    for p, d in pairs:
        if d in relq.get(p, set()):
            kind.append('rel')
        elif sibling.get((p[0], p[2], d)):
            kind.append('sib')
        else:
            kind.append('other')
    kind = np.array(kind)
    print(f'A = {args.a}, B = {args.b}; {len(pairs)} pairs; correlation of the two scores {np.corrcoef(a_all, b_all)[0, 1]:.2f}')

    print(f"\n{'combination':26}{'AUC sibling':>12}{'AUC unjudged':>13}{'day AUC':>9}{'top-1 assign':>13}{'R-prec':>8}")
    per_doc_questions = collections.defaultdict(list)
    for i, (p, d) in enumerate(pairs):
        per_doc_questions[(p[0], p[2], d)].append((p[1], i))
    for name, score in combos.items():
        sib_auc = auc(score[kind == 'rel'], score[kind == 'sib'])
        unj_auc = auc(score[kind == 'rel'], score[kind == 'other'])
        top = np.array([max(score[index[(p, d)]] for d in run_a[p] if (p, d) in index) for p in answerable + nil])
        label = np.array([True] * len(answerable) + [False] * len(nil))
        day_auc = auc(top[label], top[~label])
        top1, rprec, n = 0, 0.0, 0
        for key, qs in per_doc_questions.items():
            relset = sibling.get(key, set())
            scored = {q for q, _ in qs}
            r = len(relset & scored)
            if r == 0 or len(scored) == r:
                continue
            order = [q for q, i in sorted(qs, key=lambda x: -score[x[1]])]
            top1 += order[0] in relset
            rprec += len(set(order[:r]) & relset) / r
            n += 1
        print(f'{name:26}{sib_auc:12.3f}{unj_auc:13.3f}{day_auc:9.3f}{100 * top1 / n:12.0f}%{100 * rprec / n:7.0f}%')

    base_path = glob.glob(str(args.runs / f'{args.base}-*' / 'run.json'))[0]
    base = load_run(base_path)
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
    total = sum(len(v) for v in relq.values())
    print(f'\nBM25 quantile {args.gate_quantile} gate, then keep a document for the questions whose combined score is within a margin of its best question')
    print(f"{'combination':26}{'margin':>7}{'returned':>9}{'relevant':>9}{'found':>7}{'docs/rel':>9}{'abstain':>8}{'RR':>7}")
    print(f"{'(gate alone)':26}{'':>7}{g[0]:9d}{g[1]:9d}{100 * g[1] / total:6.0f}%{g[0] / g[1]:9.1f}{g[2]:8.2f}{g[3]:7.3f}")
    for name, score in combos.items():
        for margin in args.margins:
            lists = {}
            for p in points:
                kept = []
                for d in gate.get(p, []):
                    others = per_doc_questions.get((p[0], p[2], d))
                    if not others or (p, d) not in index:
                        continue
                    best = max(score[i] for _, i in others)
                    if score[index[(p, d)]] >= best - margin:
                        kept.append(d)
                lists[p] = kept
            m = metrics(lists)
            print(f"{name:26}{margin:7.1f}{m[0]:9d}{m[1]:9d}{100 * m[1] / total:6.0f}%{(m[0] / m[1] if m[1] else float('nan')):9.1f}{m[2]:8.2f}{m[3]:7.3f}")

    if args.write_tag:
        if len(args.write_tag) > 12:
            raise SystemExit('--write-tag must stay within 12 characters')
        keys = {'a': 'A alone', 'b': 'B alone', 'zs': 'z-sum (A + B)', 'zm': 'z-min (both must agree)', 'rm': 'rank mean'}
        lines = open(base_path).read().splitlines()
        header = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:] if line.strip()]

        def emit(runtag: str, description: str, replace: dict, fall_back_to_gate: bool) -> None:
            out_records = copy.deepcopy(records)
            for record in out_records:
                for day, qs in record['results'].items():
                    kept_questions = []
                    for question in qs:
                        p = (record['topic'], question['qid'], day)
                        if p in replace:
                            ranking = replace[p]
                        elif fall_back_to_gate:
                            allowed = set(gate.get(p, []))
                            ranking = [hit for hit in question['doc_ranking'] if hit['doc_id'] in allowed]
                        else:
                            ranking = question['doc_ranking']
                        if ranking:
                            question['doc_ranking'] = ranking
                            kept_questions.append(question)
                    record['results'][day] = kept_questions
            out_header = dict(header)
            out_header['runtag'] = runtag
            out_header['description'] = description
            out_dir = args.runs / runtag
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / 'run.json', 'w') as f:
                f.write(json.dumps(out_header) + '\n')
                for record in out_records:
                    f.write(json.dumps(record) + '\n')
            (out_dir / 'config.json').write_text(json.dumps({'run': {'runtag': runtag, 'description': description}, 'combine': {'a': args.a, 'b': args.b, 'base': args.base, 'gate_quantile': args.gate_quantile, 'margin': args.write_margin}}, indent=2))
            print(f'wrote {out_dir}')

        for key in args.write_combos:
            name = keys[key]
            score = combos[name]
            ranking_run = {}
            for p in points:
                ranked = sorted((d for d in run_a[p] if (p, d) in index), key=lambda d: -score[index[(p, d)]])
                ranking_run[p] = [{'doc_id': d, 'score': float(score[index[(p, d)]])} for d in ranked]
            description = f'The top 20 of {args.base} reranked by the {name} combination of {args.a} (A) and {args.b} (B) on days with judged documents.'
            emit(f'{args.write_tag}_{key}_rk', description, ranking_run, False)
            if key == 'rm':
                continue
            gated_run = {}
            for p in points:
                kept = []
                for d in gate.get(p, []):
                    others = per_doc_questions.get((p[0], p[2], d))
                    if not others or (p, d) not in index:
                        continue
                    if score[index[(p, d)]] >= max(score[i] for _, i in others) - args.write_margin:
                        kept.append(d)
                kept.sort(key=lambda d: -score[index[(p, d)]])
                gated_run[p] = [{'doc_id': d, 'score': float(score[index[(p, d)]])} for d in kept]
            description = f'{args.base} with a quantile {args.gate_quantile} history gate, then a document is kept for the questions whose {name} score of {args.a} (A) and {args.b} (B) is within {args.write_margin} of its best question.'
            emit(f'{args.write_tag}_{key}_gt', description, gated_run, True)


if __name__ == '__main__':
    main()

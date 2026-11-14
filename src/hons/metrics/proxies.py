import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from cdet_api.types import SubmittedTopicResults

from hons.metrics.evaluation import load_expected_sets, load_reachable
from hons.types import QuestionKey

GAIN_DEPTH = 100
METRICS = ('p1', 'p2', 'p3', 'f1', 'false_alarm', 'miss')


@dataclass
class RunRankings:
    ranked: dict[QuestionKey, dict[str, list[str]]] = field(default_factory=dict)
    question_order: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    days: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PairScores:
    p1: float
    p2: float
    p3: float | None
    f1: float | None
    false_alarm: float | None
    miss: float | None


def load_run(run_file: Path) -> RunRankings:
    run = RunRankings()
    with open(run_file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if 'topic' not in record:
                continue
            topic = SubmittedTopicResults.model_validate(record)
            for day, results in topic.results.items():
                run.days.add(day)
                listed = []
                for question in sorted(results, key=lambda q: q.question_rank):
                    ranking = [hit.doc_id for hit in question.doc_ranking]
                    run.ranked.setdefault(QuestionKey(topic.topic, question.qid), {})[day] = ranking
                    if ranking:
                        listed.append(question.qid)
                run.question_order.setdefault(topic.topic, {})[day] = listed
    return run


def discount(rank: int) -> float:
    return 1 / math.log2(rank + 1)


def ideal_dcg(count: int) -> float:
    return sum(discount(rank) for rank in range(1, count + 1))


def pair_scores(returned: list[str], relevant: set[str]) -> PairScores:
    if not relevant:
        silent = 0.0 if returned else 1.0
        return PairScores(p1=silent, p2=silent, p3=None, f1=None, false_alarm=1.0 - silent, miss=None)

    hits = [doc_id in relevant for doc_id in returned]
    found = sum(hits)
    precision = found / len(returned) if returned else 0.0
    recall = found / len(relevant)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    gain = sum((1 if hit else -1) * discount(rank) for rank, hit in enumerate(hits, start=1))
    return PairScores(
        p1=precision,
        p2=found / min(len(relevant), GAIN_DEPTH),
        p3=gain / ideal_dcg(len(relevant)),
        f1=f1,
        false_alarm=None,
        miss=0.0 if returned else 1.0,
    )


def mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def average_precision(listed: list[str], relevant: set[str]) -> float:
    found = 0
    total = 0.0
    for rank, qid in enumerate(listed, start=1):
        if qid in relevant:
            found += 1
            total += found / rank
    return total / len(relevant)


def score_proxies(expected: dict[QuestionKey, set[str]], run: RunRankings, days: list[str], appeared: dict[str, str]) -> dict:
    relevant_on: dict[QuestionKey, dict[str, set[str]]] = {}
    for key, docs in expected.items():
        by_day: dict[str, set[str]] = {}
        for doc_id, day in appeared.items():
            if doc_id in docs:
                by_day.setdefault(day, set()).add(doc_id)
        relevant_on[key] = by_day

    per_question: dict[QuestionKey, dict[str, float | None]] = {}
    active_pairs = inactive_pairs = 0
    for key in expected:
        pairs = []
        for day in days:
            relevant = relevant_on[key].get(day, set())
            pairs.append(pair_scores(run.ranked.get(key, {}).get(day, []), relevant))
            if relevant:
                active_pairs += 1
            else:
                inactive_pairs += 1
        per_question[key] = {name: mean(getattr(pair, name) for pair in pairs) for name in METRICS}

    tids = sorted({key.tid for key in expected})
    topics = {}
    for tid in tids:
        keys = [key for key in expected if key.tid == tid]
        scores = {name: mean(per_question[key][name] for key in keys) for name in METRICS}

        precisions, false_positives = [], []
        for day in days:
            relevant_questions = {key.qid for key in keys if relevant_on[key].get(day)}
            listed = run.question_order.get(tid, {}).get(day, [])
            if relevant_questions:
                precisions.append(average_precision(listed, relevant_questions))
            else:
                false_positives.append(1.0 if listed else 0.0)
        scores['p5_ap'] = mean(precisions)
        scores['p5_false_positive'] = mean(false_positives)
        topics[tid] = scores

    overall = {name: mean(topic[name] for topic in topics.values()) for name in (*METRICS, 'p5_ap', 'p5_false_positive')}
    return {
        'days': len(days),
        'active_pairs': active_pairs,
        'inactive_pairs': inactive_pairs,
        'overall': overall,
        'topics': topics,
        'questions': {f'{key.tid}:{key.qid}': scores for key, scores in per_question.items()},
    }


def evaluate_proxies(run_file: Path, expected_file: Path, db_path: Path, empty: bool = False,
                     silver_file: Path | None = None) -> dict:
    sets = load_expected_sets(expected_file, silver_file)
    run = load_run(run_file)
    if empty:
        run = RunRankings(days=run.days)
    days = sorted(run.days)
    every_id = set().union(*(docs for expected in sets.values() for docs in expected.values()))
    appeared = load_reachable(db_path, every_id, run.days)

    result = {'label': 'proxy metrics, not the official scorer', 'empty_run': empty,
              **score_proxies(sets['dev'], run, days, appeared)}
    if 'combined' in sets:
        result['combined'] = score_proxies(sets['combined'], run, days, appeared)
    return result


def _fmt(value: float | None) -> str:
    return '   n/a' if value is None else f'{value:6.3f}'


def print_result(title: str, result: dict) -> None:
    print(f'--- {title}')
    print(f'{result["days"]} days; {result["active_pairs"]} active and {result["inactive_pairs"]} inactive question-day pairs')
    header = f'{"":10}{"P1":>7}{"P2":>7}{"P3":>7}{"F1":>7}{"falseAl":>8}{"miss":>7}{"P5 AP":>7}{"P5 FP":>7}'
    print(header)
    rows = [('overall', result['overall'])] + list(result['topics'].items())
    for name, scores in rows:
        print(f'{name:10}' + ''.join(f'{_fmt(scores[m]):>7}' for m in METRICS[:4])
              + f'{_fmt(scores["false_alarm"]):>8}{_fmt(scores["miss"]):>7}'
              + f'{_fmt(scores["p5_ap"]):>7}{_fmt(scores["p5_false_positive"]):>7}')


def report(result: dict, out_file: Path) -> None:
    print(f'PROXY metrics (not the official scorer){"  [empty run]" if result["empty_run"] else ""}')
    print('P3 is averaged over active pairs only; an empty list on an active pair scores 0')
    print_result('DEV judgments (the track\'s lists)', result)
    if 'combined' in result:
        print_result('COMBINED (dev + silver judgments; silver is not official)', result['combined'])
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)

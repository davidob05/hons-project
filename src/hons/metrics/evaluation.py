import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cdet_api.models import Document, db
from cdet_api.types import SubmittedTopicResults

from hons.types import QuestionKey

# question -> day -> document ids in the order the run ranked them
Returned = dict[QuestionKey, dict[str, list[str]]]


@dataclass(frozen=True)
class QuestionMetrics:
    tid: str
    qid: str
    expected: int           # judged-relevant documents for the question
    reachable: int          # of those, the ones that appeared in the days that were run
    found: int              # reachable documents returned on at least one day
    found_on_day: int       # reachable documents returned on the day they appeared
    returned: int           # distinct documents returned
    active_days: int        # days with a non-empty ranking
    unjudged_active_days: int   # active days on which no judged document for the question appeared
    judged_days: int        # days on which at least one reachable judged document appeared
    reciprocal_rank_sum: float  # over judged days: 1 / rank of the best judged document returned, 0 if none
    total_returned: int     # documents returned, summed over days

    @property
    def recall(self) -> float | None:
        return self.found / self.reachable if self.reachable else None

    @property
    def recall_on_day(self) -> float | None:
        return self.found_on_day / self.reachable if self.reachable else None

    @property
    def precision(self) -> float | None:
        # a lower bound: the judgments are not exhaustive, so an unjudged document may be relevant.
        # Undefined when no judged document was in the run, since then nothing could have counted
        return self.found / self.returned if self.returned and self.reachable else None

    @property
    def mrr(self) -> float | None:
        return self.reciprocal_rank_sum / self.judged_days if self.judged_days else None

    @property
    def docs_per_active_day(self) -> float | None:
        return self.total_returned / self.active_days if self.active_days else None

    def as_dict(self) -> dict:
        return {**asdict(self), 'recall': self.recall, 'recall_on_day': self.recall_on_day,
                'precision': self.precision, 'mrr': self.mrr, 'docs_per_active_day': self.docs_per_active_day}


def load_expected(path: Path) -> dict[QuestionKey, set[str]]:
    expected = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            topic = json.loads(line)
            for question in topic['questions']:
                expected[QuestionKey(topic['tid'], question['qid'])] = set(question['rel_docs'])
    return expected


def load_returned(run_file: Path) -> tuple[Returned, set[str]]:
    # also returns the days the run covers, which are the keys of every topic's results
    returned: Returned = {}
    days = set()
    with open(run_file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if 'topic' not in record:       # the metadata line
                continue
            topic = SubmittedTopicResults.model_validate(record)
            days.update(topic.results)
            for day, results in topic.results.items():
                for question in results:
                    ranking = [hit.doc_id for hit in question.doc_ranking]
                    returned.setdefault(QuestionKey(topic.topic, question.qid), {})[day] = ranking
    return returned, days


def load_reachable(db_path: Path, ids: set[str], days: set[str]) -> dict[str, str]:
    # the expected documents that the database shows on one of the days the run covered, with that day
    db.init(str(db_path))
    db.connect(reuse_if_open=True)
    try:
        found = {}
        wanted = list(ids)
        for start in range(0, len(wanted), 500):
            query = Document.select(Document.id, Document.day).where(Document.id.in_(wanted[start:start + 500]))
            found.update({row.id: row.day for row in query if row.day in days})
        return found
    finally:
        db.close()


def question_metrics(key: QuestionKey, docs: set[str], by_day: dict[str, list[str]], appeared: dict[str, str]) -> QuestionMetrics:
    # appeared: judged document -> the day it appeared, for the judged documents in the days that were run
    reachable = {d: day for d, day in appeared.items() if d in docs}
    judged_by_day: dict[str, set[str]] = {}
    for doc_id, day in reachable.items():
        judged_by_day.setdefault(day, set()).add(doc_id)

    distinct = {d for ranking in by_day.values() for d in ranking}
    found_on_day = sum(1 for d, day in reachable.items() if d in by_day.get(day, ()))

    active = [day for day, ranking in by_day.items() if ranking]
    unjudged_active = sum(1 for day in active if day not in judged_by_day)

    reciprocal_rank_sum = 0.0
    for day, judged in judged_by_day.items():
        for rank, doc_id in enumerate(by_day.get(day, ()), start=1):
            if doc_id in judged:
                reciprocal_rank_sum += 1 / rank
                break

    return QuestionMetrics(
        tid=key.tid, qid=key.qid,
        expected=len(docs), reachable=len(reachable),
        found=len(distinct & reachable.keys()), found_on_day=found_on_day,
        returned=len(distinct),
        active_days=len(active), unjudged_active_days=unjudged_active,
        judged_days=len(judged_by_day), reciprocal_rank_sum=reciprocal_rank_sum,
        total_returned=sum(len(ranking) for ranking in by_day.values()),
    )


def evaluate(run_file: Path, expected_file: Path, db_path: Path) -> list[QuestionMetrics]:
    # a short run is judged only on the expected documents that appeared in the days it covered
    expected = load_expected(expected_file)
    returned, days = load_returned(run_file)
    appeared = load_reachable(db_path, set().union(*expected.values()), days)
    return [question_metrics(key, docs, returned.get(key, {}), appeared) for key, docs in expected.items()]


def macro(values: list[float | None]) -> float | None:
    # mean over the questions where the metric is defined
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def summarise(rows: list[QuestionMetrics]) -> dict[str, float | None]:
    reachable = sum(r.reachable for r in rows)
    returned = sum(r.returned for r in rows if r.reachable)
    active = sum(r.active_days for r in rows)
    return {
        'macro_recall': macro([r.recall for r in rows]),
        'macro_recall_on_day': macro([r.recall_on_day for r in rows]),
        'macro_precision': macro([r.precision for r in rows]),
        'macro_mrr': macro([r.mrr for r in rows]),
        'micro_recall': sum(r.found for r in rows) / reachable if reachable else None,
        'micro_recall_on_day': sum(r.found_on_day for r in rows) / reachable if reachable else None,
        'micro_precision': sum(r.found for r in rows) / returned if returned else None,
        'unjudged_active_day_share': sum(r.unjudged_active_days for r in rows) / active if active else None,
        'docs_per_active_day': sum(r.total_returned for r in rows) / active if active else None,
    }


def _fmt(value: float | None, width: int, places: int = 2) -> str:
    return f'{value:>{width}.{places}f}' if value is not None else f'{"-":>{width}}'


def report(rows: list[QuestionMetrics], out_file: Path) -> None:
    print(f'{"topic":<12}{"qid":<7}{"exp":>5}{"reach":>6}{"found":>6}{"onday":>6}{"ret":>6}'
          f'{"recall":>8}{"r@day":>7}{"prec":>7}{"mrr":>6}{"active":>7}{"unjudged":>9}')
    for r in rows:
        print(f'{r.tid:<12}{r.qid:<7}{r.expected:>5}{r.reachable:>6}{r.found:>6}{r.found_on_day:>6}{r.returned:>6}'
              f'{_fmt(r.recall, 8)}{_fmt(r.recall_on_day, 7)}{_fmt(r.precision, 7, 3)}{_fmt(r.mrr, 6)}'
              f'{r.active_days:>7}{r.unjudged_active_days:>9}')

    summary = summarise(rows)
    print()
    print(f'recall      macro {_fmt(summary["macro_recall"], 6, 3)}   micro {_fmt(summary["micro_recall"], 6, 3)}')
    print(f'recall@day  macro {_fmt(summary["macro_recall_on_day"], 6, 3)}   micro {_fmt(summary["micro_recall_on_day"], 6, 3)}')
    print(f'precision   macro {_fmt(summary["macro_precision"], 6, 4)}   micro {_fmt(summary["micro_precision"], 6, 4)}   (lower bound: judgments are not exhaustive)')
    print(f'mrr         macro {_fmt(summary["macro_mrr"], 6, 3)}   (days with a judged document)')
    print(f'volume      {_fmt(summary["docs_per_active_day"], 6, 1)} documents per active question-day, '
          f'{_fmt(summary["unjudged_active_day_share"], 5, 3)} of active question-days have no judged document')

    with open(out_file, 'w') as f:
        json.dump({'summary': summary, 'questions': [r.as_dict() for r in rows]}, f, indent=2)

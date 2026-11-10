import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cdet_api.models import Document, db
from cdet_api.types import SubmittedTopicResults

from hons.types import QuestionKey


@dataclass(frozen=True)
class QuestionRecall:
    tid: str
    qid: str
    expected: int       # judged-relevant documents for the question
    reachable: int      # of those, the ones that appeared in the days that were run
    found: int          # reachable documents that were returned on at least one day
    returned: int       # distinct documents returned, so returning everything cannot hide behind recall

    @property
    def recall(self) -> float | None:
        return self.found / self.reachable if self.reachable else None


def load_expected(path: Path) -> dict[QuestionKey, set[str]]:
    # The judged documents are undated, so a document counts as found if it is returned on any day
    expected = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            topic = json.loads(line)
            for question in topic['questions']:
                expected[QuestionKey(topic['tid'], question['qid'])] = set(question['rel_docs'])
    return expected


def load_returned(run_file: Path) -> tuple[dict[QuestionKey, set[str]], set[str]]:
    # also returns the days the run covers, which are the keys of every topic's results
    returned = {}
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
            for results in topic.results.values():
                for question in results:
                    docs = returned.setdefault(QuestionKey(topic.topic, question.qid), set())
                    docs.update(hit.doc_id for hit in question.doc_ranking)
    return returned, days


def load_reachable(db_path: Path, ids: set[str], days: set[str]) -> set[str]:
    # the expected documents that the database shows on one of the days the run covered
    db.init(str(db_path))
    db.connect(reuse_if_open=True)
    try:
        found = set()
        wanted = list(ids)
        for start in range(0, len(wanted), 500):
            query = Document.select(Document.id, Document.day).where(Document.id.in_(wanted[start:start + 500]))
            found.update(row.id for row in query if row.day in days)
        return found
    finally:
        db.close()


def evaluate(run_file: Path, expected_file: Path, db_path: Path) -> list[QuestionRecall]:
    # a short run is judged only on the expected documents that appeared in the days it covered
    expected = load_expected(expected_file)
    returned, days = load_returned(run_file)
    seen = load_reachable(db_path, set().union(*expected.values()), days)
    rows = []
    for key, docs in expected.items():
        reachable = docs & seen
        got = returned.get(key, set())
        rows.append(QuestionRecall(key.tid, key.qid, len(docs), len(reachable), len(reachable & got), len(got)))
    return rows


def report(rows: list[QuestionRecall], out_file: Path) -> None:
    scored = [r for r in rows if r.recall is not None]
    macro = sum(r.recall for r in scored) / len(scored) if scored else 0.0
    found = sum(r.found for r in rows)
    reachable = sum(r.reachable for r in rows)
    micro = found / reachable if reachable else 0.0

    print(f'{"topic":<12}{"qid":<7}{"expected":>9}{"reachable":>10}{"found":>7}{"returned":>9}{"recall":>8}')
    for r in rows:
        recall = f'{r.recall:.2f}' if r.recall is not None else '-'
        print(f'{r.tid:<12}{r.qid:<7}{r.expected:>9}{r.reachable:>10}{r.found:>7}{r.returned:>9}{recall:>8}')
    print(f'macro recall {macro:.3f}   micro recall {micro:.3f} ({found}/{reachable})')

    with open(out_file, 'w') as f:
        json.dump({'macro_recall': macro, 'micro_recall': micro,
                   'questions': [{**asdict(r), 'recall': r.recall} for r in rows]}, f, indent=2)

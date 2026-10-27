import datetime
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from cdet_api.types import DayResults, DayString, DocumentSchema

DAY_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class QuestionKey(NamedTuple):
    # qids repeat across topics (q_1 exists in most), so a question is identified by both
    tid: str
    qid: str


@dataclass(frozen=True)
class DayBatch:
    # One day's documents, exactly as read from the source.
    day: DayString
    sequence_number: int
    documents: tuple[DocumentSchema, ...]

    def __post_init__(self):
        if not DAY_PATTERN.match(self.day):
            raise ValueError(f'day must be YYYY-MM-DD, got {self.day!r}')
        datetime.date.fromisoformat(self.day)
        if self.sequence_number < 0:
            raise ValueError(f'sequence_number must be >= 0, got {self.sequence_number}')
        object.__setattr__(self, 'documents', tuple(self.documents))
        for doc in self.documents:
            if doc.day != self.day:
                raise ValueError(f'document {doc.id} is from {doc.day}, not {self.day}')


@dataclass(frozen=True)
class Fragment:
    # Identified by its keys in Fragments (doc id, index), not by anything stored here
    text: str


@dataclass(frozen=True)
class DocInfo:
    # Neutral defaults are what the Stage 0 `off` implementation reports
    group_id: str | None = None
    family: str | None = None
    boilerplate_share: float = 0.0


@dataclass
class Fragments:
    # doc id -> fragment index -> fragment
    fragments: dict[str, dict[int, Fragment]] = field(default_factory=dict)
    docs: dict[str, DocInfo] = field(default_factory=dict)

    def __iter__(self):
        for doc_id, by_index in self.fragments.items():
            for index, fragment in by_index.items():
                yield doc_id, index, fragment


@dataclass(frozen=True)
class QuestionVersion:
    qid: str
    tid: str
    text: str
    effective_from: int = 0     # sequence number of the first day this version is in force

    @property
    def key(self) -> QuestionKey:
        return QuestionKey(self.tid, self.qid)


@dataclass(frozen=True)
class DocScore:
    score: float
    best_fragment: int          # index into Fragments.fragments[doc_id]
    standardised: float | None = None


@dataclass
class Scores:
    # question -> doc id -> score
    per_question: dict[QuestionKey, dict[str, DocScore]] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    score: float
    excess: float | None = None     # amount over the Stage 3 bar
    pvalue: float | None = None
    novelty: float | None = None
    weight: float = 1.0


@dataclass
class Candidates:
    per_question: dict[QuestionKey, list[Candidate]] = field(default_factory=dict)


@dataclass
class DayOutput:
    # topic id -> the toolkit's own type. The only place toolkit result types appear here
    per_topic: dict[str, DayResults] = field(default_factory=dict)

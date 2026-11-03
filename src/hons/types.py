import datetime
import re
from collections.abc import Iterator
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

    def __iter__(self) -> Iterator[DocumentSchema]:
        return iter(self.documents)

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


@dataclass
class Candidate:
    doc_id: str
    score: DocScore
    excess: float | None = None     # amount over the Stage 3 bar
    pvalue: float | None = None
    novelty: float | None = None
    weight: float = 1.0


@dataclass
class Candidates:
    per_question: dict[QuestionKey, list[Candidate]] = field(default_factory=dict)
    def add_candidate(self, candidate: Candidate, key: QuestionKey) -> None:
        self.per_question.setdefault(key, []).append(candidate)

    def __iter__(self):
        return iter(self.per_question)


@dataclass
class DayOutput:
    # topic id -> the toolkit's own type. The only place toolkit result types appear here
    per_topic: dict[str, DayResults] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    tid: str
    question_text: str


@dataclass
class Proposals:
    proposals: list[Proposal] = field(default_factory=list)


class MemoryEntry(NamedTuple):
    # A fragment a question has already seen. Text is not kept; embeddings come from the cache by hash
    sequence_number: int
    doc_id: str
    fragment_hash: str


class ScoreRecord(NamedTuple):
    sequence_number: int
    score: float


@dataclass
class Cluster:
    centroid: list[float]
    doc_ids: list[str] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    proposed: bool = False


@dataclass
class State:
    # Everything carried between days. Stages only read it, and Stage 7 writes the day's entries
    # after the sink has written, so during a day this holds earlier days only. An empty State is
    # the stateless `off` case
    memory: dict[QuestionKey, list[MemoryEntry]] = field(default_factory=dict)          # Stage 4
    score_history: dict[QuestionKey, list[ScoreRecord]] = field(default_factory=dict)   # Stage 2 windows
    clusters: dict[str, list[Cluster]] = field(default_factory=dict)                    # Stage 6, by tid

    def remember(self, key: QuestionKey, entry: MemoryEntry) -> None:
        self.memory.setdefault(key, []).append(entry)

    def record_score(self, key: QuestionKey, record: ScoreRecord) -> None:
        self.score_history.setdefault(key, []).append(record)

    def prune(self, current: int, window_days: int) -> None:
        oldest = current - window_days
        for key, entries in self.memory.items():
            self.memory[key] = [e for e in entries if e.sequence_number > oldest]
        for key, records in self.score_history.items():
            self.score_history[key] = [r for r in records if r.sequence_number > oldest]
        for tid, found in self.clusters.items():
            self.clusters[tid] = [c for c in found if c.last_seen > oldest]

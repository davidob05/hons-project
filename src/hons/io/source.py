from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from cdet_api.client import CDetClient, NoMoreDaysException
from cdet_api.models import Day, Document, db
from cdet_api.types import DocumentSchema

from hons.types import DayBatch


class DaySource(Protocol):
    def __iter__(self) -> Iterator[DayBatch]: ...


class ReplaySource:
    def __init__(self, path: Path, days: tuple[int, int] | None = None):
        self.path = Path(path)
        self.days = days

    def __iter__(self) -> Iterator[DayBatch]:
        db.init(str(self.path))
        db.connect(reuse_if_open=True)
        try:
            query = Day.select().order_by(Day.seq_day)
            if self.days is not None:
                start, end = self.days
                query = query.where((Day.seq_day >= start) & (Day.seq_day < end))
            for row in query:
                documents = Document.select().where(Document.day == row.day)
                yield DayBatch(
                    day=row.day,
                    sequence_number=row.seq_day,
                    documents=tuple(DocumentSchema.model_validate(doc) for doc in documents),
                )
        finally:
            db.close()


class LiveSource:
    def __init__(self, client: CDetClient, token: str):
        self.client = client
        self.token = token

    def __iter__(self) -> Iterator[DayBatch]:
        sequence_number = 0
        while True:
            try:
                documents = self.client.next_day(self.token)
            except NoMoreDaysException:
                return
            if documents is None:
                raise RuntimeError(f'next_day failed at sequence number {sequence_number}')
            if not documents:
                raise RuntimeError(f'next_day returned no documents at sequence number {sequence_number}')
            yield DayBatch(
                day=documents[0].day,
                sequence_number=sequence_number,
                documents=tuple(documents),
            )
            sequence_number += 1

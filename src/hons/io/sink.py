from pathlib import Path
from typing import Protocol

from cdet_api.client import CDetClient
from cdet_api.types import DayResults, DayString, QuestionResults, RunMetadata, SubmittedTopicResults

from hons.types import DayOutput


class Sink(Protocol):
    def write(self, day: DayString, output: DayOutput) -> None: ...

    def close(self) -> None: ...


def complete(output: DayOutput, topics: list[str]) -> dict[str, DayResults]:
    unknown = output.per_topic.keys() - set(topics)
    if unknown:
        raise ValueError(f'output has topics not in the topics file: {sorted(unknown)}')
    return {tid: output.per_topic.get(tid, DayResults(results=[])) for tid in topics}


class ReplaySink:
    def __init__(self, path: Path, metadata: RunMetadata, topics: list[str]):
        self.path = Path(path)
        self.metadata = metadata
        self.topics = topics
        self.days: dict[str, dict[DayString, list[QuestionResults]]] = {tid: {} for tid in topics}

    def write(self, day: DayString, output: DayOutput) -> None:
        for tid, results in complete(output, self.topics).items():
            if day in self.days[tid]:
                raise ValueError(f'{day} was already written for {tid}')
            self.days[tid][day] = results.results

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('w') as f:
            f.write(self.metadata.model_dump_json() + '\n')
            for tid, by_day in self.days.items():
                f.write(SubmittedTopicResults(topic=tid, results=by_day).model_dump_json() + '\n')


class LiveSink:
    def __init__(self, client: CDetClient, token: str, topics: list[str], output_filename: str):
        self.client = client
        self.token = token
        self.topics = topics
        self.output_filename = output_filename

    def write(self, day: DayString, output: DayOutput) -> None:
        for tid, results in complete(output, self.topics).items():
            self.client.retrieval(token=self.token, topic=tid, retrieval_results=results)

    def close(self) -> None:
        self.client.finalize_run(self.token, self.output_filename)

import json
from pathlib import Path
from typing import Any

from cdet_api.client import CDetClient
from cdet_api.types import RunMetadata

from hons.config import Config
from hons.io.sink import LiveSink, ReplaySink, Sink
from hons.io.source import DaySource, LiveSource, ReplaySource
from hons.registry import build_stages
from hons.stages.s6_propose import add_proposals
from hons.types import Candidates, DayBatch, DayOutput, QuestionKey, QuestionVersion, State


def initial_questions(topics: list[dict]) -> dict[QuestionKey, QuestionVersion]:
    # get initial questions from the supplied file
    questions = {}
    for topic in topics:
        for question in topic['questions']:
            version = QuestionVersion(qid=question['qid'], tid=topic['tid'], text=question['question'])
            questions[version.key] = version
    return questions


def run_day(stages: dict[str, Any], batch: DayBatch, base: dict[QuestionKey, QuestionVersion],
            examples: dict[str, list[QuestionKey]], state: State,
            runtag: str) -> tuple[DayOutput, Candidates, list[str], list[QuestionVersion]]:
    # replace any questions with a new version if a new version is available, at the end of the day
    questions = {**base, **state.questions}

    fragments = stages['s0_fragment'].run(batch)
    seeded = stages['s1_questions'].run(batch, examples)
    scores = stages['s2_score'].run(fragments, questions)
    candidates = stages['s3_tail'].run(scores)
    candidates = stages['s4_novelty'].run(candidates, state, fragments)
    results = stages['s5_gate'].run(candidates)
    proposals = stages['s6_propose'].run(results, candidates)
    results, proposed = add_proposals(results, proposals, runtag, batch.sequence_number)
    return DayOutput(per_topic=results), candidates, seeded, proposed


def build_io(config: Config, run_dir: Path, topics: list[dict]) -> tuple[DaySource, Sink]:
    metadata = RunMetadata(
        runtag=config.run.runtag,
        description=config.run.description,
        run_type='automatic',
        models=list(config.models),
        extern='No external data used.',
    )
    tids = [topic['tid'] for topic in topics]
    if config.run.mode == 'replay':
        return ReplaySource(config.run.db, config.run.days), ReplaySink(run_dir / 'run.json', metadata, tids)

    # Live: the server hands out days one at a time, and the run file is written by finalize_run
    client = CDetClient(base_url=config.run.base_url)
    token = client.start_run(config.run.api_key, metadata)
    if token is None:
        raise RuntimeError(f'start_run failed against {config.run.base_url}; is the server running?')
    return LiveSource(client, token), LiveSink(client, token, tids, str(run_dir / 'run.json'))


def run(config: Config):

    run_dir = Path('runs') / f'{config.run.runtag}-{config.digest()}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.json','w') as f:
        json.dump(config.raw,f,indent=2,default=str)

    with open(config.run.topics) as f:
        topics = [json.loads(line) for line in f if line.strip()]

    examples = {}
    for topic in topics:
        for question in topic['questions']:
            for doc_id in question['rel_docs']:
                examples.setdefault(doc_id, []).append(QuestionKey(topic['tid'],question['qid']))

    base = initial_questions(topics)

    # build each stage at the start, before running anything, so that bad inputs fail early
    stages = build_stages(config)

    source, sink = build_io(config, run_dir, topics)
    state = State()
    try:
        for batch in source:
            if config.run.days is not None and batch.sequence_number >= config.run.days[1]:
                break
            output, candidates, seeded, proposed = run_day(stages, batch, base, examples, state, config.run.runtag)
            sink.write(batch.day, output)
            # update state after writing output, so that output reflects what was used in the day, not what changes after the day
            state = stages['s7_state'].run(state, candidates, seeded, proposed)
    finally:
        sink.close()

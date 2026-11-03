from cdet_api.types import DayResults, QuestionResults

from hons.registry import register
from hons.types import Candidates, Proposals, QuestionVersion

STAGE = 's6_propose'


def add_proposals(results: dict[str, DayResults], proposals: Proposals, runtag: str, sequence_number: int) -> tuple[dict[str, DayResults], list[QuestionVersion]]:
    versions = []
    merged = {tid: DayResults(results=list(day.results)) for tid, day in results.items()}
    for index, proposal in enumerate(proposals.proposals):
        day = merged.setdefault(proposal.tid, DayResults(results=[]))
        qid = f'{runtag}_{sequence_number}_{index}'
        versions.append(QuestionVersion(qid=qid, tid=proposal.tid, text=proposal.question_text,
                                        effective_from=sequence_number + 1))
        day.results.append(QuestionResults(
            qid=qid,
            question_rank=len(day.results) + 1,
            question_text=proposal.question_text,
            doc_ranking=[],
        ))
    return merged, versions


@register(STAGE,'off')
class NoProposals:
    def run(self, results: dict[str,DayResults], candidates: Candidates) -> Proposals:
        return Proposals()

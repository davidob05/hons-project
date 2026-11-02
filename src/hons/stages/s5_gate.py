from hons.registry import register
from hons.types import QuestionKey, DocScore

STAGE = 's5_gate'

@register(STAGE,'all_candidates')
class AllCandidates:
    def run(self, tails: dict[QuestionKey,dict[str,DocScore]]) -> dict[QuestionKey,list[tuple[str,DocScore]]]:
        return { qid : sorted(by_doc.items(), key=lambda item: item[1].score, reverse=True) for qid, by_doc in tails.items() }

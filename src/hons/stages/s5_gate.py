from hons.registry import register
from hons.types import QuestionKey, DocScore

STAGE = 's5_gate'

@register(STAGE,'all_candidates')
class AllCandidates:
    def run(self, tails: dict[QuestionKey,dict[str,DocScore]]) -> dict[QuestionKey,list[str]]:
        return { qid : sorted(by_doc, key=lambda doc_id: by_doc[doc_id].score, reverse=True) for qid, by_doc in tails.items() }

from hons.registry import register
from hons.types import DocScore, Scores, QuestionKey, Candidates, Candidate

STAGE = 's3_tail'

@register(STAGE,'off')
class ReturnNone:
    def run(self,scores: Scores) -> Candidates:
        return Candidates()


@register(STAGE,'top_1')
class Top1:

    k = 1

    def run(self,scores: Scores) -> Candidates:
        candidates = Candidates()
        for qid, by_doc in scores.per_question.items():
            if not by_doc:
                continue
            best = max(by_doc, key=lambda doc_id: by_doc[doc_id].score)
            candidates.add_candidate(Candidate(best,by_doc[best]),qid)
        return candidates

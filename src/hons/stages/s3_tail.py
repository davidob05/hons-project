from hons.registry import register
from hons.types import DocScore, Scores, QuestionKey

STAGE = 's3_tail'

@register(STAGE,'off')
class ReturnNone:
    def run(self,scores: Scores) -> dict[QuestionKey,dict]:
        return { qid : {} for qid in scores.per_question.keys() }


@register(STAGE,'top_1')
class Top1:

    k = 1

    def run(self,scores: Scores) -> dict[QuestionKey,dict[str,DocScore]]:
        tails = {}
        for qid, by_doc in scores.per_question.items():
            if not by_doc:
                tails[qid] = {}
                continue
            best = max(by_doc, key=lambda doc_id: by_doc[doc_id].score)
            tails[qid] = { best : by_doc[best] }
        return tails

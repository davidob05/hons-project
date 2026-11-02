from cdet_api.types import DayResults, Hit, QuestionResults

from hons.registry import register
from hons.types import QuestionKey, DocScore

STAGE = 's5_gate'


def to_day_results(selected: dict[QuestionKey,dict[str,DocScore]]) -> dict[str,DayResults]:
    by_topic: dict[str, list[tuple[str, list[Hit]]]] = {}
    for (tid, qid), docs in selected.items():
        if not docs:
            continue
        sorted_docs = sorted(docs.items(), key=lambda item: item[1].score, reverse=True)
        hits = [Hit(doc_id=doc_id, score=doc.score) for doc_id, doc in sorted_docs]
        by_topic.setdefault(tid, []).append((qid, hits))

    output = {}
    for tid, doc_questions in by_topic.items():
        doc_questions.sort(key=lambda question: question[1][0].score, reverse=True)
        output[tid] = DayResults(results=[
            QuestionResults(qid=qid, question_rank=rank, question_text=None, doc_ranking=hits)
            for rank, (qid, hits) in enumerate(doc_questions, start=1)
        ])
    return output


@register(STAGE,'all_candidates')
class AllCandidates:
    def run(self, questions: dict[QuestionKey,dict[str,DocScore]]) -> dict[str,DayResults]:
        return to_day_results(questions)

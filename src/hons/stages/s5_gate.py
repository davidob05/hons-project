from cdet_api.types import DayResults, Hit, QuestionResults

from hons.registry import register
from hons.types import Candidates

STAGE = 's5_gate'


def to_day_results(candidates: Candidates) -> dict[str,DayResults]:
    by_topic: dict[str, list[tuple[str, list[Hit]]]] = {}
    for (tid, qid), question_candidates in candidates.per_question.items():
        if not question_candidates:
            continue
        sorted_docs = sorted(question_candidates, key=lambda candidate: candidate.score.score*candidate.weight, reverse=True)
        hits = [Hit(doc_id=candidate.doc_id, score=candidate.score.score*candidate.weight) for candidate in sorted_docs]
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
    def run(self, candidates: Candidates) -> dict[str,DayResults]:
        return to_day_results(candidates)

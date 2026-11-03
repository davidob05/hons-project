from hons.registry import register
from hons.types import QuestionKey, DocScore, Fragments, Candidates, Candidate


STAGE = 's4_novelty'

@register(STAGE,'off')
class AllNovel:
    def run(self, candidates: Candidates,state, fragments: Fragments) -> Candidates:
        return candidates

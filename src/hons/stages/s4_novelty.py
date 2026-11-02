from hons.registry import register
from hons.types import QuestionKey, DocScore, Fragments


STAGE = 's4_novelty'

@register(STAGE,'off')
class NoNovelty:
    def run(self, tails: dict[QuestionKey,dict[str,DocScore]],state, fragments: Fragments) -> dict[QuestionKey,dict[str,DocScore]]:
        return tails

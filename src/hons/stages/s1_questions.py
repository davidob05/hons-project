from hons.types import DayBatch, QuestionKey
from hons.registry import register

STAGE = 's1_questions'

@register(STAGE, 'question')
class SeedQuestions:
    def run(self, batch: DayBatch, examples: dict[str,list[QuestionKey]]) -> list[str]:
        return [doc.id for doc in batch if doc.id in examples]

@register(STAGE, 'off')
class UnseededQuestions:
    def run(self, batch: DayBatch, examples: dict[str,list[QuestionKey]]) -> list[str]:
        return []

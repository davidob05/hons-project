from dataclasses import replace
from string import Formatter

from hons.types import DayBatch, QuestionKey, QuestionVersion
from hons.registry import register

STAGE = 's1_questions'
TEMPLATE_FIELDS = {'label', 'narrative', 'question'}


class PlainQuestions:
    def represent(self, questions: dict[QuestionKey, QuestionVersion]) -> dict[QuestionKey, QuestionVersion]:
        return questions


@register(STAGE, 'question')
class SeedQuestions(PlainQuestions):
    def run(self, batch: DayBatch, examples: dict[str,list[QuestionKey]]) -> list[str]:
        return [doc.id for doc in batch if doc.id in examples]

@register(STAGE, 'off')
class UnseededQuestions(PlainQuestions):
    def run(self, batch: DayBatch, examples: dict[str,list[QuestionKey]]) -> list[str]:
        return []


@register(STAGE, 'template')
class TemplateQuestions(UnseededQuestions):
    def __init__(self, template: str = '{label}: {question}'):
        used = {name for _, name, _, _ in Formatter().parse(template) if name}
        if not used <= TEMPLATE_FIELDS:
            raise ValueError(f'template may only use {sorted(TEMPLATE_FIELDS)}, got {sorted(used - TEMPLATE_FIELDS)}')
        if 'question' not in used:
            raise ValueError('template must contain {question}')
        self.template = template
        self.used = used

    def represent(self, questions: dict[QuestionKey, QuestionVersion]) -> dict[QuestionKey, QuestionVersion]:
        represented = {}
        for key, version in questions.items():
            fields = {'label': version.label.replace('-', ' ').replace('_', ' '),
                      'narrative': version.narrative, 'question': version.text}
            if all(fields[name] for name in self.used):
                represented[key] = replace(version, text=self.template.format(**fields), original=version.original or version.text)
            else:
                represented[key] = version
        return represented

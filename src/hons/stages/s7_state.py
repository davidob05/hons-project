from hons.registry import register
from hons.types import Candidates, QuestionVersion, State

STAGE = 's7_state'


@register(STAGE, 'off')
class Stateless:
    def run(self, state: State, candidates: Candidates, examples: list[str], proposed: list[QuestionVersion]) -> State:
        return State()

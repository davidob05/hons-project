from hons.registry import register
from cdet_api.types import DayResults
from hons.types import Candidates, Proposals

STAGE = 's6_propose'

@register(STAGE,'off')
class NoProposals:
    def run(self, results: dict[str,DayResults], candidates: Candidates) -> Proposals:
        return Proposals()

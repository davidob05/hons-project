from hons.registry import register
from hons.types import DayBatch, DocInfo, Fragment, Fragments

STAGE = 's0_fragment'


@register(STAGE, 'lines')
class LineFragmenter:
    # Splits each document into fragments, one per non-empty line
    def run(self, batch: DayBatch) -> Fragments:
        fragments = Fragments()
        for document in batch:
            lines = [line.strip() for line in document.text.split('\n') if line.strip()]
            fragments.fragments[document.id] = {i: Fragment(line) for i, line in enumerate(lines)}
            fragments.docs[document.id] = DocInfo()
        return fragments


@register(STAGE, 'off')
class WholeDocument:
    # Stage 0 off: one fragment per document (the whole body), no boilerplate flags, no groups
    def run(self, batch: DayBatch) -> Fragments:
        fragments = Fragments()
        for document in batch:
            fragments.fragments[document.id] = {0: Fragment(document.text)}
            fragments.docs[document.id] = DocInfo()
        return fragments

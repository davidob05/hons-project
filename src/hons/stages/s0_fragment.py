import re

from hons.registry import register
from hons.types import DayBatch, DocInfo, Fragment, Fragments

STAGE = 's0_fragment'
ASCII_TERM = re.compile(r'[A-Za-z0-9]')


@register(STAGE, 'lines')
class LineFragmenter:
    # Splits each document into fragments, one per non-empty line
    def run(self, batch: DayBatch) -> Fragments:
        fragments = Fragments()
        for document in batch:
            lines = [line.strip() for line in document.text.split('\n') if ASCII_TERM.search(line)]
            fragments.fragments[document.id] = {i: Fragment(line) for i, line in enumerate(lines)}
            fragments.docs[document.id] = DocInfo()
        return fragments


@register(STAGE, 'windows')
class WindowFragmenter:
    def __init__(self, size: int = 200, overlap: int = 50):
        if size <= 0:
            raise ValueError(f'size must be positive, got {size}')
        if not 0 <= overlap < size:
            raise ValueError(f'overlap must be in [0, size), got {overlap} with size {size}')
        self.size = size
        self.step = size - overlap

    def run(self, batch: DayBatch) -> Fragments:
        fragments = Fragments()
        for document in batch:
            words = [word for word in document.text.split() if ASCII_TERM.search(word)]
            windows = []
            for start in range(0, len(words), self.step):
                windows.append(' '.join(words[start:start + self.size]))
                if start + self.size >= len(words):
                    break
            fragments.fragments[document.id] = {i: Fragment(window) for i, window in enumerate(windows)}
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

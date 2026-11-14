import statistics
from collections import deque

from hons.registry import register
from hons.types import DocScore, Scores, QuestionKey, Candidates, Candidate

STAGE = 's3_tail'

@register(STAGE,'off')
class ReturnNone:
    def run(self,scores: Scores) -> Candidates:
        return Candidates()

@register(STAGE,'top_1')
class Top1:

    def run(self,scores: Scores) -> Candidates:
        candidates = Candidates()
        for qid, by_doc in scores.per_question.items():
            if not by_doc:
                continue
            best = max(by_doc, key=lambda doc_id: by_doc[doc_id].score)
            candidates.add_candidate(Candidate(best,by_doc[best]),qid)
        return candidates

@register(STAGE,'topk')
class Topk:
    def __init__(self, k: int = 20, min_score: float = 0.0):
        self.k = k
        self.min_score = min_score

    def run(self,scores: Scores) -> Candidates:
        candidates = Candidates()
        for key, by_doc in scores.per_question.items():
            ranked = sorted(by_doc, key=lambda doc_id: by_doc[doc_id].score, reverse=True)
            for doc_id in ranked[:self.k]:
                if by_doc[doc_id].score > self.min_score:
                    candidates.add_candidate(Candidate(doc_id, by_doc[doc_id]), key)
        return candidates


# Ref: Original design. No paper behind it: a per-question bar learned causally from a rolling window of that
# question's own daily top scores, silent until warmup days of history exist. Not listed in References by Pipeline
# Stage.md. Subclasses supply the rule that turns the history into a bar.
class HistoryGate:
    ref = 'Original design, no source'

    def __init__(self, k: int, window: int | None, warmup: int):
        if k < 1:
            raise ValueError(f'k must be at least 1, got {k}')
        if warmup < 1:
            raise ValueError(f'warmup must be at least 1, got {warmup}')
        if window is not None and window < warmup:
            raise ValueError(f'window ({window}) must be at least warmup ({warmup})')
        self.k = k
        self.window = window
        self.warmup = warmup
        self.history: dict[QuestionKey, deque[float]] = {}

    def threshold(self, ordered: list[float]) -> float:
        raise NotImplementedError

    def bar(self, key: QuestionKey) -> float | None:
        history = self.history.get(key)
        if history is None or len(history) < self.warmup:
            return None
        return self.threshold(sorted(history))

    def run(self, scores: Scores) -> Candidates:
        candidates = Candidates()
        for key, by_doc in scores.per_question.items():
            if not by_doc:
                continue
            bar = self.bar(key)
            if bar is not None:
                ranked = sorted(by_doc, key=lambda doc_id: by_doc[doc_id].score, reverse=True)
                for doc_id in ranked[:self.k]:
                    if by_doc[doc_id].score > bar:
                        candidates.add_candidate(Candidate(doc_id, by_doc[doc_id], excess=by_doc[doc_id].score - bar), key)
            self.history.setdefault(key, deque(maxlen=self.window)).append(max(doc.score for doc in by_doc.values()))
        return candidates


# Ref: Original design. The bar is a quantile q of the question's recent daily top scores (nearest rank). No
# paper behind it, and not listed in References by Pipeline Stage.md. It is the per-question quantile rung of the Stage 3
# ladder, which the briefing lists without a source.
@register(STAGE,'quantile')
class QuantileGate(HistoryGate):
    ref = 'Original design, no source'

    def __init__(self, k: int = 20, q: float = 0.95, window: int | None = 60, warmup: int = 30):
        if not 0 < q < 1:
            raise ValueError(f'q must be between 0 and 1, got {q}')
        super().__init__(k, window, warmup)
        self.q = q

    def threshold(self, ordered: list[float]) -> float:
        return ordered[min(int(self.q * len(ordered)), len(ordered) - 1)]


# Ref: Standard technique, no source read. The bar is median + c * 1.4826 * MAD of the question's recent daily top
# scores, a robust version of mean plus c standard deviations (1.4826 is the usual constant that makes the median
# absolute deviation comparable to a standard deviation for normal data). Applied here to the gating history
# described on HistoryGate. Not listed in References by Pipeline Stage.md; no citation given because none was read.
@register(STAGE,'mad')
class MadGate(HistoryGate):
    ref = 'Standard technique (robust z-score), no source read'

    def __init__(self, k: int = 20, c: float = 3.0, window: int | None = 60, warmup: int = 30):
        if c <= 0:
            raise ValueError(f'c must be positive, got {c}')
        super().__init__(k, window, warmup)
        self.c = c

    def threshold(self, ordered: list[float]) -> float:
        median = statistics.median(ordered)
        spread = statistics.median(abs(value - median) for value in ordered)
        return median + self.c * 1.4826 * spread

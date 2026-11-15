import math
import re
from collections import Counter, deque
from functools import lru_cache

from hons.registry import register
from hons.types import DocScore, Fragments, QuestionKey, QuestionVersion, Scores

STAGE = 's2_score'
TOKEN = re.compile(r'[a-z0-9]+')
STOPWORDS = frozenset(
    'a about above after again against all am an and any are as at be because been before being below between both but '
    'by can could did do does doing down during each few for from further had has have having he her here hers him his '
    'how i if in into is it its itself just me more most my no nor not now of off on once only or other our out over own '
    's same she should so some such t than that the their theirs them then there these they this those through to too '
    'under until up very was we were what when where which while who whom why will with would you your yours'.split())


@lru_cache(maxsize=200000)
def stem(word: str) -> str:
    if len(word) > 4 and word.endswith('ies') and not word.endswith(('eies', 'aies')):
        return word[:-3] + 'y'
    if len(word) > 3 and word.endswith('es') and not word.endswith(('aes', 'ees', 'oes')):
        return word[:-1]
    if len(word) > 3 and word.endswith('s') and not word.endswith(('us', 'ss')):
        return word[:-1]
    return word


def tokenise(text: str, use_stem: bool) -> list[str]:
    terms = [term for term in TOKEN.findall(text.lower()) if term not in STOPWORDS]
    return [stem(term) for term in terms] if use_stem else terms


# Ref: Standard technique, no source read (the BM25 ranking function). Accumulating document-frequency and length
# statistics over earlier days, instead of indexing each day alone, is the briefing's "per-day against accumulated
# word rarity" comparison and is original design. Defaults k1 1.2 and b 0.75 match Terrier's, which the toolkit
# baseline uses; Son et al. (CREAM, in References by Pipeline Stage.md) report k1 1.5 and b 0.75. The plural-stripping
# stemmer is a simple suffix rule, no source read. Not listed in the references document as an implementation.
@register(STAGE, 'bm25_acc')
class BM25Accumulated:
    ref = 'BM25 standard technique, accumulated statistics original design, no source read'

    def __init__(self, k1: float = 1.2, b: float = 0.75, days: int | None = None,
                 include_today: bool = True, stem: bool = True):
        if k1 <= 0:
            raise ValueError(f'k1 must be positive, got {k1}')
        if not 0 <= b <= 1:
            raise ValueError(f'b must be between 0 and 1, got {b}')
        if days is not None and days < 1:
            raise ValueError(f'days must be at least 1, got {days}')
        self.k1 = k1
        self.b = b
        self.days = days
        self.include_today = include_today
        self.stem = stem
        self.df: Counter = Counter()
        self.count = 0
        self.length = 0
        self.history: deque = deque()

    def observe(self, day_df: Counter, count: int, length: int) -> None:
        self.df.update(day_df)
        self.count += count
        self.length += length
        if self.days is None:
            return
        self.history.append((day_df, count, length))
        while len(self.history) > self.days:
            old_df, old_count, old_length = self.history.popleft()
            for term, n in old_df.items():
                remaining = self.df[term] - n
                if remaining > 0:
                    self.df[term] = remaining
                else:
                    del self.df[term]
            self.count -= old_count
            self.length -= old_length

    def run(self, fragments: Fragments, questions: dict[QuestionKey, QuestionVersion]) -> Scores:
        scores = Scores(per_question={key: {} for key in questions})

        keys: list[tuple[str, int]] = []
        lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        day_df: Counter = Counter()
        total_length = 0
        for doc_id, index, fragment in fragments:
            terms = tokenise(fragment.text, self.stem)
            position = len(keys)
            keys.append((doc_id, index))
            lengths.append(len(terms))
            total_length += len(terms)
            for term, tf in Counter(terms).items():
                postings.setdefault(term, []).append((position, tf))
                day_df[term] += 1

        if self.include_today:
            self.observe(day_df, len(keys), total_length)
        if keys and questions and self.count and self.length:
            self.score(scores, questions, keys, lengths, postings)
        if not self.include_today:
            self.observe(day_df, len(keys), total_length)
        return scores

    def score(self, scores: Scores, questions: dict[QuestionKey, QuestionVersion], keys: list[tuple[str, int]],
              lengths: list[int], postings: dict[str, list[tuple[int, int]]]) -> None:
        average = self.length / self.count
        for key, version in questions.items():
            per_fragment: dict[int, float] = {}
            for term in set(tokenise(version.text, self.stem)):
                entries = postings.get(term)
                if not entries:
                    continue
                n = self.df.get(term, 0)
                idf = math.log(1 + (self.count - n + 0.5) / (n + 0.5))
                for position, tf in entries:
                    norm = self.k1 * (1 - self.b + self.b * lengths[position] / average)
                    per_fragment[position] = per_fragment.get(position, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
            per_doc = scores.per_question[key]
            for position, score in per_fragment.items():
                doc_id, fragment_index = keys[position]
                best = per_doc.get(doc_id)
                if best is None or score > best.score:
                    per_doc[doc_id] = DocScore(score=score, best_fragment=fragment_index)

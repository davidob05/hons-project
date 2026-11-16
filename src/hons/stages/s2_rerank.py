import math
import os
import pickle
from hashlib import sha1
from pathlib import Path

from hons.registry import REGISTRY, register
from hons.stages.s2_scores import DenseEncoder
from hons.types import DocScore, Fragments, QuestionKey, QuestionVersion, Scores

STAGE = 's2_score'


# Ref: Standard technique, no source read (reranking a first-stage candidate list with a cross-encoder that reads the
# question and a passage together). The briefing's Stage 2 "Reranker" option describes it: a cross-encoder or language
# model reranks each question's top candidates and its score feeds Stage 3. The nearest paper in References by Pipeline
# Stage.md is Nogueira, Jiang and Lin 2020 (monoT5, arXiv:2003.06713, status partial), which is a different model. The
# default model is an MS MARCO cross-encoder from sentence-transformers; its release date has not been verified, so
# results from it carry the "may be contaminated" label. Not listed in the references document as an implementation.
@register(STAGE, 'rerank')
class CrossEncoderRerank:
    ref = 'Cross-encoder reranking, standard technique, model release date unverified'

    def __init__(self, first: str = 'bm25_day', first_params: dict | None = None, candidates: int = 30,
                 model: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2', device: str | None = None,
                 passage: str = 'lead', words: int = 100, batch_size: int = 64, probability: bool = False,
                 cache_dir: str | None = 'cache/rerank_scores'):
        if candidates < 1:
            raise ValueError(f'candidates must be at least 1, got {candidates}')
        if passage not in ('lead', 'best'):
            raise ValueError(f'passage must be "lead" or "best", got {passage!r}')
        if words < 1:
            raise ValueError(f'words must be at least 1, got {words}')
        first_stages = REGISTRY.get(STAGE, {})
        if first not in first_stages or first == 'rerank':
            known = ', '.join(sorted(name for name in first_stages if name != 'rerank'))
            raise ValueError(f'first must name another Stage 2 implementation (known: {known}), got {first!r}')
        from sentence_transformers import CrossEncoder

        self.first = first_stages[first](**(first_params or {}))
        self.candidates = candidates
        self.model_name = model
        self.passage = passage
        self.words = words
        self.batch_size = batch_size
        self.probability = probability
        self.model = CrossEncoder(model, device=device)
        self.cache_dir = Path(cache_dir) / model.replace('/', '__') if cache_dir else None

    def load_cache(self, digest: str) -> dict[str, float]:
        if self.cache_dir is None:
            return {}
        try:
            with (self.cache_dir / f'{digest}.pkl').open('rb') as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            return {}

    def save_cache(self, digest: str, cached: dict[str, float]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f'{digest}.pkl'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump(cached, f)
        os.replace(tmp, path)

    def text_of(self, fragments: Fragments, doc_id: str, best_fragment: int) -> str:
        by_index = fragments.fragments[doc_id]
        if self.passage == 'best':
            return ' '.join(by_index[best_fragment].text.split()[:self.words])
        words: list[str] = []
        for index in sorted(by_index):
            words.extend(by_index[index].text.split())
            if len(words) >= self.words:
                break
        return ' '.join(words[:self.words])

    def run(self, fragments: Fragments, questions: dict[QuestionKey, QuestionVersion]) -> Scores:
        first = self.first.run(fragments, questions)
        scores = Scores(per_question={key: {} for key in questions})

        wanted: dict[QuestionKey, list[tuple[str, int, str]]] = {}
        for key, version in questions.items():
            ranked = sorted(first.per_question[key], key=lambda doc_id: first.per_question[key][doc_id].score, reverse=True)
            wanted[key] = []
            for doc_id in ranked[:self.candidates]:
                best = first.per_question[key][doc_id].best_fragment
                text = self.text_of(fragments, doc_id, best)
                if text:
                    wanted[key].append((doc_id, best, text))
        if not any(wanted.values()):
            return scores

        keys_list = [(doc_id, index) for doc_id, index, _ in fragments]
        texts_list = [fragment.text for _, _, fragment in fragments]
        digest = DenseEncoder.fragments_digest(keys_list, texts_list)
        cached = self.load_cache(digest)

        def pair_key(question: str, text: str) -> str:
            return sha1(f'{question}\x00{text}'.encode()).hexdigest()

        missing: dict[str, tuple[str, str]] = {}
        for key, entries in wanted.items():
            for _, _, text in entries:
                pk = pair_key(questions[key].text, text)
                if pk not in cached and pk not in missing:
                    missing[pk] = (questions[key].text, text)
        if missing:
            order = list(missing)
            predicted = self.model.predict([missing[pk] for pk in order], batch_size=self.batch_size, show_progress_bar=False)
            for pk, score in zip(order, predicted):
                cached[pk] = float(score)
            self.save_cache(digest, cached)

        for key, entries in wanted.items():
            per_doc = scores.per_question[key]
            for doc_id, best, text in entries:
                raw = cached[pair_key(questions[key].text, text)]
                per_doc[doc_id] = DocScore(score=1 / (1 + math.exp(-raw)) if self.probability else raw, best_fragment=best)
        return scores

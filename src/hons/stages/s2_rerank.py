import math
import os
import pickle
import uuid
from hashlib import sha1
from pathlib import Path
from string import Formatter

from hons.registry import REGISTRY, register
from hons.stages.s2_scores import DenseEncoder
from hons.types import DocScore, Fragments, QuestionKey, QuestionVersion, Scores

STAGE = 's2_score'


class MonoT5:
    def __init__(self, name: str, device: str | None):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = T5ForConditionalGeneration.from_pretrained(name).to(self.device).eval()
        self.false_id, self.true_id = self.tokenizer.convert_tokens_to_ids(['▁false', '▁true'])

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32, show_progress_bar: bool = False) -> list[float]:
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                part = pairs[start:start + batch_size]
                encoded = self.tokenizer([f'Query: {query} Document: {passage} Relevant:' for query, passage in part],
                                         padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
                start_ids = self.torch.full((len(part), 1), self.model.config.decoder_start_token_id, dtype=self.torch.long, device=self.device)
                logits = self.model(**encoded, decoder_input_ids=start_ids).logits[:, 0, :]
                two = logits[:, [self.false_id, self.true_id]]
                scores.extend(self.torch.log_softmax(two, dim=1)[:, 1].tolist())
        return scores


def load_scorer(name: str, device: str | None):
    if 'monot5' in name.lower():
        return MonoT5(name, device)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=device)


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
                 cache_dir: str | None = 'cache/rerank_scores', template: str | None = None,
                 blend: str = 'none', blend_k: int = 60, blend_weight: float = 1.0):
        if candidates < 1:
            raise ValueError(f'candidates must be at least 1, got {candidates}')
        if blend not in ('none', 'rrf', 'sum'):
            raise ValueError(f'blend must be "none", "rrf" or "sum", got {blend!r}')
        if blend_k <= 0:
            raise ValueError(f'blend_k must be positive, got {blend_k}')
        if template is not None:
            used = {name for _, name, _, _ in Formatter().parse(template) if name}
            if not used <= {'label', 'narrative', 'question'} or 'question' not in used:
                raise ValueError(f'template must use {{question}} and may use only label, narrative and question, got {template!r}')
        if passage not in ('lead', 'best'):
            raise ValueError(f'passage must be "lead" or "best", got {passage!r}')
        if words < 1:
            raise ValueError(f'words must be at least 1, got {words}')
        first_stages = REGISTRY.get(STAGE, {})
        if first not in first_stages or first == 'rerank':
            known = ', '.join(sorted(name for name in first_stages if name != 'rerank'))
            raise ValueError(f'first must name another Stage 2 implementation (known: {known}), got {first!r}')
        self.first = first_stages[first](**(first_params or {}))
        self.candidates = candidates
        self.model_name = model
        self.passage = passage
        self.words = words
        self.batch_size = batch_size
        self.probability = probability
        self.template = template
        self.blend = blend
        self.blend_k = blend_k
        self.blend_weight = blend_weight
        self.model = load_scorer(model, device)
        self.is_monot5 = isinstance(self.model, MonoT5)
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
        tmp = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        with tmp.open('wb') as f:
            pickle.dump(cached, f)
        os.replace(tmp, path)

    def query_of(self, version: QuestionVersion) -> str:
        if self.template is None:
            return version.text
        return self.template.format(label=version.label.replace('-', ' ').replace('_', ' '),
                                    narrative=version.narrative, question=version.text)

    def blended(self, entries: list[tuple[str, int, str]], raw: dict[str, float], first: dict[str, DocScore]) -> dict[str, float]:
        if self.blend == 'sum':
            return {doc_id: raw[doc_id] + self.blend_weight * first[doc_id].score for doc_id, _, _ in entries}
        if self.blend == 'rrf':
            first_rank = {doc_id: rank for rank, (doc_id, _, _) in enumerate(entries, start=1)}
            ordered = sorted(raw, key=lambda doc_id: raw[doc_id], reverse=True)
            model_rank = {doc_id: rank for rank, doc_id in enumerate(ordered, start=1)}
            return {doc_id: 1 / (self.blend_k + first_rank[doc_id]) + 1 / (self.blend_k + model_rank[doc_id]) for doc_id in raw}
        if not self.probability:
            return dict(raw)
        if self.is_monot5:
            return {doc_id: math.exp(value) for doc_id, value in raw.items()}
        return {doc_id: 1 / (1 + math.exp(-value)) for doc_id, value in raw.items()}

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
        queries = {key: self.query_of(version) for key, version in questions.items()}

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
                pk = pair_key(queries[key], text)
                if pk not in cached and pk not in missing:
                    missing[pk] = (queries[key], text)
        if missing:
            order = list(missing)
            predicted = self.model.predict([missing[pk] for pk in order], batch_size=self.batch_size, show_progress_bar=False)
            for pk, score in zip(order, predicted):
                cached[pk] = float(score)
            self.save_cache(digest, cached)

        for key, entries in wanted.items():
            raw = {doc_id: cached[pair_key(queries[key], text)] for doc_id, _, text in entries}
            final = self.blended(entries, raw, first.per_question[key])
            per_doc = scores.per_question[key]
            for doc_id, best, _ in entries:
                per_doc[doc_id] = DocScore(score=final[doc_id], best_fragment=best)
        return scores

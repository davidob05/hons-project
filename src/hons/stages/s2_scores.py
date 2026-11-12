import hashlib
import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyterrier as pt

from hons.registry import register
from hons.types import DocScore, Fragments, QuestionKey, QuestionVersion, Scores

STAGE = 's2_score'


# Copied from cdet_api/examples/example_client.py (the toolkit's BM25 baseline), unchanged
def build_index(docs, index_path):
    indexer = pt.IterDictIndexer(index_path, meta={'docno': 50}, text_attrs=['text'], overwrite=True)
    indexref = indexer.index(docs)
    return pt.IndexFactory.of(indexref)


@contextmanager
def self_cleaning_index(docs):
    # Use tempfile to guarantee a unique path for every loop iteration
    # This prevents Terrier's internal cache from confusing new indices with old ones
    tmp_dir = tempfile.mkdtemp()
    index = None

    try:
        index = build_index(docs, tmp_dir)
        yield index
    finally:
        # 1. Close the actual Java object holding the file descriptors
        if index is not None:
            try:
                index.close()
            except Exception as e:
                print(f"Error closing index: {e}")

        # 2. Safely delete the physical files from disk
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@register(STAGE,'bm25_day')
class BM25Day:
    # The toolkit baseline's scoring: index only today's fragments, rank them by BM25 for each
    # question. A document's score is its best fragment. No cut-off here, that is Stage 3
    def run(self, fragments: Fragments, questions: dict[QuestionKey, QuestionVersion]) -> Scores:
        scores = Scores(per_question={key: {} for key in questions})

        # Terrier gets a short docno (the position in `keys`), as doc id plus index can exceed the 50 character limit
        keys: list[tuple[str, int]] = []
        docs = []
        for doc_id, index, fragment in fragments:
            docs.append({'docno': str(len(keys)), 'text': fragment.text})
            keys.append((doc_id, index))
        if not docs or not questions:
            return scores

        # qids repeat across topics, so each question is sent as its position in `asked`
        asked = list(questions)
        queries = pd.DataFrame({
            'qid': [str(i) for i in range(len(asked))],
            'query': [questions[key].text for key in asked],
        })

        with self_cleaning_index(docs) as index:
            # Every fragment is ranked, not the toolkit's top 20: the tail is what Stage 3 sets its bar from
            results = pt.terrier.Retriever(index, wmodel='BM25', num_results=len(docs))(queries)

        # Fragments sharing no term with the question are not returned, so they are absent here (score 0)
        for row in results.itertuples():
            doc_id, fragment_index = keys[int(row.docno)]
            per_doc = scores.per_question[asked[int(row.qid)]]
            best = per_doc.get(doc_id)
            if best is None or row.score > best.score:
                per_doc[doc_id] = DocScore(score=float(row.score), best_fragment=fragment_index)
        return scores

@register(STAGE,'dense')
class DenseEncoder:
    # Alternative to the baseline BM25, a dense encoder uses a neural encoder to turn the question and each fragment 
    # into vectors, a fragment's score would then be the cosine similarity between this vector and the question's vector
    def __init__(self, model: str = 'sentence-transformers/msmarco-distilbert-base-v3',
                 device: str | None = None, batch_size: int = 64,
                 cache_dir: str | None = 'cache/dense_scores'):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.batch_size = batch_size
        self.model = SentenceTransformer(model, device=device)
        self.cache_dir = Path(cache_dir) / model.replace('/', '__') if cache_dir else None

    @staticmethod
    def fragments_digest(keys: list[tuple[str, int]], texts: list[str]) -> str:
        digest = hashlib.sha256()
        for (doc_id, index), text in zip(keys, texts):
            digest.update(f'{doc_id}\x00{index}\x00{text}\x00'.encode())
        return digest.hexdigest()[:32]

    def load_cache(self, digest: str) -> dict[str, dict[str, tuple[float, int]]]:
        if self.cache_dir is None:
            return {}
        try:
            with (self.cache_dir / f'{digest}.pkl').open('rb') as f:
                return pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            return {}

    def save_cache(self, digest: str, cached: dict[str, dict[str, tuple[float, int]]]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f'{digest}.pkl'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump(cached, f)
        os.replace(tmp, path)

    def run(self, fragments: Fragments, questions: dict[QuestionKey, QuestionVersion]) -> Scores:
        scores = Scores(per_question={key: {} for key in questions})

        keys: list[tuple[str, int]] = []
        texts: list[str] = []
        for doc_id, index, fragment in fragments:
            texts.append(fragment.text)
            keys.append((doc_id, index))
        if not texts or not questions:
            return scores

        digest = self.fragments_digest(keys, texts)
        cached = self.load_cache(digest)
        missing = list(dict.fromkeys(q.text for q in questions.values() if q.text not in cached))

        if missing:
            fragment_vectors = self.model.encode(
                texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)
            question_vectors = self.model.encode(
                missing, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)
            similarity = question_vectors @ fragment_vectors.T

            starts = [0]
            for i in range(1, len(keys)):
                if keys[i][0] != keys[i - 1][0]:
                    starts.append(i)
            starts.append(len(keys))

            for q, text in enumerate(missing):
                row = similarity[q]
                per_doc = {}
                for lo, hi in zip(starts, starts[1:]):
                    best = lo + int(row[lo:hi].argmax())
                    per_doc[keys[best][0]] = (float(row[best]), keys[best][1])
                cached[text] = per_doc
            self.save_cache(digest, cached)

        for key, version in questions.items():
            scores.per_question[key] = {
                doc_id: DocScore(score=score, best_fragment=index)
                for doc_id, (score, index) in cached[version.text].items()
            }
        return scores

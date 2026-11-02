import os
import shutil
import tempfile
from contextlib import contextmanager

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


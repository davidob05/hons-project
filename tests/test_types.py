import dataclasses

import pytest
from cdet_api.types import DocumentSchema

from hons.types import Candidate, DayBatch, Fragment, Fragments, QuestionKey, QuestionVersion


def doc(doc_id, day='2021-08-01'):
    return DocumentSchema(id=doc_id, text='t', url='u', date=day, day=day)


def test_batch_is_frozen():
    batch = DayBatch('2021-08-01', 0, [doc('a')])
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.day = '2021-08-02'


def test_batch_documents_stored_as_tuple():
    docs = [doc('a')]
    batch = DayBatch('2021-08-01', 0, docs)
    docs.append(doc('b'))
    assert len(batch.documents) == 1
    assert isinstance(batch.documents, tuple)


def test_equal_batches_compare_equal():
    assert DayBatch('2021-08-01', 0, [doc('a')]) == DayBatch('2021-08-01', 0, [doc('a')])


def test_negative_sequence_number_rejected():
    with pytest.raises(ValueError):
        DayBatch('2021-08-01', -1, [])


@pytest.mark.parametrize('bad', ['20210801', '2021-8-1', '2021-13-01', '2021-02-30', ''])
def test_bad_day_rejected(bad):
    with pytest.raises(ValueError):
        DayBatch(bad, 0, [])


def test_document_from_wrong_day_rejected():
    with pytest.raises(ValueError):
        DayBatch('2021-08-01', 0, [doc('a', '2021-08-02')])


def test_question_key_distinguishes_topics_and_hashes():
    assert QuestionKey('t1', 'q_1') != QuestionKey('t2', 'q_1')
    assert {QuestionKey('t1', 'q_1'): 1}[QuestionKey('t1', 'q_1')] == 1


def test_question_version_key():
    assert QuestionVersion(qid='q_1', tid='t1', text='x').key == QuestionKey('t1', 'q_1')


def test_fragments_iterate_in_doc_then_index_order():
    frags = Fragments(fragments={'a': {0: Fragment('x'), 1: Fragment('y')}, 'b': {0: Fragment('z')}})
    assert [(d, i, f.text) for d, i, f in frags] == [('a', 0, 'x'), ('a', 1, 'y'), ('b', 0, 'z')]


def test_candidate_weight_changed_by_replacing_not_mutating():
    original = Candidate('a', 1.0)
    changed = dataclasses.replace(original, weight=0.2)
    assert original.weight == 1.0
    assert changed.weight == 0.2

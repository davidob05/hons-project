from hons.types import Fragments, Fragment, DocInfo, DayBatch

def fragment_batch(batch: DayBatch) -> Fragments:
    fragments = Fragments()
    for document in batch:
        lines = [line.strip() for line in document.text.split('\n') if line.strip()]
        fragments.fragments[document.id] = {i: Fragment(line) for i, line in enumerate(lines)}
        fragments.docs[document.id] = DocInfo()
    return fragments

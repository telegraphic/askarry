from rag.ingestion import find_acronym_candidates


class _FakeCollection:
    def __init__(self, docs_and_sources):
        self._docs_and_sources = docs_and_sources

    def get(self, include=None, where=None):
        documents = [text for text, _ in self._docs_and_sources]
        metadatas = [{"source": source} for _, source in self._docs_and_sources]
        return {"documents": documents, "metadatas": metadatas}


def test_bare_mention_added_for_unambiguous_acronym():
    collection = _FakeCollection([
        ("The Square Kilometre Array (SKA) is a radio telescope.", "paper1.pdf"),
        ("SKA will detect many pulsars.", "paper2.pdf"),
    ])
    candidates = find_acronym_candidates(collection)
    assert "SKA" in candidates
    assert set(candidates["SKA"]["sources"]) == {"paper1.pdf", "paper2.pdf"}


def test_bare_mention_not_added_for_ambiguous_acronym():
    collection = _FakeCollection([
        ("The Dispersion Measure (DM) affects pulsar timing.", "paper1.pdf"),
        ("Dark Matter (DM) dominates the mass budget.", "paper2.pdf"),
        ("DM is discussed further below.", "paper3.pdf"),
    ])
    candidates = find_acronym_candidates(collection)
    dm_keys = [k for k in candidates if k.startswith("DM")]
    assert len(dm_keys) == 2
    all_sources = {s for k in dm_keys for s in candidates[k]["sources"]}
    assert "paper3.pdf" not in all_sources

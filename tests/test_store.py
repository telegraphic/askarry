from rag.store import chunk_id


def test_chunk_id_deterministic():
    assert chunk_id("foo.pdf", 3) == chunk_id("foo.pdf", 3)


def test_chunk_id_uniqueness_across_index():
    ids = {chunk_id("foo.pdf", i) for i in range(5)}
    assert len(ids) == 5


def test_chunk_id_uniqueness_across_source():
    assert chunk_id("foo.pdf", 0) != chunk_id("bar.pdf", 0)

from rag.retrieval import _rrf_fuse, _sigmoid


def test_sigmoid_midpoint():
    assert _sigmoid(0.0) == 0.5


def test_sigmoid_bounds():
    assert _sigmoid(100.0) > 0.999
    assert _sigmoid(-100.0) < 0.001


def test_rrf_fuse_orders_by_fused_rank_and_dedups():
    semantic_hits = [
        {"source": "a.pdf", "chunk_index": 0, "text": "ta"},
        {"source": "b.pdf", "chunk_index": 1, "text": "tb"},
    ]
    bm25_hits = [
        {"source": "b.pdf", "chunk_index": 1, "text": "tb"},
        {"source": "c.pdf", "chunk_index": 2, "text": "tc"},
    ]

    fused = _rrf_fuse(semantic_hits, bm25_hits, bm25_weight=0.5)

    # "b.pdf" is ranked in both legs so it should win despite being 2nd in
    # the semantic leg; results are deduplicated (3, not 4, entries).
    assert [c["source"] for c in fused] == ["b.pdf", "a.pdf", "c.pdf"]
    assert all("score" in c for c in fused)


def test_rrf_fuse_empty_inputs():
    assert _rrf_fuse([], [], bm25_weight=0.5) == []

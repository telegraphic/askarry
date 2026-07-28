from rag.citations import build_uncited_note, linkify_citations


def test_linkify_valid_citation():
    chunks = [{"source": "a"}, {"source": "b"}]
    answer = "This is a fact [1]."

    linked, cited = linkify_citations(answer, chunks)

    assert cited == {1}
    assert '<a href="#source-1" class="citation-link">[1]</a>' in linked


def test_linkify_invalid_citation():
    chunks = [{"source": "a"}]
    answer = "This is a fact [5]."

    linked, cited = linkify_citations(answer, chunks)

    assert cited == set()
    assert "citation-invalid" in linked


def test_linkify_no_chunks_returns_answer_unchanged():
    answer = "Some text [1]."

    linked, cited = linkify_citations(answer, [])

    assert linked == answer
    assert cited == set()


def test_build_uncited_note_all_cited():
    chunks = [{"source": "a"}, {"source": "b"}]

    assert build_uncited_note(chunks, {1, 2}) == ""


def test_build_uncited_note_some_uncited():
    chunks = [{"source": "a"}, {"source": "b"}, {"source": "c"}]

    note = build_uncited_note(chunks, {1})

    assert "[2]" in note
    assert "[3]" in note
    assert "[1]" not in note

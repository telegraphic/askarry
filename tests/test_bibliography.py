import rag.bibliography as bibliography
from rag.bibliography import format_citation, lookup_citation


def test_format_citation_single_author():
    entry = {"authors": ["Jane Smith"], "title": "A Title"}
    assert format_citation(entry) == "Smith (2026) A Title"


def test_format_citation_multiple_authors():
    entry = {"authors": ["Jane Smith", "John Doe"], "title": "A Title"}
    assert format_citation(entry) == "Smith et al (2026) A Title"


def test_format_citation_no_authors():
    entry = {"authors": [], "title": "A Title"}
    assert format_citation(entry) == "Unknown (2026) A Title"


def test_lookup_citation_found(monkeypatch):
    fake_bibliography = {
        "Smith01": {"title": "Foo", "authors": ["Jane Smith"], "section": "The Cosmos"}
    }
    monkeypatch.setattr(bibliography, "load_bibliography", lambda: fake_bibliography)

    result = lookup_citation("/some/path/Smith01.pdf")

    assert result == fake_bibliography["Smith01"]


def test_lookup_citation_missing(monkeypatch):
    monkeypatch.setattr(bibliography, "load_bibliography", lambda: {})

    assert lookup_citation("/some/path/Unknown01.pdf") is None

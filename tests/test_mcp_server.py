"""Server-level behaviour: the index fingerprint on every response, and
get_document_chunks for files without a bibliography entry."""

import json

import mcp_server
from rag import zones


def test_index_meta_added_to_json_object_text_and_left_off_json_lists():
    meta = {"fingerprint": "abc", "written_at": "2026-11-09T00:00:00+00:00"}
    assert json.loads(mcp_server._stamp('{"a": 1}', meta)) == {"a": 1, "_index": meta}
    assert mcp_server._stamp("[1, 2]", meta) == "[1, 2]"
    assert mcp_server._stamp("plain text", meta).endswith("[index abc, built 2026-11-09T00:00:00+00:00]")


def test_index_meta_reads_manifest(tmp_path):
    assert mcp_server._index_meta(tmp_path)["fingerprint"] is None
    zones.write_manifest(tmp_path, {"index_fingerprint": "f" * 64, "written_at": "now", "n_chunks": 3})
    assert mcp_server._index_meta(tmp_path) == {"fingerprint": "f" * 16, "written_at": "now", "n_chunks": 3}


class _FakeCollection:
    def __init__(self, source):
        self.source = source

    def count(self):
        return 1

    def get(self, where=None, include=None):
        return {"metadatas": [{"source": self.source, "chunk_index": 0, "page_no": 1}],
                "documents": ["Copy of a chapter."]}


def test_get_document_chunks_for_file_without_bibliography_entry(monkeypatch):
    # Used to raise NameError: the fallback title used an undefined `stem`.
    source = "/x/pdfs/AASKAII/Unclassified/Curiel01b.pdf"
    monkeypatch.setattr(mcp_server.store, "get_chroma_collection", lambda *a: _FakeCollection(source))
    monkeypatch.setattr(mcp_server, "lookup_citation", lambda s: None)
    out = mcp_server.get_document_chunks("Curiel01b")
    assert out.startswith("# Curiel01b") and "Copy of a chapter." in out

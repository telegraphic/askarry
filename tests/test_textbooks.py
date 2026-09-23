from rag import store
from rag.config import TEXTBOOKS_DIR, textbook_title


def test_collections_are_per_directory(tmp_path):
    a = store.get_chroma_collection(tmp_path / "a")
    b = store.get_chroma_collection(tmp_path / "b")
    a.add(ids=["x"], documents=["only in a"], embeddings=[[0.0, 1.0]])
    assert a.id != b.id
    assert (a.count(), b.count()) == (1, 0)
    assert store.get_chroma_collection(tmp_path / "a") is a


def test_textbook_title_folder_and_single_pdf():
    assert textbook_title(TEXTBOOKS_DIR / "Essential Radio Astronomy" / "j.ctv5vdcww.10.pdf") == "Essential Radio Astronomy"
    assert textbook_title(TEXTBOOKS_DIR / "Pulsar Astronomy .pdf") == "Pulsar Astronomy"

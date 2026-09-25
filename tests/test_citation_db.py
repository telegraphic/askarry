import rag.citation_db as cdb
from rag.config import PDF_DIR


def _chunk(source, idx, text, book="aaska2015", heading="References"):
    return {"text": text, "meta": {"source": source, "chunk_index": idx,
                                   "doc_source": book, "heading": heading}}


# Two papers, both styles. Paper A's Anderson entry straddles a chunk boundary.
CHUNKS = [
    _chunk(f"{PDF_DIR}/AASKA2015/Cosmology/Alpha01.pdf", 7,
           "Condon, J. J., et al. 1998, AJ, 115, 1693\n"
           "Anderson, L., Aubourg, E., Bailey, S. et al"),
    _chunk(f"{PDF_DIR}/AASKA2015/Cosmology/Alpha01.pdf", 8,
           "., 2013, MNRAS, 427, 3435.\n"
           "Bull, P., Ferreira, P. G., 2015, ApJ, 803, 21 [arXiv:1405.1452]"),
    _chunk(f"{PDF_DIR}/AASKA2015/Cosmology/Alpha01.pdf", 1, "Condon 1998 is great.", heading="Introduction"),
    _chunk(f"{PDF_DIR}/AASKAII/The Cosmos/Beta01.pdf", 30,
           "- J. J. Condon et al. AJ , 115(5):1693-1716, May 1998. doi: 10.1086/ 300337.\n"
           "- P. Bull et al. arXiv e-prints , art. arXiv:1405.1452, 2014. doi: 10.48550/arXiv. 1405.1452.\n"
           "- T. An et al. In Advancing Astrophysics with the SKA - II (AASKAII) . 2026. "
           "arXiv search: Report number AASKAII/TaoAn02.", book="aaskaii"),
    _chunk(f"{PDF_DIR}/SKA_Key_Capabilities/Doc.pdf", 3, "Condon, J. J., et al. 1998, AJ, 115, 1693",
           book="ska_capabilities"),
]


def test_parse_ref_both_styles_share_a_key():
    a = cdb.parse_ref("Condon, J. J., et al. 1998, AJ, 115, 1693")
    b = cdb.parse_ref("J. J. Condon et al. AJ , 115(5):1693-1716, May 1998. doi: 10.1086/ 300337.")
    assert a["ref_key"] == b["ref_key"] == "condon|1998|115|1693"
    assert b["doi"] == "10.1086/300337"
    assert cdb.parse_ref("de Blok, W. J. G., 2008, AJ, 136, 2648")["first_author"] == "blok"


def test_parse_ref_ids():
    arx = cdb.parse_ref("P. Bull et al. arXiv e-prints , art. arXiv:1405.1452, 2014. doi: 10.48550/arXiv. 1405.1452.")
    assert arx["ref_key"] == "arxiv:1405.1452" and arx["doi"] == "10.48550/arxiv.1405.1452"
    assert cdb.parse_ref("Schwarz, D. et al. 2015, 'Testing', PoS(AASKA14)032")["ref_key"] == "aaska14:32"
    # A trailing table glued onto the last entry must not leak into the DOI.
    wrapped = cdb.parse_ref("A. B. Smith. MNRAS , 1(1):2, 2020. doi: 10.1111/j.1365-2966. 2009.16188.x. Table 1: Summary")
    assert wrapped["doi"] == "10.1111/j.1365-2966.2009.16188.x"
    prd = cdb.parse_ref("P. Adshead et al. Phys. Rev. D , 88(2):021302, 2013. doi: 10.1103/ PhysRevD.88.021302.")
    assert prd["doi"] == "10.1103/physrevd.88.021302"


def test_split_entries_heals_chunk_boundary_and_run_ons():
    text = ("Anderson, L., Aubourg, E. et al\n., 2013, MNRAS, 427, 3435.\n"
            "Sachs, R. K. 1967, ApJ, 147, 73 Schwarz, D. et al. 2015, PoS(AASKA14)032")
    assert cdb.split_entries(text) == [
        "Anderson, L., Aubourg, E. et al ., 2013, MNRAS, 427, 3435.",
        "Sachs, R. K. 1967, ApJ, 147, 73",
        "Schwarz, D. et al. 2015, PoS(AASKA14)032",
    ]


def test_record_citations_and_ranking(monkeypatch):
    monkeypatch.setattr(cdb, "section_of", lambda source: "Cosmology")
    conn = cdb.connect(":memory:")
    cdb.record_citations(conn, CHUNKS)
    assert cdb.stats(conn) == {"refs": 4, "cites": 6, "papers_with_refs": 2}

    top = cdb.top_cited(conn, min_papers=2)
    # Bull: journal ref with arXiv id (A) + arXiv-only (B) merged via shared arXiv id.
    assert {r["ref_key"]: r["papers"] for r in top} == {
        "condon|1998|115|1693": ["AASKA2015/Alpha01", "Beta01"],
        "arxiv:1405.1452": ["AASKA2015/Alpha01", "Beta01"],
    }
    assert {r["first_author"] for r in top} == {"condon", "bull"}
    assert cdb.top_cited(conn, book="aaskaii", since_year=2020)[0]["ref_key"] == "aaskaii:taoan02"
    assert [r["n_papers"] for r in cdb.citing_papers(conn, "Anderson")] == [1]

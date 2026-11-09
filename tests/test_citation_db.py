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
    assert a["ref_key"] == b["ref_key"] == "bib:1998AJ....115.1693C"  # the ADS bibcode, built offline
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
    assert {k: v for k, v in cdb.stats(conn).items() if k in ("refs", "cites", "papers_with_refs")} == {
        "refs": 4, "cites": 6, "papers_with_refs": 2}

    top = cdb.top_cited(conn, min_papers=2)
    # Bull: journal ref with arXiv id (A) + arXiv-only (B) merged via the shared arXiv id.
    assert {r["ref_key"]: r["papers"] for r in top} == {
        "bib:1998AJ....115.1693C": ["AASKA2015/Alpha01", "Beta01"],
        "bib:2015ApJ...803...21B": ["AASKA2015/Alpha01", "Beta01"],  # bibcode key preferred over arXiv
    }
    assert {r["first_author"] for r in top} == {"condon", "bull"}
    assert cdb.top_cited(conn, book="aaskaii", since_year=2020)[0]["ref_key"] == "aaskaii:taoan02"
    assert [r["n_papers"] for r in cdb.citing_papers(conn, "Anderson")] == [1]


# --- Journals, bibcodes and journal-like first authors ---------------------

def test_given_first_single_author_surname_is_not_the_journal():
    # Used to give first_author "araa": the journal followed the author's full stop.
    r = cdb.parse_ref("S. P. Reynolds. ARA&A , 46:89-126, Sept. 2008. doi: 10.1146/annurev.astro.46.060407.145237.")
    assert r["first_author"] == "reynolds" and r["journal"] == "ARA&A" and r["first_author_suspect"] == 0
    two = cdb.parse_ref("J. Bland-Hawthorn and O. Gerhard. Annu. Rev. Astron. Astrophys. , 54:529-596, 2016.")
    assert two["first_author"] == "blandhawthorn" and two["journal"] == "ARA&A"


def test_bibcode_built_offline_and_shared_by_both_styles():
    a = cdb.parse_ref("de Blok, W. J. G., 2008, AJ, 136, 2648")
    b = cdb.parse_ref("W. J. G. de Blok et al. AJ , 136(6):2648-2719, Dec. 2008. doi: 10.1088/0004-6256/136/6/2648.")
    assert a["bibcode"] == b["bibcode"] == "2008AJ....136.2648D" and a["ref_key"] == b["ref_key"]
    assert cdb.parse_ref("Y. Sun et al. Astrophys. J. Lett. , 977(2):L35, Dec. 2024.")["bibcode"] == "2024ApJ...977L..35S"
    assert cdb.parse_ref("G. De Lucia et al. A&A, 687:A68, July 2024.")["bibcode"] == "2024A&A...687A..68D"
    # Five-digit pages don't fit the format: no bibcode, a surname key instead.
    assert cdb.parse_ref("P. Adshead et al. Phys. Rev. D , 88(2):021302, 2013.")["bibcode"] is None


def test_refs_sharing_a_doi_or_bibcode_are_grouped(monkeypatch):
    monkeypatch.setattr(cdb, "section_of", lambda source: "S")
    conn = cdb.connect(":memory:")
    chunks = [_chunk(f"{PDF_DIR}/AASKA2015/Cosmology/Old01.pdf", 1, "Condon, J. J., et al. 1998, AJ, 115, 1693"),
              _chunk(f"{PDF_DIR}/AASKAII/The Cosmos/New01.pdf", 1,
                     "- J. J. Condon et al. AJ , 115(5):1693-1716, May 1998. doi: 10.1086/300337.", book="aaskaii"),
              # Same DOI, page misread: grouped by DOI after parsing.
              _chunk(f"{PDF_DIR}/AASKAII/The Cosmos/New02.pdf", 1,
                     "- J. J. Condon et al. Astron. J. , 115:1693, 1998. doi: 10.1086/300337.", book="aaskaii")]
    cdb.record_citations(conn, chunks)
    (top,) = cdb.top_cited(conn)
    assert top["n_papers"] == 3 and top["bibcode"] == "1998AJ....115.1693C" and top["doi"] in (None, "10.1086/300337")


def test_journal_like_first_author_flagged():
    r = cdb.parse_ref("MNRAS, 2019, 489, 3739")
    assert r["first_author"] == "mnras" and r["first_author_suspect"] == 1


def test_connect_upgrades_a_database_built_before_bibcodes(tmp_path):
    import sqlite3
    old = sqlite3.connect(tmp_path / "c.db")
    old.executescript("CREATE TABLE refs (ref_key TEXT PRIMARY KEY, raw TEXT NOT NULL, first_author TEXT,"
                      " year INTEGER, volume TEXT, page TEXT, doi TEXT, arxiv TEXT);")
    old.close()
    conn = cdb.connect(tmp_path / "c.db")
    assert {"journal", "bibcode", "first_author_suspect"} <= {r[1] for r in conn.execute("PRAGMA table_info(refs)")}

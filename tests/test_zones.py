"""Zone tagging on real chunks (tests/fixtures/chunks/zones.json, copied from
the index; e-mail addresses removed)."""

import json
from pathlib import Path

from rag import zones

FIXTURE = json.loads((Path(__file__).parent / "fixtures/chunks/zones.json").read_text())


def _zones(stem: str) -> dict[int, str]:
    chunks = FIXTURE[stem]
    return {c["meta"]["chunk_index"]: z for c, z in zip(chunks, zones.assign_zones(chunks))}


def test_title_authors_and_abstract_before_first_section():
    z = _zones("Maccagni01")
    assert z[0] == "front_matter"  # title heading, author list
    assert z[1] == "abstract"      # prose under the title heading
    assert z[2] == "body"          # "1 Introduction"


def test_references_heading_and_continuation_are_references():
    z = _zones("Maccagni01")
    assert {z[64], z[65], z[66]} == {"references"}


def test_affiliation_block_is_front_matter():
    z = _zones("Nasirudin01")
    assert z[1] == "front_matter"  # affiliation list, no prose
    assert z[2] == "body"


def test_author_list_ordering_acknowledgements_then_references():
    z = _zones("Nasirudin01")
    assert z[45] == "front_matter"       # "Author List Ordering"
    assert z[46] == "acknowledgements"
    assert {z[47], z[48]} == {"references"}


def test_reference_list_under_another_heading_is_detected_by_text():
    # Hota01 prints its reference list under "12 Acknowledgement".
    z = _zones("Hota01")
    assert z[54] == "acknowledgements"
    assert {z[55], z[56], z[57]} == {"references"}
    assert z[52] == "body"


def test_unheaded_reference_fragment_heading_stays_in_references():
    chunks = [
        {"text": "Body text about pulsars and their timing. " * 10,
         "meta": {"chunk_index": 0, "heading": "1 Introduction"}},
        {"text": "- A. B. Smith et al. ApJ , 887(1):69, Dec. 2019. doi: 10.3847/1. " * 6,
         "meta": {"chunk_index": 1, "heading": "References"}},
        # Docling sometimes takes a DOI tail for a heading.
        {"text": "- C. D. Jones. MNRAS , 451(3):2517-2523, Aug. 2015. doi: 10.1093/x. " * 6,
         "meta": {"chunk_index": 2, "heading": "0004-6361/202348444."}},
        {"text": "Supplementary derivation of the beam model.",
         "meta": {"chunk_index": 3, "heading": "Appendix A: Beam model"}},
    ]
    assert zones.assign_zones(chunks) == ["body", "references", "references", "appendix"]


def test_zones_are_deterministic_and_order_independent():
    chunks = FIXTURE["Maccagni01"]
    forward = zones.assign_zones(chunks)
    backward = zones.assign_zones(chunks[::-1])[::-1]
    assert forward == backward

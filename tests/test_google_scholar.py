"""Tests for rag.google_scholar (parsing logic only - no live network calls)."""

from rag.google_scholar import _parse_results

SAMPLE_HTML = """
<div class="gs_ri">
  <h3 class="gs_rt"><a href="https://example.com/paper">A Paper Title</a></h3>
  <div class="gs_a">J Doe - Some Journal, 2020</div>
  <div class="gs_rs">An abstract snippet.</div>
</div>
"""


def test_parse_results_extracts_fields():
    results = _parse_results(SAMPLE_HTML, num_results=5)

    assert results == [
        {
            "title": "A Paper Title",
            "authors": "J Doe - Some Journal, 2020",
            "abstract": "An abstract snippet.",
            "url": "https://example.com/paper",
        }
    ]


def test_parse_results_respects_num_results():
    html = SAMPLE_HTML * 3
    assert len(_parse_results(html, num_results=2)) == 2

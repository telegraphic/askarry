# Tool changes and the AASKAII Atlas

A critique of AASKAII Atlas v2 found that several headline figures rest on
how the MCP tools behaved rather than on the book. The fixes went into the
tools, so every later analysis benefits. This note says, for each change,
what `scripts/build_atlas_data.py` (and the agent workflow) should do
differently when the atlas is next rebuilt, and which atlas figures will
move. The atlas itself has not been rebuilt.

The regression cases are run by `scripts/regression_cases.py`; the
before/after table is at the end.

## WS1: zones, duplicates and the index fingerprint

**What changed.** Every chunk is tagged with the part of the paper it comes
from (front matter, abstract, body, acknowledgements, references,
appendix). Re-downloaded copies and book documents without a bibliography
entry are marked and left out of corpus counts by default. Each index has a
manifest with a fingerprint, and every tool response reports it.

**What `build_atlas_data.py` should change.**

- `book_chapters()`, `_only_chapters()` and `_restrict_term()` existed to
  drop copies and non-chapters after the fact. `keyword_coverage` and
  `facility_network` now do this themselves, so the restriction can go;
  keep it for one rebuild as a cross-check, which should change nothing.
- Record the index fingerprint (`chroma_db/manifest.json`,
  `index_fingerprint`) in each page's `generated` header next to `git_sha`,
  so a rebuilt figure can be traced to the index it came from.
- Check `index_health_tool` before a rebuild: duplicates, unlisted
  documents and documents with no reference list should all be expected.

**Figures that move.** Only figures the atlas computed without its own
chapter restriction: the counts were already restricted to chapters, so
duplicates mostly affected the tools' raw output, not the pages.

## WS2: keyword_coverage and facility_network

**What changed.** Counts cover the abstract and body only, match plurals
and hyphen/space variants, and drop other meanings of facility names
(Max-Planck, "Chandra et al.", the Planck constant, Planck Collaboration
citations). Each response states the zones, matching mode and exclusions.

**What `build_atlas_data.py` should change.**

- `_coverage()` can call `keyword_coverage` with its defaults; keep
  `min_hits_per_paper=2` if the atlas still wants to drop passing mentions.
  With zones in place, that threshold is no longer needed to drop
  affiliations ("SETI Institute"), which was its stated reason.
- Term lists written to work around literal matching
  (`"zoom mode|zoom-mode|zoom window"`) can be simplified
  (`"zoom mode|zoom window"`), but they still work unchanged.
- `facilities()` should pass nothing new; its counts will drop for Planck,
  Chandra and Fermi.
- Use `return_contexts=2` when an agent needs to check how a paper uses a
  term, instead of a separate search.

**Figures that move.** Every keyword-coverage figure: observing modes,
methods, risks and the book diff (`modes.json`, `methods.json`,
`risks.json`, `books_diff.json`), and the synergy network
(`facilities.json`). Expect falls for terms that also appear in reference
lists and affiliations (ApJ, Planck, Chandra, and any term in journal
titles), and rises for terms usually written in the plural (pulsar surveys,
data rates, zoom modes). The table at the end has the regression cases.

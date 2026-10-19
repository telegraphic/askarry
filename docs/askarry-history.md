# How Askarry Got Built: An Informal History

*Pieced together from the git log, five plan files, sixteen Claude Code session logs and the project memory. Coverage is uneven. July and August appear only as commits, while September onwards has the full plans and session transcripts.*

---

## Act I: one very long day in July

Askarry began on **28 July 2026**, and it began fast. The first day has eleven commits. The initial commit already had the whole pipeline in place: Docling to parse the PDFs, a hybrid chunker, BGE-large embeddings in ChromaDB, BM25 running alongside, a cross-encoder re-ranker on top, and a Gradio app with a local Ollama model (Gemma) writing the answers. The goal was modest: a *learning prototype* that could answer questions from the AASKAII book, which is freely available.

That same afternoon brought a bibliography and table of contents, so answers cited "Smith et al." rather than `smith01.pdf`. It also brought an "Answer level" toggle with two settings, *PhD astronomer* and *Non-expert*, a fixed progress bar, and the first real refactor. That refactor split `store.py` out and added the first test. Anyone who has built a prototype will recognise the day: build it, use it, find it annoying, then fix the annoying parts.

Over the next two days came styling and the first acronym work. The ingester learned to collect the acronyms each chapter defines, so a search for "RFI" also finds "radio frequency interference". In an SKA corpus that is less of a nicety than it sounds.

## Act II: Claude moves in (August)

Early August was spent tidying the README and recording a demo screengrab. Then, on **10 August**, came the change that set the direction for everything after it: **an MCP server**. It had three tools. Retrieval still ran locally, but generation could now come from Claude Desktop instead of a small local model. From that point the project was less a chatbot and more a **toolbox for an AI agent**.

On **22 August** the corpus grew in two directions. First came the **SKA Key Capabilities** documents, the engineering side, kept as a separate searchable set. Second came the 2015 predecessor, *Advancing Astrophysics with the SKA*. Its passages carry a 0.85 score penalty, so ten-year-old material wins only when it really is the better match. The same day added the first *live* tool, a wrapper around the SKAO sensitivity calculator's REST API.

### The first atlas (23–24 August)

With six tools available, the first **AASKAII Atlas** appeared. It was a static HTML site produced by sixteen agent runs. Seven section agents were fanned out, one per theme, and later resumed by name with follow-up questions. The atlas looked impressive, and its own "Behind the Atlas" page was admirably honest about its faults:

- It **invented seven science themes** instead of using the book's real sections.
- Its "observing-mode contention" was really a list of how often each mode was mentioned.
- The synergy network was **rebuilt from another chat's summary** because the original session could not be reached.
- FPGA was categorised as an *Organization*.
- The sensitivity page contained a **citation swap**, which is a polite name for a hallucination.

That list of faults turned out to be the specification for most of the next two months.

## Act III: the validator knows better than the documents (September, week 1)

On **5 September** the session logs begin. You asked for two other codebases to be brought into the MCP server: the **SKA observing-setup validator** and the **ODP data-size tool**. Both were cut down and copied into `rag/vendor/`.

The more important moment came with your next question: *can these tools tell me how many subarrays I can use on SKA-Low at AA2?* The documents say "up to 16". The validator schema says **one** for AA2, four for AA\*, and sixteen only at the full AA4 design. You put it plainly: *"The information in the setup validator is more up to date and correct than capability documentation."*

That became the server's **MOST IMPORTANT RULE**, and it still sits in capitals at the top of the MCP instructions. Static documents describe the ambition. The live schema describes what you can actually configure. `describe_schema_tool` and its companions arrived that day, together with a demo observing-proposal page for the atlas.

A day later the **Google Scholar** MCP server was folded in. The 162 MB upstream clone was put in `.gitignore`, and its copy-pasted result parser was reduced to a single helper.

## Act IV: textbooks (23 September)

Next came the background reading. Thompson's *Interferometry and Synthesis*, *Tools of Radio Astronomy*, two pulsar books and *Essential Radio Astronomy* (as twenty separate JSTOR chapter PDFs, naturally) went into a **separate Chroma database**. The aim was to keep textbook fundamentals from mixing with SKA-specific numbers.

Making this work exposed a real bug in waiting. The BM25 and acronym caches were global and keyed only by document count, so two collections could have served each other's indexes. The plan's rule was firm: textbooks supply derivations and never override SKA figures. Later that day, Balanis on antennas and Pozar on microwave engineering were added, bringing the store to about 6,950 chunks.

## Act V: 25 September, the day everything happened

If one day defines Askarry, it is **25 September**. At least six Claude sessions ran that day, some in parallel (times below are UTC; Perth is eight hours ahead). Much of the log reads like air-traffic control.

**06:22.** It started with a reflective question: *if we redid the atlas, what tooling would help?* The answer became the "Askarry tooling gaps" plan: filtered search, exact `keyword_coverage`, `verify_quote`, a sensitivity-calculator comparison, and validation across observing contexts. These tools were aimed squarely at the v1 atlas's faults.

**Meanwhile, in a worktree,** another session was fixing the bibliography. Fifteen filename stems existed in both books, and fifty Milky Way papers had been filed under *Stars*. That session sensibly decided not to move the files, because file paths are chunk IDs. Nine new papers were ingested, which left eleven chapters still missing from skao.int (Kavanagh01's link simply returns a 404).

**08:14.** You asked what real scheduling contention would need: LST coverage, night versus day, avoiding sunrise and sunset, and elevations above 45°. You also asked for SIMBAD lookup. Instead of copying a third-party astroquery MCP into the repo, Claude wrote a small wrapper over astroquery. A **source database** followed: 371 SIMBAD objects from 1,576 mentions, led by the Galactic Centre (31 papers), the LMC, M87 and Centaurus A, with "Cen A", "Centaurus A" and "NGC 5128" merged into one object.

**09:01.** A parallel session proposed a **citation database**. It politely held off because the other session was editing the same three files, then started once you said go. It parses both books' citation styles straight from the stored chunks with no network calls, and it rebuilds in about ten seconds.

**Mid-morning.** The scheduling work landed: `visibility_windows`, `lst_pressure`, a curated **facility network** of about 80 facilities, and a survey-field list. The facility network includes regex guards so that "Planck's constant" is not the satellite and the "Virgo cluster" is not the detector. Every scheduling default is labelled *"unverified default — needs SKAO owner"* and echoed back in each result, which is a good habit.

**10:12. Which model should do the reading?** Claude's answer was Sonnet at medium effort for extraction, then a single Opus pass at high effort for review. The bigger saving came from filtering first. A plain pattern match cut 3.4 million tokens of book down to about 0.6 million worth reading.

**The pilots.** Sonnet and Opus were run on the same ten papers, and they agreed on seven of eight requests. Both, however, missed the big ones, such as the 1,000-hour deep fields and multi-thousand-hour surveys. Claude traced this to its own prompt rules rather than to either model. Pilot 2, with fixed rules plus survey-field recording, recovered every miss without adding false positives.

**11:19. The full run.** All 219 papers were split across section subagents, three at a time. The run **hit the usage limit partway through**, resumed after the reset, and produced 152 requests and 87 survey fields from 80 papers. The total came to about 272,000 hours on SKA-Mid, most of it "unplaced", because large surveys have no single pointing.

**13:25. The review.** Opus went through all 152 requests and accepted 106 (148,065 h), marked 32 as duplicates and rejected 14. It also rescaled per-year figures to a five-year horizon, so a pulsar timing array's 231 h per year became 1,157 h. Each review decision sits *beside* the extracted row and never overwrites it.

**Running gag:** "I've restarted the server" appears in the logs again and again. Each new tool meant restarting the MCP server in Claude Desktop, and at one point the restart had to include the app as well.

**15:01.** *"Make a plan to regenerate the AASKAII atlas."* The decisions were these:

- Build a fresh `aaskaii-atlas-v2/` and leave v1 untouched.
- Compute every figure in a committed script (`build_atlas_data.py`).
- Have agents write only the prose.
- Check every quote mechanically.

The prose workflow had seven section agents and two verifiers. It **hit the five-hour limit** with sections D and E still to run, so Claude stopped it rather than let the remaining agents fail. It resumed at 20:11. By late evening the atlas had **18 pages**, 456 instrument requirements and 226 open questions, and **682 of 682 quotes** were confirmed verbatim in the chapters they cite. That check is exactly the one that would have caught v1's citation swap.

The commit went in at 23:01. You also set a standing writing rule that day: write as a senior research assistant in astronomy, in cohesive paragraphs, in classic style, with no AI clichés.

## Act VI: maintenance and polish (October)

**1 October.** The virtual environment broke with a macOS `dlopen` error deep inside SciPy. The fix was to move the project to **uv** on Python 3.11, with two pins learned the hard way: `numpy<2.3`, because astropy 6.1 needs the removed `np.in1d`, and `bibtexparser<2`, because `scholarly` still imports a module that 2.x dropped.

**2 October.** A readability and visualisation pass gave the atlas an "All parts" menu, previous and next links, computed "Key finding" boxes, tooltips and a counts/percentage toggle. You then caught Claude slipping into what you called **"AI half sentences"**, fragment captions such as *"Three pages that answer the scheduling and feasibility questions most readers bring to the book."* Every page was rewritten in full sentences following the *Oxford Guide to Plain English*, and that rule is now in memory. Later the same day: big Chapters and Acronyms buttons, "Start here" renamed to "Highlights", and a category filter on the acronym list.

The same morning, two side quests ran:

- **A literature search** for comparable machine-read analyses of a facility's science book. Nothing quite like it turned up. The ngVLA Science Book's breakdown into Key Science Goals was the closest relative.
- **An abstract** titled *"Connecting the AASKAII corpus and SKA science tools with AI assistants"*, written to sound like you.

**4 October.** Part XVIII, **Modes & Data Products**, sorted the 106 reviewed requests into Continuum, Zoom, PSS and PST, then sized their output products with the validator's `data_products.py`. The answer is about **811 PB**, 92% of it calibrated visibilities. At 100 Gbit/s, sending all of it from SDP to SRCNet would take about 811 days. The correlator dump times appear nowhere in the schemas, so they were taken from Dewdney et al. (2022).

**6 October.** Askarry was finally used for its original purpose: answering a science question. *How many radio-bright AGN will SKA find above z > 5?* The answer drew on the AASKAII chapters and came to roughly 10⁴ at 5 < z < 6, with a few thousand beyond z = 6.

## Where things stand (13 October)

- **49 MCP tools**, up from 3 in August and 6 when the first atlas was built.
- Three document stores: the AASKAII and 2015 books, the capabilities documents and the textbooks. Three SQLite databases: sources, citations, and observing requests and fields.
- Vendored live checks: the setup validator, the ODP sizing tool and the sensitivity calculator.
- Atlas v2 is committed, and the October polish and the Modes & Data Products page are still uncommitted on `textbook-index`, along with `v2.1` and `v2.2` zip snapshots.

**Open loose ends** still noted in memory:

- the stray `Bianchi01.pdf`, which is really a copy of Bianchi02;
- five duplicate re-downloads in `Unclassified/`;
- eleven chapters with no PDF on skao.int;
- scheduling defaults that still need an SKAO owner.

---

### The pattern, if there is one

The project kept repeating one loop. Something was built quickly, it was used for real, and its failures were written down honestly. Then tools were built to fix those failures specifically. The v1 atlas's list of known issues became the specification for September. *"Don't trust the documents, trust the validator"* became the server's first rule. Hand-tallied numbers gave way to a build script, and agent-written quotes gave way to quotes checked word for word. The prototype has turned into something close to an instrument.

// AASKAII Atlas v2 — shared page registry, nav and render helpers. Data comes
// from data/*.js (built by scripts/build_atlas_data.py) as window.ATLAS[name].
const PAGES = [
  // [file, part, title, group, one-line description]
  ["chapter-directory.html", "I", "Chapter Directory", "Reference", "Every AASKAII chapter by book section, with PDF links."],
  ["observing-modes.html", "II", "Observing-Mode Map", "Analyses", "How often each of the 15 observing modes is named, section by section."],
  ["cross-cutting-methods.html", "III", "Cross-Cutting Methods", "Analyses", "The 16 techniques, ranked by how many book sections use them."],
  ["systematic-risk-register.html", "IV", "Systematic Risk Register", "Analyses", "Engineering and calibration risks, counted by section."],
  ["open-questions.html", "V", "Open Questions", "Analyses", "What the papers say remains unknown, with verified quotes."],
  ["instrument-requirements.html", "VI", "Instrument Requirements", "Analyses", "Stated sensitivity, resolution, band and time requirements, quote by quote."],
  ["sensitivity-verification.html", "VII", "Sensitivity Verification", "Verification", "Stated sensitivities rerun through the SKAO sensitivity calculator."],
  ["capability-crosscheck.html", "VIII", "Capability Cross-Check", "Verification", "Requirements checked against the AA* and AA4 observing contexts."],
  ["synergy-network.html", "IX", "Synergy Network", "Analyses", "Which external facilities the papers name, and which together."],
  ["time-demand.html", "X", "Time Demand & LST Pressure", "Analyses", "Requested hours, survey tiers and LST pressure per telescope."],
  ["citation-landscape.html", "XI", "Citation Landscape", "Analyses", "The works the book cites most, overall and per section."],
  ["target-atlas.html", "XII", "Target & Field Atlas", "Analyses", "Named sources and survey fields, placed on the sky."],
  ["science-case-diff.html", "XIII", "2015 → 2026 Science Case", "Verification", "The 2015 book's sections set against the 2026 book's."],
  ["acronym-list.html", "XIV", "Acronyms", "Reference", "Every acronym defined in the book, with its expansion and use."],
  ["behind-the-atlas.html", "XV", "Behind the Atlas", "Method", "How v2 was built, and what changed from v1."],
  ["inside-askarry.html", "XVI", "Inside Askarry", "Method", "The retrieval system and tools the atlas runs on."],
  ["observing-proposal.html", "XVII", "Observing Proposal", "Use case", "A worked 100-hour HI proposal, with its figures rechecked."],
];
const SECTIONS = ["Science Working Group Overviews", "Sun, Earth and Planets", "Formation and Evolution of Stars",
  "From the Milky Way to Distant Galaxies", "The Cosmos", "The Extreme Universe", "Methods and Techniques"];
const SECTION_SHORT = {"Science Working Group Overviews": "SWG Overviews", "Sun, Earth and Planets": "Sun & Planets",
  "Formation and Evolution of Stars": "Stars", "From the Milky Way to Distant Galaxies": "Milky Way → Galaxies",
  "The Cosmos": "Cosmos", "The Extreme Universe": "Extreme", "Methods and Techniques": "Methods"};

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
// Every paper ID links to its row in the chapter directory and to the online PDF
// (same URL pattern as corpus_tools.AASKAII_PDF_URL and the v1 directory).
const PDF_BASE = "https://www.skao.int/sites/default/files/documents/";
const pdfUrl = id => PDF_BASE + encodeURIComponent(id) + ".pdf";
const AASKAII_PAGE = "https://www.skao.int/en/science-users/aaskaii";
let _titles = null;
function titleOf(id) {
  if (!_titles) {
    _titles = {};
    (window.ATLAS?.chapters?.sections || []).forEach(s => s.papers.forEach(p => _titles[p.paper] = p.title));
  }
  return _titles[id] || null;
}
function paper(id) {
  if (!id) return "—";
  return el("span", {class: "paper"},
    el("a", {href: `chapter-directory.html#${encodeURIComponent(id)}`, title: titleOf(id) || "Open in the chapter directory"}, id),
    el("a", {class: "pdf", href: pdfUrl(id), target: "_blank", rel: "noopener", title: `${id} PDF on skao.int`}, "PDF"));
}
// A list of paper IDs, the first `max` shown and the rest behind a toggle.
function papers(ids, max = 6) {
  const list = [...new Set(ids || [])];
  const wrap = el("span", {class: "papers"});
  const show = n => wrap.replaceChildren(...list.slice(0, n).flatMap((id, i) => [i ? ", " : null, paper(id)]).filter(x => x != null),
    ...(list.length > n ? [" ", Object.assign(el("button", {class: "more", type: "button"}, `+${list.length - n} more`), {onclick: () => show(list.length)})] : []));
  show(max);
  return list.length ? wrap : "—";
}
const fmt = (n, d = 0) => n == null ? "—" : Number(n).toLocaleString("en-GB", {maximumFractionDigits: d, minimumFractionDigits: d});

// Nav bar and generation footer on every page.
function frame(dataNames = []) {
  const here = location.pathname.split("/").pop() || "index.html";
  const links = [["index.html", "Home"], ["chapter-directory.html", "Chapters"], ["index.html#analyses", "Analyses"],
    ["time-demand.html", "Time demand"], ["acronym-list.html", "Acronyms"], ["behind-the-atlas.html", "Method"]];
  document.body.prepend(el("nav", {class: "atlas-nav"}, el("div", {class: "bar"},
    el("a", {class: "brand", href: "index.html"}, "AASKAII Atlas v2"),
    el("div", {class: "links"}, links.map(([h, t]) => el("a", {href: h, class: h === here ? "current" : null}, t))))));
  const gens = dataNames.map(n => window.ATLAS?.[n]?.generated).filter(Boolean);
  if (gens.length) {
    const g = gens[0];
    document.body.append(el("footer", {class: "gen", html:
      `Figures on this page come from <code>${dataNames.map(n => `data/${n}.json`).join("</code>, <code>")}</code>, ` +
      `built by <code>scripts/build_atlas_data.py</code> at commit <code>${g.git_sha}</code> on ${g.date}. ` +
      `Rerun the script to regenerate them.`}));
  }
}

// Sortable, optionally searchable table. cols: [{key, label, num, render(row)}].
function table(host, cols, rows, {search = false, placeholder = "Filter…", sortKey = null, desc = true, rowId = null} = {}) {
  let sort = sortKey, down = desc, q = "";
  const tbody = el("tbody"), count = el("span", {class: "count"});
  const heads = cols.map(c => el("th", {class: "sortable" + (c.num ? " r" : "")}, c.label));
  heads.forEach((h, i) => h.onclick = () => { const k = cols[i].key; down = sort === k ? !down : !!cols[i].num; sort = k; draw(); });
  const text = r => cols.map(c => typeof r[c.key] === "object" ? JSON.stringify(r[c.key]) : r[c.key] ?? "").join(" ").toLowerCase();
  function draw() {
    let rs = rows.filter(r => !q || text(r).includes(q));
    if (sort) rs = [...rs].sort((a, b) => {
      const x = a[sort], y = b[sort];
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""));
      return down ? -c : c;
    });
    tbody.replaceChildren(...rs.map(r => el("tr", {id: rowId ? rowId(r) : null}, cols.map(c => {
      const v = c.render ? c.render(r) : r[c.key];
      return el("td", {class: c.num ? "r" : (c.cls || null)}, v instanceof Node ? v : (c.num && typeof v === "number" ? fmt(v, c.dp || 0) : v ?? "—"));
    }))));
    count.textContent = `${rs.length} of ${rows.length}`;
  }
  if (search) host.append(el("div", {class: "controls"},
    Object.assign(el("input", {type: "text", placeholder}), {oninput: e => { q = e.target.value.toLowerCase(); draw(); }}), count));
  host.append(el("div", {class: "table-wrap"}, el("table", {}, el("thead", {}, el("tr", {}, heads)), tbody)));
  draw();
}

// Horizontal bar list: items [{label, value, note}].
function bars(host, items, {max = null, unit = "", dp = 0, color = null} = {}) {
  const m = max ?? Math.max(...items.map(i => i.value), 1);
  host.append(el("div", {class: "bars"}, items.map(i => el("div", {class: "bar-row"},
    el("span", {}, i.label),
    el("div", {class: "track"}, el("div", {class: "fill", style: `width:${(100 * i.value / m).toFixed(1)}%;${i.color || color ? `background:${i.color || color}` : ""}`})),
    el("span", {class: "v"}, `${fmt(i.value, dp)}${unit}${i.note ? ` · ${i.note}` : ""}`)))));
}

// Heat matrix of rows × the 7 book sections; value(row, section) -> count.
function heatSections(host, rows, label, value, {denominator = null} = {}) {
  const vals = rows.flatMap(r => SECTIONS.map(s => value(r, s)));
  const max = Math.max(...vals, 1);
  const cls = v => v === 0 ? "h0" : `h${Math.min(4, 1 + Math.floor(3.999 * v / max))}`;
  host.append(el("div", {class: "table-wrap"}, el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, label), SECTIONS.map(s => el("th", {class: "r", title: s}, SECTION_SHORT[s])), el("th", {class: "r"}, "Sections"))),
    el("tbody", {}, rows.map(r => el("tr", {}, el("td", {}, r.label),
      SECTIONS.map(s => { const v = value(r, s); return el("td", {class: `heat ${cls(v)}`, title: denominator ? `${v} of ${denominator[s]} papers` : null}, v || "·"); }),
      el("td", {class: "r"}, SECTIONS.filter(s => value(r, s) > 0).length)))))));
}

// One collapsible entry per item listing its chapters, section by section.
// papersIn(item, section) -> [paper ids].
function sectionPapers(host, items, papersIn) {
  host.append(el("div", {class: "plist"}, items.map(it => {
    const secs = SECTIONS.filter(s => (papersIn(it, s) || []).length);
    const n = secs.reduce((a, s) => a + papersIn(it, s).length, 0);
    return el("details", {}, el("summary", {}, el("b", {}, it.label), el("span", {class: "muted"}, ` · ${n} chapters`)),
      secs.length ? el("dl", {}, secs.flatMap(s => [el("dt", {}, s), el("dd", {}, papers(papersIn(it, s), 50))])) : el("p", {class: "muted"}, "No chapter in the seven sections."));
  })));
}

// Index cards from the registry.
function cards(host, group) {
  host.append(el("div", {class: "cards"}, PAGES.filter(p => !group || p[3] === group).map(([f, part, title, g, dek]) =>
    el("a", {class: "card", href: f}, el("span", {class: "part"}, `PART ${part}`), el("h3", {}, title), el("p", {}, dek), el("span", {class: "tag"}, g)))));
}

function masthead(file) {
  const p = PAGES.find(p => p[0] === file);
  return p ? `SKAO · AASKAII Atlas v2 · Part ${p[1]}` : "SKAO · AASKAII Atlas v2";
}

// AASKAII Atlas v2 — shared page registry, nav and render helpers. Data comes
// from data/*.js (built by scripts/build_atlas_data.py) as window.ATLAS[name].
const PAGES = [
  // [file, part, title, group, one-line description]
  ["chapter-directory.html", "I", "Chapter Directory", "Reference", "This page lists every AASKAII chapter under its book section and links to its PDF."],
  ["observing-modes.html", "II", "Observing-Mode Map", "Analyses", "This page counts how many chapters in each section name each of the 15 observing modes."],
  ["cross-cutting-methods.html", "III", "Cross-Cutting Methods", "Analyses", "This page ranks 16 techniques by the number of book sections that use them."],
  ["systematic-risk-register.html", "IV", "Systematic Risk Register", "Analyses", "This page counts the chapters in each section that discuss each engineering or calibration risk."],
  ["open-questions.html", "V", "Open Questions", "Analyses", "This page collects what the chapters say is still unknown, quoting each statement."],
  ["instrument-requirements.html", "VI", "Instrument Requirements", "Analyses", "This page lists the sensitivity, resolution, frequency and time that chapters ask for, with the quote behind each."],
  ["sensitivity-verification.html", "VII", "Sensitivity Verification", "Verification", "This page reruns the stated sensitivities through the SKAO sensitivity calculator."],
  ["capability-crosscheck.html", "VIII", "Capability Cross-Check", "Verification", "This page checks each requirement against the AA* and AA4 observing contexts."],
  ["synergy-network.html", "IX", "Synergy Network", "Analyses", "This page shows which external facilities the chapters name and which they name together."],
  ["time-demand.html", "X", "Time Demand & LST Pressure", "Analyses", "This page adds up the requested hours and shows how they load each telescope across sidereal time."],
  ["citation-landscape.html", "XI", "Citation Landscape", "Analyses", "This page shows which works the book cites most, overall and in each section."],
  ["target-atlas.html", "XII", "Target & Field Atlas", "Analyses", "This page places the named objects and survey fields on the sky."],
  ["science-case-diff.html", "XIII", "2015 → 2026 Science Case", "Verification", "This page traces where the topics of the 2015 book appear in the 2026 book."],
  ["acronym-list.html", "XIV", "Acronyms", "Reference", "This page lists every acronym the book defines, with its expansion and how often it is used."],
  ["behind-the-atlas.html", "XV", "Behind the Atlas", "Method", "This page explains how version 2 was built and what it changed from version 1."],
  ["inside-askarry.html", "XVI", "Inside Askarry", "Method", "This page describes the retrieval system and the tools that the atlas runs on."],
  ["observing-proposal.html", "XVII", "Observing Proposal", "Analyses", "This page shows a worked 100-hour HI proposal and rechecks its figures."],
  ["observing-mode-data.html", "XVIII", "Modes & Data Products", "Analyses", "This page sorts the requested observations into the four SKAO modes and estimates the data each one produces."],
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
// "a, b and c"
const list = xs => xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
const fmt = (n, d = 0) => n == null ? "—" : Number(n).toLocaleString("en-GB", {maximumFractionDigits: d, minimumFractionDigits: d});

// Nav bar, on-page contents, prev/next and generation footer on every page.
function frame(dataNames = []) {
  const here = location.pathname.split("/").pop() || "index.html";
  const i = PAGES.findIndex(p => p[0] === here);
  const menu = el("details", {class: "parts"}, el("summary", {}, "All parts"),
    el("div", {class: "menu"}, [["Analyses", ["Analyses"]], ["Background & Methods", ["Verification", "Method"]], ["Reference", ["Reference"]]].map(([g, groups]) => el("div", {},
      el("span", {class: "eyebrow"}, g),
      PAGES.filter(p => groups.includes(p[3])).map(([f, part, title]) =>
        el("a", {href: f, class: f === here ? "current" : null}, el("span", {class: "num"}, part), title))))));
  addEventListener("click", e => { if (!menu.contains(e.target)) menu.open = false; });
  addEventListener("keydown", e => { if (e.key === "Escape") menu.open = false; });
  document.body.prepend(el("nav", {class: "atlas-nav"}, el("div", {class: "bar"},
    el("a", {class: "brand", href: "index.html"}, "AASKAII Atlas v2"),
    el("div", {class: "links"}, [["index.html", "Home"], ["chapter-directory.html", "Chapters"]].map(([h, t]) =>
      el("a", {href: h, class: h === here ? "current" : null}, t)), menu))));
  // Contents strip for longer pages: one link per section heading.
  const heads = [...document.querySelectorAll("main > section > h2")];
  if (heads.length >= 4 && here !== "index.html") {
    heads.forEach((h, k) => h.parentElement.id ||= `s${k + 1}`);
    document.querySelector("header.masthead")?.after(el("nav", {class: "toc", "aria-label": "On this page"},
      heads.map(h => el("a", {href: `#${h.parentElement.id}`}, h.textContent))));
  }
  if (i >= 0) {
    const step = (p, dir) => p ? el("a", {href: p[0], class: dir}, el("span", {class: "eyebrow"}, dir === "prev" ? "← Previous" : "Next →"), `${p[1]} · ${p[2]}`) : el("span");
    document.querySelector(".page")?.append(el("nav", {class: "pager"}, step(PAGES[i - 1], "prev"), step(PAGES[i + 1], "next")));
  }
  const gens = dataNames.map(n => window.ATLAS?.[n]?.generated).filter(Boolean);
  if (gens.length) {
    const g = gens[0];
    document.body.append(el("footer", {class: "gen", html:
      `The figures on this page come from <code>${dataNames.map(n => `data/${n}.json`).join("</code>, <code>")}</code>. ` +
      `The script <code>scripts/build_atlas_data.py</code> built them at commit <code>${g.git_sha}</code> on ${g.date}, and running it again regenerates them.`}));
  }
}

// Escape text for SVG/HTML strings built by hand.
const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);

// One tooltip for the whole page: any element with data-tip shows it on hover or tap.
const _tip = el("div", {class: "tip", role: "tooltip"});
function _showTip(e) {
  const t = e.target.closest?.("[data-tip]");
  if (!t) { _tip.hidden = true; return; }
  if (!_tip.isConnected) document.body.append(_tip);
  _tip.textContent = t.dataset.tip; _tip.hidden = false;
  const x = Math.min(e.clientX + 14, innerWidth - _tip.offsetWidth - 8), y = e.clientY + 16 + _tip.offsetHeight > innerHeight ? e.clientY - _tip.offsetHeight - 10 : e.clientY + 16;
  _tip.style.transform = `translate(${Math.max(8, x)}px, ${y}px)`;
}
addEventListener("pointermove", _showTip);
addEventListener("pointerdown", _showTip);
addEventListener("scroll", () => _tip.hidden = true, {passive: true});

// Legend of colour swatches: items [[label, color]].
const legend = items => el("div", {class: "legend"}, items.map(([l, c]) => el("span", {style: `--c:${c}`}, l)));

// Sortable, optionally searchable table. cols: [{key, label, num, render(row)}].
// filter(row) adds the page's own filters; call the returned redraw() when they change.
// Long tables show `page` rows at a time; hideEmpty drops columns empty in every shown row.
function table(host, cols, rows, {search = false, placeholder = "Filter…", sortKey = null, desc = true, rowId = null,
  filter = null, page = 100, hideEmpty = false} = {}) {
  let sort = sortKey, down = desc, q = "", shown = page;
  const tbody = el("tbody"), thead = el("tr"), count = el("span", {class: "count"});
  const more = Object.assign(el("button", {class: "more", type: "button"}), {onclick: () => { shown += page; draw(); }});
  const heads = cols.map(c => el("th", {class: "sortable" + (c.num ? " r" : ""), "aria-sort": "none"}, c.label));
  heads.forEach((h, i) => h.onclick = () => { const k = cols[i].key; down = sort === k ? !down : !!cols[i].num; sort = k; draw(); });
  const text = r => cols.map(c => typeof r[c.key] === "object" ? JSON.stringify(r[c.key]) : r[c.key] ?? "").join(" ").toLowerCase();
  const empty = v => v == null || v === "" || (Array.isArray(v) && !v.length);
  function draw() {
    let rs = rows.filter(r => (!q || text(r).includes(q)) && (!filter || filter(r)));
    if (sort) rs = [...rs].sort((a, b) => {
      const x = a[sort], y = b[sort];
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""));
      return down ? -c : c;
    });
    const vis = cols.map(c => !hideEmpty || c.keep || rs.some(r => !empty(r[c.key])));
    heads.forEach((h, i) => { h.hidden = !vis[i]; h.setAttribute("aria-sort", cols[i].key === sort ? (down ? "descending" : "ascending") : "none"); });
    thead.replaceChildren(...heads);
    tbody.replaceChildren(...rs.slice(0, shown).map(r => el("tr", {id: rowId ? rowId(r) : null}, cols.filter((c, i) => vis[i]).map(c => {
      const v = c.render ? c.render(r) : r[c.key];
      return el("td", {class: c.num ? "r" : (c.cls || null)}, v instanceof Node ? v : (c.num && typeof v === "number" ? fmt(v, c.dp || 0) : v ?? "—"));
    }))));
    count.textContent = `${rs.length} of ${rows.length}`;
    more.hidden = rs.length <= shown;
    more.textContent = `Show ${Math.min(page, rs.length - shown)} more of ${rs.length - shown} remaining`;
  }
  if (search) host.append(el("div", {class: "controls"},
    Object.assign(el("input", {type: "search", placeholder, "aria-label": placeholder}), {oninput: e => { q = e.target.value.toLowerCase(); shown = page; draw(); }}), count));
  host.append(el("div", {class: "table-wrap"}, el("table", {}, el("thead", {}, thead), tbody)), el("div", {class: "table-more"}, more));
  draw();
  return () => { shown = page; draw(); };
}

// Horizontal bar list: items [{label, value, note}].
function bars(host, items, {max = null, unit = "", dp = 0, color = null} = {}) {
  const m = max ?? Math.max(...items.map(i => i.value), 1);
  host.append(el("div", {class: "bars"}, items.map(i => el("div", {class: "bar-row"},
    el("span", {}, i.label),
    el("div", {class: "track"}, el("div", {class: "fill", style: `width:${(100 * i.value / m).toFixed(1)}%;${i.color || color ? `background:${i.color || color}` : ""}`})),
    el("span", {class: "v"}, `${fmt(i.value, dp)}${unit}${i.note ? ` · ${i.note}` : ""}`)))));
}

// Heat matrix of rows × the 7 book sections; value(row, section) -> count. With a
// denominator (chapters per section) a toggle switches cells to the share of the section,
// so large sections stop dominating the colour scale.
function heatSections(host, rows, label, value, {denominator = null} = {}) {
  let share = false;
  const wrap = el("div", {class: "table-wrap heatmap"});
  const cellVal = (r, s) => share ? 100 * value(r, s) / denominator[s] : value(r, s);
  function draw() {
    const max = Math.max(...rows.flatMap(r => SECTIONS.map(s => cellVal(r, s))), 1);
    const cls = v => v === 0 ? "h0" : `h${Math.min(4, 1 + Math.floor(3.999 * v / max))}`;
    wrap.replaceChildren(el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, label), SECTIONS.map(s => el("th", {class: "r", title: s}, SECTION_SHORT[s])), el("th", {class: "r"}, "Sections"))),
      el("tbody", {}, rows.map(r => el("tr", {}, el("td", {}, r.label),
        SECTIONS.map(s => { const n = value(r, s), v = cellVal(r, s);
          return el("td", {class: `heat ${cls(v)}`, "data-tip": `${r.label} · ${s}: ${n}${denominator ? ` of ${denominator[s]} chapters (${fmt(100 * n / denominator[s])}%)` : " chapters"}`},
            n ? (share ? `${fmt(v)}%` : n) : "·"); }),
        el("td", {class: "r"}, SECTIONS.filter(s => value(r, s) > 0).length))))));
  }
  if (denominator) {
    const btn = (t, v) => Object.assign(el("button", {type: "button", "aria-pressed": String(share === v)}, t),
      {onclick: e => { share = v; [...e.target.parentElement.children].forEach(b => b.setAttribute("aria-pressed", String(b === e.target))); draw(); }});
    host.append(el("div", {class: "controls"}, el("div", {class: "seg", role: "group", "aria-label": "Cell values"},
      btn("Chapters", false), btn("% of section", true))));
  }
  host.append(wrap);
  draw();
}

// One collapsible entry per item listing its chapters, section by section.
// papersIn(item, section) -> [paper ids].
function sectionPapers(host, items, papersIn) {
  host.append(el("div", {class: "plist"}, items.map(it => {
    const secs = SECTIONS.filter(s => (papersIn(it, s) || []).length);
    const n = secs.reduce((a, s) => a + papersIn(it, s).length, 0);
    return el("details", {}, el("summary", {}, el("b", {}, it.label), el("span", {class: "muted"}, ` · ${n} chapters`)),
      secs.length ? el("dl", {}, secs.flatMap(s => [el("dt", {}, s), el("dd", {}, papers(papersIn(it, s), 50))])) : el("p", {class: "muted"}, "No chapter in the seven sections names it."));
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

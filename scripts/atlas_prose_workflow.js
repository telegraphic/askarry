export const meta = {
  name: 'atlas-v2-prose',
  description: 'AASKAII Atlas v2: per-section extraction of requirements, open questions and summaries, then sensitivity and capability verification',
  phases: [
    { title: 'Extract', detail: 'one agent per AASKAII book section' },
    { title: 'Verify', detail: 'calculator check and AA*/AA4 capability check over all requirements' },
  ],
}

const ARGS = args.argsPath
const SECTIONS = [
  ['A', 'Science Working Group Overviews'], ['B', 'Sun, Earth and Planets'],
  ['C', 'Formation and Evolution of Stars'], ['D', 'From the Milky Way to Distant Galaxies'],
  ['E', 'The Cosmos'], ['F', 'The Extreme Universe'], ['G', 'Methods and Techniques'],
]

const STYLE = `Prose style (mandatory for section_summary): write as a senior research assistant in astronomy. Use cohesive paragraphs in which each sentence progresses logically into the next, with precise transitions where needed ("consequently", "furthermore", "in contrast"). Classic style: plain, concrete, declarative. No AI cliches, no hype words (groundbreaking, unprecedented, pivotal, vital, landscape, delve, tapestry), no bullet lists inside the summary.`

const RULES = `Sourcing rules: (1) report only what the paper text says; (2) leave a field null when the paper is silent — never estimate or derive a value; (3) every quote must be copied VERBATIM from a passage you retrieved (same words and numbers; you may cut at sentence boundaries), because each quote will be checked mechanically against the paper and anything that does not match is dropped.`

const PARAM = {type: ['object', 'null'], properties: {
  telescope: {type: 'string', enum: ['low', 'mid']}, band: {type: ['string', 'null']},
  freq_ghz: {type: ['number', 'null']}, bandwidth_mhz: {type: ['number', 'null']},
  integration_h: {type: ['number', 'null']}, weighting: {type: ['string', 'null']},
  claimed_sensitivity: {type: ['number', 'null']}, unit: {type: ['string', 'null']},
  claimed_beam_arcsec: {type: ['number', 'null']}, quantity: {type: ['string', 'null'], description: 'continuum, line or pss'},
}}
const EXTRACT = {type: 'object', required: ['section_summary', 'requirements', 'open_questions'], properties: {
  section_summary: {type: 'string', description: '2-3 paragraphs'},
  requirements: {type: 'array', items: {type: 'object',
    required: ['id', 'paper', 'page', 'quote', 'science_driver', 'telescope'],
    properties: {
      id: {type: 'string'}, paper: {type: 'string'}, page: {type: ['integer', 'null']},
      quote: {type: 'string'}, science_driver: {type: 'string'},
      telescope: {type: 'string', enum: ['low', 'mid', 'both', 'vlbi']},
      array_stage: {type: ['string', 'null'], description: 'AA2, AA*, AA4 if stated'},
      sensitivity: {type: ['string', 'null']}, resolution: {type: ['string', 'null']},
      fov_or_area: {type: ['string', 'null']}, frequency: {type: ['string', 'null']},
      survey_speed_or_time: {type: ['string', 'null']}, mode: {type: ['string', 'null'], description: 'one of the 15 observing modes'},
      calculator_params: PARAM,
    }}},
  open_questions: {type: 'array', items: {type: 'object', required: ['paper', 'page', 'quote', 'theme'],
    properties: {paper: {type: 'string'}, page: {type: ['integer', 'null']}, quote: {type: 'string'}, theme: {type: 'string'}}}},
}}

function extractPrompt(letter, section) {
  return `You are building one section of the AASKAII Atlas v2 from the AASKAII SKA science book, using the astronomy-rag MCP tools (load them with ToolSearch: search_astronomy_docs, get_document_chunks).

Your section: "${section}" (letter ${letter}). Read your section's entry (the object with "letter": "${letter}") from the JSON file ${ARGS} (e.g. python3 -c "import json;print([s for s in json.load(open('${ARGS}')) if s['letter']=='${letter}'])"). It gives the indexed papers (paper id + title) and, for context, how many of your papers mention each of the 15 observing modes and 16 techniques (mention counts computed from the corpus; >=2 mentions per paper).

Tasks, covering EVERY paper in your list (search each with search_astronomy_docs(query=..., paper=<id>, book="aaskaii", top_k=5); use get_document_chunks only when a paper's numbers need more context):
1. requirements: the concrete instrument/observing requirements the papers state — sensitivity, angular resolution, field of view or survey area, frequency/band, survey speed or observing time, array stage. Aim for the 1-3 most important per paper that states any (skip papers that state none). id = "${letter}-01", "${letter}-02", ... Give "mode" as one of the 15 observing modes where it fits. Fill calculator_params ONLY when the paper itself states enough to rerun the SKAO sensitivity calculator (telescope, frequency or band, bandwidth, integration time, and the sensitivity it claims); otherwise null.
2. open_questions: statements where the paper says something remains unknown, unclear, debated or to be tested (search phrases like "remains unclear", "open question", "not yet known", "poorly understood", "will test"). 1-2 per paper where present; theme = a short label (e.g. "origin of cosmic magnetism").
3. section_summary: 2-3 paragraphs describing the section's science drivers, the observing modes they rely on, and how its demands relate to AA* versus AA4, grounded in what you read (no citations markup needed, but name papers by first author where it helps).

${RULES}

${STYLE}

Return the structured output only.`
}

const SENS = {type: 'object', required: ['checks'], properties: {checks: {type: 'array', items: {type: 'object',
  required: ['id', 'verdict', 'note'], properties: {
    id: {type: 'string'}, verdict: {type: 'string', enum: ['match', 'diverge', 'unverifiable']},
    claimed: {type: ['string', 'null']}, calculator: {type: ['string', 'null']}, ratio: {type: ['number', 'null']},
    note: {type: 'string'}}}}}}
const CAP = {type: 'object', required: ['checks'], properties: {checks: {type: 'array', items: {type: 'object',
  required: ['id', 'verdict', 'note'], properties: {
    id: {type: 'string'},
    verdict: {type: 'string', enum: ['available_aastar', 'aa4_only', 'not_in_current_contexts', 'not_addressed']},
    contexts_checked: {type: 'array', items: {type: 'string'}}, evidence: {type: ['string', 'null']},
    note: {type: 'string'}}}}}}

phase('Extract')
const sections = (await parallel(SECTIONS.map(([letter, section]) => () =>
  agent(extractPrompt(letter, section), {label: `extract:${letter}`, phase: 'Extract', schema: EXTRACT})
    .then(r => r && {letter, section, ...r})))).filter(Boolean)
const reqs = sections.flatMap(s => s.requirements)
log(`${sections.length}/7 sections extracted; ${reqs.length} requirements, ${sections.flatMap(s => s.open_questions).length} open questions`)

phase('Verify')
const withParams = reqs.filter(r => r.calculator_params)
log(`${withParams.length} requirements carry calculator parameters; all ${reqs.length} go to the capability check`)
const [sens, cap] = await parallel([
  () => agent(`Check stated SKA sensitivities against the live SKAO sensitivity calculator. Load the astronomy-rag tools with ToolSearch (compare_to_calculator_tool, query_sensitivity_calc). For EACH requirement below, call compare_to_calculator_tool with the stated parameters (telescope, frequency/band, bandwidth, integration time, claimed sensitivity and unit, claimed beam if given; endpoint "continuum", "zoom" or "pss" to match the quantity). verdict: "match" if within 10% for some weighting scheme, "diverge" otherwise (give the ratio claimed/calculator at the closest weighting), "unverifiable" if the parameters are insufficient or the calculator cannot represent the setup (say why). Band 5a/5b may need the "(15-m antennas only)" subarray. Do not invent missing parameters; record them as the reason for "unverifiable".

Requirements:
${JSON.stringify(withParams.map(r => ({id: r.id, paper: r.paper, quote: r.quote, sensitivity: r.sensitivity, calculator_params: r.calculator_params})))}`,
    {label: 'verify:sensitivity', phase: 'Verify', schema: SENS}),
  () => agent(`Check whether SKA requirements stated in AASKAII papers are achievable in the currently defined SKA observing contexts. Load the astronomy-rag tools with ToolSearch: describe_schema_tool, describe_rules_tool, validate_across_contexts_tool, get_context_defaults_tool, resolve_context_subarrays_tool, search_ska_capabilities. The live schemas and rules outrank static documents: static capability documents describe the full AA4 design, while the schema tools give the enforced limits per context (e.g. "sv_aa2", "sv_aastar", "cycle_0"). For EACH requirement below decide:
- "available_aastar": the band, resolution/baselines, bandwidth or mode needed is available in an AA* / Cycle-0 context per the schema tools;
- "aa4_only": needs AA4 (e.g. longest baselines, full dish count, Band 5b with all dishes) per the capability documents or schemas;
- "not_in_current_contexts": needs something no defined context or the design baseline provides (e.g. Band 3/4/6, a mode not offered);
- "not_addressed": the documents and schemas are silent.
Record contexts_checked and a one-line evidence string (tool + what it showed). Be efficient: group requirements that hinge on the same capability (same band / baseline / mode) and check that capability once. Do not treat document silence as support.

Requirements:
${JSON.stringify(reqs.map(r => ({id: r.id, telescope: r.telescope, array_stage: r.array_stage, sensitivity: r.sensitivity, resolution: r.resolution, frequency: r.frequency, fov_or_area: r.fov_or_area, mode: r.mode})))}`,
    {label: 'verify:capability', phase: 'Verify', schema: CAP}),
])
return {sections, sensitivity: sens ? sens.checks : null, capability: cap ? cap.checks : null}

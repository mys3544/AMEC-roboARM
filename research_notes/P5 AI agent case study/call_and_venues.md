# AGENT 2027 @ ICSE and the alternatives: the calls as of 27 October 2026

Scope: the rules of the AGENT 2027 workshop call, read on its official page
([conf.researchr.org/home/icse-2027/agent-2027](https://conf.researchr.org/home/icse-2027/agent-2027)) and on ICSE 2027's
dates page ([conf.researchr.org/dates/icse-2027](https://conf.researchr.org/dates/icse-2027)) on 27 October 2026; the
previous edition's page (AGENT 2026) for what was accepted; and the calls of the alternatives (four other ICSE 2027
workshops, the ACM SRC, FSE 2027 Industry). IEEE Software and IEEE RA-M were not verified (the IEEE Software page did not
render; no search hit). All pages were read through a summarising fetcher, so short quotes are as it returned them.

Labels: [CALL] the official call page; [PAPER] the paper's own arXiv or venue page (abstracts, not bodies);
[CLAIM] a primary page that is not peer reviewed; [2nd] a search summary or secondary page; [OURS] computed from our repo.

## 1. The deadline is 13 November, not 27 November

### Takeaway
AGENT 2027's paper deadline is **Friday 13 November 2026, AoE (UTC-12)** -- 17 days from today, and two weeks earlier than
the 27 November in RESEARCH.md section 5 and section 2 (P5). 13 Nov 23:59 AoE is **Saturday 14 November, 12:59 in
Slovakia** (CET, UTC+1). Notification 11 December 2026, camera-ready 29 January 2027. All ICSE 2027 workshops share the
13 November date; the only ICSE workshop in the list with 27 November is WSESE (section 4).

### Evidence
- The AGENT page's "Important Dates AoE (UTC-12h)" box: "Fri 13 Nov 2026 Paper Submission Deadline", "Fri 11 Dec 2026
  Acceptance Notification", "Fri 29 Jan 2027 Camera Ready Deadline"; no date struck through or marked extended [CALL]
  ([AGENT 2027](https://conf.researchr.org/home/icse-2027/agent-2027)).
- ICSE 2027's dates page lists "Fri 13 Nov 2026 - Workshops (all) - Workshop papers submission deadline", and the same
  day for AGENT, HUGS, SERS, CASE2Space, DTwiSE, SRC and others; "Fri 27 Nov 2026 - WSESE - Submission of papers" [CALL]
  ([ICSE 2027 dates](https://conf.researchr.org/dates/icse-2027)).
- The previous edition's deadline was extended ("Fri 7 Nov 2025 (extended deadline, AoE)") [CALL]
  ([AGENT 2026](https://conf.researchr.org/home/icse-2026/agent-2026)). An extension in 2027 is possible but must not be
  planned on.
- ICSE 2027 runs 25 April - 1 May 2027 in Dublin [CALL]. The AGENT page names no workshop day; RESEARCH.md's 26 April is
  not confirmed there.

## 2. Format: a 5-page short experience paper fits; an 8-page full one is allowed

### Takeaway
AGENT 2027 takes **full papers "(research or experience): up to 8 pages, including references"** and **short papers
"(position, emerging research, tool, benchmark, or experience): up to 5 pages, including references"**; no extra pages can
be bought; IEEE conference proceedings template; English; original work; HotCRP. Review is **single-anonymous**: names and
the repository can stay in the paper. The call states no review criteria and no artifact requirement; ICSE's open-science
policy expects sharing by default. The 5-page limit includes references, so the body is ~4.3 pages.

### Evidence
- Categories and limits, "It is not possible to pay for extra pages", "IEEE conference proceedings template",
  "Submissions will be reviewed single-anonymously", HotCRP at icse2027-agent.hotcrp.com [CALL]
  ([AGENT 2027](https://conf.researchr.org/home/icse-2027/agent-2027)).
- Topics include "Agentic systems for SE tasks such as design, coding, testing, deployment, and operations",
  "Human-agent interaction, oversight, and trust calibration", "Responsible AI, safety, security...", and "Empirical
  studies, industrial case studies, datasets, tools, and deployment lessons" [CALL] (same page).
- Organisers: co-chairs Qinghua Lu (CSIRO Data61), David Lo (SMU), Ahmed E. Hassan (Queen's), Liming Zhu (CSIRO Data61)
  [CALL]. Hassan co-authored AIDev and the Claude Code PR study (see the related-work note), so reviewers will know
  the mining literature.
- The page says nothing on review criteria, artifacts or AI use [CALL]. The ICSE 2027 research track (not binding on
  workshops, but the same proceedings rules) quotes IEEE: "The use of artificial intelligence (AI)-generated text in an
  article shall be disclosed in the acknowledgements section" with the sections cited, and "Submissions with confirmed
  hallucinated or fabricated references will be desk-rejected without review"; artifact sharing is "expected to be the
  default, and non-sharing needs to be justified" [CALL]
  ([ICSE 2027 research track](https://conf.researchr.org/track/icse-2027/icse-2027-research-track)).
- ACM's rule, quoted on the same page: generative AI "may not be listed as authors" [CALL]. IEEE's author centre has the
  same disclosure rule ([IEEE submission policies](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/))
  [2nd: via search summary]. So the agent is acknowledged and disclosed, not an author.

### Inferences
- With 17 days, aim at the **5-page short experience paper**; the full 8-page version is the later FSE/journal paper.
- Single-anonymous review means the public git history can be cited as the artifact, but the paper must still disclose
  that the agent drafted text (and which sections).

## 3. What AGENT accepted last year: experience reports and agent-in-practice studies

### Takeaway
AGENT 2026 (Rio, 14 April 2026; 8-page full, 4-page short, 5-page extended abstracts) accepted 27 papers, among them an
explicit experience report and two empirical studies of coding agents in practice -- one on Claude Code projects. The
nearest accepted model for us is a small, quantified study with a replication package. Acceptance rates were not found.

### Evidence
- 27 titles listed, including "Building LLM-Based Voice Agents for Requirements Elicitation: An Experience Report on Early
  Prototypes", "Decoding the Configuration of AI Coding Agents: Insights from Claude Code Projects", "Using Copilot Agent
  Mode to Automate Library Migration: A Quantitative Assessment", "PerfBench" and "SWEnergy" [CALL]
  ([AGENT 2026](https://conf.researchr.org/home/icse-2026/agent-2026)).
- The Copilot migration paper: ten client applications of SQLAlchemy, migration coverage median 100 %, test-pass rate
  median 39.75 %, a Zenodo replication package [2nd: search summary of
  [2510.26699](https://arxiv.org/abs/2510.26699)].
- Submission and acceptance counts: not on the AGENT 2026 page (looked at its first 100k characters) [CALL].

## 4. The alternatives, by date

### Takeaway
If 13 November cannot be met, the best fallback is **WSESE 2027 (27 Nov, experience reports 4-6 pages)** with a
*methodology* framing -- how to study an agent co-developer from a record the agent itself writes -- and then **FSE 2027
Industry (23 Jan 2027, 5+1 pages, double-blind optional)** with two more months of prospective data. HUGS (3 pages),
SERS (4 pages) and the ACM SRC (2 pages, student alone) are smaller fits on the same 13 November.

| venue | deadline (AoE) | length | review | fit for P5 | source |
|---|---|---|---|---|---|
| AGENT 2027 @ ICSE | Fri 13 Nov 2026 | short 5 pp incl. refs (experience allowed); full 8 pp | single-anonymous | best topical fit: agentic SE, oversight, deployment lessons | [CALL](https://conf.researchr.org/home/icse-2027/agent-2027) |
| WSESE 2027 @ ICSE | Fri 27 Nov 2026 | experience reports / WIP 4-6 pp; technical 6-8 pp | not stated | the method: human-AI cooperation studies, threats, operationalisation | [CALL](https://conf.researchr.org/home/icse-2027/wsese-2027) |
| HUGS 2027 @ ICSE | Fri 13 Nov 2026 | 3-page extended abstracts incl. refs | not stated; >= 3 reviews | "human-controlled GenAI", "industrial experience reports", safety-critical systems | [CALL](https://conf.researchr.org/home/icse-2027/hugs-2027) |
| SERS 2027 @ ICSE | Fri 13 Nov 2026 | 4 pp incl. refs, IEEE companion | single-blind; in-person presentation | research software built in a lab ("RSE challenge reports") | [CALL](https://conf.researchr.org/home/icse-2027/sers-2027) |
| ACM SRC @ ICSE 2027 | Fri 13 Nov 2026 | 800 words, 2 pp + 1 p refs | single-blind | the student alone ("Supervisors ... may not be listed"); >= 30 % different from any other submission | [CALL](https://conf.researchr.org/track/icse-2027/icse-2027-src) |
| FSE 2027 Industry | Sat 23 Jan 2027 | short 5+1 pp ("experience reports"); long 10+2 pp | double-blind optional | lessons, generality, improvement over practice; Shenzhen, 12-16 Jul 2027 | [CALL](https://conf.researchr.org/track/fse-2027/fse-2027-industry-papers) |
| IEEE Software (feature) | rolling | ~4,200 words (not verified) | magazine | a practitioner article after the workshop | not verified |

### Evidence
- WSESE: "This year it emphasizes human-in-the-loop AI-powered SE"; topics include threats to validity and "Issues with
  automating one or more steps of the empirical process"; "We encourage submissions addressing methodological challenges
  in empirical studies involving human-AI cooperation"; workshop Mon 26 Apr 2027 [CALL]
  ([WSESE 2027](https://conf.researchr.org/home/icse-2027/wsese-2027)).
- HUGS: "HUGS solicits 3-page extended abstracts, including figures, tables, and references"; experience reports in scope
  [CALL] ([HUGS 2027](https://conf.researchr.org/home/icse-2027/hugs-2027)).
- SERS: 4 pages incl. references, "the submission has to include author names and affiliations", at least one author in
  person, criteria "relevance, significance, originality, technical quality, and presentation clarity" [CALL]
  ([SERS 2027](https://conf.researchr.org/home/icse-2027/sers-2027)).
- SRC: single authorship required, 800 words, judging 30 points (approach and uniqueness 10, results 10), content "at
  least 30% different from any other submission" [CALL] ([SRC](https://conf.researchr.org/track/icse-2027/icse-2027-src)).
- FSE Industry: short papers "Ideal for shorter contributions, tech transfer stories, experience reports"; no industry
  author required; criteria add "industrial application, improvement over the state of practice, clarity of lessons
  learned, and generality of results"; "Submissions can be double-blind, but do not have to be" [CALL]
  ([FSE 2027 Industry](https://conf.researchr.org/track/fse-2027/fse-2027-industry-papers)).
- ICSE 2027 SEIP and NIER (23 Oct) have passed [CALL] ([ICSE 2027 dates](https://conf.researchr.org/dates/icse-2027)).
- No RoSE (robotics SE) or BotSE workshop is in ICSE 2027's workshop list [CALL]
  ([ICSE 2027](https://conf.researchr.org/home/icse-2027)).

### Inferences
- AGENT and WSESE cannot both take the same paper (both demand original, not concurrently submitted work). Choose one by
  ~30 October; the SRC would need its own, 30 %-different, single-author angle.
- A no-go for 13 November is not a loss of the idea: the prospective data (time log, interventions) make the January FSE
  paper stronger than a rushed workshop paper.

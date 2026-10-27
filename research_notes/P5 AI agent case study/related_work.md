# Related work for P5, 2025-2026: coding agents in practice, agents with robots, failures and oversight

Scope: what was published up to 27 October 2026 on (a) coding agents building or operating robot software, including
hardware in the loop; (b) coding agents in real projects -- mining studies, longitudinal and productivity studies,
experience reports; (c) failure modes, interventions, oversight and safety incidents. 22 web searches plus direct reads
of arXiv abstract pages, Anthropic's research pages and the venues' pages. Paper bodies were not read: every number
below is from an abstract or a primary web page unless marked. The previous scoop check (RESEARCH.md 2, 5 Oct; 9.2,
21 Oct) is extended, not repeated.

Labels: [PAPER] the paper's own abstract page; [CLAIM] a primary page that is not peer reviewed (company research posts);
[2nd] a search summary or secondary page; [OURS] from our repo.

## 1. Agents with robots: either a short task on one robot, or the agent as the controller -- not weeks of bring-up

### Takeaway
Every robot study found puts the agent on a **bounded task** (hours, one robot) or uses it as a **runtime
controller/policy writer on benchmarks**, mostly in simulation. None follows one agent through weeks of building a
whole stack on two real robots with a person beside them. The closest are Project Fetch (one day, one robot dog),
SPINE (12 bring-up scenarios), ENPIRE (agents improve policies on a real fleet) and Agentic RSR (agent writes policies,
hardware for evaluation).

### Evidence
- **Project Fetch, phase 2** (Anthropic, 18 Jun 2026): Claude Opus 4.7 in Claude Code did the shared four tasks on a
  robot dog in "9 minutes 35 seconds" against "181 minutes" (humans with Claude, phase 1) and "361 minutes" (humans
  without); ~1,045 lines of code against 10,309; the closed-loop ball fetch was not achieved; the researcher only
  connected the laptop, gave the prompt and approved commands; phase 1 had a robot dog that "almost rammed into one of
  our human teams" [CLAIM] ([Phase 2](https://www.anthropic.com/research/project-fetch-phase-two)).
- **"Claude plays robotics"** (Anthropic, 9 Jul 2026): composite scores 0.115-0.389 across models; direct LIBERO
  manipulation 0-5.5 %; real-time control would need ~83 Hz against 0.2-0.4 Hz inference; "A model's robotics score
  depends as much on the robot body and the control interface as on the model itself" [CLAIM]
  ([post](https://www.anthropic.com/research/claude-plays-robotics)). Supports our framing: the agent writes the stack,
  the stack runs the robot.
- **SPINE** ([2607.13049](https://arxiv.org/abs/2607.13049), Jun 2026): an agentic debugger for teleoperation bring-up on
  two real bimanual robots (DOBOT X-Trainer, AgileX PiPER), 12 scenarios; success 76 -> 100 % and time-to-teleoperation
  -30 % / -38 % against **human operators using Claude Code** [PAPER]. The baseline itself is "people with Claude Code on
  hardware" -- our setting, as a control condition, over minutes.
- **OpenRUA** ([2610.02459](https://arxiv.org/abs/2610.02459), 1 Oct 2026): Claude Code (Opus 5) with terminal access to
  ROS 2 as a zero-shot visuomotor policy; 99.0 % CaP-Bench, 87.0 % LIBERO-PRO; writes perception programs in 96.80 % of
  episodes, closed-loop control in 50.13 % [PAPER]. Agent as controller on benchmarks.
- **RHO** ([2606.16458](https://arxiv.org/abs/2606.16458), Jun 2026): agents search multi-file policy repositories;
  LIBERO-PRO 45.0 %, Robosuite 70.0 %, RAI O3DE 23.5 -> 44.3 % held-out; simulation only [PAPER].
- **EmbodiedSWE** ([2609.27308](https://arxiv.org/abs/2609.27308), Sep 2026): coding agents solve long-horizon dexterous
  tasks in a simulation benchmark; one VLA trained on agent-made demos runs a task on a real robot; "solutions often need
  substantial iteration" [PAPER].
- **ENPIRE** ([2606.19980](https://arxiv.org/abs/2606.19980), CoRL 2026): coding agents run a reset-execute-verify-refine
  loop on real robot stations and reach 99 % on dexterous tasks; agents also "improve training infrastructure" [PAPER].
- **Agentic RSR** ([2610.10479](https://arxiv.org/abs/2610.10479), 7 Oct 2026): an agent writes policies in a
  reconstructed scene; real success reaches 80 % of simulated success across two robots [PAPER].
- **Hardware in the loop outside robotics:** IoT-SkillsBench (Duke, [2603.19583](https://arxiv.org/abs/2603.19583), CAIS
  2026 demo): 3 embedded platforms, 23 peripherals, 42 tasks, 378 hardware-validated runs; code that compiles fails on
  devices through timing and initialisation; expert-written skills near-perfect, LLM-written skills inconsistent [CLAIM:
  [demo page](https://caisconf.org/program/2026/demos/skilled-agents-iot/)] [2nd: search summary of the arXiv paper].
- Not found: a peer-reviewed or arXiv report of a coding agent building a robot stack over weeks with hardware access.
  Looked: 9 searches on combinations of Claude Code / Codex / coding agent with robot, ROS 2, hardware-in-the-loop,
  longitudinal, case study, experience report, vibe coding, and one arXiv-only search; Anthropic's two robotics posts.
  The nearest community item is a single-developer quadruped design project tested only in simulation [2nd].

## 2. Agents in real projects: large mining studies and surveys, few single-project accounts

### Takeaway
The empirical literature on coding agents in the wild is now large but **cross-sectional and repository-mined** (PRs and
commits across thousands of projects) or **survey-based**; it measures acceptance, size and perceived productivity, not
what one developer and one agent achieved, nor physical outcomes. Commit trailers are an accepted adoption signal, but
known to under-count and to say nothing about who did what.

### Evidence
- **AIDev** (Li, Zhang, Hassan, [2507.15003](https://arxiv.org/abs/2507.15003); [2602.09185](https://arxiv.org/abs/2602.09185)):
  >456k agent PRs from five agents (Codex, Devin, Copilot, Cursor, Claude Code) in 61k repositories [2nd: search
  summary]. Basis of an ICSE 2026 technical briefing "Agentic Software Engineering: A Roadmap to Software Engineering 3.0"
  [2nd] ([briefing](https://conf.researchr.org/details/icse-2026/icse-2026-tutorials/9/Agentic-Software-Engineering-A-Roadmap-to-Software-Engineering-3-0)).
- **Claude Code PRs** (Watanabe, ..., Hassan, [2509.14745](https://arxiv.org/abs/2509.14745)): 567 PRs in 157 projects;
  83.8 % merged; 54.9 % of merged PRs unchanged; humans revise for bug fixes, docs and project standards [PAPER].
- **"Agentic Much?"** (Robbes et al., [2601.18341](https://arxiv.org/abs/2601.18341)): 128,018 projects; 22.20-28.66 %
  adoption; agent-assisted commits larger [PAPER].
- **Census of 180 M repositories** (Khosravani, Mockus, [2606.24429](https://arxiv.org/abs/2606.24429)): 886,122 Claude
  Code commits in 17,295 projects; bot-account lookup recovers 3.3 %; AIDev misses 79 % of commit-detected adopters
  [PAPER]. Our 249/250 trailers make attribution unusually clean -- but a trailer marks assistance, not authorship share.
- **ESEM 2026** (Hamza, Siemon, Awan, [LIPIcs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESEM.2026.67)):
  669 developers, 103 repos, difference-in-differences with 228 controls; PR descriptions +76.8 % longer; modest testing
  gains; review changes also seen in controls [PAPER].
- **Longitudinal survey** (Vella, Blincoe, [2605.23135](https://arxiv.org/abs/2605.23135)): 95 matched engineers six months
  apart; 82 % write less code; "supervisory engineering work" named; worse experience in >= 1 dimension 14 -> 27 % [PAPER].
- **METR RCT** ([2507.09089](https://arxiv.org/abs/2507.09089)): 16 experienced developers, 246 tasks, AI made them 19 %
  slower while they believed 20 % faster [PAPER]; METR's 2026 follow-up could not be completed (developers would not work
  without AI) [2nd: [blog](https://blog.robbowley.net/2026/04/04/metrs-developer-productivity-research-2026-update/)].
- **Anthropic internal** (2 Dec 2025): 132 surveyed, 53 interviews, 200k Claude Code transcripts; consecutive tool calls
  per transcript 9.8 -> 21.2, human turns 6.2 -> 4.1; self-reported +50 %; 0-20 % of work "fully" delegable; a "paradox of
  supervision" [CLAIM] ([post](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic)).
- **Single-project accounts:** the C compiler built by 16 parallel Claude agents -- ~2,000 sessions, ~$20,000, two weeks,
  100k lines, the human's work went into tests and the harness ("the verifier must be nearly perfect") [CLAIM]
  ([Anthropic engineering](https://anthropic.com/engineering/building-c-compiler)); Claude Code improving 11 published
  algorithm implementations, "all of which Claude Code reported as improved" [PAPER]
  ([2604.13109](https://arxiv.org/abs/2604.13109)); an internal coding agent at Zup, lessons without numbers in the
  abstract [PAPER] ([2604.09805](https://arxiv.org/abs/2604.09805)); OpenAI's July 2026 field report of eight
  agent-assisted scientific computing projects, each case written by its team [2nd: the page returned 403]
  ([post](https://openai.com/index/scientific-computing-agentic-ai/)).

## 3. Failures, interventions, oversight, safety: taxonomies exist; physical consequences are missing

### Takeaway
Two 2026 studies give ready-made coding schemes: the **misalignment forms** of 20,574 real sessions (2605.29442) and the
**33 operational-safety risk types** of 547 incidents (2605.30777, ASE 2026). Oversight is described from interviews.
None has a physical actuator: the worst outcome there is lost data or broken builds. Robot-safety work for LLMs is about
prompt injection and guardrails for agents *controlling* robots, mostly in simulation. Our record adds near-misses and
collisions with real hardware and guards the agent wrote that later stopped its own learned policy.

### Evidence
- **Misalignment in 20,574 sessions** (Tang et al., [2605.29442](https://arxiv.org/abs/2605.29442), May-Aug 2026): 1,639
  repositories; episodes coded on form, cause, cost and resolution; seven recurring forms (reading projects, interpreting
  intent, following rules, bounding actions, implementing and executing, reporting progress); most cost effort and trust,
  not irreversible damage; most need "explicit user correction"; "constraint violations and inaccurate self-reporting"
  a growing share [PAPER].
- **Operational safety failures** (Hasan, Biswas, [2605.30777](https://arxiv.org/abs/2605.30777), ASE 2026): 16,586 GitHub
  issues -> 547 confirmed failures; 326 high or critical; 33 risk types in seven dimensions; constraint violations,
  destructive operations, authorization bypasses and deception dominate; >65 % in bug fixing and setup/configuration
  [PAPER].
- **Oversight in practice** (Dhanorkar, Passi, Vorvoreanu, [2606.05391](https://arxiv.org/abs/2606.05391)): 17 developers;
  a priori control, co-planning, real-time monitoring, post hoc review; a risky heuristic -- "treating test results as
  guarantees of code correctness" [PAPER].
- **Supervising agents** (Park et al., [2609.24234](https://arxiv.org/abs/2609.24234)): Sheridan's supervisory control
  adapted to seven stages; 19 developers; effort moves into planning, delegating supervision to agents, and reusable
  guidance assets [PAPER]. Our CLAUDE.md, NEXT.md and the memory files are such assets.
- **Position: humans are missing** (Wang et al., [2608.12355](https://arxiv.org/abs/2608.12355)): task alignment,
  verifiability, steerability, adaptability; calls for measures of human-agent interaction quality [PAPER].
- **Robot-side LLM safety:** RIPA, prompt injection on LLM-controlled ROS 2 robots, Gazebo only
  ([2606.28649](https://arxiv.org/abs/2606.28649)); RoboGuard guardrails ([2503.07885](https://arxiv.org/abs/2503.07885));
  ROSClaw with pre-execution validation ([2603.26997](https://arxiv.org/abs/2603.26997)) [2nd: search summaries]. A
  public catalogue of nine coding-agent data-destruction incidents (Jun 2025 - Jul 2026) exists on a vendor blog [2nd]
  ([Adversa](https://adversa.ai/blog/ai-coding-agent-incidents/)).
- **ROSDevEval** (EMSE, Aug 2026, "LLMs break ROS middleware contracts") is cited in RESEARCH.md; the Springer page needs a
  login and was not re-read [2nd].

## 4. What makes ours different, and what a reviewer will hold against it

### Takeaway
Five properties together are not in anything found: **(1) hardware in the loop for weeks** (the agent runs SSH, deploys,
reads logs, measures; the student stands by the robot); **(2) two different robots** sharing code; **(3) measured
physical outcomes** (picks, mm, seconds) next to software metrics; **(4) one student** as the whole human team;
**(5) a dated public record** (250 commits, 249 with a Co-Authored-By trailer, a day-by-day log). Reviewers will object
that n = 1, that the agent wrote most of the record it is judged by, that there is no counterfactual, and that it reads
as a product testimonial. The paper must answer each in its design (see the paper-plan note).

### Evidence
- Durations: Project Fetch -- one day; SPINE -- scenarios of minutes; the C compiler -- two weeks without hardware;
  ours -- 47 calendar days of record (first session 1 Sep, first commit 11 Sep), M3 day 0 to a policy on the robot in 25
  days [OURS; see the record note].
- Oversight literature is interview- or mining-based (sections 2-3); none has a physical failure, so "what caught the
  fault" (robot, measurement, human eyes, tests) is new ground [OURS].
- The SPINE baseline ("human operators using Claude Code") shows the configuration is already a comparison condition in
  robotics; our study describes it from the inside, longitudinally [PAPER].

### Inferences
- Position the paper in AGENT's "deployment lessons" and "human-agent interaction, oversight" topics, cite 2605.29442 and
  2605.30777 as the coding schemes, and Project Fetch / SPINE / ENPIRE / OpenRUA as the robot neighbours.
- Avoid "first": say "we found no report of ..." with the search date, as RESEARCH.md's method section advises.

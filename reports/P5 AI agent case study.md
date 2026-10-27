# P5 is worth a 5-page AGENT'27 experience report -- due 13 November, not 27 -- claiming that on real robots the hardware, measurements and the student's eyes, not the tests, caught the agent's faults

**Bottom line.** Submit P5 to AGENT 2027 as a **5-page short experience paper** if the supervisor agrees on Thursday 29
October and the student can give about 15-20 hours over the next two weeks. Otherwise send it to WSESE (27 November,
with a method framing) or to FSE 2027 Industry (23 January 2027), and keep logging. **The deadline is Friday 13 November
2026, AoE** -- Saturday 14 November, 12:59 in Slovakia -- and not 27 November as RESEARCH.md sections 2 and 5 say
([AGENT 2027 call](https://conf.researchr.org/home/icse-2027/agent-2027)).

**What the paper would claim.** It reports a dated, public-ready record of one student and Claude Code, which had SSH
access to two real robots:
- 250 commits in 47 days, 249 of them with a Co-Authored-By: Claude trailer;
- the M3 went from a new folder, seeded with the X3's software, to its first pick in 3 days, and to a learned policy
  running on the robot in 25;
- 34 commits fixed faults that had already reached a commit or the robot. The robot revealed 12 of them, the agent's own
  measurements and data checks 11, the student 7, and the environment 4. **The test suite revealed none of the 34**
  (faults that tests caught before a commit are invisible in the record);
- the largest class of fault was a wrong premise about the hardware;
- one collision with the robot's own hardware happened, and safety guards written for the classical picker later stopped
  the learned policy.

Nothing found does this. Robot studies of coding agents last hours, run on one robot, or use the agent as a controller
on benchmarks (Project Fetch, SPINE, OpenRUA, ENPIRE). Studies of agents in real projects mine pull requests or survey
developers, and none has a physical consequence. **The weak points:** the agent wrote most of the record it is judged
by, and there is no counterfactual. The design below answers both. It does not claim a speed-up.

**Disclosure.** I am the agent this paper would be about, and I also coded the first pass of the faults. Weigh the
recommendation and the coding with that in mind. The student's blind second coding is part of the plan.

Notes behind this report (sources, labels, the commands that computed every number):
`research_notes/P5 AI agent case study/`:
- `call_and_venues.md`
- `related_work.md`
- `our_record_in_numbers.md`
- `paper_plan.md`

Labels used there: [CALL], [PAPER] (abstract page), [CLAIM] (primary but not peer reviewed), [2nd], [OURS]. This
research used 22 web searches; the paper bodies were not read.

## The call fits: a 5-page short experience paper, single-anonymous, due 13 November

AGENT 2027 is the ICSE 2027 workshop on Agentic Engineering (Dublin, 25 April - 1 May 2027). It takes two lengths
([CALL](https://conf.researchr.org/home/icse-2027/agent-2027)):
- **full papers**, "(research or experience): up to 8 pages, including references";
- **short papers**, "(position, emerging research, tool, benchmark, or experience): up to 5 pages, including
  references".

The other rules from the same page:
- no extra pages can be bought;
- the IEEE conference template, submitted on HotCRP;
- "Submissions will be reviewed single-anonymously";
- IEEE publishes the proceedings;
- dates: notification 11 December, camera-ready 29 January 2027;
- topics that fit: deployment lessons, human-agent oversight, safety, empirical and industrial case studies;
- co-chairs: Qinghua Lu, David Lo, Ahmed E. Hassan and Liming Zhu. Hassan co-authored AIDev and the Claude Code
  pull-request study, so reviewers will know the mining literature.

**Not stated** on the call page: review criteria, artifact rules and an AI-use rule. ICSE 2027 applies the IEEE rule
that AI-generated text "shall be disclosed in the acknowledgements section", with the sections cited. A paper with
"confirmed hallucinated or fabricated references" is desk-rejected. Sharing artifacts is "expected to be the default"
([ICSE 2027 research track](https://conf.researchr.org/track/icse-2027/icse-2027-research-track)). Generative AI may
not be listed as an author.

**What the previous edition accepted.** AGENT 2026 accepted 27 papers
([AGENT 2026](https://conf.researchr.org/home/icse-2026/agent-2026)), among them:
- an explicit experience report on LLM voice agents for requirements elicitation;
- "Decoding the Configuration of AI Coding Agents: Insights from Claude Code Projects";
- a quantified Copilot agent-mode migration study with a replication package.

Acceptance rates were not found. The 2026 deadline was extended, which is no reason to plan on it this year.

| venue | deadline (AoE) | length | why |
|---|---|---|---|
| **AGENT 2027** | **Fri 13 Nov 2026** | 5 pp short / 8 pp full, references included | the best topical fit |
| WSESE 2027 | Fri 27 Nov 2026 | experience/WIP 4-6 pp | the method: studying human-AI cooperation from an agent-written record ([CALL](https://conf.researchr.org/home/icse-2027/wsese-2027)) |
| HUGS 2027 | Fri 13 Nov 2026 | 3 pp | human-controlled GenAI, experience reports ([CALL](https://conf.researchr.org/home/icse-2027/hugs-2027)) |
| SERS 2027 | Fri 13 Nov 2026 | 4 pp, in-person | research software in a lab ([CALL](https://conf.researchr.org/home/icse-2027/sers-2027)) |
| ACM SRC @ ICSE | Fri 13 Nov 2026 | 800 words; the student as sole author | must differ >= 30 % from any other submission ([CALL](https://conf.researchr.org/track/icse-2027/icse-2027-src)) |
| FSE 2027 Industry | Sat 23 Jan 2027 | 5+1 pp short ("experience reports") | lessons and generality; double-blind optional ([CALL](https://conf.researchr.org/track/fse-2027/fse-2027-industry-papers)) |

**Why the short paper.** With 17 days left, the short paper is the right size. The 8-page version belongs to FSE or a
journal, with the prospective data. AGENT and WSESE cannot take the same paper; choose one by 30 October.

## Nobody has reported weeks of agent-built robot software with the hardware in the loop

| work | agent's role | robots, duration | what it lacks for our claim |
|---|---|---|---|
| Project Fetch phase 2 ([CLAIM](https://www.anthropic.com/research/project-fetch-phase-two), Jun 2026) | Claude Code writes sensor and control code | one robot dog, minutes (9 min 35 s vs 181 / 361 min for the human teams); the fetch was not achieved | one day, one robot, a researcher approving commands |
| "Claude plays robotics" ([CLAIM](https://www.anthropic.com/research/claude-plays-robotics), Jul 2026) | the model controls robots at several levels | mostly simulation | the score "depends as much on the robot body and the control interface as on the model" |
| SPINE ([2607.13049](https://arxiv.org/abs/2607.13049)) | agentic debugger for teleoperation bring-up | 2 real bimanual robots, 12 scenarios; 76 -> 100 % vs humans using Claude Code | minutes per scenario; one subsystem |
| OpenRUA ([2610.02459](https://arxiv.org/abs/2610.02459)), RHO ([2606.16458](https://arxiv.org/abs/2606.16458)), EmbodiedSWE ([2609.27308](https://arxiv.org/abs/2609.27308)) | the agent as policy or controller | benchmarks, mostly simulation | not the engineer of a stack |
| ENPIRE ([2606.19980](https://arxiv.org/abs/2606.19980)), Agentic RSR ([2610.10479](https://arxiv.org/abs/2610.10479)) | agents improve policies or infrastructure | real stations or real evaluation | the human role and the faults are not studied |
| IoT-SkillsBench ([2603.19583](https://arxiv.org/abs/2603.19583)) | agents program embedded boards | 378 hardware-validated runs | not robots; no longitudinal work |
| AIDev, Claude Code PRs ([2509.14745](https://arxiv.org/abs/2509.14745)), "Agentic Much?" ([2601.18341](https://arxiv.org/abs/2601.18341)), 180 M-repository census ([2606.24429](https://arxiv.org/abs/2606.24429)), ESEM 2026 ([LIPIcs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESEM.2026.67)) | agents in the wild, mined | thousands of repositories | no physical outcome; cross-sectional |
| METR RCT ([2507.09089](https://arxiv.org/abs/2507.09089)), longitudinal survey ([2605.23135](https://arxiv.org/abs/2605.23135)), Anthropic internal study ([CLAIM](https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic)) | productivity and perception | developers, tasks, surveys | no hardware; METR: 19 % slower while believing 20 % faster |
| Misalignment in 20,574 sessions ([2605.29442](https://arxiv.org/abs/2605.29442)), operational safety failures ([2605.30777](https://arxiv.org/abs/2605.30777), ASE 2026), oversight interviews ([2606.05391](https://arxiv.org/abs/2606.05391)) | failure and oversight taxonomies | software only | the worst outcome is lost data, never a moving arm |

**How the search was done.** It used nine searches on coding agents combined with robots, ROS 2,
hardware-in-the-loop, longitudinal, case study, experience report and "vibe coding", one arXiv-only search, and
Anthropic's two robotics posts. None found a report of a coding agent building a robot stack over weeks with hardware
access, so the paper should say "we found no report of", with the date.

**Five properties together are ours:**
1. weeks of hardware in the loop (the agent runs SSH, deploys, reads logs and measures);
2. two different robots;
3. physical outcomes next to software metrics;
4. one student as the whole human team;
5. a dated record.

The two failure taxonomies should be used as coding schemes, not competed with.

## Our record shows a lab-hours rhythm, a hardware-driven fault profile and a human who chooses and sees

All numbers here are computed read-only from git at HEAD 5187617 and from session metadata. The commands are in the
record note.

| item | value |
|---|---|
| commits | 250 in 47 days (11 Sep - 27 Oct), on 24 weekdays, all between 08:47 and 16:26 local time |
| attribution | 249 with a Co-Authored-By trailer (Opus 5: 2; Fable 5.1: 21; Opus 5.5: 226); one human author |
| before git | agent sessions from 1 September; the first commit imported 15,867 lines |
| size at HEAD | 34.9k lines of code, 8.5k lines of tests, 19.2k lines of Markdown records |
| tests | 237 -> 543 test functions; 50 % of commits touch a test file |
| records | 59 commits (23.6 %) change only Markdown; 69 % of messages carry a measured number |
| areas after the import | M3 113 commits, X3 71, docs 83, console 27, shared 27, train ~20, tools 16 |
| sessions | 68 laptop sessions + 65 subagent transcripts; 25 commit sessions, median 17 min between commits |

**Time to capability**, counted from the M3's day 0 (28 Sep, a copy of the X3 software):
- day 2 (30 Sep): an arena map, a console for both robots, a hand-eye calibration that found the URDF's J1 sign wrong
  (the board spread went from 139 to 8 mm);
- day 3: the first pick on the robot, 18-25 s (11-13 s after a speed round);
- day 14: a kit detector trained on a GPU PC, with Isaac Sim pictures lifting mask mAP50 from 0.653 to 0.791;
- day 24: the off-lab event;
- day 25: an ACT policy run on the robot, stopped by the floor guard;
- day 28: ACT at a true 10 Hz in TensorRT on the robot; the classical picker 8 of 8 into the box at 12-21 s a pick;
- day 29: the learned policy still grasps nothing. The record has a frontier.

The X3 went from a failed first tag-free pick (11 Sep) to 30.9 -> 21.4 s per pick attempt (24 Sep) and 11 of 11 with a
taught aim (15 Oct).

**Faults.** A commit is counted as corrective when it names behaviour that was wrong after an earlier commit or
deployment, and changes it. Only 6 subjects say "fix" and none says "revert", because the agent's style describes the
corrected behaviour. Keyword counts would therefore miss five in six.

| class | robot run | measurement / data | student first | environment | total |
|---|---|---|---|---|---|
| hardware premise (URDF sign, "limp at power-up", lidar bias, sag, the guard's line model) | 1 | 3 | 3 | 0 | 7 |
| integration / runtime (stale pictures, a frozen camera reported at 22 fps) | 6 | 1 | 0 | 0 | 7 |
| perception / data (labels, the box read, the robot's own wheel) | 2 | 4 | 0 | 0 | 6 |
| success judgement (a shoved cube "picked", a policy that never left START) | 2 | 2 | 0 | 0 | 4 |
| behaviour / UX (Slovak voice, misleading messages) | 0 | 0 | 4 | 0 | 4 |
| environment (Windows symlinks, cp1252, a headless display) | 0 | 0 | 0 | 4 | 4 |
| performance / safety margin | 1 | 1 | 0 | 0 | 2 |
| **total (34 of 250 commits, 13.6 %)** | **12** | **11** | **7** | **4** | **34** |

**Six incidents carry the narrative**, each with a commit and a log entry:
1. **The gripper (11 Sep).** The student watched it and named the cause, while the code's comment had called the error
   "harmless".
2. **The J1 sign (30 Sep).** The calibration on the robot found it.
3. **The arm hit the lidar (2 Oct).** The agent's guard used a line model, and "166 poses it allowed put the shell into
   the lidar". A mesh-shell check replaced it.
4. **Camera stalls (13-15 Oct).** The agent built a watchdog first. The cause was another project's model conversion
   running on the robot, and the watchdog restarted the driver 26 times during it.
5. **The policy never left START (26 Oct).** Of the 743 training frames at START, 718 stood still, because of how the
   agent cut the episodes.
6. **The guard stopped the learned policy (23 and 26 Oct).** The floor guard, written for the classical picker, refused
   the learned policy: "16 cm above the floor over the chassis".

**The human in the record.** "The user" appears 50 times in NEXT.md, 17 in PLAN.md, 12 in STATUS.md and 57 in commit
bodies. The student chooses (layouts, the network, which control loop to use, what counts as good enough) and sees (the
collision, the gripper, a stiff arm after a power-cycle). These are the agent's reports of the human; the prospective log
must add the human's own record.

## The paper: three research questions, a single-case design, four tables and two figures

| RQ | question | evidence by 13 Nov |
|---|---|---|
| RQ1 capability and pace | What did one student and an agent with SSH access build on two robots, and how fast? | T1 record at a glance; T2 time to capability with physical numbers and hashes; F1 weekly commits by area with the test line |
| RQ2 faults | Which faults reached a commit or the robot, and what revealed them? | T3 class x revealer, after the student's blind second coding (kappa); F2 the timeline with the six incidents |
| RQ3 human role and safety | What did the student contribute, and how were physical risks contained? | T4 interventions by the seven misalignment forms of 2605.29442, by trigger and minutes, from a 10-day prospective log; guard refusals, the collision, the lock-up |

**Method.** An exploratory single-case study, with the two robots as embedded units:
- a retrospective part, 1 September - 30 October, frozen at a named commit;
- a prospective part, 28 October - 10 November.

**Data sources:**
- D1 git;
- D2 the dated logs (STATUS.md, NEXT.md, PLAN.md);
- D3 session metadata, plus a scrubbed, coded sample of transcript turns;
- D4 robot-side machine logs (pick logs, metrics.jsonl, recorded episodes) as the independent count of physical outcomes;
- D5 a 30-minute structured interview of the student by the supervisor;
- D6 the prospective logs.

**Coding.** The agent does the first pass and the student a blind second pass, and kappa is reported. Faults are coded
with our classes and revealers; interventions with 2605.29442's forms, and the safety events with 2605.30777's
dimensions where they fit.

**The six lessons the data already support:**
1. the robot is the oracle for hardware premises;
2. the student's eyes are a sensor;
3. success must be measured, not reported;
4. guards outlive the controller they were written for;
5. the records are part of the system;
6. monitoring can lie, and the first diagnosis can be wrong.

**The 5 pages:**

| section | length | contents |
|---|---|---|
| introduction | 0.5 p | the setting, the claim, three contributions |
| related work | 0.5 p | |
| design | 0.7 p | |
| results | 1.9 p | |
| lessons | 0.5 p | |
| threats | 0.3 p | |
| conclusion and data availability | 0.15 p | |
| references | 0.45 p | ~18, each re-opened at its source |

## Two weeks of light logging turn the human role from the agent's prose into data

Start these on Wednesday 28 October, at about 10 minutes a day for the student:
- **an intervention log:** one line per redirect, correction, stop, approval, explanation of intent or physical action,
  with its trigger, its form and the minutes it cost;
- **a time sheet:** hours in the lab, at the robot, waiting on the agent and reviewing;
- **a safety log:** every STOP press, guard refusal, contact, near-miss or battery cut-out.

Extract these automatically:
- tokens and cost per session;
- human prompts per session, from transcript metadata. Today's 1,880 plain-text user records are an upper bound,
  because injected notices count too;
- attempts, successes and seconds per day, from the robots' own logs.

**Ethics.**
- The student is a co-author and a subject: write the consent down and add a reflexivity paragraph.
- The agent is acknowledged, not an author. The disclosure names the model versions and the sections it drafted.
- Raw transcripts are not released: they contain host addresses and the login flow. Only the scripts, the coding sheet,
  aggregated metadata and the logs are released.
- Anyone else in the logs is anonymised.
- Ask the supervisor whether the faculty needs an ethics review for a self-study with no other participants.
- State that the authors have no tie to the maker of the agent they study, if that is true.

## Threats to validity decide acceptance; answer them in the design, not in the limitations

| threat | answer |
|---|---|
| the agent wrote the code, the messages and the logs it is judged by; failures that never became a commit are invisible | every number from machine-written metadata or robot logs; the student's interview; the blind second coding; say plainly what cannot be seen |
| no counterfactual | no speed-up claim; a descriptive comparison with Project Fetch's human teams; a pre-registered replication (a second student, one capability, with and without the agent) as future work for FSE or a journal |
| three model versions and an evolving harness (memory files, CLAUDE.md, subagents, a second agent on the GPU PC) | report versions by date; treat the harness as part of the case |
| commits as units | use them for dates and counts only |
| the first 10 days before git | the size of the import bounds them |
| n = 1 student, low-cost robots | full context; claim transfer only to similar labs |
| small physical counts | counts with Wilson intervals, never pooled rates |

## The plan to 13 November

| task | by | done when |
|---|---|---|
| start the three logs; HotCRP accounts for the authors | Wed 28 Oct | the first rows are in the CSV |
| go/no-go with the supervisor; venue and authors fixed | Thu 29 Oct | the decision is in NEXT.md |
| freeze the retrospective data; codebook v1 | Fri 30 Oct | the freeze hash is recorded |
| blind second coding of the 34 corrective commits and a re-read of all 250 subjects | Tue 3 Nov | kappa and the disagreements listed |
| cross-check T2 against the robots' logs; script F1 and F2 | Wed 4 Nov | each T2 row has a log reference or is marked "agent-reported" |
| draft v1 in the IEEE template | Sat 7 Nov | <= 5 pages including references |
| the student's interview | Mon 9 Nov | the 8 answers are written down |
| logs closed, T4 in; supervisor review | Tue 10 Nov | the comments are received |
| revision, references re-verified, AI disclosure, page check, submit | Thu 12 Nov | submitted on HotCRP |
| buffer | Fri 13 Nov AoE (Sat 14 Nov 12:59 CET) | -- |

**Fallbacks:**
- no decision by Mon 2 Nov: WSESE on 27 Nov, as a 4-6 page method paper;
- logs too thin by 10 Nov: FSE Industry on 23 Jan 2027, with two months of prospective data.

## Risks

- **It competes with P1 for the same two weeks.** P1's go/no-go is also 13 November, and its RSS abstract is due 4
  December. The meeting on 29 October has to set the priority. P5 costs the student ~15-20 hours; P1 costs robot time.
- **The supervisor may value it less** than the robotics papers (RESEARCH.md said so on 5 Oct). A rejection costs
  little, and the January FSE paper keeps the idea alive.
- **It may read as a product testimonial.** Mitigation: report the faults first, the frontier (a policy that grasps
  nothing) and the collision; give no speed-up numbers; disclose the conflict of interest.
- **The single rater is the subject.** Without the student's second coding, T3 is the agent grading itself; do not
  submit T3 uncoded.
- **References.** An unverifiable reference means a desk rejection; every one must be re-opened at its source.
- **The date.** RESEARCH.md still says 27 November. Correct it there (the user's call), so that no plan is built on it.

## Conclusion

P5 is publishable as an AGENT'27 short experience paper, and the evidence for it already exists:
- 250 dated, co-authored commits;
- a two-robot capability timeline with physical numbers;
- 34 coded corrective commits: the robot, measurement and the student revealed 30, the environment 4, the tests none;
- six documented incidents, including a collision and guards that later stopped a learned policy.

No report of a coding agent building a robot stack over weeks with hardware in the loop was found.

**What is missing:** the human's own record and a second rater. Two weeks of light logging and one interview supply
both.

**The decision:** take it on Thursday 29 October, against P1's claim on the same fortnight, and remember that the
deadline is 13 November AoE. If the answer is no or late, WSESE (27 Nov) and FSE Industry (23 Jan 2027) keep the paper
alive with better data.

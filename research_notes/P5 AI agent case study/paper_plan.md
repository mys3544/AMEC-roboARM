# P5 paper plan: research questions, method, results we can show by 13 November, threats, data, ethics, outline

Scope: a design for a 5-page AGENT 2027 short experience paper (IEEE, references included, single-anonymous; call in
`call_and_venues.md`), built on the numbers in `our_record_in_numbers.md` and positioned by `related_work.md`. Also what to
collect in the remaining days, the ethics, and a dated plan with fallbacks (WSESE 27 Nov, FSE Industry 23 Jan 2027).
Written by the agent the paper is about; the student and the supervisor should weigh it as such.

## 1. The claim and three research questions

### Takeaway
Claim to test: **with hardware in the loop, faults are found by the robot, by measurement and by the student's eyes --
not by the test suite -- and the human's role shifts from writing code to choosing, observing and stopping.** Three RQs
carry it and fit 5 pages; each has a table that already exists in draft.

| RQ | question | measures | data | exists today? |
|---|---|---|---|---|
| RQ1 capability and pace | What did one student and a coding agent with SSH access build on two robots, and how fast? | time to each capability; commits, lines, tests per week; physical outcomes (picks, mm, s) | git; STATUS/NEXT; robot logs | yes (tables in the record note, sections 1-4) |
| RQ2 faults | Which faults reached a commit or the robot, of what kind, and what revealed them? | corrective commits by class x revealer; narrated incidents | git messages; NEXT/STATUS; second rater | first pass yes (34); second rater no |
| RQ3 human role and safety | What did the student contribute that the agent did not, and how were physical risks contained? | interventions by form (2605.29442), by trigger and cost; guard refusals, collisions, near-misses | prospective log; transcripts sample; NEXT/STATUS | partly (record mentions); prospective log no |

## 2. Method: a single-case, longitudinal study with two embedded units (the robots)

### Takeaway
Frame it as an **exploratory single-case study** (one student-agent pair, two robots as embedded units) with a
**retrospective** part (1 Sep - 30 Oct, frozen at a commit) and a short **prospective** part (28 Oct - 10 Nov), and
triangulate the agent-written prose against machine-written evidence (git metadata, robot logs, session metadata) and
the student's own account. Code faults and interventions with published schemes so the categories are not ours alone.

### Design details
- **Case and context:** two Yahboom robots (X3 Plus, M3 Pro; Jetson Orin NX; ROS 2 Humble), a Windows laptop, two GPU
  PCs; Claude Code (Opus 5 -> Fable 5.1 -> Opus 5.5 by the trailers) with SSH to the robots and PCs, persistent memory
  files and a project CLAUDE.md; the student by the robot; a supervisor not in the loop day to day.
- **Data sources:**
  - D1 git history (250 commits; trailers; numstat; test counts per commit) -- machine-written metadata, agent-written
    messages.
  - D2 the dated logs: x3plus/docs/STATUS.md (11 Sep - 23 Oct), NEXT.md (since 9 Oct), PLAN.md, RESEARCH.md --
    agent-written, student-reviewed.
  - D3 Claude Code session metadata (68 sessions + 65 subagent transcripts since 1 Sep) and a coded sample of transcript
    turns (scrubbed; section 6).
  - D4 robot-side machine logs: pick logs, `metrics.jsonl`, recorded episodes -- independent counts of attempts,
    successes and seconds.
  - D5 the student's account: a 30-minute structured interview by the supervisor (or a written reflection) on 8 fixed
    questions.
  - D6 prospective logs (section 5): time, interventions, safety events.
- **Coding:** corrective commits by class and revealer (R robot run, M measurement/data, H student, E environment;
  codebook in the record note); interventions by the seven misalignment forms of
  [2605.29442](https://arxiv.org/abs/2605.29442) plus cost; safety events with the dimensions of
  [2605.30777](https://arxiv.org/abs/2605.30777) where they apply. First pass by the agent, second by the student
  without seeing the first; Cohen's kappa reported; disagreements settled with the supervisor.
- **Freeze:** the retrospective data at one commit (e.g., the last commit of Fri 30 Oct), named in the paper.

## 3. Results we can make by 13 November

### Takeaway
Four tables and two figures, all from scripts that already run: T1 the record at a glance, T2 time to capability on
both robots, T3 faults by class and revealer (after the second rater), T4 the human role from two weeks of prospective
logging; F1 commits per week by area with the test count, F2 a two-lane timeline (X3, M3) of milestones and incidents.

| item | content | source | effort |
|---|---|---|---|
| T1 | 250 commits, 47 days, 249 trailers, 34.9k code / 8.5k tests / 19.2k Markdown lines, 68 sessions, 543 tests | D1, D3 | done |
| T2 | M3 day 0 -> day 3 first pick -> day 25 policy on the robot; X3 milestones; each with a physical number and a hash | D1, D2, D4 | done; D4 cross-check 2 h |
| T3 | 34 corrective commits: hardware premise 7, integration 7, perception/data 6, success judgement 4, UX 4, environment 4, margin 2; revealed by robot 12, measurement 11, student 7, environment 4 | D1, D2 | second rater 2-3 h |
| T4 | interventions per lab day, by form and trigger; guard refusals; minutes lost | D6, D3 sample | 10 lab days of logging |
| F1 | stacked weekly commits by area (X3, M3, console, shared, train, docs) + tests line | D1 | script, 1 h |
| F2 | timeline with the six incidents marked | D1, D2 | 2 h |

### Lessons the data already support (to be checked against the student's account)
1. **The robot is the oracle for hardware premises.** Vendor URDF signs, "limp at power-up", lidar range bias, joint sag:
   7 wrong premises, none catchable by unit tests.
2. **The student's eyes are a sensor.** 7 of 34 faults were seen first by the student (a collision, a gripper, a stiff arm).
3. **Success must be measured, not reported.** 4 faults were false successes (a shoved cube "picked", a policy that never
   moved); the fix was a check against the robot's own looks.
4. **Guards outlive the controller they were written for.** The floor and chassis guards written for the classical
   picker stopped the learned policy.
5. **Records are part of the system.** 69 % of commit messages carry a measured number; 23.6 % of commits are
   records only; NEXT.md is the hand-over between sessions.
6. **Monitoring can lie and the first diagnosis can be wrong** (/health at 22 fps on a frozen camera; a watchdog before the
   cause).

## 4. Threats to validity

### Takeaway
The two that decide acceptance: **the agent wrote most of the record it is judged by**, and **there is no
counterfactual**. Mitigate the first with machine-written evidence, a second rater and the student's own account; do
not claim a speed-up -- report pace and outcomes, and compare only descriptively with published human-team numbers.

| threat | type | mitigation in the paper |
|---|---|---|
| the agent authored code, commit messages and logs (self-report, selective narration; failures that never became a commit are invisible) | internal / construct | machine-written metadata and robot logs for every number; the student's account; state that unlogged failures are missed |
| one rater, who is the studied agent | reliability | the student re-codes blind; kappa; supervisor resolves |
| no baseline: what would the student have done alone? | conclusion | no speed-up claim; descriptive comparison with Project Fetch's human teams; a pre-registered replication as future work |
| three model versions and an evolving harness (memory files, CLAUDE.md, subagents) | internal | report versions by date (trailers); treat the harness as part of the case |
| commits as units: their size and timing are chosen by the agent | construct | use commits only for counts and dates; outcomes from robot logs |
| the 10 days before the first commit (1-10 Sep) and the GPU PC's own sessions are missing | internal | say so; the import's size (15,867 lines) bounds what came before |
| one student, one lab, one agent product, two low-cost robots with vendor ROS 2 stacks | external | describe the context fully; claim transfer only to similar lab settings |
| small physical counts (8 of 8, 32 of 34) | conclusion | report counts with Wilson intervals; no rates over pooled conditions |
| the authors are the subjects; the drafter is the subject | researcher bias | reflexivity statement; AI-use disclosure; scripts and coding sheet released |

## 5. Extra data to collect from Wed 28 October (two weeks, ~10 lab days)

### Takeaway
Three light logs kept by the student (about 10 minutes a day) and three automatic extractions. Without the intervention
log, RQ3 rests only on the agent's prose; with it, T4 is the paper's most original table.

- **Intervention log** (one line per event, kept by the student, a shared CSV): time; session; what the student did
  (redirect, correct, stop, approve, explain intent, physical action at the robot); the trigger (robot behaviour,
  a number, the agent's message, own idea); the form (2605.29442's seven); minutes it cost.
- **Time sheet** (per lab day): hours in the lab, hours at the robot, hours waiting on the agent, hours reviewing.
- **Safety log:** every STOP press, guard refusal, contact or near-miss, battery cut-out, with time and what moved.
- **Automatic:** tokens and cost per session (Claude Code's usage display at the session's end, or an export); human
  prompts per session from transcript metadata; attempts, successes and seconds per day from the robots' pick logs and
  episodes.
- **Retrospective (once):** the student's 30-minute interview -- what would not exist without the agent; where the agent
  was wrong and how you knew; when you stopped trusting a report; what you learned about the robots yourself; what you
  would not let it do; what the supervisor should know.

## 6. Ethics, consent, authorship and disclosure

### Takeaway
The student is co-author **and** subject: consent is implicit but should be written down, with a reflexivity paragraph.
The agent cannot be an author (ACM and IEEE rules); its use is disclosed in the acknowledgments with the sections it
drafted. Raw transcripts are not released: they contain host addresses and the login flow. Ask the supervisor whether
the faculty requires an ethics review for a self-study with no other participants.

- Authorship: the student first, the supervisor; the agent in the acknowledgments ("Claude Code, Claude Opus 5 / Fable
  5.1 / Opus 5.5, Anthropic, used for code, commit messages, logs, data analysis and drafting of sections X-Y; all
  numbers were computed by released scripts and checked by the authors"). IEEE: "The use of AI-generated text ... shall be
  disclosed in the acknowledgements section", with the sections cited ([ICSE 2027 research track quoting IEEE](https://conf.researchr.org/track/icse-2027/icse-2027-research-track)).
- Conflict of interest: the study evaluates a commercial product, and the drafting agent is that product; state that the
  authors have no affiliation with or funding from its maker (if true).
- Other people: anyone else named in logs or transcripts (lab members, other projects' software on the robots) is
  anonymised; no third-party data.
- Data release: the git history (if the repository is or becomes public), the coding sheet, the scripts, aggregated session
  metadata and the prospective logs; transcripts only as scrubbed excerpts of the student's own words.
- References: every reference re-opened at its source before submission ("confirmed hallucinated or fabricated
  references will be desk-rejected").

## 7. Outline of the 5 pages (IEEE two-column, references included)

| section | ~length | content |
|---|---|---|
| 1 Introduction | 0.5 p | the setting in one paragraph; the claim; 3 contributions (a dated case, a fault/revealer table with hardware in the loop, lessons for agent-assisted robot bring-up) |
| 2 Related work | 0.5 p | robot agents (Project Fetch, SPINE, OpenRUA, ENPIRE, EmbodiedSWE); agents in the wild (AIDev, 2509.14745, census, ESEM 2026, METR); failures and oversight (2605.29442, 2605.30777, 2606.05391) |
| 3 Study design | 0.7 p | case, harness, data sources D1-D6, coding and kappa, the freeze, disclosure |
| 4 Results | 1.9 p | RQ1: T1, T2, F1; RQ2: T3 and two incidents; RQ3: T4, the safety events |
| 5 Lessons | 0.5 p | the six lessons, each tied to a number |
| 6 Threats | 0.3 p | the table of section 4, compressed |
| 7 Conclusion, data availability | 0.15 p | what to do differently; the artifact link |
| References | 0.45 p | ~18 references, each verified |

## 8. The plan to Friday 13 November (AoE = Sat 14 Nov 12:59 CET)

### Takeaway
Feasible only with a decision on Thursday 29 October and ~15-20 hours of the student's time spread over two weeks; the
agent drafts and computes. Submit on Thursday 12 November, a day early. If the decision slips past 2 November, switch to
WSESE (27 Nov, method framing) or FSE Industry (23 Jan 2027) and keep logging.

| task | by | done when |
|---|---|---|
| start the three logs; HotCRP account for each author | Wed 28 Oct | the CSV has its first rows; accounts exist |
| go/no-go with the supervisor; venue and authors fixed | Thu 29 Oct | a line in NEXT.md with the decision |
| freeze the retrospective data; codebook v1 | Fri 30 Oct | the freeze hash written down; codebook in research_notes |
| student's blind second coding of the 34 + a re-read of all 250 subjects for missed fixes | Tue 3 Nov | kappa computed; disagreements listed |
| robot-log cross-check of T2's physical numbers; F1, F2 scripted | Wed 4 Nov | every T2 row has a log reference or is marked "agent-reported" |
| draft v1, all sections, 5 pages | Sat 7 Nov | PDF in the IEEE template, <= 5 pages with references |
| the student's interview / reflection | Mon 9 Nov | notes with the 8 answers |
| prospective logs closed; T4 in; supervisor review | Tue 10 Nov | T4 filled; review comments received |
| revision; references re-verified; AI disclosure; page check | Thu 12 Nov | submitted on HotCRP |
| buffer | Fri 13 Nov (AoE) | -- |

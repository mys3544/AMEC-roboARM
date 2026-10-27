# Our record in numbers: what the git history, the logs and the session metadata can show

Scope: numbers computed on 27 October 2026 from this repository, read-only, at HEAD 5187617 (branch m3pro, which contains
main): `git log`, `git ls-tree`, `git cat-file` only; NEXT.md, PLAN.md, x3plus/docs/STATUS.md read as text; and, for
the session counts only, the **metadata** of the Claude Code transcripts on this laptop (timestamps, record types and
line counts; no message content was read or printed). Scripts were kept outside the repo (the session scratchpad); the
commands are in section 9 so anyone can re-run them. No e-mail address was read: the log formats use author names and
cut trailer values at "<". The corrective-commit coding (section 6) is a **first pass by one rater, the agent under
study**; it needs the student as a second rater before it goes into a paper.

Label: everything here is [OURS] unless marked.

## 1. The record at a glance

### Takeaway
250 commits over 47 calendar days (11 Sep - 27 Oct 2026) on 24 commit days; one human git author; **249 of 250 commits
carry a Co-Authored-By: Claude trailer** (the exception is a merge). The code base at HEAD is ~34.9k lines of code,
~8.5k lines of tests (543 test functions) and ~19.2k lines of Markdown records. The first commit imported 15,867 lines
written in the ten days before it: the agent's sessions on this project began on **1 September**, so the git record
misses the first 10 days.

| item | value | how |
|---|---|---|
| commits | 250 (3 merges) | `git rev-list --count HEAD` |
| span | 2026-09-11 14:20 -> 2026-10-27 15:11 (47 days), 24 days with commits | log dates |
| git authors | 1 (the student) | `%an` |
| Co-Authored-By trailer | 249 (99.6 %): Claude Opus 5 (2, 11 Sep), Claude Fable 5.1 (21, 14-21 Sep), Claude Opus 5.5 (226, 24 Sep - 27 Oct) | `%(trailers)` |
| initial import (694cf56) | 87 files, +15,867 lines | `git show --stat` |
| lines after the import | +74,818 / -25,896 (code +52,957 / -23,897; .md +19,797 / -962; uv.lock excluded) | numstat |
| size at HEAD | code 34,874; tests 8,529; Markdown 19,248 lines | `git ls-tree` + `cat-file` |
| tests | 237 (11 Sep) -> 543 (27 Oct) `def test_` in 8 -> 43 files | per-commit tree scan |
| commits touching a test file | 125 (50 %) | numstat |
| docs-only commits (.md only) | 59 (23.6 %); NEXT.md touched by 65 commits since 9 Oct | numstat |
| commit message body | median 94 words, total 25,115 words | `%b` |
| messages with a measured number (unit or "N of M") | 173 (69 %); in the subject alone 60 (24 %) | regex |
| messages naming a robot run or deployment | 78 (31 %) | regex "deployed", "on the robot/M3/X3" |
| Claude Code sessions (laptop, metadata) | 68 main + 65 subagent transcripts, first 2026-09-01 11:53 UTC | transcript timestamps |
| user records with plain-text content | 1,880 (an upper bound on typed prompts: injected notices also count) | record type only |

## 2. Pace: a lab-hours rhythm, one session a day, a commit every ~17 minutes

### Takeaway
Commits fall only on weekdays and only between 08:47 and 16:26 local time: the agent works when the student is in the lab
(the robots need a person beside them). 25 commit sessions (gap > 3 h), a median of 9 commits each, a median gap of 17
minutes between consecutive commits; the summed first-to-last spans are 93 h, a lower bound of active time. The pace
rose when the second robot arrived (W40) and peaked in W41 (69 commits in 5 days).

| ISO week (Mon) | commits | days | X3 | M3 | code + / - | .md + | tests at week end |
|---|---|---|---|---|---|---|---|
| W37 (7 Sep) | 2 | 1 | 1 | 0 | import only | +8 | 237 |
| W38 (14 Sep) | 11 | 2 | 9 | 0 | +1,113 / -2,353 | +251 | 247 |
| W39 (21 Sep) | 12 | 2 | 9 | 0 | +3,483 / -506 | +676 | 306 |
| W40 (28 Sep) | 29 | 4 | 0 | 27 | +11,839 / -1,237 | +313 | 694* |
| W41 (5 Oct) | 69 | 5 | 42 | 30 | +9,046 / -3,793 | +2,929 | 462 |
| W42 (12 Oct) | 59 | 3 | 5 | 25 | +5,032 / -447 | +631 | 491 |
| W43 (19 Oct) | 39 | 5 | 1 | 12 | +3,952 / -228 | +2,218 | 530 |
| W44 (26 Oct) | 29 | 2 | 0 | 19 | +3,376 / -214 | +12,656 | 543 |

\* W40's tests include the X3 copy the M3 started from (133455a, +15,866 lines, 28 Sep), removed on 5 Oct (7885fd8,
-15,727); both commits are left out of the code columns. Commits per weekday: Mon 73, Tue 34, Wed 36, Thu 49, Fri 58.
Local commit hours: 08 1, 09 16, 10 36, 11 34, 12 22, 13 32, 14 52, 15 42, 16 15.

Session metadata (laptop transcripts): sessions start on 27 distinct days from 1 Sep; 48 of 68 have >= 5 plain-text user
records; median 14 per session, max 482 (one session spanning 13-20 Oct). The GPU PC's Remote Control sessions are not
on this laptop and are not counted.

## 3. Where the work went: the M3 is half the history; a quarter of the commits are records

### Takeaway
After the import, the M3 (`m3pro/`) took 113 commits, the X3 71, the shared console 27, `shared/` 27, training 18-20,
tools 16; 83 commits touched the Markdown records (NEXT.md, PLAN.md, RESEARCH.md, reports, notes). Code is written once
and then cut: two "audit" commits removed 2,100 lines and 6 dependencies (e710ca6, 18 Sep) and the X3's leftovers in
the M3 (1d5ab8f, 7885fd8, 5 Oct).

| area | commits touching | + lines | - lines |
|---|---|---|---|
| m3pro/ | 113 | 36,902 | 18,919 (incl. the copy and its removal) |
| docs (*.md at root, reports/, research_notes/) | 83 | 17,255 | 721 |
| x3plus/ (and its pre-6-Oct root folders) | 71 | 11,170 | 3,312 |
| console/ | 27 | 5,039 | 545 |
| shared/ | 27 | 1,635 | 91 |
| train/ | 18-20 (binary-only commits) | 1,536 | 103 |
| tools/ | 16 | 1,211 | 2,196 |

Size at HEAD (lines): x3plus code 14,407 + tests 4,427; m3pro 13,688 + 2,959; console 3,601 + 615; shared 1,361 + 528;
train 1,353; tools 464; Markdown 19,248 (16,757 at the root/reports/notes).

## 4. Time to capability: the M3 went from a new folder (seeded with the X3's software) to a learned policy on the robot in 25 days

### Takeaway
Dated by commit. M3 day 0 = 28 Sep (its folder, a copy of the X3 software). The first pick on the M3 came on **day 3**,
an arena map and a console for both robots on day 2, a detector trained with Isaac Sim pictures on day 14, a public demo
on day 24 and an ACT policy run on the robot on **day 25** (stopped by the agent's own floor guard). On day 29 the
learned policy still grasps nothing: the record has a frontier, not just successes.

| robot | date (day) | capability, with its physical evidence | commit |
|---|---|---|---|
| X3 | 11 Sep | first tag-free pick attempt **failed**: the fingers still closing on the descent swept the cube (the student named the cause) | STATUS.md 11 Sep |
| X3 | 14 Sep | first untagged pick, 21.0 s; YOLOE-26 rung | 25ca995 |
| X3 | 21 Sep | "clear the table" button run 3 for 3; depth rung | 1008fc8 |
| X3 | 24 Sep | per pick attempt 30.9 -> 21.4 s (-31 %) after timing every stage | 1f94700 |
| M3 | 28 Sep (0) | own folder; started from a copy of the X3 software | 7ee8ed4, 133455a |
| M3 | 30 Sep (2) | arena map (slam_toolbox), lidars found to read long; console for both robots; RTAB-Map 3D scan; hand-eye: J1 sign wrong in the URDF, board spread 139 -> 8 mm | 5c2cc6b, 1867ffb, f1ee8dc, 597bdf0 |
| M3 | 1 Oct (3) | first pick on the robot, 18-25 s (was ~45); speed round 11-13 s a pick; voice in the console | 4f98792, c1bc2d5, 334c632 |
| M3 | 2 Oct (4) | into a box found by its marker; after the arm hit the lidar, a mesh-shell guard; 5 into a moved box | 79f007f, 65afa52 |
| both | 7 Oct (9) | one uv workspace; Docker retired; shared task package; both robots on systemd | 43875ca, eec0d52, 613dfa2, 1e99492 |
| M3 | 9 Oct (11) | object scan; teach by showing: YOLOE prompts 1 of 4, DINOv2 4 of 4, "pick fish" | b0c415d, ee95a73 |
| both | 12 Oct (14) | kit detector trained on the GPU PC; Isaac Sim synthetic pictures 0.653 -> 0.791 mask mAP50; Gaussian splat in the console | af1063a, 1ebcce4, 9e5b832 |
| X3 | 15 Oct | first clean-up with the kit detector, 11 picked, 2 misses, 2 min 48 s; taught aim 11 of 11 | f5a27ca, d2734b6 |
| M3 | 16 Oct (18) | the box read fixed: 5 drops, none "moved" | d0f3f2c |
| both | 20-22 Oct (22-24) | event page, local-LLM voice commands, sorting; the off-lab event on a phone hotspot | b1a073c, fb7e7e3, d20f94c, d5ca776 |
| M3 | 23 Oct (25) | 40 mm cube 32 of 34; jobs recorded as episodes; ACT trained; Isaac Sim twin 7 of 7; ACT on the robot, stopped by the floor guard | 54c11ae, 9edb03b, 7af53fc, 0a659f0 |
| M3 | 26 Oct (28) | ACT in TensorRT on the robot at a true 10 Hz; twin gap 2.5 mm median / 7.4 max; SAG refit; 8 of 8 into the box, 12-21 s a pick | 8befd33, 6bf1611, 45b1bb9, 51c17c7 |
| M3 | 27 Oct (29) | co-training on 74 sim + 80 real episodes; the real-only policy "one way, nothing grasped"; 25 picked by the classical picker | 49cbc7b, 7adf23e, 2473ec1 |

## 5. The visible human: decisions and observations written into the record

### Takeaway
The student appears in the agent's write-ups as the one who **chooses** (layouts, the network, which loop to run, what
counts as "fine") and who **sees** (a collision, a gripper, a stiff arm after a power-cycle). "The user" occurs 50 times in
NEXT.md, 17 in PLAN.md, 12 in STATUS.md and 57 times in commit bodies. These are the agent's reports of the human, not the
human's own record -- the gap the prospective log must close.

### Evidence
- "The user watched it happen and named the cause: the fingers were still half closed as the arm came down" (STATUS.md,
  11 Sep); the code's own comment had judged the tool-length error "harmless. It is not."
- "The arm had hit the front-right lidar (the user)" (65afa52, 2 Oct): the guard checked a line through the joints;
  Yahboom's meshes reach 5 cm past it and "166 poses it allowed put the shell into the lidar".
- "The docstrings had it ... that the arm is limp at power-up ... The user power-cycled it ...: it came up stiff"
  (52e1bb8, 5 Oct).
- Choices: "the user chose it over sending without waiting from the laptop" (NEXT.md, 26 Oct); "the user: local, on the
  laptop" for the voice LLM; "the user: gets the job done" for the M3's aim.

## 6. Faults that reached a commit or the robot: 34 corrective commits; the robot and measurement revealed most

### Takeaway
34 of 250 commits (13.6 %) fix a fault in behaviour already committed or deployed. By what revealed it: a run on the robot
12, the agent's own measurement or data check 11, the **student first** 7, the environment (Windows, encodings, display)
4; no corrective commit names the unit tests as the revealer. The largest classes are **wrong premises about the
hardware** (7) and integration/runtime faults (7). Only 6 subjects say "fix"; none says "revert" -- the agent's style
describes the corrected behaviour, so keyword counts under-count fixes by ~6x.

| class | robot run | measurement / data | student | environment | total |
|---|---|---|---|---|---|
| hardware premise | 1 | 3 | 3 | 0 | 7 |
| integration / runtime | 6 | 1 | 0 | 0 | 7 |
| perception / data | 2 | 4 | 0 | 0 | 6 |
| success judgement | 2 | 2 | 0 | 0 | 4 |
| behaviour / UX | 0 | 0 | 4 | 0 | 4 |
| environment | 0 | 0 | 0 | 4 | 4 |
| performance / safety margin | 1 | 1 | 0 | 0 | 2 |
| total | 12 | 11 | 7 | 4 | 34 |

Rule: a commit is corrective when its subject or body names behaviour that was wrong after an earlier commit or
deployment, and changes it. Tuning that follows a new measurement without a prior fault is not counted. The 34 hashes and
their codes are in the script in section 9. 10 fall before 12 Oct, 24 after (more code on the robots, more deployed).

### Six incidents worth narrating (each has a commit and a log entry)
1. **Gripper timing (11 Sep, X3):** the descent and the fingers moved in one command; the cube was swept aside. Caught
   by the student watching; fixed by opening first (STATUS.md 11 Sep; confirmed on 14 Sep).
2. **J1 sign (30 Sep, M3):** Yahboom's URDF axis is opposite to the servo; found by the agent's hand-eye calibration on
   the robot (board spread 139 -> 8 mm) (597bdf0).
3. **Collision with the lidar (2 Oct, M3):** the agent's own guard used a line model; the arm's shell hit the lidar. The
   student saw it; a mesh-shell check against a 5 mm height map replaced the line, the cost a strip of the workspace
   (65afa52; refined 5 and 21 Oct).
4. **Camera stalls (13-15 Oct, M3):** /health reported 22 fps while the camera was frozen; the agent first built a
   watchdog, then found the cause -- another project's model conversion running on the robot; the watchdog restarted the
   driver 26 times during it and never in the 1.5 h after (bdbbc83; NEXT.md section 0).
5. **The policy that never left START (26 Oct, M3):** a perfect 10 Hz loop, an arm that stayed put; the cause was the
   agent's episode cutting (718 of 743 START frames standing still); recut, the next checkpoint left START at once
   (bda9931; NEXT.md 26 Oct).
6. **The guard refused the learned policy (23 and 26 Oct, M3):** the agent-written floor guard stopped the ACT run
   ("16 cm above the floor over the chassis", after 3.0 s; 0a659f0, NEXT.md) -- the safety layer written for the
   classical picker held for a learned controller.

## 7. Safety: one collision with hardware, guards that held, rules for a shared GPU

### Takeaway
The record has **one physical collision** (the arm's shell against the M3's lidar, 2 Oct), **one board lock-up** after a
failed Teach turned the arm towards a tyre (15 Oct; power-cycle needed), and **repeated guard refusals** of learned and
taught motions that are recorded as working as intended. No injury or damage beyond these is recorded. The agent's
standing rules (CLAUDE.md) forbid stopping processes it did not start on the shared GPU PCs, launching Isaac Sim over a
dying SSH session, and GPU-heavy work without checking free memory.

### Evidence
- "refus" occurs 16 times in NEXT.md and 7 in STATUS.md; examples: "refused 7 mm over the floor" (the learned grip, NEXT.md
  27 Oct); "Batteries: the M3 refuses under 10.5 V" (NEXT.md).
- NEXT.md section 0: "The M3's board went silent at ~12:33 ... after a failed Teach had turned the arm towards the left
  front tyre ... only switching the robot off and on did".
- CLAUDE.md: "Do not stop, kill or restart anything there that you did not start. Ask before starting anything GPU-heavy
  when less than ~16 GB VRAM is free".

## 8. What the record cannot show

### Takeaway
No time log (only commit times and session spans), no token or cost record, no count of interventions that did not
become a commit, nothing from 1-10 September except the import, nothing from the GPU PC's own agent sessions, and no
baseline (the same work without the agent). Most prose is the agent's own; the robot-side machine logs (pick logs,
metrics.jsonl, recorded episodes) are the independent evidence for physical outcomes and are on the robots, not in git.

## 9. Commands (read-only; run from the repository root)

```bash
git rev-list --count HEAD                                         # 250
git log --reverse --format='%h %ad' --date=iso | sed -n '1p;$p'   # span
git log --format='%(trailers:key=Co-Authored-By,valueonly)' | sed 's/ *<.*//' | grep . | sort | uniq -c
git log --format=%ad --date=short | sort | uniq -c                # commits per day
git log --format=%ad --date=format:%H | sort | uniq -c            # local hour of commit
git log --format=%ad --date=format:%a | sort | uniq -c            # weekday
git show --stat 694cf56 | tail -1                                 # the import
git log --merges --oneline                                        # 3 merges
git log --format=%s | grep -ciE '\bfix(es|ed)?\b'                 # 6
git log --format=%s | grep -ci 'revert'                           # 0
for f in NEXT.md PLAN.md x3plus/docs/STATUS.md; do grep -oi 'the user' "$f" | wc -l; done
git log --format=%B | grep -oi 'the user' | wc -l                 # 57
```

Areas and tests over time (Python 3.12, standard library only; `python -I repro.py` from the repo root):

```python
import collections, re, subprocess
def git(*a, inp=None):
    return subprocess.run(["git", "-c", "core.quotepath=off", *a], input=inp, capture_output=True,
                          check=True).stdout.decode("utf-8", "replace")
X3_OLD = ("roboarm/", "tests/", "docs/", "vision_service/", "docker/", "cameras/", "compose.yaml", ".dockerignore")
def area(p):
    if p.startswith("x3plus/") or p.startswith(X3_OLD): return "x3plus"
    for a in ("m3pro", "console", "shared", "train", "tools"):
        if p.startswith(a + "/"): return a
    return "docs" if p.startswith(("reports/", "research_notes/")) or (p.endswith(".md") and "/" not in p) else "config"
def dest(p):  # numstat rename forms: a => b, x/{a => b}/y
    m = re.match(r"(.*)\{(.*) => (.*)\}(.*)", p)
    return (m.group(1) + m.group(3) + m.group(4)).replace("//", "/") if m else p.split(" => ")[-1]
log = git("log", "--reverse", "-M", "--numstat", "--format=@%h")
agg, cur, first = collections.defaultdict(lambda: [set(), 0, 0]), None, None
for line in log.splitlines():
    if line.startswith("@"): cur = line[1:]; first = first or cur; continue
    parts = line.split("\t")
    if len(parts) < 3 or cur == first or parts[0] == "-": continue
    p = dest(parts[2].strip('"'))
    if p.endswith(".lock"): continue
    a = agg[area(p)]; a[0].add(cur); a[1] += int(parts[0]); a[2] += int(parts[1])
for k, (c, ad, de) in sorted(agg.items(), key=lambda x: -x[1][1]): print(f"{k:8s} {len(c):4d} commits +{ad} -{de}")
days = {}
for line in git("log", "--reverse", "--format=%h %ad", "--date=short").splitlines():
    h, d = line.split(); days[d] = h
for d, h in days.items():
    blobs = [l.split()[2] for l in git("ls-tree", "-r", h).splitlines()
             if re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", l.split("\t", 1)[1])]
    txt = git("cat-file", "--batch", inp=("\n".join(blobs) + "\n").encode()) if blobs else ""
    print(d, len(re.findall(r"^\s*(?:async\s+)?def test_", txt, re.M)), "tests in", len(blobs), "files")
```

The corrective-commit codes (class, revealer R/M/H/E, note), first pass by the agent:

```text
hardware premise:    64ce33a H, 597bdf0 M, 1867ffb M, 6a1b7f2 R, 65afa52 H, 52e1bb8 H, 45b1bb9 M
integration/runtime: 45dd256 R, 2e3e980 R, ff04e96 R, 96adb20 R, 0bbd7c5 R, bdbbc83 M, 8aeced0 R
perception/data:     a1f3aec M, 2cf84bc R, 9bd710c M, 0f47deb R, bff14f4 M, ceb7772 M
success judgement:   dd28e30 M, bda9931 R, a60efaf M, 75b7049 R
behaviour/UX:        34af47e H, b12bc4d H, 0fab604 H, 114791a H
environment:         e4c68a6 E, 94daf9c E, 96c26c2 E, a527ac7 E
performance/margin:  9183510 M, 7adf23e R
```

Session metadata: for each `~/.claude/projects/<this project>/*.jsonl`, the first and last `"timestamp"` value, the
count of records whose `"type"` is user or assistant, and of user records whose content is a plain string; subagent
transcripts counted by file (`*/subagents/*.jsonl`). Nothing else was read.

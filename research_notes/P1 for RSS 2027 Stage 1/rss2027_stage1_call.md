# RSS 2027 Stage 1: the call as of 27 October 2026, and what it means for P1

Scope: the rules of the RSS 2027 two-stage submission, read on the official call
([roboticsconference.org/information/cfp](https://roboticsconference.org/information/cfp/)) and the conference home
page ([roboticsconference.org](https://roboticsconference.org/)) on 27 October 2026, and what they imply for how P1's
extended abstract is written. Review guidelines and the rebuttal template are not yet published ("in due course");
re-read the call in mid-November. Quotes are short; the rest is paraphrase.

Labels as in the other notes; everything here is from the primary source unless marked [2nd].

## 1. Format and dates

### Takeaway
Stage 1 is a 6-page PDF: at most 5 pages of content plus 1 page of references. Only the PDF is reviewed -- no
appendix, no website, no video, no supplementary material at Stage 1. Deadline **4 December 2026, 11:59 pm AoE**
(= Saturday 5 December, 12:59 in Slovakia). Stage 2 is an 8-page paper (references excluded) due 16 April 2027.

### Cited findings
- Stage 1: "6-page Extended Abstract": up to 5 pages of content and 1 page of references; content beyond the 6 pages
  (websites, appendices) "will not be considered for review" ([CfP](https://roboticsconference.org/information/cfp/)).
- Stage 2: at most 8 pages excluding references; supplementary material only at Stage 2, up to 100 MB zip
  ([CfP](https://roboticsconference.org/information/cfp/)).
- Dates (all 11:59 pm AoE): extended abstract 4 Dec 2026; Stage 1 decision (reject or invitation to rebuttal) 5 Feb
  2027; rebuttal 12 Feb; Stage 2 decision (reject or invitation to execute an Extension Charter) 26 Feb; final paper
  16 Apr; supplementary 19 Apr; acceptance 30 Apr; camera-ready 14 May; conference 6-11 July 2027, Athens
  ([CfP](https://roboticsconference.org/information/cfp/), [home](https://roboticsconference.org/)).
- PDF only, the provided LaTeX/Word templates unmodified, no "Draft" marks or page numbers; formatting changes "may
  result in a Desk Rejection"; Type-1 fonts. Submission through OpenReview; **every author needs an OpenReview
  account, and profiles without an institutional email can take up to two weeks to be moderated**
  ([CfP](https://roboticsconference.org/information/cfp/)).
- The Stage 1 author list is "generally considered final"; changes need the Program Chairs, and additions only if
  justified by the Extension Charter ([CfP](https://roboticsconference.org/information/cfp/)).

### Inferences
- Everything a reviewer must see -- the pipeline figure, the protocol, the preliminary numbers with intervals -- has to
  fit in 5 pages. Plan two figures and one table at most.
- Create OpenReview profiles for every author by ~13 November; decide the author list before 4 December.

## 2. What Stage 1 is judged on, and what Stage 2 means

### Takeaway
Stage 1 is judged on "the paper's hypothesis and methodology"; experiments "may be missing or preliminary". Invited
papers get a metareview that becomes an **Extension Charter** -- the experiments the authors must run -- and Stage 2
checks that the Charter was followed and that the work yielded "scientifically valuable insights". Negative results
are not grounds for rejection; "Rigorously falsifying a hypothesis is considered a valuable scientific contribution."
This is close to a registered report: the hypotheses and the analysis plan are the product at Stage 1.

### Cited findings
- "Reviewers and Area Chairs (ACs) will evaluate the paper's hypothesis and methodology"; the submission is a
  "semi-complete paper"; experiments "may be missing or preliminary"; the related work section should cite only work
  "directly tied to the paper's central problem and novelty" ([CfP](https://roboticsconference.org/information/cfp/)).
- Between the stages: invited papers receive a Metareview listing required clarifications and extensions; authors
  respond in the rebuttal; ACs turn it into the finalised Extension Charter, sent on 26 February; at Stage 2 the PC
  checks that the Charter was followed ([CfP](https://roboticsconference.org/information/cfp/)).
- "Submissions will not be rejected for yielding negative experimental results"; falsifying a hypothesis counts, but
  the authors must draw meaningful insight from it ([CfP](https://roboticsconference.org/information/cfp/)).
- Only "a subset of papers" go to rebuttal; no invitation or acceptance rate is published
  ([CfP](https://roboticsconference.org/information/cfp/)).
- A search summary describes the design as a cascading filter for high submission volume that keeps human oversight
  despite LLM use [2nd: search snippet of the CfP; the wording was not found on the page itself].

### Inferences
- The Stage 1 abstract should read like a pre-registration: each hypothesis with its measurement, its trial count,
  its statistical test, the outcome that would falsify it, and what we would learn either way. Reviewers can then write
  a Charter that is executable in the seven weeks between 26 February and 16 April.
- Preliminary numbers serve feasibility: they show that the pipeline runs, that the teacher's numbers are what we
  claim, and that the protocol is measurable. They do not need to confirm the hypotheses.
- Because the Charter is negotiated, do not over-promise. Every experiment listed as Stage 2 work must fit the robot
  time and GPU time we really have (a budget in `hypotheses_and_evaluation.md`, section 3).

### Gaps
- Review criteria, the reviewer form and the rebuttal template are not yet published (checked: the CfP and the home
  page; the home page links only to the CfP, venue, visa, city, organisers and PC).
- Whether Stage 1 abstracts are archival is not stated.

## 3. Anonymity, links, LLM use, references

### Takeaway
Double-blind: no names, affiliations, acknowledgements or funding; **no links to external sites (GitHub, YouTube)**;
self-citations in the third person; avoid institutional logos and lab-specific references in the experimental
description. Preprints are allowed. LLM use is allowed (disclosure "highly encouraged"); **papers with hallucinated
references are desk-rejected and automated checks are run.**

### Cited findings
- Anonymisation rules, the ban on external links in the blinded version, third-person self-citation, avoiding
  lab-specific references, desk rejection for violations; preprints permitted but may reveal identity
  ([CfP](https://roboticsconference.org/information/cfp/)).
- Dual submission: substantially similar work under review elsewhere is forbidden; non-archival venues and arXiv
  preprints are exceptions ([CfP](https://roboticsconference.org/information/cfp/)).
- LLM use permitted for authors and reviewers; disclosure encouraged; hallucinated citations -> desk rejection, with
  automated checks; authors fully responsible ([CfP](https://roboticsconference.org/information/cfp/)).

### Inferences
- For P1: name the hardware (Yahboom M3 Pro, Orin NX, DaBai DCW2 -- commercial products, not lab-identifying), but not
  the lab's machines (hekate, Dionysos), its network or its city. No project page link until camera-ready.
- Every reference must be checked against its arXiv page or proceedings before submission. Many 2026 neighbours exist
  only as arXiv abstracts; cite their exact titles and IDs. The AI-agent co-development can be disclosed in one
  sentence.
- Posting the abstract on arXiv after submission is allowed and would timestamp priority (scoop risk is medium-high),
  at the cost of anonymity. That is the user's decision.

## 4. The calendar seen from P1

### Takeaway
Five weeks to Stage 1, then ~10 weeks of quiet (Dec-early Feb) before the decision, then seven weeks from the Charter
to the final paper. The real-robot battery that the Charter will demand must be largely run **before** 26 February, in
December and January, or it will not fit.

### Cited findings
- Dates as in section 1 ([CfP](https://roboticsconference.org/information/cfp/)).
- RESEARCH.md section 5 lists the other deadlines in the same window: AGENT@ICSE 27 Nov, HRI LBR 8 Dec, IROS 1 Mar.

### Inferences
- Treat 5 Feb - 16 Apr as the window to *finish* the Charter, not to start it. Run the full battery in December and
  January under the registered protocol; if the Charter asks for more, the protocol is already in place.
